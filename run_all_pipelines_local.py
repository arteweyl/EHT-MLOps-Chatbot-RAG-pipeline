import os
import sys
import json
import numpy as np
import pickle
from datetime import datetime, timezone

# Add current folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.simulator import EHTSimulator
from src.calibrator import EHTCalibrator
from src.reconstruction import EHTReconstructor
from src.evaluator import EHTEvaluator
from src.mlops_registry import EHTModelRegistry
from src.rag_index import EHTCatalogRAGIndex

BASE_DIR = "/tmp/eht_mlops_airflow_demo"
os.makedirs(BASE_DIR, exist_ok=True)

# Shared data paths
ingest_path = os.path.join(BASE_DIR, 'ingested_data.pkl')
cal_path = os.path.join(BASE_DIR, 'calibrated_data.pkl')
recon_path = os.path.join(BASE_DIR, 'reconstructed_image.pkl')
report_path = os.path.join(BASE_DIR, 'evaluation_report.pkl')

def run_pipeline_1(simulate_drift=False):
    """
    Runs Pipeline 1: EHT Image Reconstruction with Quality Gate.
    """
    print("\n" + "="*50)
    print("RUNNING PIPELINE 01: IMAGE RECONSTRUCTION")
    print("="*50)
    
    # 1. Ingestion
    print("[1/5] Ingesting VLBI observation tracks...")
    sim = EHTSimulator()
    baselines_uv, baseline_names = sim.generate_uv_coverage(hours=8, num_points=100)
    gt_image, _, _ = sim.generate_black_hole_model(
        grid_size=64, fov_uas=120, ring_rad_uas=20.0, ring_width_uas=4.0, asymmetry=0.5, phi_0_deg=135
    )
    true_vis = sim.sample_visibilities(gt_image, fov_uas=120, baselines_uv=zip(baselines_uv, baseline_names))
    
    with open(ingest_path, 'wb') as f:
        pickle.dump({
            'gt_image': gt_image,
            'true_visibilities': true_vis,
            'fov_uas': 120,
            'grid_size': 64
        }, f)
        
    # 1b. Validation Schema
    print("[1b/5] Validating Data Schema...")
    if len(true_vis) != 15:
        raise ValueError("Schema validation failed: Missing baseline antennas data.")
    print("--> Data schema is valid.")
        
    # 2. Calibration
    print("[2/5] Calibrating data (Atmosphere noise + Phase Self-Calibration)...")
    cal = EHTCalibrator(random_seed=42)
    
    # If simulating drift, we double the atmospheric noise to simulate a bad calibration
    noise_level = 0.08 if simulate_drift else 0.03
    phase_std = 2.2 if simulate_drift else 1.2
    
    corrupted_vis = cal.corrupt_visibilities(true_vis, thermal_noise_level=noise_level, station_phase_noise_std=phase_std)
    calibrated_vis = cal.self_calibrate(corrupted_vis, model_image=gt_image, fov_uas=120)
    
    with open(cal_path, 'wb') as f:
        pickle.dump({
            'corrupted_vis': corrupted_vis,
            'calibrated_vis': calibrated_vis,
            'fov_uas': 120
        }, f)
        
    # 3. Reconstruction
    print("[3/5] Reconstructing image via RML optimization (Analytical Gradients)...")
    recon = EHTReconstructor(grid_size=64, fov_uas=120)
    reconstructed, loss = recon.reconstruct(
        calibrated_data=calibrated_vis,
        alpha_tv=0.05,
        alpha_entropy=0.005,
        max_iter=60
    )
    
    with open(recon_path, 'wb') as f:
        pickle.dump({
            'reconstructed_image': reconstructed,
            'alpha_tv': 0.05,
            'alpha_entropy': 0.005,
            'loss': loss
        }, f)
        
    # 4. Evaluation
    print("[4/5] Evaluating image fidelity and fitting GRMHD template library...")
    evaluator = EHTEvaluator(grid_size=64, fov_uas=120)
    metrics = evaluator.calculate_metrics(reconstructed, gt_image)
    
    grmhd_lib = evaluator.generate_grmhd_library()
    fit_result = evaluator.fit_grmhd_model(reconstructed, grmhd_lib)
    
    eval_data = {
        'metrics': metrics,
        'fit_result': {
            'estimated_mass_10_9': float(fit_result['estimated_mass_10_9']),
            'estimated_spin': float(fit_result['estimated_spin']),
            'fit_correlation': float(fit_result['fit_correlation'])
        }
    }
    
    with open(report_path, 'wb') as f:
        pickle.dump(eval_data, f)
        
    # 5. Registry & Quality Gate Promotion
    print("[5/5] Checking MLOps Quality Gates and registering model...")
    registry = EHTModelRegistry(registry_dir=BASE_DIR)
    
    # Gate check
    fid = metrics['fidelity_score']
    fit_corr = fit_result['fit_correlation']
    mass = fit_result['estimated_mass_10_9']
    
    is_valid = (fid >= 0.70 and fit_corr >= 0.75 and 5.0 <= mass <= 8.0)
    
    if is_valid:
        run_record = registry.log_run(
            parameters={'alpha_tv': 0.05, 'alpha_entropy': 0.005, 'calibration': 'Phase-Self-Cal'},
            metrics=metrics,
            reconstructed_image=reconstructed,
            fit_result=eval_data['fit_result']
        )
        status = 'PRODUCTION'
        promo_log = run_record['promotion_log']
        run_id = run_record['run_id']
    else:
        status = 'FAILED_VALIDATION'
        promo_log = "Reconstruction NCC or GRMHD correlation is too low. Mass range checks failed."
        run_id = f"failed_run_{int(datetime.now().timestamp())}"
        
    # Write metadata to root for local UI access
    web_data = {
        'run_id': run_id,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': status,
        'metrics': metrics,
        'fit_result': eval_data['fit_result'],
        'promotion_log': promo_log
    }
    
    # Save both in root (for UI) and in BASE_DIR (for execution backup)
    for folder in [BASE_DIR, os.path.dirname(os.path.abspath(__file__))]:
        with open(os.path.join(folder, 'last_run_web.json'), 'w') as f:
            json.dump(web_data, f, indent=4)
            
    print(f"--> Pipeline 1 completed. Run Status: {status}. Fidelity NCC: {metrics['fidelity_score']:.4f}")
    return web_data


def run_pipeline_2(simulate_severe_drift=True):
    """
    Runs Pipeline 2: Drift Monitoring.
    """
    print("\n" + "="*50)
    print("RUNNING PIPELINE 02: DRIFT MONITORING")
    print("="*50)
    
    # 1. Load reference amplitudes
    print("[1/3] Loading calibration reference baseline...")
    sim = EHTSimulator()
    baselines_uv, baseline_names = sim.generate_uv_coverage(hours=8, num_points=100)
    gt_image, _, _ = sim.generate_black_hole_model()
    true_vis = sim.sample_visibilities(gt_image, fov_uas=120, baselines_uv=zip(baselines_uv, baseline_names))
    
    ref_amplitudes = []
    for name, b in true_vis.items():
        ref_amplitudes.extend(np.abs(b['vis']))
    ref_amplitudes = np.array(ref_amplitudes)
    
    # 2. Extract current scan
    print("[2/3] Capturing current telescope scan visibilities...")
    calibrator = EHTCalibrator(random_seed=123)
    
    # Severe atmospheric noise creates amplitude/phase drift
    noise = 0.09 if simulate_severe_drift else 0.03
    phase_std = 2.4 if simulate_severe_drift else 1.1
    
    corrupted = calibrator.corrupt_visibilities(true_vis, thermal_noise_level=noise, station_phase_noise_std=phase_std)
    
    cur_amplitudes = []
    for name, b in corrupted.items():
        cur_amplitudes.extend(np.abs(b['vis']))
    cur_amplitudes = np.array(cur_amplitudes)
    
    # 3. Calculate PSI
    print("[3/3] Calculating Population Stability Index (PSI) drift metric...")
    # Inline definition of calculate_psi to avoid dynamic imports
    quantiles = np.linspace(0, 100, 11)
    breakpoints = np.percentile(ref_amplitudes, quantiles)
    breakpoints = np.unique(breakpoints)
    
    ref_counts, _ = np.histogram(ref_amplitudes, bins=breakpoints)
    cur_counts, _ = np.histogram(cur_amplitudes, bins=breakpoints)
    
    ref_perc = ref_counts / max(ref_counts.sum(), 1)
    cur_perc = cur_counts / max(cur_counts.sum(), 1)
    
    eps = 1e-6
    ref_perc = np.where(ref_perc == 0, eps, ref_perc)
    cur_perc = np.where(cur_perc == 0, eps, cur_perc)
    
    psi_values = (cur_perc - ref_perc) * np.log(cur_perc / ref_perc)
    psi = float(np.sum(psi_values))
    
    status = 'ALERT' if psi >= 0.20 else 'OK'
    
    monitoring_report = {
        'psi': psi,
        'threshold': 0.20,
        'status': status,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save both in root (for UI) and in BASE_DIR (for backup)
    for folder in [BASE_DIR, os.path.dirname(os.path.abspath(__file__))]:
        with open(os.path.join(folder, 'last_monitor_web.json'), 'w') as f:
            json.dump(monitoring_report, f, indent=4)
            
    print(f"--> Pipeline 2 completed. Drift PSI: {psi:.4f}. Status: {status}")
    
    # Trigger training DAG if PSI >= threshold
    if status == 'ALERT':
        print("\n[MLOps Trigger] Calibration Drift detected (PSI >= 0.20). TriggerDagRunOperator fired!")
        run_pipeline_1(simulate_drift=False) # trigger retraining/re-calibration
    else:
        print("\n[MLOps Trigger] Drift is OK (PSI < 0.20). Retraining skipped.")
        
    return monitoring_report


def run_pipeline_3():
    """
    Runs Pipeline 3: RAG Catalog Index Refresh.
    """
    print("\n" + "="*50)
    print("RUNNING PIPELINE 03: RAG CATALOG REFRESH")
    print("="*50)
    
    # 1. Read latest approved model
    print("[1/3] Fetching latest approved model from Registry...")
    registry_db_path = os.path.join(BASE_DIR, 'model_registry.json')
    if not os.path.exists(registry_db_path):
        print("--> No approved model found in registry database. Running Pipeline 1 first...")
        run_pipeline_1(simulate_drift=False)
        
    with open(registry_db_path, 'r') as f:
        db = json.load(f)
        
    prod_run_id = db['active_production_model']
    prod_run = next((r for r in db['runs'] if r['run_id'] == prod_run_id), None)
    
    # 2. Add to RAG Index (embeddings generation)
    print("[2/3] Chunking text metadata and generating semantic embeddings...")
    rag = EHTCatalogRAGIndex(index_dir=BASE_DIR)
    num_chunks = rag.add_model_to_index(
        run_id=prod_run['run_id'],
        parameters=prod_run['parameters'],
        metrics=prod_run['metrics'],
        fit_result=prod_run['fit_result']
    )
    
    # Copy index file to root for easy UI queries
    root_dir = os.path.dirname(os.path.abspath(__file__))
    import shutil
    shutil.copy(rag.index_path, os.path.join(root_dir, 'eht_rag_index.json'))
    
    # 3. Freshness Validation
    print("[3/3] Validating database RAG index freshness...")
    # Check if latest run is in RAG docs
    with open(rag.index_path, 'r') as f:
        index_data = json.load(f)
        
    contains_run = any(doc['metadata']['run_id'] == prod_run['run_id'] for doc in index_data['documents'])
    if not contains_run:
        raise ValueError("RAG Index validation failed: fresh documents missing.")
        
    status_report = {
        'fresh': True,
        'index_path': os.path.join(root_dir, 'eht_rag_index.json'),
        'document_count': len(index_data['documents']),
        'last_run_id': prod_run['run_id'],
        'validated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save both in root (for UI) and in BASE_DIR (for backup)
    for folder in [BASE_DIR, root_dir]:
        with open(os.path.join(folder, 'last_rag_web.json'), 'w') as f:
            json.dump(status_report, f, indent=4)
            
    print(f"--> Pipeline 3 completed. RAG index refreshed. Total documents indexed: {len(index_data['documents'])}.")
    return status_report


if __name__ == "__main__":
    # Run all 3 pipelines in order
    run_pipeline_1(simulate_drift=False)
    run_pipeline_2(simulate_severe_drift=True)
    run_pipeline_3()
    print("\nAll 3 Pipelines ran successfully locally!")
