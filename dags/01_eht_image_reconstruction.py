"""
DAG 01: Pipeline de Otimização e Reconstrução da Imagem (RML) com Quality Gate.
"""

from __future__ import annotations

import os
import sys
import pickle
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone

try:
    from airflow.sdk import dag, task, get_current_context
    from airflow.providers.standard.operators.python import BranchPythonOperator
except ImportError:
    from airflow.decorators import dag, task
    from airflow.operators.python import BranchPythonOperator, get_current_context

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulator import EHTSimulator
from src.calibrator import EHTCalibrator
from src.reconstruction import EHTReconstructor
from src.evaluator import EHTEvaluator
from src.mlops_registry import EHTModelRegistry

try:
    import pendulum
    START_DATE = pendulum.datetime(2026, 7, 1, tz="UTC")
except Exception:
    START_DATE = datetime(2026, 7, 1)

BASE_DIR = Path("/tmp/eht_mlops_airflow_demo")

def current_run_date() -> str:
    try:
        context = get_current_context()
        if context.get("ds"):
            return str(context["ds"])
        for key in ("logical_date", "data_interval_start", "run_after"):
            value = context.get(key)
            if value:
                if hasattr(value, "to_date_string"):
                    return value.to_date_string()
                return value.date().isoformat()
    except Exception:
        pass
    return datetime.now(timezone.utc).date().isoformat()


@dag(
    dag_id="01_eht_image_reconstruction",
    start_date=START_DATE,
    schedule=None, # Triggered manually or by Drift Monitor
    catchup=False,
    tags=["mlops", "reconstruction", "quality_gate"],
    default_args={"retries": 1},
)
def eht_image_reconstruction_dag():
    
    @task(task_id="ingest_vlbi_data")
    def ingest_vlbi_data() -> str:
        run_date = current_run_date()
        run_dir = BASE_DIR / f"run_date={run_date}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ingest_filepath = run_dir / "ingested_data.pkl"
        
        sim = EHTSimulator()
        baselines_uv, baseline_names = sim.generate_uv_coverage(hours=8, num_points=100)
        gt_image, _, _ = sim.generate_black_hole_model(
            grid_size=64, fov_uas=120, ring_rad_uas=20.0, ring_width_uas=4.0, asymmetry=0.5, phi_0_deg=135
        )
        true_vis = sim.sample_visibilities(gt_image, fov_uas=120, baselines_uv=zip(baselines_uv, baseline_names))
        
        with open(ingest_filepath, 'wb') as f:
            pickle.dump({
                'gt_image': gt_image,
                'true_visibilities': true_vis,
                'fov_uas': 120,
                'grid_size': 64
            }, f)
            
        print(f"VLBI Data Ingestion completed. Saved to {ingest_filepath}.")
        return str(ingest_filepath)

    @task(task_id="validate_data_schema")
    def validate_data_schema(ingest_filepath: str) -> str:
        import numpy as np
        with open(ingest_filepath, 'rb') as f:
            data = pickle.load(f)
        
        true_vis = data['true_visibilities']
        if data['grid_size'] != 64 or data['fov_uas'] != 120:
            raise ValueError("Schema validation failed: incorrect image matrix grid parameters.")
            
        if 'ALMA-LMT' not in true_vis:
            raise ValueError("Schema validation failed: Missing critical ALMA-LMT baseline.")
            
        for name, baseline in true_vis.items():
            if np.isnan(baseline['vis']).any():
                raise ValueError(f"NaN values detected in baseline {name}")
                
        print("Schema validation successful.")
        return ingest_filepath

    @task(task_id="calibrate_data")
    def calibrate_data(ingest_filepath: str) -> str:
        with open(ingest_filepath, 'rb') as f:
            data = pickle.load(f)
            
        true_vis = data['true_visibilities']
        gt_image = data['gt_image']
        fov_uas = data['fov_uas']
        
        calibrator = EHTCalibrator(random_seed=42)
        corrupted_vis = calibrator.corrupt_visibilities(true_vis, thermal_noise_level=0.03, station_phase_noise_std=1.2)
        closure_phases = calibrator.compute_closure_phases(corrupted_vis)
        calibrated_vis = calibrator.self_calibrate(corrupted_vis, model_image=gt_image, fov_uas=fov_uas)
        
        cal_filepath = Path(ingest_filepath).parent / "calibrated_data.pkl"
        with open(cal_filepath, 'wb') as f:
            pickle.dump({
                'corrupted_vis': corrupted_vis,
                'closure_phases': closure_phases,
                'calibrated_vis': calibrated_vis,
                'fov_uas': fov_uas
            }, f)
            
        print(f"Calibration completed. Saved to {cal_filepath}.")
        return str(cal_filepath)

    @task(task_id="reconstruct_image")
    def reconstruct_image(cal_filepath: str) -> str:
        with open(cal_filepath, 'rb') as f:
            data = pickle.load(f)
            
        calibrated_vis = data['calibrated_vis']
        fov_uas = data['fov_uas']
        
        recon = EHTReconstructor(grid_size=64, fov_uas=fov_uas)
        reconstructed_image, final_loss = recon.reconstruct(
            calibrated_data=calibrated_vis,
            alpha_tv=0.05,
            alpha_entropy=0.005,
            max_iter=60
        )
        
        recon_filepath = Path(cal_filepath).parent / "reconstructed_image.pkl"
        with open(recon_filepath, 'wb') as f:
            pickle.dump({
                'reconstructed_image': reconstructed_image,
                'alpha_tv': 0.05,
                'alpha_entropy': 0.005,
                'loss': final_loss
            }, f)
            
        print(f"Reconstruction completed. Loss: {final_loss:.6f}")
        return str(recon_filepath)

    @task(task_id="evaluate_reconstruction")
    def evaluate_reconstruction(recon_filepath: str, ingest_filepath: str) -> Dict[str, Any]:
        with open(ingest_filepath, 'rb') as f:
            ingest_data = pickle.load(f)
        gt_image = ingest_data['gt_image']
        fov_uas = ingest_data['fov_uas']
        
        with open(recon_filepath, 'rb') as f:
            recon_data = pickle.load(f)
        reconstructed = recon_data['reconstructed_image']
        
        evaluator = EHTEvaluator(grid_size=64, fov_uas=fov_uas)
        metrics = evaluator.calculate_metrics(reconstructed, gt_image)
        
        grmhd_lib = evaluator.generate_grmhd_library()
        fit_result = evaluator.fit_grmhd_model(reconstructed, grmhd_lib)
        
        eval_data = {
            'metrics': metrics,
            'fit_result': {
                'estimated_mass_10_9': float(fit_result['estimated_mass_10_9']),
                'estimated_spin': float(fit_result['estimated_spin']),
                'fit_correlation': float(fit_result['fit_correlation']),
            },
            'reconstructed_image_path': recon_filepath
        }
        
        report_filepath = Path(recon_filepath).parent / "evaluation_report.pkl"
        with open(report_filepath, 'wb') as f:
            pickle.dump(eval_data, f)
            
        metrics_json_path = Path(recon_filepath).parent / "metrics.json"
        metrics_json_path.write_text(json.dumps(eval_data, indent=2), encoding="utf-8")
        
        return eval_data

    def choose_next_step(**context) -> str:
        ti = context["ti"]
        eval_data = ti.xcom_pull(task_ids="evaluate_reconstruction")
        
        fid = eval_data["metrics"]["fidelity_score"]
        fit_corr = eval_data["fit_result"]["fit_correlation"]
        mass = eval_data["fit_result"]["estimated_mass_10_9"]
        
        is_valid = (fid >= 0.70 and fit_corr >= 0.75 and 5.0 <= mass <= 8.0)
        if is_valid:
            return "register_model"
        return "reject_model"

    @task(task_id="register_model")
    def register_model(eval_data: Dict[str, Any]) -> str:
        recon_filepath = eval_data['reconstructed_image_path']
        with open(recon_filepath, 'rb') as f:
            recon_data = pickle.load(f)
        reconstructed = recon_data['reconstructed_image']
        
        registry = EHTModelRegistry(registry_dir=str(BASE_DIR))
        run_record = registry.log_run(
            parameters={
                'alpha_tv': 0.05,
                'alpha_entropy': 0.005,
                'observation_date': current_run_date(),
                'calibration_method': 'Phase-Self-Cal'
            },
            metrics=eval_data['metrics'],
            reconstructed_image=reconstructed,
            fit_result=eval_data['fit_result']
        )
        
        web_data_path = BASE_DIR / 'last_run_web.json'
        web_data = {
            'run_id': run_record['run_id'],
            'timestamp': run_record['timestamp'],
            'status': 'PRODUCTION',
            'metrics': eval_data['metrics'],
            'fit_result': eval_data['fit_result'],
            'promotion_log': run_record['promotion_log']
        }
        with open(web_data_path, 'w') as f:
            json.dump(web_data, f, indent=4)
            
        print("Model promoted to PRODUCTION.")
        return run_record['run_id']

    @task(task_id="reject_model")
    def reject_model(eval_data: Dict[str, Any]) -> str:
        recon_filepath = eval_data['reconstructed_image_path']
        rejection_report = {
            'status': 'FAILED_VALIDATION',
            'reason': 'Reconstruction NCC below threshold or Mass bounds violated.',
            'metrics': eval_data['metrics'],
            'fit_result': eval_data['fit_result']
        }
        
        report_path = Path(recon_filepath).parent / "rejection_report.json"
        report_path.write_text(json.dumps(rejection_report, indent=2), encoding="utf-8")
        
        web_data_path = BASE_DIR / 'last_run_web.json'
        web_data = {
            'run_id': f"failed_run_{int(datetime.now(timezone.utc).timestamp())}",
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'FAILED_VALIDATION',
            'metrics': eval_data['metrics'],
            'fit_result': eval_data['fit_result'],
            'promotion_log': rejection_report['reason']
        }
        with open(web_data_path, 'w') as f:
            json.dump(web_data, f, indent=4)
            
        print("Model REJECTED.")
        return str(report_path)

    # DAG Connections
    ingest_path = ingest_vlbi_data()
    validated_path = validate_data_schema(ingest_path)
    cal_path = calibrate_data(validated_path)
    recon_path = reconstruct_image(cal_path)
    eval_d = evaluate_reconstruction(recon_path, ingest_path)
    
    quality_gate = BranchPythonOperator(
        task_id="quality_gate",
        python_callable=choose_next_step,
    )
    
    approved = register_model(eval_d)
    rejected = reject_model(eval_d)
    
    eval_d >> quality_gate
    quality_gate >> [approved, rejected]

# DAG Instantiation
eht_image_reconstruction_dag()
