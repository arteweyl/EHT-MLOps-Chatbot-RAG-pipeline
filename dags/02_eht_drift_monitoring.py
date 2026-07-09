"""
DAG 02: Pipeline de Monitoramento de Drift Instrumental e Retreino Automatizado.

calcula o Population Stability Index (PSI) das visibilidades e
dispara o pipeline de reconstrução se houver drift atmosférico elevado.
"""

from __future__ import annotations

import os
import sys
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict
from datetime import datetime, timezone

try:
    from airflow.sdk import dag, task, get_current_context
    from airflow.providers.standard.operators.python import BranchPythonOperator
    from airflow.providers.standard.operators.empty import EmptyOperator
    from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
except ImportError:
    from airflow.decorators import dag, task
    from airflow.operators.python import BranchPythonOperator, get_current_context
    from airflow.operators.empty import EmptyOperator
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulator import EHTSimulator
from src.calibrator import EHTCalibrator

try:
    import pendulum
    START_DATE = pendulum.datetime(2026, 7, 1, tz="UTC")
except Exception:
    START_DATE = datetime(2026, 7, 1)

BASE_DIR = Path("/tmp/eht_mlops_airflow_demo")
PSI_THRESHOLD = 0.20

def current_run_date() -> str:
    try:
        context = get_current_context()
        if context.get("ds"):
            return str(context["ds"])
    except Exception:
        pass
    return datetime.now(timezone.utc).date().isoformat()

def calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Computes the Population Stability Index (PSI) between reference and current visibility amplitudes.
    """
    quantiles = np.linspace(0, 100, bins + 1)
    breakpoints = np.percentile(reference, quantiles)
    breakpoints = np.unique(breakpoints)
    
    if len(breakpoints) < 3:
        return 0.0
        
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)
    
    ref_perc = ref_counts / max(ref_counts.sum(), 1)
    cur_perc = cur_counts / max(cur_counts.sum(), 1)
    
    # Avoid division by zero
    eps = 1e-6
    ref_perc = np.where(ref_perc == 0, eps, ref_perc)
    cur_perc = np.where(cur_perc == 0, eps, cur_perc)
    
    psi_values = (cur_perc - ref_perc) * np.log(cur_perc / ref_perc)
    return float(np.sum(psi_values))


@dag(
    dag_id="02_eht_drift_monitoring",
    start_date=START_DATE,
    schedule="@daily",
    catchup=False,
    tags=["mlops", "monitoring", "drift"],
    default_args={"retries": 1},
)
def eht_drift_monitoring_dag():
    
    @task(task_id="extract_reference_data")
    def extract_reference_data() -> str:
        """
        Loads baseline reference visibility data (without calibration drifts).
        """
        run_date = current_run_date()
        run_dir = BASE_DIR / "monitoring" / f"run_date={run_date}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ref_path = run_dir / "reference_amplitudes.pkl"
        
        sim = EHTSimulator()
        baselines_uv, baseline_names = sim.generate_uv_coverage(hours=8, num_points=100)
        gt_image, _, _ = sim.generate_black_hole_model()
        true_vis = sim.sample_visibilities(gt_image, fov_uas=120, baselines_uv=zip(baselines_uv, baseline_names))
        
        # Collect all visibility amplitudes
        amplitudes = []
        for name, baseline in true_vis.items():
            amplitudes.extend(np.abs(baseline['vis']))
            
        with open(ref_path, 'wb') as f:
            pickle.dump(np.array(amplitudes), f)
            
        return str(ref_path)

    @task(task_id="extract_current_scan")
    def extract_current_scan() -> str:
        """
        Extracts current daily scan, simulating instrumental drift.
        Drift is simulated by increasing phase and amplitude gains errors.
        """
        run_date = current_run_date()
        run_dir = BASE_DIR / "monitoring" / f"run_date={run_date}"
        run_dir.mkdir(parents=True, exist_ok=True)
        scan_path = run_dir / "current_amplitudes.pkl"
        
        sim = EHTSimulator()
        baselines_uv, baseline_names = sim.generate_uv_coverage(hours=8, num_points=100)
        gt_image, _, _ = sim.generate_black_hole_model()
        true_vis = sim.sample_visibilities(gt_image, fov_uas=120, baselines_uv=zip(baselines_uv, baseline_names))
        
        # Corrupt data with significant drift (high station phase/amplitude errors)
        calibrator = EHTCalibrator(random_seed=int(datetime.now().timestamp()) % 1000)
        
        # Simulate severe atmospheric drift (higher gain errors and thermal noise)
        corrupted = calibrator.corrupt_visibilities(true_vis, thermal_noise_level=0.08, station_phase_noise_std=2.2)
        
        amplitudes = []
        for name, baseline in corrupted.items():
            amplitudes.extend(np.abs(baseline['vis']))
            
        with open(scan_path, 'wb') as f:
            pickle.dump(np.array(amplitudes), f)
            
        return str(scan_path)

    @task(task_id="calculate_drift_psi")
    def calculate_drift_psi(ref_path: str, scan_path: str) -> Dict[str, float | str]:
        """
        Computes the PSI of current scan visibilities against the reference data.
        """
        with open(ref_path, 'rb') as f:
            reference = pickle.load(f)
            
        with open(scan_path, 'rb') as f:
            current = pickle.load(f)
            
        psi = calculate_psi(reference, current, bins=10)
        
        monitoring_report = {
            'psi': psi,
            'threshold': PSI_THRESHOLD,
            'status': 'ALERT' if psi >= PSI_THRESHOLD else 'OK',
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        report_path = Path(scan_path).parent / "drift_report.json"
        report_path.write_text(json.dumps(monitoring_report, indent=2), encoding="utf-8")
        
        # Write temporary visualizer file
        web_monitor_path = BASE_DIR / 'last_monitor_web.json'
        with open(web_monitor_path, 'w') as f:
            json.dump(monitoring_report, f, indent=4)
            
        print(f"Computed PSI: {psi:.4f}. Status: {monitoring_report['status']}")
        return monitoring_report

    def choose_next_step(**context) -> str:
        ti = context["ti"]
        report = ti.xcom_pull(task_ids="calculate_drift_psi")
        
        if report["psi"] >= PSI_THRESHOLD:
            print(f"Drift detected (PSI = {report['psi']:.4f} >= {PSI_THRESHOLD}). Triggering Retraining...")
            return "trigger_retraining"
        else:
            print(f"Drift in acceptable limits (PSI = {report['psi']:.4f}). Skipping retraining.")
            return "drift_ok"

    # Tasks instantiations
    ref_p = extract_reference_data()
    scan_p = extract_current_scan()
    report = calculate_drift_psi(ref_p, scan_p)
    
    quality_gate = BranchPythonOperator(
        task_id="drift_gate",
        python_callable=choose_next_step,
    )
    
    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="01_eht_image_reconstruction",
        conf={"retrain_reason": "Calibration drift detected via PSI monitoring"},
    )
    
    drift_ok = EmptyOperator(task_id="drift_ok")
    
    # Dependencies
    report >> quality_gate
    quality_gate >> [trigger_retraining, drift_ok]

# Instantiate DAG
eht_drift_monitoring_dag()
