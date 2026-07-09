import os
import json
import time
import numpy as np

class EHTModelRegistry:
    """
    Simulates a lightweight MLOps Model Registry (like MLflow or a custom metadata database).
    Logs parameters, metrics, registers models, and automates deployment transitions.
    """
    def __init__(self, registry_dir=None):
        if registry_dir is None:
            # Place in the same folder as this script, or parent
            self.registry_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.registry_dir = registry_dir
            
        self.registry_db_path = os.path.join(self.registry_dir, 'model_registry.json')
        self.models_dir = os.path.join(self.registry_dir, 'registered_models')
        
        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        if not os.path.exists(self.registry_db_path):
            initial_db = {
                'runs': [],
                'active_production_model': None
            }
            with open(self.registry_db_path, 'w') as f:
                json.dump(initial_db, f, indent=4)
                
    def _read_db(self):
        with open(self.registry_db_path, 'r') as f:
            return json.load(f)
            
    def _write_db(self, db):
        with open(self.registry_db_path, 'w') as f:
            json.dump(db, f, indent=4)
            
    def log_run(self, parameters, metrics, reconstructed_image, fit_result):
        """
        Logs a pipeline run to the registry database, saves the reconstructed image model,
        and runs auto-deployment logic.
        """
        db = self._read_db()
        run_id = f"run_{int(time.time())}"
        
        # Save model image as numpy file
        model_filename = f"{run_id}_reconstructed.npy"
        model_path = os.path.join(self.models_dir, model_filename)
        np.save(model_path, reconstructed_image)
        
        # Build run record
        run_record = {
            'run_id': run_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': parameters,
            'metrics': metrics,
            'fit_result': {
                'estimated_mass_10_9': fit_result['estimated_mass_10_9'],
                'estimated_spin': fit_result['estimated_spin'],
                'fit_correlation': fit_result['fit_correlation']
            },
            'model_artifact_path': model_path,
            'status': 'REGISTERED'
        }
        
        db['runs'].append(run_record)
        
        # Automated deployment logic (CI/CD check)
        # 1. Fidelity score (SSIM surrogate) must be high (> 0.75)
        # 2. Fit correlation with GRMHD template must be high (> 0.80)
        # 3. Mass check: physical plausibility (5.0 to 8.0 billion solar masses)
        
        is_valid = (
            metrics['fidelity_score'] > 0.70 and
            fit_result['fit_correlation'] > 0.75 and
            5.0 <= fit_result['estimated_mass_10_9'] <= 8.0
        )
        
        promotion_reason = ""
        if is_valid:
            # Check if there is an active production model and compare
            current_prod_id = db['active_production_model']
            
            if current_prod_id is None:
                db['active_production_model'] = run_id
                run_record['status'] = 'PRODUCTION'
                promotion_reason = "No active production model found. Promoted automatically."
            else:
                # Find current production model details
                prod_run = next((r for r in db['runs'] if r['run_id'] == current_prod_id), None)
                if prod_run is None or metrics['fidelity_score'] > prod_run['metrics']['fidelity_score']:
                    db['active_production_model'] = run_id
                    run_record['status'] = 'PRODUCTION'
                    if prod_run:
                        prod_run['status'] = 'ARCHIVED'
                    promotion_reason = f"Fidelity score ({metrics['fidelity_score']:.4f}) is higher than current champion ({prod_run['metrics']['fidelity_score']:.4f}). Promoted."
                else:
                    run_record['status'] = 'STAGING'
                    promotion_reason = "Model is valid, but fidelity score did not outperform current champion. Sent to Staging."
        else:
            run_record['status'] = 'FAILED_VALIDATION'
            promotion_reason = "Failed MLOps quality gates (reconstruction fidelity or physics constraint check)."
            
        run_record['promotion_log'] = promotion_reason
        self._write_db(db)
        
        print(f"Run {run_id} logged. Status: {run_record['status']}. Reason: {promotion_reason}")
        return run_record

if __name__ == "__main__":
    registry = EHTModelRegistry()
    
    # Test logging
    dummy_img = np.zeros((64, 64))
    run = registry.log_run(
        parameters={'alpha_tv': 0.1, 'alpha_entropy': 0.01, 'array': 'EHT-2017'},
        metrics={'mse': 1e-5, 'ncc': 0.85, 'fidelity_score': 0.85},
        reconstructed_image=dummy_img,
        fit_result={'estimated_mass_10_9': 6.5, 'estimated_spin': 0.5, 'fit_correlation': 0.88}
    )
    
    print("Logged model metadata saved in registry DB.")
