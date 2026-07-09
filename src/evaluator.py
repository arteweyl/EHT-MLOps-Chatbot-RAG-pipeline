import numpy as np

class EHTEvaluator:
    """
    Evaluates the reconstructed image quality and performs model matching
    against a grid of theoretical GRMHD (General Relativistic Magnetohydrodynamic) simulation templates
    to infer the physical properties of the black hole (Mass and Spin).
    Ref: [5] Physical Origin of the Asymmetric Ring, [6] The Shadow and Mass of the Central Black Hole.
    """
    def __init__(self, grid_size=64, fov_uas=120):
        self.grid_size = grid_size
        self.fov_uas = fov_uas
        
    def calculate_metrics(self, reconstructed, ground_truth):
        """
        Calculates image reconstruction quality metrics: Mean Squared Error (MSE),
        Normalized Cross-Correlation (NCC), and structural similarity.
        """
        # Ensure normalization
        recon_norm = reconstructed / np.sum(reconstructed)
        gt_norm = ground_truth / np.sum(ground_truth)
        
        # Mean Squared Error (MSE)
        mse = np.mean((recon_norm - gt_norm)**2)
        
        # Normalized Cross-Correlation (NCC)
        numerator = np.sum((recon_norm - np.mean(recon_norm)) * (gt_norm - np.mean(gt_norm)))
        denominator = np.sqrt(np.sum((recon_norm - np.mean(recon_norm))**2) * np.sum((gt_norm - np.mean(gt_norm))**2))
        ncc = numerator / denominator if denominator > 0 else 0.0
        
        # Simple structural similarity surrogate (pixel correlation)
        ssim_surrogate = max(0.0, ncc) # simplified for didactic purposes
        
        return {
            'mse': mse,
            'ncc': ncc,
            'fidelity_score': ssim_surrogate
        }
        
    def generate_grmhd_library(self):
        """
        Generates a grid of synthetic GRMHD template images representing different physical parameters.
        Parameters:
          - Mass (M): in units of 10^9 Solar Masses (controls ring radius)
            Values: [5.5, 6.0, 6.5, 7.0, 7.5]
          - Spin (a): dimensionless spin parameter (controls asymmetry / Doppler beaming)
            Values: [-0.94, 0.0, 0.5, 0.94]
        """
        from .simulator import EHTSimulator
        sim = EHTSimulator()
        
        library = []
        
        # Grid values
        mass_grid = [5.5, 6.0, 6.5, 7.0, 7.5] # 10^9 Solar Masses
        spin_grid = [-0.94, 0.0, 0.5, 0.94] # -1 to 1 (negative indicates retrograde accretion disk)
        
        # Base scale factor: M87* shadow size is ~42 micro-arcseconds (radius ~21 uas) for M = 6.5e9 M_sun
        base_mass = 6.5
        base_radius = 20.0
        
        for M in mass_grid:
            # Radius is directly proportional to Mass (R = 2 G M / c^2)
            ring_rad = base_radius * (M / base_mass)
            
            for a in spin_grid:
                # Asymmetry increases with spin (Doppler beaming of frame dragging)
                # Spin also dictates position angle offset slightly, but let's map it directly to asymmetry
                asymmetry = 0.2 + 0.6 * np.abs(a)
                # Retrograde vs prograde spins affect brightness peak direction (relative to disk rotation)
                phi_0 = 135.0 if a >= 0 else 315.0
                
                # Generate model
                template, _, _ = sim.generate_black_hole_model(
                    grid_size=self.grid_size,
                    fov_uas=self.fov_uas,
                    ring_rad_uas=ring_rad,
                    ring_width_uas=4.5,
                    asymmetry=asymmetry,
                    phi_0_deg=phi_0
                )
                
                library.append({
                    'mass': M,
                    'spin': a,
                    'image': template
                })
                
        return library

    def fit_grmhd_model(self, reconstructed, library):
        """
        Fits the reconstructed image to the GRMHD template library by finding the model that
        maximizes cross-correlation (or minimizes MSE).
        This is a template matching regression workflow to estimate black hole properties.
        """
        best_score = -1.0
        best_model = None
        
        # Normalize reconstructed image
        recon_norm = reconstructed - np.mean(reconstructed)
        recon_std = np.std(reconstructed)
        
        for model in library:
            template = model['image']
            temp_norm = template - np.mean(template)
            temp_std = np.std(template)
            
            # Cross correlation
            if recon_std > 0 and temp_std > 0:
                corr = np.mean(recon_norm * temp_norm) / (recon_std * temp_std)
            else:
                corr = -1.0
                
            if corr > best_score:
                best_score = corr
                best_model = model
                
        return {
            'estimated_mass_10_9': best_model['mass'],
            'estimated_spin': best_model['spin'],
            'fit_correlation': best_score,
            'best_template_image': best_model['image']
        }

if __name__ == "__main__":
    from simulator import EHTSimulator
    sim = EHTSimulator()
    img, _, _ = sim.generate_black_hole_model(grid_size=64, ring_rad_uas=20.0, asymmetry=0.5, phi_0_deg=135)
    
    evaluator = EHTEvaluator(grid_size=64)
    lib = evaluator.generate_grmhd_library()
    print(f"Generated {len(lib)} GRMHD templates in the library.")
    
    # We will "corrupt" the ground truth slightly and evaluate
    noisy_img = img + np.random.normal(0, 0.005, img.shape)
    noisy_img = np.maximum(0, noisy_img)
    noisy_img = noisy_img / np.sum(noisy_img)
    
    metrics = evaluator.calculate_metrics(noisy_img, img)
    print("Metrics:")
    print(f"  MSE: {metrics['mse']:.6e}")
    print(f"  NCC: {metrics['ncc']:.4f}")
    
    fit = evaluator.fit_grmhd_model(noisy_img, lib)
    print("GRMHD Fitting Result:")
    print(f"  Estimated Mass: {fit['estimated_mass_10_9']} billion solar masses (Expected ~6.5)")
    print(f"  Estimated Spin: {fit['estimated_spin']} (Expected 0.5)")
    print(f"  Fit Correlation: {fit['fit_correlation']:.4f}")
