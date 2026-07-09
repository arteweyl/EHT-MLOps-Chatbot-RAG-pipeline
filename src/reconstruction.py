import numpy as np
from scipy.optimize import minimize

class EHTReconstructor:
    """
    Reconstructs the black hole image from sparse VLBI visibility data using Regularized Maximum Likelihood (RML).
    Ref: [4] Imaging the Central Supermassive Black Hole, [7] Computational Imaging for VLBI Image Reconstruction.
    """
    def __init__(self, grid_size=64, fov_uas=120):
        self.grid_size = grid_size
        self.fov_uas = fov_uas
        
    def _compute_model_vis(self, image_1d, baselines, u_grid, fft_shift_idx):
        """
        Computes the model complex visibilities for a given candidate image (1D array).
        """
        image_2d = image_1d.reshape((self.grid_size, self.grid_size))
        
        # FFT of candidate image
        # Note: we use np.fft.fft2 and shift it
        fft_img = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image_2d)))
        
        model_vis = {}
        for name, u_pts, v_pts in baselines:
            # Interpolate FFT values
            vis_list = []
            for u, v in zip(u_pts, v_pts):
                u_idx = np.argmin(np.abs(u_grid - u))
                v_idx = np.argmin(np.abs(u_grid - v))
                vis_list.append(fft_img[v_idx, u_idx])
            model_vis[name] = np.array(vis_list)
            
        return model_vis

    def reconstruct(self, calibrated_data, alpha_tv=0.1, alpha_entropy=0.01, max_iter=40):
        """
        Reconstructs the image by solving an optimization problem:
        Minimize: Chi^2_data + alpha_tv * TV(I) + alpha_entropy * Entropy(I)
        Subject to: I_xy >= 0 (positivity constraint)
        """
        # Prepare baselines and grid coordinates
        baselines = []
        for name, data in calibrated_data.items():
            baselines.append((name, data['u'], data['v']))
            
        # Define spatial frequency grid coordinates
        # Matching simulator resolution
        uas_to_rad = 4.8481368e-12
        dx = self.fov_uas * uas_to_rad / self.grid_size
        u_max = 1.0 / (2.0 * dx)
        u_grid = np.linspace(-u_max, u_max, self.grid_size) / 1e6 # in Mlambda
        
        # Prior image (broad Gaussian centered)
        x = np.linspace(-self.fov_uas/2, self.fov_uas/2, self.grid_size)
        y = np.linspace(-self.fov_uas/2, self.fov_uas/2, self.grid_size)
        X, Y = np.meshgrid(x, y)
        prior = np.exp(-0.5 * ((X**2 + Y**2) / (30.0)**2)) # 30 uas standard deviation
        prior = prior / np.sum(prior)
        prior_1d = prior.flatten()
        
        # Initial guess: flat image or the prior
        initial_guess = prior_1d.copy()
        
        # Cache indices for faster FFT interpolation inside optimization loop
        # We find nearest pixel indices for u,v in advance
        cached_baselines = []
        for name, u_pts, v_pts in baselines:
            u_indices = [np.argmin(np.abs(u_grid - u)) for u in u_pts]
            v_indices = [np.argmin(np.abs(u_grid - v)) for v in v_pts]
            cached_baselines.append((name, u_indices, v_indices, calibrated_data[name]['vis']))
            
        # Define Objective function and gradient to minimize
        def objective_and_grad(image_1d):
            # Enforce flux normalization: sum of pixels = 1.0
            total_flux = np.sum(image_1d)
            if total_flux > 0:
                img_norm = image_1d / total_flux
            else:
                img_norm = image_1d
                
            img_2d = img_norm.reshape((self.grid_size, self.grid_size))
            
            # 1. Forward model FFT
            shift1 = np.fft.ifftshift(img_2d)
            fft2d = np.fft.fft2(shift1)
            fft_img = np.fft.fftshift(fft2d)
            
            # Data Term (Chi-squared mismatch)
            chi_sq = 0.0
            grad_fft = np.zeros((self.grid_size, self.grid_size), dtype=complex)
            
            for name, u_idx, v_idx, obs_vis in cached_baselines:
                # Retrieve model visibilities
                model_vis = fft_img[v_idx, u_idx]
                
                # Difference (model - observed)
                diff = model_vis - obs_vis
                chi_sq += np.sum(np.abs(diff)**2)
                
                # Accumulate Fourier-plane gradient
                for idx, (v_i, u_i) in enumerate(zip(v_idx, u_idx)):
                    grad_fft[v_i, u_i] += 2.0 * diff[idx]
                    
            # Backpropagate gradient through FFT operations (adjoint operations)
            grad_fft2d = np.fft.ifftshift(grad_fft)
            grad_shift1 = np.fft.ifft2(grad_fft2d) * (self.grid_size * self.grid_size)
            grad_img_2d = np.fft.fftshift(grad_shift1)
            grad_data_1d = np.real(grad_img_2d).flatten()
            
            # Apply derivative of normalization: d(x/sum(x)) = (sum(x) - x) / sum(x)^2
            if total_flux > 0:
                grad_data_1d = (grad_data_1d * total_flux - np.sum(grad_data_1d * image_1d)) / (total_flux**2)
                
            # 2. Total Variation (TV) Regularizer (Anisotropic edge-preserving smoothness)
            diff_x = np.diff(img_2d, axis=1)
            diff_y = np.diff(img_2d, axis=0)
            tv = np.sum(np.abs(diff_x)) + np.sum(np.abs(diff_y))
            
            grad_tv_2d = np.zeros((self.grid_size, self.grid_size))
            grad_tv_2d[:, :-1] -= np.sign(diff_x)
            grad_tv_2d[:, 1:] += np.sign(diff_x)
            grad_tv_2d[:-1, :] -= np.sign(diff_y)
            grad_tv_2d[1:, :] += np.sign(diff_y)
            
            grad_tv_1d = grad_tv_2d.flatten()
            if total_flux > 0:
                grad_tv_1d = (grad_tv_1d * total_flux - np.sum(grad_tv_1d * image_1d)) / (total_flux**2)
            
            # 3. Maximum Entropy Regularizer
            eps = 1e-12
            entropy = np.sum(img_norm * np.log((img_norm + eps) / (prior_1d + eps)))
            
            grad_entropy_1d = np.log((img_norm + eps) / (prior_1d + eps)) + 1.0
            if total_flux > 0:
                grad_entropy_1d = (grad_entropy_1d * total_flux - np.sum(grad_entropy_1d * image_1d)) / (total_flux**2)
            
            # Total Loss and Gradient
            loss = chi_sq + alpha_tv * tv + alpha_entropy * entropy
            grad = grad_data_1d + alpha_tv * grad_tv_1d + alpha_entropy * grad_entropy_1d
            
            return loss, grad
            
        # Positivity bounds for each pixel: I_xy >= 0
        bounds = [(0, None) for _ in range(self.grid_size * self.grid_size)]
        
        # Optimize using L-BFGS-B with analytical gradient (jac=True)
        res = minimize(
            objective_and_grad, 
            initial_guess, 
            jac=True,
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False}
        )
        
        # Reshape final reconstructed image
        reconstructed_image = res.x / np.sum(res.x) # Normalized
        reconstructed_image_2d = reconstructed_image.reshape((self.grid_size, self.grid_size))
        
        return reconstructed_image_2d, res.fun

if __name__ == "__main__":
    from simulator import EHTSimulator
    from calibrator import EHTCalibrator
    
    sim = EHTSimulator()
    img, _, _ = sim.generate_black_hole_model(grid_size=64)
    uv, names = sim.generate_uv_coverage()
    data = sim.sample_visibilities(img, 120, zip(uv, names))
    
    cal = EHTCalibrator()
    corrupted = cal.corrupt_visibilities(data)
    calibrated = cal.self_calibrate(corrupted, img, 120)
    
    recon = EHTReconstructor(grid_size=64)
    reconstructed, loss = recon.reconstruct(calibrated, alpha_tv=0.01, alpha_entropy=0.001)
    
    print(f"Reconstructed image size: {reconstructed.shape}")
    print(f"Final Optimization Loss: {loss:.6f}")
