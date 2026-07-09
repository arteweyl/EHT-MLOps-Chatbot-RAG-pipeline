import numpy as np

class EHTCalibrator:
    """
    Simulates instrumental and atmospheric phase/gain corruptions and performs calibration.
    Ref: [3] Data Processing and Calibration, [7] Computational Imaging for VLBI Image Reconstruction.
    """
    def __init__(self, random_seed=42):
        self.rng = np.random.default_rng(random_seed)
        
    def corrupt_visibilities(self, sampled_data, thermal_noise_level=0.02, station_phase_noise_std=1.5):
        """
        Corrupts true visibilities with station-specific phase errors, gain fluctuations, and baseline thermal noise.
        """
        corrupted_data = {}
        
        # Get list of unique stations from the baseline names
        stations = set()
        for name in sampled_data.keys():
            st1, st2 = name.split('-')
            stations.add(st1)
            stations.add(st2)
        stations = list(stations)
        
        # Generate station-specific phase errors for each observation time index
        num_points = len(next(iter(sampled_data.values()))['amp'])
        station_phases = {st: self.rng.uniform(-np.pi, np.pi, num_points) * station_phase_noise_std for st in stations}
        # Station amplitude gains (averaging around 1.0, with some drift)
        station_gains = {st: self.rng.normal(1.0, 0.1, num_points) for st in stations}
        
        for name, data in sampled_data.items():
            st1, st2 = name.split('-')
            u, v = data['u'], data['v']
            true_vis = data['vis']
            
            # Apply station-based corruptions: V_obs = g1 * g2 * exp(i * (theta1 - theta2)) * V_true
            corrupted_vis = np.zeros(num_points, dtype=complex)
            for t in range(num_points):
                g1, g2 = station_gains[st1][t], station_gains[st2][t]
                theta1, theta2 = station_phases[st1][t], station_phases[st2][t]
                
                # Add atmospheric path delay / phase error and gain error
                gain_corr = g1 * g2 * np.exp(1j * (theta1 - theta2))
                
                # Add thermal noise (Gaussian noise in both real and imaginary parts)
                thermal_noise = self.rng.normal(0, thermal_noise_level) + 1j * self.rng.normal(0, thermal_noise_level)
                
                corrupted_vis[t] = true_vis[t] * gain_corr + thermal_noise
                
            corrupted_data[name] = {
                'u': u,
                'v': v,
                'amp': np.abs(corrupted_vis),
                'phase': np.angle(corrupted_vis),
                'vis': corrupted_vis
            }
            
        return corrupted_data

    def compute_closure_phases(self, corrupted_data):
        """
        Computes closure phases for triangles of stations.
        Closure phase: Phi_123 = Phase(1-2) + Phase(2-3) + Phase(3-1)
        This quantity is invariant to station-based phase corruptions (like atmospheric delay).
        """
        closure_data = {}
        stations = set()
        for name in corrupted_data.keys():
            st1, st2 = name.split('-')
            stations.add(st1)
            stations.add(st2)
        stations = sorted(list(stations))
        
        # Helper to find baseline data with correct sign (swapping baseline conjugates)
        def get_vis_phase(stA, stB):
            name1 = f"{stA}-{stB}"
            name2 = f"{stB}-{stA}"
            if name1 in corrupted_data:
                return corrupted_data[name1]['phase']
            elif name2 in corrupted_data:
                # V_ji = V_ij* (complex conjugate), so phase(ji) = -phase(ij)
                return -corrupted_data[name2]['phase']
            else:
                return None

        # Triangles of stations
        for i in range(len(stations)):
            for j in range(i+1, len(stations)):
                for k in range(j+1, len(stations)):
                    stA, stB, stC = stations[i], stations[j], stations[k]
                    p1 = get_vis_phase(stA, stB)
                    p2 = get_vis_phase(stB, stC)
                    p3 = get_vis_phase(stC, stA) # phase(C-A) = -phase(A-C)
                    
                    if p1 is not None and p2 is not None and p3 is not None:
                        # Closure phase calculation
                        c_phase = p1 + p2 + p3
                        # Wrap to [-pi, pi]
                        c_phase = np.arctan2(np.sin(c_phase), np.cos(c_phase))
                        
                        triangle_name = f"{stA}-{stB}-{stC}"
                        closure_data[triangle_name] = c_phase
                        
        return closure_data

    def self_calibrate(self, corrupted_data, model_image, fov_uas):
        """
        Performs a basic phase self-calibration. It takes corrupted visibility data,
        calculates the expected "model" visibilities using the model_image,
        and solves for the station phase errors to calibrate the visibility phases.
        """
        from .simulator import EHTSimulator
        sim = EHTSimulator()
        
        # Get ground-truth or model expected visibilities
        baselines = []
        for name, data in corrupted_data.items():
            baselines.append(((data['u'], data['v']), name))
            
        model_vis_dict = sim.sample_visibilities(model_image, fov_uas, baselines)
        
        calibrated_data = {}
        stations = set()
        for name in corrupted_data.keys():
            st1, st2 = name.split('-')
            stations.add(st1)
            stations.add(st2)
        stations = sorted(list(stations))
        
        num_points = len(next(iter(corrupted_data.values()))['amp'])
        
        # Simple Phase Self-Calibration (least squares solver for station phases)
        # We solve for theta_i at each time step t that minimizes:
        # Sum_{i,j} | Phase_obs(i-j) - (theta_i - theta_j) - Phase_model(i-j) |^2
        # Let's implement an iterative phase solver:
        calibrated_vis_dict = {}
        for name in corrupted_data.keys():
            calibrated_vis_dict[name] = np.zeros(num_points, dtype=complex)
            
        for t in range(num_points):
            # Phase solver using coordinate descent
            theta = {st: 0.0 for st in stations} # Initial guesses
            
            for iteration in range(15): # few iterations are enough for didactic purposes
                for st in stations:
                    sum_complex = 0.0 + 0.0j
                    count = 0
                    for name, data in corrupted_data.items():
                        st1, st2 = name.split('-')
                        if st == st1:
                            other = st2
                            sign = 1.0
                        elif st == st2:
                            other = st1
                            sign = -1.0
                        else:
                            continue
                            
                        phi_obs = data['phase'][t]
                        phi_model = model_vis_dict[name]['phase'][t]
                        
                        # Accumulate complex phasor to handle 2pi phase wrapping
                        res_phase = sign * (phi_obs - phi_model) + theta[other]
                        sum_complex += np.exp(1j * res_phase)
                        count += 1
                        
                    if count > 0 and np.abs(sum_complex) > 0:
                        theta[st] = np.angle(sum_complex)
            
            # Calibrate observed visibilities with solved station phases
            for name, data in corrupted_data.items():
                st1, st2 = name.split('-')
                phi_obs = data['phase'][t]
                amp_obs = data['amp'][t]
                
                cal_phase = phi_obs - (theta[st1] - theta[st2])
                cal_vis = amp_obs * np.exp(1j * cal_phase)
                
                calibrated_vis_dict[name][t] = cal_vis
                
        for name, data in corrupted_data.items():
            calibrated_data[name] = {
                'u': data['u'],
                'v': data['v'],
                'amp': data['amp'], # Amplitude remains uncalibrated or self-calibrated (here we keep observed)
                'phase': np.angle(calibrated_vis_dict[name]),
                'vis': calibrated_vis_dict[name]
            }
            
        return calibrated_data

if __name__ == "__main__":
    from simulator import EHTSimulator
    sim = EHTSimulator()
    img, _, _ = sim.generate_black_hole_model()
    uv, names = sim.generate_uv_coverage()
    data = sim.sample_visibilities(img, 120, zip(uv, names))
    
    cal = EHTCalibrator()
    corrupted = cal.corrupt_visibilities(data)
    closures = cal.compute_closure_phases(corrupted)
    
    print(f"Computed {len(closures)} closure phase triangles.")
    first_tri = list(closures.keys())[0]
    print(f"Closure phase for {first_tri} at index 0: {closures[first_tri][0]:.4f} rad")
    
    # Run self-calibration
    calibrated = cal.self_calibrate(corrupted, img, 120)
    print("Self-calibration complete!")
