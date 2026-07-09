import numpy as np
import matplotlib.pyplot as plt

class EHTSimulator:
    """
    Simulates a VLBI telescope array, generates u-v coverage based on Earth's rotation,
    and creates synthetic interferometric visibility data from a ground-truth relativistic black hole image.
    Ref: [2] Array and Instrumentation, [3] Data Processing and Calibration.
    """
    def __init__(self, target_dec_deg=12.391, wavelength_m=1.3e-3):
        # M87* Declination: +12.391 degrees
        self.dec = np.radians(target_dec_deg)
        self.wavelength = wavelength_m
        
        # Define telescope stations (latitude, longitude in degrees)
        self.stations = {
            'ALMA': {'lat': -23.029, 'lon': -67.755},
            'LMT': {'lat': 18.985, 'lon': -97.315},
            'SMT': {'lat': 32.701, 'lon': -109.892},
            'SMA': {'lat': 19.823, 'lon': -155.478},
            'IRAM': {'lat': 37.066, 'lon': -3.393},
            'SPT': {'lat': -90.000, 'lon': 0.000}
        }
        
    def generate_uv_coverage(self, hours=8, num_points=100):
        """
        Generates u-v coordinates in mega-wavelengths (Mlambda) for all baselines as Earth rotates.
        """
        station_names = list(self.stations.keys())
        num_stations = len(station_names)
        
        # Convert lat/lon to geocentric coordinates (XYZ) in meters
        # Simplified spherical Earth model
        R_earth = 6378100.0 # meters
        xyz = {}
        for name, pos in self.stations.items():
            lat = np.radians(pos['lat'])
            lon = np.radians(pos['lon'])
            x = R_earth * np.cos(lat) * np.cos(lon)
            y = R_earth * np.cos(lat) * np.sin(lon)
            z = R_earth * np.sin(lat)
            xyz[name] = np.array([x, y, z])
            
        # Observation time range (Hour Angles from -hours/2 to hours/2)
        ha_range = np.linspace(-hours/2, hours/2, num_points) * (np.pi / 12.0) # hours to radians
        
        baselines_uv = []
        baseline_names = []
        
        # Projection matrix from geocentric XYZ coordinates to u-v plane
        # For a given declination dec and hour angle ha
        for i in range(num_stations):
            for j in range(i + 1, num_stations):
                st1, st2 = station_names[i], station_names[j]
                # Baseline vector in XYZ
                bx = xyz[st1][0] - xyz[st2][0]
                by = xyz[st1][1] - xyz[st2][1]
                bz = xyz[st1][2] - xyz[st2][2]
                
                u_pts = []
                v_pts = []
                
                for ha in ha_range:
                    # Check mutual visibility: zenith angle < 80 degrees (elevation > 10 degrees)
                    # For simplicity in this didactic model, we check if stations can see the source
                    # (source vector s = [cos(dec)cos(ha), -cos(dec)sin(ha), sin(dec)] in local coordinate frames)
                    # Let's assume visibility or simply calculate u,v and add a flag for horizon checks
                    
                    # Transformation to (u, v, w) coordinates
                    # Ref: Thompson, Moran, & Swenson (Interferometry and Synthesis in Radio Astronomy)
                    u = (np.sin(ha) * bx + np.cos(ha) * by) / self.wavelength
                    v = (-np.sin(self.dec)*np.cos(ha)*bx + np.sin(self.dec)*np.sin(ha)*by + np.cos(self.dec)*bz) / self.wavelength
                    
                    # Normalize to mega-wavelengths (Mlambda)
                    u_pts.append(u / 1e6)
                    v_pts.append(v / 1e6)
                
                baselines_uv.append((np.array(u_pts), np.array(v_pts)))
                baseline_names.append(f"{st1}-{st2}")
                
        return baselines_uv, baseline_names

    def generate_black_hole_model(self, grid_size=128, fov_uas=120, ring_rad_uas=20.0, ring_width_uas=4.0, asymmetry=0.6, phi_0_deg=135):
        """
        Creates a 2D model representing a relativistic black hole shadow (asymmetric crescent).
        Ref: [1] The Shadow of the Supermassive Black Hole, [5] Physical Origin of the Asymmetric Ring.
        
        fov_uas: Field of view in micro-arcseconds
        ring_rad_uas: Radius of the photon ring in micro-arcseconds
        ring_width_uas: Width of the emission ring in micro-arcseconds
        asymmetry: Controls Doppler beaming (0 = symmetric ring, 1 = maximum asymmetry)
        phi_0_deg: Position angle of the brightness peak (degrees East of North)
        """
        x = np.linspace(-fov_uas/2, fov_uas/2, grid_size)
        y = np.linspace(-fov_uas/2, fov_uas/2, grid_size)
        X, Y = np.meshgrid(x, y)
        
        R = np.sqrt(X**2 + Y**2)
        Phi = np.arctan2(X, Y) # Position angle in radians (East of North)
        
        # Gaussian ring radial profile
        radial_profile = np.exp(-0.5 * ((R - ring_rad_uas) / (ring_width_uas / 2.355))**2)
        
        # Doppler beaming asymmetry: peak at phi_0
        phi_0 = np.radians(phi_0_deg)
        beaming = 1.0 + asymmetry * np.cos(Phi - phi_0)
        
        # Final intensity map
        intensity = radial_profile * beaming
        
        # Normalize flux to 1.0 Jansky (Jy)
        intensity = intensity / np.sum(intensity)
        
        return intensity, x, y

    def sample_visibilities(self, image, fov_uas, baselines_uv):
        """
        Samples the Fourier Transform of the ground-truth image at the u-v coordinates.
        fov_uas: Field of View in micro-arcseconds (controls scaling between image grid and spatial frequencies)
        """
        grid_size = image.shape[0]
        
        # Compute 2D Fourier Transform (centered)
        # Shift zero-frequency component to center of spectrum
        fft_img = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))
        
        # Resolution in u, v coordinates
        # 1 uas = 1e-6 * pi / (180 * 3600) radians = 4.8481368e-12 rad
        uas_to_rad = 4.8481368e-12
        dx = fov_uas * uas_to_rad / grid_size
        
        # Maximum frequency measured in wavelengths
        u_max = 1.0 / (2.0 * dx)
        u_grid = np.linspace(-u_max, u_max, grid_size) / 1e6 # in Mega-wavelengths
        
        # We will interpolate the complex FFT values at the measured u, v baseline points
        sampled_data = {}
        
        for (u_pts, v_pts), name in baselines_uv:
            vis_amplitudes = []
            vis_phases = []
            
            for u, v in zip(u_pts, v_pts):
                # Find nearest grid indices in the Fourier plane
                u_idx = np.argmin(np.abs(u_grid - u))
                v_idx = np.argmin(np.abs(u_grid - v))
                
                val = fft_img[v_idx, u_idx] # Note: y corresponds to row (v), x to col (u)
                vis_amplitudes.append(np.abs(val))
                vis_phases.append(np.angle(val))
                
            sampled_data[name] = {
                'u': u_pts,
                'v': v_pts,
                'amp': np.array(vis_amplitudes),
                'phase': np.array(vis_phases),
                # Complex visibility: V = Amp * exp(i * phase)
                'vis': np.array(vis_amplitudes) * np.exp(1j * np.array(vis_phases))
            }
            
        return sampled_data

if __name__ == "__main__":
    # Test script to make sure the simulator runs
    sim = EHTSimulator()
    img, x, y = sim.generate_black_hole_model()
    uv, names = sim.generate_uv_coverage()
    data = sim.sample_visibilities(img, 120, zip(uv, names))
    print(f"Simulated {len(names)} baselines successfully!")
    print(f"Sample baseline ALMA-LMT has {len(data['ALMA-LMT']['amp'])} data points.")
