# Spectral Sirens Tutorial

## Introduction

Spectral Sirens is a Python package for performing spectral siren cosmology using the population of compact binary coalescences detected by gravitational wave (GW) detectors. This tutorial will guide you through the installation, basic concepts, and usage of the code.

## What are Spectral Sirens?

Spectral sirens is a method for measuring cosmological parameters (like the Hubble constant H₀) using the entire mass distribution of gravitational wave sources, rather than relying on individual electromagnetic counterparts. The method uses hierarchical Bayesian inference to constrain cosmology from the population of detected compact binary mergers.

## Installation

### Prerequisites

The code requires the IGWN conda environment. To install it:

1. Follow the instructions at: https://computing.docs.ligo.org/conda/environments/igwn/

### Basic Installation

```bash
git clone https://github.com/ezquiaga/spectral_sirens.git
cd spectral_sirens
pip install .
```

### GPU Support (Optional but Recommended for Inference)

For running Bayesian inference on GPUs, you need JAX and NumPyro with GPU support:

1. Install CUDA first (required)
2. See JAX docs: https://github.com/google/jax#pip-installation-gpu-cuda
3. See NumPyro docs: https://num.pyro.ai/en/latest/getting_started.html

You can use the provided conda environment file:

```bash
conda env create -f envs/inference_gpu.yml
conda activate inference_gpu
```

## Package Structure

The package consists of several key modules:

- `spectral_sirens.bayesian_inference`: Hierarchical Bayesian inference using JAX/NumPyro
- `spectral_sirens.cosmology`: Cosmological calculations and utilities
- `spectral_sirens.detectors`: Detector sensitivity curves and network configurations
- `spectral_sirens.gw_population`: GW source population models
- `spectral_sirens.gw_rates`: Merger rate calculations
- `spectral_sirens.utils`: General utility functions

## Tutorial 1: Detector Sensitivity

This example shows how to compute signal-to-noise ratios (SNR) for different GW detectors.

### Loading Detector Sensitivity Curves

```python
import numpy as np
from spectral_sirens.detectors import sensitivity_curves as sc

# Load detector power spectral densities (PSDs)
Sn_O3, fmin_O3, fmax_O3 = sc.detector_psd('O3')      # LIGO/Virgo O3 run
Sn_O4, fmin_O4, fmax_O4 = sc.detector_psd('O4')      # LIGO/Virgo O4 run
Sn_Aplus, fmin_Aplus, fmax_Aplus = sc.detector_psd('A+')    # A+ design
Sn_Asharp, fmin_Asharp, fmax_Asharp = sc.detector_psd('A#')  # A# design
Sn_CE, fmin_CE, fmax_CE = sc.detector_psd('CE-40')   # Cosmic Explorer
```

### Computing SNR

```python
from spectral_sirens.utils import gwutils

# Parameters
fmin = 10.0        # Minimum frequency (Hz)
Tobs = 1.0         # Observation time (years)
detector = 'A#'    # Detector name
based = 'ground'   # 'ground' or 'space' based
snr_th = 8.0       # SNR threshold for detection

# Binary parameters
mass1 = 10.0       # Primary mass (solar masses)
mass2 = 10.0       # Secondary mass (solar masses)
DL = 100.0         # Luminosity distance (Mpc)

# Compute SNR
snr = gwutils.snr(mass1, mass2, DL, fmin, Tobs, detector, based)
print(f'SNR in {detector} detector: {snr}')
```

### Plotting Sensitivity Curves

```python
import matplotlib.pyplot as plt

# Create frequency arrays
nfs = 100
fs_O3 = np.logspace(np.log10(fmin_O3), np.log10(fmax_O3), nfs)
fs_O4 = np.logspace(np.log10(fmin_O4), np.log10(fmax_O4), nfs)
fs_Aplus = np.logspace(np.log10(fmin_Aplus), np.log10(fmax_Aplus), nfs)

# Plot
plt.figure(figsize=(10, 6))
plt.loglog(fs_O3, Sn_O3(fs_O3), label='O3')
plt.loglog(fs_O4, Sn_O4(fs_O4), label='O4')
plt.loglog(fs_Aplus, Sn_Aplus(fs_Aplus), label='A+')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [Hz$^{-1}$]')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Tutorial 2: Horizon Distance

Calculate the maximum distance at which equal-mass binaries can be detected.

```python
from spectral_sirens.detectors import horizon

# Load sensitivity
Sn_O4, fmin_O4, fmax_O4 = sc.detector_psd('O4')
based = 'ground'
snr_th = 8.0
Tobs = 1.0

# Set up mass and frequency grids
nfs = 100
nms = 100
fs_O4 = np.logspace(np.log10(fmin_O4), np.log10(fmax_O4), nfs)
masses = np.logspace(np.log10(0.5), 5, nms)

# Compute horizon distance
zhor_max_O4 = horizon.zhor_max(masses, fs_O4, Sn_O4, snr_th, Tobs, based)

# Plot
plt.figure(figsize=(10, 6))
plt.loglog(masses, zhor_max_O4)
plt.xlabel('Mass [M$_\\odot$]')
plt.ylabel('Maximum redshift')
plt.title('Horizon distance for O4')
plt.grid(True, alpha=0.3)
plt.show()
```

## Tutorial 3: Mock Population

Generate a mock population of GW detections with realistic selection effects.

### Setting Fiducial Parameters

```python
# Cosmology
H0_fid = 67.66      # Hubble constant (km/s/Mpc)
Om0_fid = 0.30966   # Matter density parameter

# Merger rate (Madau & Dickinson 2014 star formation rate)
r0_fid = 30.0       # Local merger rate (Gpc^-3 yr^-1)
alpha_z_fid = 2.7   # Redshift evolution
zp_fid = 1.9        # Peak redshift
beta_fid = 2.9      # High-z suppression

# Primary mass distribution (power-law + peak)
mmin_pl_fid = 0.01      # Minimum mass (M☉)
mmax_pl_fid = 150.0     # Maximum mass (M☉)
alpha_fid = -3.4        # Power-law slope
sig_m1_fid = 3.6        # Gaussian peak width
mu_m1_fid = 34.0        # Gaussian peak location
f_peak_fid = 1.4e-8     # Peak fraction

# Mass ratio
bq_fid = 1.1            # Mass ratio parameter

# Detector settings
detector = 'O4'
fmin = 10.0
Tobs_fid = 1.0
snr_th = 8.0
```

### Generating Mock Data

```python
import time
from spectral_sirens.gw_population import gwpop

# Number of sources
n_sources = 1000
n_detections = 100
n_samples = 100  # Posterior samples per detection

# Generate mock population
starttime = time.time()
m1z_samples, m2z_samples, dL_samples, pdraw_samples = mock_population(
    n_sources,
    n_detections,
    n_samples,
    H0_fid,
    Om0_fid,
    mmin_pl_fid,
    mmax_pl_fid,
    alpha_fid,
    sig_m1_fid,
    mu_m1_fid,
    f_peak_fid,
    mMin_filter_fid,
    mMax_filter_fid,
    dmMin_filter_fid,
    dmMax_filter_fid,
    zp_fid,
    alpha_z_fid,
    beta_fid,
    snr_th,
    fmin,
    Tobs_fid,
    detector,
    based
)
print(f'Time taken = {time.time() - starttime} seconds')
```

### Saving Data

```python
import h5py

filename = 'mock_catalog.hdf5'
with h5py.File(filename, 'w') as f:
    f.create_dataset('m1z_samples', data=m1z_samples)
    f.create_dataset('m2z_samples', data=m2z_samples)
    f.create_dataset('dL_samples', data=dL_samples)
    f.create_dataset('pdraw_samples', data=pdraw_samples)
    
    # Store metadata
    f.attrs['n_detections'] = n_detections
    f.attrs['n_samples'] = n_samples
    f.attrs['H0_fid'] = H0_fid
    f.attrs['Om0_fid'] = Om0_fid
    f.attrs['detector'] = detector
```

## Tutorial 4: Injection Campaign

Set up an injection campaign to compute selection effects.

### Define Injection Parameters

```python
# Injection range
zmin_inj, zmax_inj = 1e-3, 15.0
mmin_inj, mmax_inj = 1e-1, 100.0
alpha_inj = -0.2  # Injection power-law slope

# Compute grid
mzmin_inj = mmin_inj
mzmax_inj = mmax_inj * (1 + zmax_inj)
```

### Run Injections

```python
from spectral_sirens.cosmology import gwcosmo

n_injections = 10000

# Draw from injection distribution
m1z_inj = gwpop.draw_powerlaw(mzmin_inj, mzmax_inj, alpha_inj, n_injections)
m2z_inj = gwpop.draw_powerlaw(mzmin_inj, m1z_inj, alpha_inj, n_injections)

# Draw redshifts
z_inj = gwpop.draw_redshift_uniform_comoving(zmin_inj, zmax_inj, n_injections, H0_fid, Om0_fid)

# Convert to luminosity distance
dL_inj = gwcosmo.dL_from_z(z_inj, H0_fid, Om0_fid)

# Compute SNRs
detectorSn, fmin_detect, fmax_detect = sc.detector_psd(detector)
snr_obs = gwutils.snr(m1z_inj, m2z_inj, dL_inj, fmin, Tobs_fid, detector, based)

# Apply selection
detected = snr_obs > snr_th
m1z_det = m1z_inj[detected]
m2z_det = m2z_inj[detected]
dL_det = dL_inj[detected]

print(f'Detection efficiency: {np.sum(detected)/n_injections:.2%}')
```

## Tutorial 5: Bayesian Inference

Perform hierarchical Bayesian inference to constrain cosmological and population parameters.

### Prepare Data

```python
import jax.numpy as jnp
from spectral_sirens.bayesian_inference import inference

# Load mock data (from Tutorial 3)
with h5py.File('mock_catalog.hdf5', 'r') as f:
    m1z_obs = jnp.array(f['m1z_samples'][:])
    m2z_obs = jnp.array(f['m2z_samples'][:])
    dL_obs = jnp.array(f['dL_samples'][:])
    pdraw_obs = jnp.array(f['pdraw_samples'][:])
```

### Set Up Inference

```python
import numpyro
from numpyro.infer import MCMC, NUTS

# Set random seed
numpyro.set_host_device_count(4)  # Use 4 parallel chains

# Define model (simplified example)
def model(m1z, m2z, dL):
    # Priors
    H0 = numpyro.sample('H0', numpyro.distributions.Uniform(20, 140))
    Om0 = numpyro.sample('Om0', numpyro.distributions.Uniform(0.1, 0.5))
    
    # Population parameters
    alpha = numpyro.sample('alpha', numpyro.distributions.Uniform(-4, 2))
    
    # Likelihood (simplified)
    # ... (full implementation in the package)
    
# Run MCMC
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=4)
mcmc.run(rng_key, m1z_obs, m2z_obs, dL_obs)

# Get results
samples = mcmc.get_samples()
print(f"H0 = {jnp.mean(samples['H0']):.2f} ± {jnp.std(samples['H0']):.2f}")
```

## Advanced Features

### Custom Detector Networks

You can define custom detector networks for multi-detector analysis:

```python
from spectral_sirens.detectors import pw_network

# Define network configuration
network_config = {
    'detectors': ['H1', 'L1', 'V1'],  # LIGO Hanford, Livingston, Virgo
    'snr_threshold': 8.0,
    'network_snr': True
}
```

### Custom Mass Distributions

Implement custom mass distribution models:

```python
def custom_mass_model(m1, params):
    # Your custom model
    # e.g., broken power law, multi-peak, etc.
    return probability
```

## Tips and Best Practices

1. **GPU Usage**: For large catalogs (>100 events), use GPU acceleration
2. **Convergence**: Always check MCMC convergence diagnostics
3. **Injection Studies**: Run injection campaigns to validate your analysis
4. **Selection Effects**: Always account for detector selection effects
5. **Priors**: Choose physically motivated priors based on current observations

## Common Issues and Solutions

### Issue: JAX GPU not detected
**Solution**: Ensure CUDA is properly installed and JAX was installed with GPU support

### Issue: Memory errors during inference
**Solution**: Reduce batch size or number of parallel chains

### Issue: Poor MCMC convergence
**Solution**: Increase warmup steps, adjust step size, or reparameterize the model

## References

If you use this code, please cite:

1. Chen, Ezquiaga & Gupta (2024): "Cosmography with next-generation gravitational wave detectors" [arXiv:2402.03120]
2. Ezquiaga & Holz (2022): "Spectral Sirens: Cosmology from the Full Mass Distribution of Compact Binaries" [Phys. Rev. Lett. 129, 061102]

## Additional Resources

- **Examples folder**: Check the `examples/` directory for Jupyter notebooks with detailed examples
- **Zenodo data**: Mock catalogs and sensitivity data at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10655745.svg)](https://doi.org/10.5281/zenodo.10655745)
- **Documentation**: See individual module docstrings for detailed API documentation

## Support

For questions or issues:
- Open an issue on GitHub: https://github.com/ezquiaga/spectral_sirens/issues
- Contact: jose.ezquiaga@nbi.ku.dk

---

**Happy spectral siren hunting! 🌊✨**
