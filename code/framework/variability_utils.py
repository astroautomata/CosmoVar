# variability_utils.py
"""
Shared utilities for quasar variability analysis pipeline.

Contains:
- SED and filter infrastructure (JAX-accelerated)
- Cosmology calculations (distance modulus)
- K-corrections and magnitude conversions
- Effective wavelength calculations

All functions optimized with JAX JIT compilation where possible.
"""

# Enable 64-bit precision BEFORE importing JAX modules
# Required for luminosities ~10^45 erg/s which overflow float32
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd
import os

# ==============================================================================
# GLOBAL STATE
# ==============================================================================
passband_data = None  # Set by calling code via su.passband_data = su.load_passband_data(...)

# ==============================================================================
# CONSTANTS
# ==============================================================================
LN10 = np.log(10.0)
MPC_TO_CM = 3.0856775814671916e24
C_LIGHT = 299792.458  # km/s

# ==============================================================================
# PHOTOMETRY CONFIGURATION
# ==============================================================================
# Band photometry configuration for magnitude calculations.
#
# For Gaia bands:
#   Source: https://gea.esac.esa.int/archive/documentation/GEDR3/Data_processing/chap_cu5pho/cu5pho_sec_photProc/cu5pho_ssec_photCal.html
#   - zp_mag: SZP_VEG from Table 5.3 (synthetic zero point for VEGAMAG system)
#   - flux_scale: 100.0 to convert erg/s/cm²/Å → W/m²/nm (since 1 W/m²/nm = 100 erg/s/cm²/Å)
#   - Magnitude formula: m_VEG = -2.5 * log10(<f_λ>) + SZP_VEG, where <f_λ> is in W/m²/nm
#   - In code: M = -2.5 * log10(F / flux_scale) - zp_mag
#
# For ZTF bands (AB system):
#   Source: http://svo2.cab.inta-csic.es/theory/fps/index.php?id=Palomar/ZTF.g&&mode=browse&gname=Palomar&gname2=ZTF#filter
#   - zp_mag: -2.5 * log10(F_zp) where F_zp is the AB zero-point flux in erg/cm²/s/Å
#   - flux_scale: 1.0 (no scaling needed, flux already in erg/cm²/s/Å)
#   - AB magnitude formula: m_AB = -2.5 * log10(F_λ / F_zp) = -2.5 * log10(F_λ) - zp_mag
#   - ZTF g: F_zp = 4.75724e-9 erg/cm²/s/Å → zp_mag = -2.5 * log10(4.75724e-9) = +20.807
#   - ZTF r: F_zp = 2.64344e-9 erg/cm²/s/Å → zp_mag = -2.5 * log10(2.64344e-9) = +21.444
#
BAND_PHOTOMETRY = {
    # Gaia - flux in erg/s/cm²/Å, divide by 100 to get W/m²/nm, then apply SZP_VEG
    'GPb':   {'zp_mag': 26.48986, 'flux_scale': 100.0},
    'BPPb':  {'zp_mag': 25.96551, 'flux_scale': 100.0},
    'RPPb':  {'zp_mag': 27.21639, 'flux_scale': 100.0},
    # ZTF AB - flux in erg/cm²/s/Å, no scaling, ZP = -2.5*log10(F_zp)
    'ZTF_g': {'zp_mag': 20.807, 'flux_scale': 1.0},  # F_zp = 4.75724e-9 erg/cm²/s/Å
    'ZTF_r': {'zp_mag': 21.444, 'flux_scale': 1.0},  # F_zp = 2.64344e-9 erg/cm²/s/Å
}

# Helper functions to get photometry parameters from BAND_PHOTOMETRY
def get_zp_mag(band):
    """Get zero-point magnitude for a band from BAND_PHOTOMETRY."""
    return BAND_PHOTOMETRY[band]['zp_mag']

def get_flux_scale(band):
    """Get flux scale factor for a band from BAND_PHOTOMETRY."""
    return BAND_PHOTOMETRY[band]['flux_scale']

# Default MW extinction parameter
R_V_DEFAULT = 3.1

# Bolometric magnitude constant: M_bol = MABS_BOL_CONSTANT - 2.5 * log10(L_bol)
MABS_BOL_CONSTANT = 4.74 - 2.5 * (jnp.log10(5.15) - 33.582)

# ==============================================================================
# PASSBAND DATA LOADING
# ==============================================================================
# Global storage for ZTF filter data (similar to passband_data for Gaia)
ztf_filter_data = {}  # Set by calling code via su.ztf_filter_data['ZTF_g'] = su.load_ztf_filter(...)

def load_passband_data(filepath):
    """
    Load Gaia passband transmission data.

    Parameters
    ----------
    filepath : str
        Path to passband.dat file

    Returns
    -------
    df : pd.DataFrame
        Passband data with columns: lambda, GPb, BPPb, RPPb, etc.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Passband file not found: {filepath}")

    colspecs = [
        (0, 7),    # lambda
        (9, 23),   # GPb
        (25, 39),  # e_GPb
        (41, 55),  # BPPb
        (57, 71),  # e_BPPb
        (73, 87),  # RPPb
        (89, 103)  # e_RPPb
    ]
    names = ['lambda', 'GPb', 'e_GPb', 'BPPb', 'e_BPPb', 'RPPb', 'e_RPPb']
    df = pd.read_fwf(filepath, colspecs=colspecs, names=names, na_values='99.99')
    df['lambda'] = df['lambda'].astype(float)
    return df


def load_ztf_filter(filepath):
    """
    Load ZTF filter transmission data.

    ZTF filter files from SVO Filter Profile Service have two columns:
    wavelength (Angstrom) and transmission.

    Source: http://svo2.cab.inta-csic.es/theory/fps/index.php?id=Palomar/ZTF.g

    Parameters
    ----------
    filepath : str
        Path to ZTF filter file (e.g., Palomar_ZTF.g.dat)

    Returns
    -------
    wavelength : np.array
        Wavelengths in Angstrom
    transmission : np.array
        Filter transmission values (0 to 1)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ZTF filter file not found: {filepath}")

    data = np.loadtxt(filepath)
    wavelength = data[:, 0]  # Already in Angstrom
    transmission = data[:, 1]
    return wavelength, transmission


# ==============================================================================
# SED INTERPOLATOR CLASS
# ==============================================================================


class JAXQSOSedInterpolator:
    """
    JAX-accelerated quasar SED interpolator.
    
    Loads a template SED and provides fast interpolation at arbitrary
    luminosities and redshifts. Uses fixed SED shape with luminosity scaling.
    
    Attributes
    ----------
    wavlen : jnp.array
        Rest-frame wavelength grid [Å]
    reference_logL3000 : float
        Reference log10(L_3000) for template normalization
    lambda_L_lambda_rest : jnp.array
        Template λL_λ spectrum [erg/s]
    """
    
    def __init__(self, filenamegrid):
        """Load SED template from NPZ file."""
        data = np.load(filenamegrid, allow_pickle=True)
        self.wavlen = jnp.array(data["wavlen"])
        self.reference_logL3000 = float(data["reference_logL3000"])
        self.lambda_L_lambda_rest = jnp.array(data["lambda_L_lambda_normalized"])
        self.Nlam = len(self.wavlen)
        
        # Storage for precomputed interpolation indices
        self.z_vals_stored = None
        self.n_quasars = None
        self.band_interp_indices_stacked = None
        self.band_interp_weights_stacked = None
        self.band_shapes = None
        
        print(f"✓ Loaded SED template: {self.Nlam} wavelengths")
        print(f"  Range: [{self.wavlen.min():.1f}, {self.wavlen.max():.1f}] Å")
        print(f"  Reference logL3000: {self.reference_logL3000:.2f}")
    
    def precompute_z_interpolation_for_quasars(self, z_vals):
        """Store redshifts for batch operations."""
        self.z_vals_stored = jnp.asarray(z_vals)
        self.n_quasars = len(self.z_vals_stored)
        print(f"✓ Stored {self.n_quasars} redshifts for batch operations")
    
    def precompute_band_interpolation(self, wave_grids_per_band, band_order):
        """
        Precompute interpolation indices for filter grids.
        
        Speeds up repeated evaluations by caching interpolation weights.
        """
        if self.z_vals_stored is None:
            raise ValueError("Must call precompute_z_interpolation_for_quasars() first!")
        
        z_vals = self.z_vals_stored
        n_quasars = len(z_vals)
        
        all_indices = []
        all_weights = []
        shapes = []
        
        for band in band_order:
            wave_grid = wave_grids_per_band[band]
            n_wave_band = len(wave_grid)
            
            # Map observer wavelengths to rest frame
            lambda_rf = wave_grid[None, :] / (1 + z_vals[:, None])
            lambda_rf_flat = lambda_rf.ravel()
            
            # Find interpolation indices
            idx_flat = jnp.searchsorted(self.wavlen, lambda_rf_flat)
            idx_flat = jnp.clip(idx_flat, 1, len(self.wavlen) - 1)
            idx = idx_flat.reshape(n_quasars, n_wave_band) - 1
            
            # Compute interpolation weights
            lambda_lower = self.wavlen[idx]
            lambda_upper = self.wavlen[idx + 1]
            weights = (lambda_rf - lambda_lower) / (lambda_upper - lambda_lower)
            
            all_indices.append(idx)
            all_weights.append(weights)
            shapes.append((n_quasars, n_wave_band))
        
        self.band_interp_indices_stacked = jnp.stack(all_indices)
        self.band_interp_weights_stacked = jnp.stack(all_weights)
        self.band_shapes = shapes
        print(f"✓ Precomputed band interpolation: {self.band_interp_indices_stacked.shape}")
    
    def interpolate_batch(self, target_logL3000_array):
        """
        Interpolate SED at target luminosities.
        
        Parameters
        ----------
        target_logL3000_array : array
            Target log10(L_3000) values [n_objects]
        
        Returns
        -------
        lambda_L_lambda : array [n_objects, n_wavelength]
            λL_λ spectra at target luminosities
        """
        target_logL3000 = jnp.asarray(target_logL3000_array)
        scale_factor = 10**(target_logL3000[:, None] - self.reference_logL3000)
        return self.lambda_L_lambda_rest[None, :] * scale_factor

def create_jax_interpolator(filenamegrid):
    """Factory function to create SED interpolator."""
    return JAXQSOSedInterpolator(filenamegrid)

# ==============================================================================
# FILTER TRANSMISSION FUNCTIONS
# ==============================================================================
@jax.jit
def _onedimension_interp(x, xp_raw, fp_raw):
    """JIT-compiled 1D interpolation using interpax."""
    import interpax
    xp = jnp.asarray(xp_raw)
    fp = jnp.asarray(fp_raw)
    return interpax.interp1d(x, xp, fp, method="linear")

@jax.jit
def _S_interp(wavelengths, lambda_transmissivity_jax, S_values_jax):
    """Interpolate filter transmission curve."""
    return _onedimension_interp(wavelengths, lambda_transmissivity_jax, S_values_jax)

def band_transmissivity_jax(band: str):
    """
    Get filter transmission curve for a band.

    Supports Gaia bands (GPb, BPPb, RPPb) and ZTF bands (ZTF_g, ZTF_r).

    Parameters
    ----------
    band : str
        Band name: 'GPb', 'BPPb', 'RPPb' (Gaia), 'ZTF_g', 'ZTF_r' (ZTF).

    Returns
    -------
    lambda_transmissivity : jnp.array
        Wavelengths [Å]
    S_values : jnp.array
        Transmission values (normalized)
    """
    band = str(band)

    # ZTF bands - use ztf_filter_data global
    if band.startswith('ZTF_'):
        if band not in ztf_filter_data:
            raise ValueError(f"ZTF filter '{band}' not loaded. "
                           f"Call su.ztf_filter_data['{band}'] = su.load_ztf_filter(path) first.")
        wavelength, transmission = ztf_filter_data[band]
        return jnp.array(wavelength), jnp.array(transmission)

    # Gaia bands - use passband_data global
    if passband_data is None:
        raise ValueError("Gaia passband data not loaded. "
                        "Call su.passband_data = su.load_passband_data(path) first.")
    mask = ~np.isnan(passband_data[band])
    lambda_transmissivity = np.array(passband_data['lambda'][mask]) * 10.0  # nm → Å
    S_values = passband_data[band][mask]
    return jnp.array(lambda_transmissivity), jnp.array(S_values)

def setup_wavelength_grid(bands, n_integration_wavelengths):
    """
    Create wavelength grids and interpolated transmission curves.
    
    Parameters
    ----------
    bands : list of str
        Band names
    n_integration_wavelengths : int
        Number of wavelength points for integration
    
    Returns
    -------
    wave_grids : dict
        Wavelength grids per band
    S_unifieds : dict
        Interpolated transmission curves per band
    """
    wave_grids = {}
    S_unifieds = {}
    
    for band in bands:
        lambda_trans, S_vals = band_transmissivity_jax(band)
        wave_grid = jnp.linspace(lambda_trans.min(), lambda_trans.max(), n_integration_wavelengths)
        S_unified = _S_interp(wave_grid, lambda_trans, S_vals)
        wave_grids[band] = wave_grid
        S_unifieds[band] = S_unified
    
    return wave_grids, S_unifieds

# ==============================================================================
# EFFECTIVE WAVELENGTH CALCULATION
# ==============================================================================
def compute_effective_lambda_rf_simplified(z_vals, wave_grids, S_unifieds, jax_interpolator,
                                          fiducial_logL3000=45.0, band='GPb'):
    """
    Compute simplified filter-weighted effective rest-frame wavelength.
    
    Uses the fact that distance and redshift factors cancel in the ratio:
    
    Starting from:
        λ_eff = ∫ λ_rf F_λ^rf(λ_rf) S_X((1+z)λ_rf) dλ_rf / ∫ F_λ^rf(λ_rf) S_X((1+z)λ_rf) dλ_rf
    
    Where F_λ^rf = L_λ^rf / (4π d_L² (1+z))
    
    The (4π d_L² (1+z)) cancels in the ratio, leaving:
        λ_eff = ∫ λ_rf L_λ^rf(λ_rf) S_X((1+z)λ_rf) dλ_rf / ∫ L_λ^rf(λ_rf) S_X((1+z)λ_rf) dλ_rf
    
    So we can work directly with luminosity density L_λ without needing distance or redshift factors!
    Returns
    -------
    lambda_rf_eff : array [n_quasars]
        Effective rest-frame wavelength [Å]
    """
    z_vals = jnp.asarray(z_vals)
    
    # Get observer-frame wavelength grid for this filter
    lambda_obs_grid = wave_grids[band]  # Observer-frame wavelengths [Å]
    
    # Get filter transmission as function of observer wavelength
    S_X = S_unifieds[band]  # S_X(λ_obs)
    
    # Get SED template: λL_λ as function of rest-frame wavelength
    lambda_L_lambda_template = jax_interpolator.interpolate_batch(jnp.full(1, fiducial_logL3000))[0]
    wavlen_template = jax_interpolator.wavlen  # Rest-frame wavelength grid [Å]
    
    # Convert to L_λ (luminosity density per unit wavelength)
    # L_λ = λL_λ / λ [erg/s/Å]
    L_lambda_template = lambda_L_lambda_template / wavlen_template
    
    def compute_for_quasar(z):
        # STEP 1: Map observer wavelengths to rest frame
        # λ_obs = (1+z) × λ_rf  →  λ_rf = λ_obs / (1+z)
        lambda_rf = lambda_obs_grid / (1 + z)  # Rest-frame wavelengths [Å]
        
        # STEP 2: Get luminosity density L_λ^rf at these rest-frame wavelengths
        # This is the intrinsic SED of the quasar
        L_lambda_rf = jnp.interp(lambda_rf, wavlen_template, L_lambda_template)  # [erg/s/Å]
        
        # STEP 3: Compute weighted average
        # The filter S_X is evaluated at observer wavelengths: S_X(λ_obs) = S_X((1+z)λ_rf)
        # This is already baked into our setup because:
        #   - S_X is defined on lambda_obs_grid
        #   - lambda_rf corresponds to those same grid points via λ_rf = λ_obs/(1+z)
        
        # Numerator: ∫ λ_rf × L_λ^rf(λ_rf) × S_X((1+z)λ_rf) dλ_rf
        numerator = jnp.trapezoid(lambda_rf * L_lambda_rf * S_X, lambda_rf)
        
        # Denominator: ∫ L_λ^rf(λ_rf) × S_X((1+z)λ_rf) dλ_rf
        denominator = jnp.trapezoid(L_lambda_rf * S_X, lambda_rf)
        
        # The (1+z) from F_λ = L_λ/(4πd_L²(1+z)) cancels in this ratio!
        # We never needed to compute d_L or divide by (1+z)
        return numerator / denominator
    
    return jax.vmap(compute_for_quasar)(z_vals)


def compute_effective_lambda_rf_simplified_single_sed_optimized(
    z_vals,
    wave_grids,
    S_unifieds,
    jax_interpolator,
    fiducial_logL3000=45.0,
    band='GPb',
):
    """
    Memory-safe equivalent of compute_effective_lambda_rf_simplified for the
    current single-template SED model.

    A shared fiducial SED avoids materialising identical template arrays
    for each source.
    """
    z_vals = jnp.asarray(z_vals)

    lambda_obs_grid = wave_grids[band]
    S_X = S_unifieds[band]

    lambda_L_lambda_template = jax_interpolator.interpolate_batch(jnp.full(1, fiducial_logL3000))[0]
    wavlen_template = jax_interpolator.wavlen
    L_lambda_template = lambda_L_lambda_template / wavlen_template

    def compute_for_quasar(z):
        lambda_rf = lambda_obs_grid / (1 + z)
        L_lambda_rf = jnp.interp(lambda_rf, wavlen_template, L_lambda_template)
        numerator = jnp.trapezoid(lambda_rf * L_lambda_rf * S_X, lambda_rf)
        denominator = jnp.trapezoid(L_lambda_rf * S_X, lambda_rf)
        return numerator / denominator

    return jax.vmap(compute_for_quasar)(z_vals)

# ==============================================================================
# K-CORRECTION CALCULATION
# ==============================================================================
def compute_K_correction(z_vals, jax_interpolator, wave_grids, S_unifieds,
                        obs_band='GPb', rest_band='GPb', verbose=False):
    """
    Compute K-correction from observed to rest-frame band.

    K_QR(z) = -2.5 log10[(1/(1+z)) × (<L_λ>_R^obs / <L_λ>_Q)] + (ZP_Q - ZP_R)

    IMPORTANT: This formula assumes obs_band and rest_band are in the SAME
    photometric system (both Gaia VEGAMAG or both ZTF AB). Cross-instrument
    K-corrections (e.g., Gaia→ZTF) would require additional terms to convert
    between magnitude systems, which is NOT implemented here.

    Typical usage:
      - Gaia G → Gaia G (same band): ZP_Q - ZP_R = 0
      - ZTF g → ZTF g (same band): ZP_Q - ZP_R = 0
      - Gaia G → Gaia BP (cross-band, same instrument): valid
      - ZTF g → Gaia G (cross-instrument): NOT VALID with this formula

    Parameters
    ----------
    z_vals : array
        Redshifts [n_quasars]
    jax_interpolator : JAXQSOSedInterpolator
        Must have precomputed z interpolation
    wave_grids : dict
        Wavelength grids
    S_unifieds : dict
        Filter curves
    obs_band : str
        Observed band (where magnitude is measured)
    rest_band : str
        Rest-frame band (for absolute magnitude)
    verbose : bool
        Print debug info

    Returns
    -------
    K_QR : array [n_quasars]
        K-correction values (float64)

    Raises
    ------
    ValueError
        If obs_band and rest_band are from different photometric systems
        (e.g., Gaia VEGAMAG and ZTF AB).

    Notes
    -----
    Memory optimization: The intermediate SED arrays are computed in float32 to reduce
    GPU memory usage by half. This is safe because:
    1. K-corrections are small numbers (typically -1 to +3 magnitudes)
    2. float32 provides ~7 significant digits, far exceeding the ~1% precision needed
    3. The final result is cast back to float64 for compatibility with the rest of the pipeline
    4. The K-correction is a ratio of integrals, so any small precision loss cancels out
    """
    # =========================================================================
    # GUARD: Prevent cross-instrument K-corrections (different mag systems)
    # =========================================================================
    GAIA_BANDS = {'GPb', 'BPPb', 'RPPb'}
    ZTF_BANDS = {'ZTF_g', 'ZTF_r'}

    obs_is_gaia = obs_band in GAIA_BANDS
    rest_is_gaia = rest_band in GAIA_BANDS
    obs_is_ztf = obs_band in ZTF_BANDS
    rest_is_ztf = rest_band in ZTF_BANDS

    if (obs_is_gaia and rest_is_ztf) or (obs_is_ztf and rest_is_gaia):
        raise ValueError(
            f"Cross-instrument K-correction not supported: {obs_band} -> {rest_band}. "
            f"Gaia bands use VEGAMAG, ZTF bands use AB. Mixing photometric systems "
            f"requires additional magnitude system conversion terms not implemented here."
        )

    n_quasars = len(z_vals)
    fiducial_logL3000 = 45.0

    if verbose:
        print(f"\n=== K-CORRECTION COMPUTATION ===")
        print(f"  N_quasars: {n_quasars}")
        print(f"  Obs band: {obs_band} → Rest band: {rest_band}")
        print(f"  Using float32 for intermediate arrays (memory optimization)")

    # =========================================================================
    # MEMORY OPTIMIZATION: Use float32 + normalization for large intermediate arrays
    # =========================================================================
    # The SED template arrays have shape (N_quasars, N_wavelengths), e.g. (60000, 163840)
    # In float64: 60000 × 163840 × 8 bytes = ~78 GB → GPU out-of-memory
    # In float32: 60000 × 163840 × 4 bytes = ~39 GB → fits in 40GB A100
    #
    # NORMALIZATION: Raw λL_λ values are ~10^44-46, which after integration give
    # numerators ~10^47-49, exceeding float32 max (~3.4×10^38). We normalize by
    # dividing by SED_NORM=10^45. Since K-correction is a RATIO of integrals:
    #   K = -2.5 * log10(numerator_R / numerator_Q / (1+z))
    # the normalization factor cancels exactly and does not affect the result.
    # =========================================================================
    SED_NORM = 1e45  # Normalization to keep values in float32 range

    # Cast inputs to float32 for memory-efficient computation
    z_vals_f32 = z_vals.astype(jnp.float32)
    wavlen_f32 = jax_interpolator.wavlen.astype(jnp.float32)

    # Get template SED, normalize in float64 FIRST, then cast to float32
    # CRITICAL: Must divide before casting! Raw λL_λ ~ 10^44-46 overflows float32 max (~3.4e38).
    # After /SED_NORM → ~10^-1 to 10^1, which is safe for float32.
    lambda_L_lambda_f64 = jax_interpolator.interpolate_batch(
        jnp.full(n_quasars, fiducial_logL3000)
    )
    lambda_L_lambda_f32 = (lambda_L_lambda_f64 / SED_NORM).astype(jnp.float32)
    # L_λ = λL_λ / λ, now ~10^-5 to 10^-3 (well within float32 range)
    L_lambda_f32 = lambda_L_lambda_f32 / wavlen_f32[None, :]

    # Filter data in float32
    lambda_Q_f32 = wave_grids[rest_band].astype(jnp.float32)
    S_Q_f32 = S_unifieds[rest_band].astype(jnp.float32)
    lambda_R_f32 = wave_grids[obs_band].astype(jnp.float32)
    S_R_f32 = S_unifieds[obs_band].astype(jnp.float32)

    # <L_λ>_Q: Filter-averaged luminosity density in rest band
    # Integrands now ~10^-2 to 10^1, numerators ~10^1 to 10^4 (safe for float32)
    L_lambda_Q_f32 = jax.vmap(lambda L: jnp.interp(lambda_Q_f32, wavlen_f32, L))(L_lambda_f32)
    integrand_Q_f32 = lambda_Q_f32[None, :] * L_lambda_Q_f32 * S_Q_f32[None, :]
    numerator_Q_f32 = jnp.trapezoid(integrand_Q_f32, lambda_Q_f32, axis=1)
    denominator_Q_f32 = jnp.trapezoid(lambda_Q_f32 * S_Q_f32, lambda_Q_f32)
    L_lambda_avg_Q_f32 = numerator_Q_f32 / denominator_Q_f32

    # <L_λ>_R^obs: Filter-averaged luminosity density in observed band
    lambda_obs_f32 = lambda_R_f32[None, :]
    lambda_rest_for_obs_f32 = lambda_obs_f32 / (1.0 + z_vals_f32[:, None])
    L_lambda_at_rest_f32 = jax.vmap(
        lambda L, lam_rf: jnp.interp(lam_rf, wavlen_f32, L)
    )(L_lambda_f32, lambda_rest_for_obs_f32)
    integrand_R_f32 = lambda_obs_f32 * L_lambda_at_rest_f32 * S_R_f32[None, :]
    numerator_R_f32 = jnp.trapezoid(integrand_R_f32, lambda_R_f32, axis=1)
    denominator_R_f32 = jnp.trapezoid(lambda_R_f32 * S_R_f32, lambda_R_f32)
    L_lambda_avg_R_obs_f32 = numerator_R_f32 / denominator_R_f32

    # K-correction formula (still in float32)
    # The SED_NORM factor cancelled in the ratio L_lambda_avg_R / L_lambda_avg_Q
    ratio_f32 = (L_lambda_avg_R_obs_f32 / L_lambda_avg_Q_f32) / (1.0 + z_vals_f32)
    ZP_Q = get_zp_mag(rest_band)
    ZP_R = get_zp_mag(obs_band)
    K_QR_f32 = -2.5 * jnp.log10(ratio_f32) + (ZP_Q - ZP_R)

    # =========================================================================
    # Cast result back to float64 for compatibility with rest of pipeline
    # =========================================================================
    # The K-correction is now a 1D array of shape (N_quasars,) with small values,
    # so converting to float64 is cheap and ensures consistency with other computations
    # (e.g., distance modulus, likelihood) that require float64 precision.
    # =========================================================================
    K_QR = K_QR_f32.astype(jnp.float64)

    if verbose:
        print(f"  K-correction range: [{K_QR.min():.3f}, {K_QR.max():.3f}]")

    return K_QR


def compute_K_correction_single_sed_optimized(
    z_vals,
    jax_interpolator,
    wave_grids,
    S_unifieds,
    obs_band='GPb',
    rest_band='GPb',
    fiducial_logL3000=45.0,
    verbose=False,
):
    """
    Memory-safe equivalent of compute_K_correction for the current single-template
    SED model.

    The current SED interpolator scales one shared template by a single
    luminosity-like normalization. For a fixed fiducial value, every row of the
    interpolated SED batch is identical, so we compute the template once and map
    only over the genuinely z-dependent part of the calculation.
    """
    GAIA_BANDS = {'GPb', 'BPPb', 'RPPb'}
    ZTF_BANDS = {'ZTF_g', 'ZTF_r'}

    obs_is_gaia = obs_band in GAIA_BANDS
    rest_is_gaia = rest_band in GAIA_BANDS
    obs_is_ztf = obs_band in ZTF_BANDS
    rest_is_ztf = rest_band in ZTF_BANDS

    if (obs_is_gaia and rest_is_ztf) or (obs_is_ztf and rest_is_gaia):
        raise ValueError(
            f"Cross-instrument K-correction not supported: {obs_band} -> {rest_band}. "
            f"Gaia bands use VEGAMAG, ZTF bands use AB. Mixing photometric systems "
            f"requires additional magnitude system conversion terms not implemented here."
        )

    z_vals_f32 = jnp.asarray(z_vals, dtype=jnp.float32)
    wavlen_f32 = jax_interpolator.wavlen.astype(jnp.float32)
    lambda_Q_f32 = wave_grids[rest_band].astype(jnp.float32)
    S_Q_f32 = S_unifieds[rest_band].astype(jnp.float32)
    lambda_R_f32 = wave_grids[obs_band].astype(jnp.float32)
    S_R_f32 = S_unifieds[obs_band].astype(jnp.float32)
    SED_NORM = 1e45

    if verbose:
        print(f"\n=== K-CORRECTION COMPUTATION (OPTIMIZED SINGLE SED) ===")
        print(f"  N_quasars: {len(z_vals_f32)}")
        print(f"  Obs band: {obs_band} -> Rest band: {rest_band}")
        print(f"  Shared fiducial SED computed once at logL3000={fiducial_logL3000}")

    lambda_L_lambda_template_f64 = jax_interpolator.interpolate_batch(
        jnp.full(1, fiducial_logL3000)
    )[0]
    lambda_L_lambda_template_f32 = (lambda_L_lambda_template_f64 / SED_NORM).astype(jnp.float32)
    L_lambda_template_f32 = lambda_L_lambda_template_f32 / wavlen_f32

    L_lambda_Q_f32 = jnp.interp(lambda_Q_f32, wavlen_f32, L_lambda_template_f32)
    integrand_Q_f32 = lambda_Q_f32 * L_lambda_Q_f32 * S_Q_f32
    numerator_Q_f32 = jnp.trapezoid(integrand_Q_f32, lambda_Q_f32)
    denominator_Q_f32 = jnp.trapezoid(lambda_Q_f32 * S_Q_f32, lambda_Q_f32)
    L_lambda_avg_Q_f32 = numerator_Q_f32 / denominator_Q_f32

    def compute_obs_band_average(z):
        lambda_rest_for_obs_f32 = lambda_R_f32 / (1.0 + z)
        L_lambda_at_rest_f32 = jnp.interp(lambda_rest_for_obs_f32, wavlen_f32, L_lambda_template_f32)
        integrand_R_f32 = lambda_R_f32 * L_lambda_at_rest_f32 * S_R_f32
        numerator_R_f32 = jnp.trapezoid(integrand_R_f32, lambda_R_f32)
        denominator_R_f32 = jnp.trapezoid(lambda_R_f32 * S_R_f32, lambda_R_f32)
        return numerator_R_f32 / denominator_R_f32

    L_lambda_avg_R_obs_f32 = jax.vmap(compute_obs_band_average)(z_vals_f32)
    ratio_f32 = (L_lambda_avg_R_obs_f32 / L_lambda_avg_Q_f32) / (1.0 + z_vals_f32)
    ZP_Q = get_zp_mag(rest_band)
    ZP_R = get_zp_mag(obs_band)
    K_QR_f32 = -2.5 * jnp.log10(ratio_f32) + (ZP_Q - ZP_R)
    K_QR = K_QR_f32.astype(jnp.float64)

    if verbose:
        print(f"  K-correction range: [{K_QR.min():.3f}, {K_QR.max():.3f}]")

    return K_QR

# ==============================================================================
# M_REST ↔ LOGL3000 MAPPING
# ==============================================================================

def compute_M_rest_logL3000_linear_coeff(jax_interpolator, wave_grids, S_unifieds, rest_band='GPb',
                                         verify=True):
    """
    Compute the linear coefficient for M_rest ↔ logL3000 conversion.

    Since L_band ∝ L_3000 (SED template is fixed), the relationship is EXACTLY linear:
        M_rest = a_band - 2.5 × logL3000_tag
        logL3000_tag = (a_band - M_rest) / 2.5

    NOTE: logL3000 here refers to the SED template TAG value (scaling parameter),
    NOT the actual monochromatic luminosity at 3000Å. This is physically meaningful
    because we normalize the SED to the continuum level, and the tag represents
    the scaling applied to the template.

    Parameters
    ----------
    jax_interpolator : JAXQSOSedInterpolator
    wave_grids : dict
    S_unifieds : dict
    rest_band : str
    verify : bool
        If True, verify that linear method matches grid interpolation

    Returns
    -------
    a_band : float
        Linear coefficient: logL3000_tag = (a_band - M_rest) / 2.5
    """
    # Template SED - uses TAG definition of logL3000
    wavlen = jax_interpolator.wavlen
    template_lamLlam = jax_interpolator.lambda_L_lambda_rest
    template_ref_logL3000 = jax_interpolator.reference_logL3000  # TAG value from template

    # Filter data
    wave_grid_rest = wave_grids[rest_band]
    S_rest = S_unifieds[rest_band]
    ZP_rest = get_zp_mag(rest_band)
    flux_scale = get_flux_scale(rest_band)
    d_10pc_cm = 10.0 * 3.0857e18

    def compute_M_rest_at_logL3000(logL3000_tag):
        """Compute M_rest for a given logL3000 TAG value."""
        # Scale template using TAG definition
        scale_factor = 10**(logL3000_tag - template_ref_logL3000)
        lambda_L_lambda = template_lamLlam * scale_factor
        L_lambda = lambda_L_lambda / wavlen

        # Interpolate onto filter grid
        L_lambda_rest = jnp.interp(wave_grid_rest, wavlen, L_lambda)

        # Filter-averaged luminosity
        numerator = jnp.trapezoid(L_lambda_rest * wave_grid_rest * S_rest, wave_grid_rest)
        denominator = jnp.trapezoid(wave_grid_rest * S_rest, wave_grid_rest)
        L_band = numerator / denominator

        # Flux at 10 pc and absolute magnitude
        # flux_scale converts erg/s/cm²/Å to the units expected by the ZP
        F_10pc = L_band / (4.0 * jnp.pi * d_10pc_cm**2)
        M_rest = -2.5 * jnp.log10(F_10pc / flux_scale) - ZP_rest
        return float(M_rest)

    # Compute at reference point
    ref_logL3000 = 45.0
    M_rest_ref = compute_M_rest_at_logL3000(ref_logL3000)

    # Linear coefficient: M_rest = a_band - 2.5 * logL3000_tag
    # So: a_band = M_rest_ref + 2.5 * ref_logL3000
    a_band = M_rest_ref + 2.5 * ref_logL3000

    # Verify linearity by checking at another point
    if verify:
        test_logL3000 = 46.0
        M_rest_test_actual = compute_M_rest_at_logL3000(test_logL3000)
        M_rest_test_linear = a_band - 2.5 * test_logL3000
        diff = abs(M_rest_test_actual - M_rest_test_linear)
        if diff > 1e-6:
            print(f"    ⚠ WARNING: Linear approximation error = {diff:.6f} mag at logL3000={test_logL3000}")
        # Also verify inverse
        logL3000_recovered = (a_band - M_rest_test_actual) / 2.5
        logL3000_diff = abs(logL3000_recovered - test_logL3000)
        if logL3000_diff > 1e-6:
            print(f"    ⚠ WARNING: logL3000 recovery error = {logL3000_diff:.6f} dex")

    return a_band


# ==============================================================================
# COSMOLOGY: DISTANCE MODULUS (OPTIMIZED)
# ==============================================================================
# ==============================================================================
# COSMOLOGY: DISTANCE MODULUS (ALWAYS FULL CPL)
# ==============================================================================
def make_distmod_fn(z_vals, n_steps):
    """
    Create distance modulus function using the full CPL w(z) parametrization.

    We ALWAYS use:
        w(z)   = w0 + wa * z / (1 + z)
        E(z)^2 = Ωm (1+z)^3 + ΩΛ(z)
        ΩΛ(z)  = (1 - Ωm) * [(1+z)^(3(1+w0+wa)) * exp(-3 wa z/(1+z))]

    Parameters
    ----------
    z_vals : array
        Redshifts of the objects.
    n_steps : int
        Number of integration steps for the non-uniform z grid.

    Returns
    -------
    distmod_fn : callable
        distmod_fn(H0, Om0, w0, wa) → μ(z) array
    """
    z_vals = jnp.asarray(z_vals)

    # Non-uniform integration grid (biased towards low z for accuracy)
    t = jnp.linspace(0.0, 1.0, n_steps)
    z_grid = t**0.7
    z_eval = z_vals[:, None] * z_grid[None, :]

    one_plus_z_eval = 1.0 + z_eval
    one_plus_z_eval_sq = one_plus_z_eval * one_plus_z_eval
    one_plus_z_eval_cube = one_plus_z_eval_sq * one_plus_z_eval

    dz_half = z_vals[:, None] * (z_grid[1:] - z_grid[:-1])[None, :] * 0.5
    one_plus_z_vals = 1.0 + z_vals

    @jax.jit
    def distmod_fn(H0, Om0, w0, wa):
        """
        Distance modulus μ(z) for given cosmology (H0, Om0, w0, wa).

        You get:
          - ΛCDM if w0 = -1, wa = 0
          - wCDM if wa = 0, w0 free
          - full CPL if wa ≠ 0
        """
        Ol0 = 1.0 - Om0

        # CPL dark energy factor
        power = 3.0 * (1.0 + w0 + wa)
        dark_energy_factor = (one_plus_z_eval ** power) * jnp.exp(
            -3.0 * wa * z_eval / one_plus_z_eval
        )

        E_z = jnp.sqrt(Om0 * one_plus_z_eval_cube + Ol0 * dark_energy_factor)
        inv_E = 1.0 / E_z

        # Comoving distance integral by trapezoid rule in z
        chi = jnp.sum(dz_half * (inv_E[:, 1:] + inv_E[:, :-1]), axis=1)

        d_L_Mpc = one_plus_z_vals * (C_LIGHT / H0) * chi
        return 5.0 * jnp.log10(d_L_Mpc) + 25.0

    return distmod_fn
