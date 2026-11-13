"""Shared photometry transforms with lazy JAX/SED loading."""
from __future__ import annotations

from framework import PROJECT_ROOT

import os
from pathlib import Path

import numpy as np

ROOT = PROJECT_ROOT
DEFAULT_SED_GRID = str(ROOT / 'assets/sed/quasar_continuum_and_lines.npz')
DEFAULT_PASSBAND_FILE = str(ROOT / 'assets/passbands/gaia_g.dat')
DEFAULT_ZTF_FILTER_DIR = str(ROOT / "assets/ZTF_filters")
DEFAULT_N_INTEGRATION_WAVELENGTH = 1000
TEMPLE2021_COSMOLOGY = {"H0": 71.0, "Om0": 0.27, "w0": -1.0, "wa": 0.0, "n_steps_z": 512}
_JAX_INTERPOLATOR_CACHE = None


def get_jax_sed_interpolator():
    """Reuse the fixed-grid interpolator; report load failures without caching them."""
    global _JAX_INTERPOLATOR_CACHE
    if _JAX_INTERPOLATOR_CACHE is None:
        from framework import variability_utils as su

        try:
            _JAX_INTERPOLATOR_CACHE = su.create_jax_interpolator(DEFAULT_SED_GRID)
        except Exception as exc:
            raise RuntimeError(f"Could not load SED template {DEFAULT_SED_GRID}: {exc}") from exc
    return _JAX_INTERPOLATOR_CACHE


def band_to_passband(band: str) -> str:
    b = str(band).strip().lower()
    if b in {"ztf_r", "r"}:
        return "ZTF_r"
    if b in {"ztf_g", "g"}:
        return "ZTF_g"
    if b in {"gaia_g", "gaia"}:
        return "GPb"
    raise ValueError(f"Unknown band '{band}'. Expected one of: ztf_r, ztf_g, gaia_g")


def real_logl3000_from_mwin(
    *,
    z: np.ndarray,
    mwin_med: np.ndarray,
    band: str,
) -> np.ndarray:
    """Convert window-averaged magnitudes using the paper's Temple cosmology."""
    z = np.asarray(z, dtype=np.float64)
    mwin_med = np.asarray(mwin_med, dtype=np.float64)
    if z.shape != mwin_med.shape:
        raise ValueError("z and mwin_med must have the same shape")

    out = np.full(z.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(z) & np.isfinite(mwin_med)
    if not np.any(ok):
        return out

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import jax.numpy as jnp
    from framework import variability_utils as su

    passband = band_to_passband(band)
    su.passband_data = su.load_passband_data(DEFAULT_PASSBAND_FILE)
    if passband.startswith("ZTF_"):
        ztf_g = os.path.join(DEFAULT_ZTF_FILTER_DIR, "Palomar_ZTF.g.dat")
        ztf_r = os.path.join(DEFAULT_ZTF_FILTER_DIR, "Palomar_ZTF.r.dat")
        if os.path.exists(ztf_g):
            su.ztf_filter_data["ZTF_g"] = su.load_ztf_filter(ztf_g)
        if os.path.exists(ztf_r):
            su.ztf_filter_data["ZTF_r"] = su.load_ztf_filter(ztf_r)
        if passband not in su.ztf_filter_data:
            raise RuntimeError(f"Missing ZTF passband data for {passband}")

    jax_interpolator = get_jax_sed_interpolator()

    wave_grids, S_unifieds = su.setup_wavelength_grid(
        [passband],
        int(DEFAULT_N_INTEGRATION_WAVELENGTH),
    )
    z_j = jnp.array(z[ok])
    Kcorr = np.asarray(
        su.compute_K_correction(
            z_j,
            jax_interpolator,
            wave_grids,
            S_unifieds,
            obs_band=passband,
            rest_band=passband,
        ),
        dtype=np.float64,
    )

    cosmo = TEMPLE2021_COSMOLOGY
    distmod_fn = su.make_distmod_fn(z_j, n_steps=int(cosmo["n_steps_z"]))
    distmod = np.asarray(
        distmod_fn(
            float(cosmo["H0"]),
            float(cosmo["Om0"]),
            float(cosmo["w0"]),
            float(cosmo["wa"]),
        ),
        dtype=np.float64,
    )
    a_band = float(
        su.compute_M_rest_logL3000_linear_coeff(
            jax_interpolator,
            wave_grids,
            S_unifieds,
            rest_band=passband,
            verify=True,
        )
    )

    out[ok] = (a_band + distmod + Kcorr - mwin_med[ok]) / 2.5
    return out
