#!/usr/bin/env python
"""
Generate the injected source population for cosmological closure tests.

Sample redshifts and luminosities and compute the variability and photometric
quantities implied by the cosmology and SED/passband model.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
from framework import PROJECT_ROOT
from framework import variability_utils as su



PARENT_SCHEMA = "fullclosure_truth_parent"
DEFAULT_OUTPUT_NAME = 'simulated_population.npz'
DEFAULT_SUMMARY_NAME = 'population_settings.json'


LAMBDA_0 = 3000.0  # Angstrom


def _map_stage1_band_to_passband(stage1_band: str) -> str:
    b = str(stage1_band).strip()
    if not b:
        raise ValueError("Empty --band; expected e.g. 'gaia_g' or 'ztf_r'.")
    bl = b.lower()
    mapping = {
        "gaia_g": "GPb",
        "gaia_gp": "GPb",
        "gaia_gpb": "GPb",
        "gaia_bp": "BPPb",
        "gaia_bpp": "BPPb",
        "gaia_rp": "RPPb",
        "gaia_rpp": "RPPb",
        "ztf_g": "ZTF_g",
        "ztf_r": "ZTF_r",
    }
    return mapping.get(bl, b)


def _sample_full_box(
    rng: np.random.Generator,
    n_objects: int,
    z_min: float,
    z_max: float,
    logl_min: float,
    logl_max: float,
    z_sampling: str,
    h0: float,
    om0: float,
    w0: float,
    wa: float,
    n_steps_mu: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw source IDs, redshifts and log10(L3000) for the injected population.
    
    Luminosity is uniform in its logarithm. Redshift is uniform either in z or in
    flat-cosmology comoving volume, with the latter inverted on the distance grid."""
    source_id = np.arange(1, n_objects + 1, dtype=np.int64)
    if z_sampling == "uniform_z":
        z = rng.uniform(z_min, z_max, size=n_objects)
    elif z_sampling == "uniform_comoving_volume":
        if not np.isfinite(z_min) or not np.isfinite(z_max) or float(z_max) <= float(z_min):
            raise ValueError(
                f"uniform_comoving_volume requires finite z_min < z_max, got z_min={z_min}, z_max={z_max}"
            )
        # Build the distance-modulus function on the same grid we later use for
        # chi(z) interpolation; otherwise the sampled redshift distribution can collapse.
        z_grid = np.linspace(float(z_min), float(z_max), int(n_steps_mu), dtype=np.float64)
        distmod_fn = su.make_distmod_fn(z_grid, n_steps=n_steps_mu)
        mu_grid = np.asarray(distmod_fn(h0, om0, w0, wa), dtype=np.float64)
        dl_grid_mpc = 10.0 ** ((mu_grid - 25.0) / 5.0)
        chi_grid = dl_grid_mpc / (1.0 + z_grid)
        chi_min = float(chi_grid[0])
        chi_max = float(chi_grid[-1])
        u = rng.uniform(0.0, 1.0, size=n_objects)
        chi = ((1.0 - u) * chi_min**3 + u * chi_max**3) ** (1.0 / 3.0)
        z = np.interp(chi, chi_grid, z_grid)
    else:
        raise ValueError(f"Unknown z sampling mode: {z_sampling}")
    logl3000 = rng.uniform(logl_min, logl_max, size=n_objects)
    return source_id, z, logl3000


def _default_scenarios() -> list[dict]:
    return [
        {"name": "full_box", "remove_highL_lowz": False, "remove_lowL_highz": False},
        {"name": "no_highL_lowz", "remove_highL_lowz": True, "remove_lowL_highz": False},
        {"name": "no_lowL_highz", "remove_highL_lowz": False, "remove_lowL_highz": True},
        {"name": "combined_cutouts", "remove_highL_lowz": True, "remove_lowL_highz": True},
    ]


def _default_analysis_scenarios() -> list[dict]:
    return [
        {"name": "full_box", "remove_highL_lowz": False, "remove_lowL_highz": False},
    ]


def _load_scenarios(scenarios_json: str | None) -> list[dict]:
    if scenarios_json is None:
        return _default_scenarios()

    path = Path(scenarios_json).expanduser()
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "scenarios" in payload:
        scenarios = payload["scenarios"]
    elif isinstance(payload, list):
        scenarios = payload
    else:
        raise ValueError("--scenarios-json must be a list or a dict with key 'scenarios'.")

    if not scenarios:
        raise ValueError("--scenarios-json provided but contains no scenarios.")

    names = []
    out = []
    for i, s in enumerate(scenarios):
        if not isinstance(s, dict):
            raise ValueError(f"Scenario index {i} is not an object.")
        if "name" not in s:
            raise ValueError(f"Scenario index {i} missing required key 'name'.")
        name = str(s["name"]).strip()
        if not name:
            raise ValueError(f"Scenario index {i} has empty name.")
        if name in names:
            raise ValueError(f"Duplicate scenario name: {name}")
        names.append(name)
        out.append(s)
    return out


def _prepare_sed_band_terms(
    *,
    stage1_band: str,
    sed_grid: str,
    passband_file: str,
    ztf_filter_dir: str,
    n_integration_wavelength: int,
    chunk_size: int,
    z: np.ndarray,
    lambda_rf_fiducial_logl3000: float,
) -> dict[str, np.ndarray | float | str]:
    """Prepare the fixed-SED mapping between luminosity, magnitude and wavelength.
    
    Use the requested fiducial luminosity for the SED integrals. Return a_band in
    M_band = a_band - 2.5*log10(L3000), per-source K-corrections and effective
    rest-frame wavelengths in Angstrom, plus the passband identifiers."""
    passband = _map_stage1_band_to_passband(stage1_band)

    su.passband_data = su.load_passband_data(passband_file)
    su.ztf_filter_data.clear()
    ztf_g_path = os.path.join(ztf_filter_dir, "Palomar_ZTF.g.dat")
    ztf_r_path = os.path.join(ztf_filter_dir, "Palomar_ZTF.r.dat")
    if os.path.exists(ztf_g_path):
        su.ztf_filter_data["ZTF_g"] = su.load_ztf_filter(ztf_g_path)
    if os.path.exists(ztf_r_path):
        su.ztf_filter_data["ZTF_r"] = su.load_ztf_filter(ztf_r_path)

    if passband.startswith("ZTF_") and passband not in su.ztf_filter_data:
        raise FileNotFoundError(
            f"{passband} filter not loaded; expected file under --ztf-filter-dir: {ztf_filter_dir}"
        )

    jax_interpolator = su.create_jax_interpolator(sed_grid)
    wave_grids, s_unifieds = su.setup_wavelength_grid([passband], n_integration_wavelength)
    a_band = float(
        su.compute_M_rest_logL3000_linear_coeff(
            jax_interpolator, wave_grids, s_unifieds, rest_band=passband
        )
    )
    # Use the same population chunks for the SED calculations.
    z = np.asarray(z, dtype=np.float64)
    k_corr = np.empty_like(z)
    lambda_rf_eff = np.empty_like(z)
    for start in range(0, z.size, chunk_size):
        rows = slice(start, start + chunk_size)
        k_corr[rows] = su.compute_K_correction_single_sed_optimized(
            z[rows], jax_interpolator, wave_grids, s_unifieds,
            obs_band=passband, rest_band=passband,
            fiducial_logL3000=lambda_rf_fiducial_logl3000,
        )
        lambda_rf_eff[rows] = su.compute_effective_lambda_rf_simplified_single_sed_optimized(
            z[rows], wave_grids, s_unifieds, jax_interpolator,
            fiducial_logL3000=lambda_rf_fiducial_logl3000, band=passband,
        )
    log10_lambda_ratio = np.log10(lambda_rf_eff / LAMBDA_0)

    return {
        "stage1_band": stage1_band,
        "passband": passband,
        "a_band": a_band,
        "k_corr": k_corr,
        "lambda_rf_eff": lambda_rf_eff,
        "log10_lambda_ratio": log10_lambda_ratio,
    }


@partial(jax.jit, static_argnames=("n_steps_mu",))
def _distance_modulus(
    z: np.ndarray,
    h0: float,
    om0: float,
    w0: float,
    wa: float,
    n_steps_mu: int,
) -> np.ndarray:
    return su.make_distmod_fn(z, n_steps=n_steps_mu)(h0, om0, w0, wa)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Generate an injected source population for cosmological closure tests."
        )
    )

    ap.add_argument("--output-dir", required=True, help="Directory to write the truth-only parent NPZ and summary JSON.")
    ap.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME, help=f"Parent NPZ filename. Default: {DEFAULT_OUTPUT_NAME}.")
    ap.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME, help=f"Summary JSON filename. Default: {DEFAULT_SUMMARY_NAME}.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--band", type=str, default="gaia_g", help="Stage-1 band label, e.g. gaia_g or ztf_r.")

    ap.add_argument("--sed-grid", required=True, help="Path to the Stage-2 SED grid NPZ.")
    ap.add_argument("--passband-file", required=True, help="Path to Gaia passband.dat.")
    ap.add_argument("--ztf-filter-dir", default="assets/ZTF_filters", help="Directory containing ZTF filter files.")
    ap.add_argument(
        "--n-integration-wavelength",
        type=int,
        default=1000,
        help="Number of wavelength grid points for SED/filter integrals.",
    )

    ap.add_argument("--n-objects", type=int, default=120000, help="Number of objects in the initial full box.")
    ap.add_argument("--chunk-size", type=int, default=50000, help="Number of objects per truth-generation chunk.")
    ap.add_argument(
        "--z-sampling",
        type=str,
        default="uniform_comoving_volume",
        choices=["uniform_comoving_volume", "uniform_z"],
        help="Redshift sampling: uniform in comoving volume or uniform in z.",
    )

    ap.add_argument("--z-min", type=float, default=0.2)
    ap.add_argument("--z-max", type=float, default=4.0)
    ap.add_argument("--logl-min", type=float, default=43.0)
    ap.add_argument("--logl-max", type=float, default=49.0)

    ap.add_argument("--H0", type=float, default=70.0)
    ap.add_argument("--Om0", type=float, default=0.3)
    ap.add_argument("--w0", type=float, default=-1.0)
    ap.add_argument("--wa", type=float, default=0.0)
    ap.add_argument("--n-steps-mu", type=int, default=1024, help="Integration steps for distance modulus calculation.")

    ap.add_argument("--logL-pivot", type=float, default=45.5, help="True Stage-2 pivot used in the D relation.")
    ap.add_argument("--D-pivot", type=float, default=-4.49, help="True D_pivot for the mock relation.")
    ap.add_argument("--d-D", type=float, default=-0.85, help="True d_D for the mock relation.")
    ap.add_argument("--alpha-D", type=float, default=2.51, help="True alpha_D for the mock relation.")
    ap.add_argument(
        "--log-sigma-int",
        type=float,
        default=np.log(0.18),
        help="True log_sigma_int for object-level intrinsic scatter in log10(D); default is ln(0.18).",
    )
    ap.add_argument(
        "--lambda-rf-fiducial-logl3000",
        type=float,
        default=45.0,
        help="Fiducial logL3000 used in compute_effective_lambda_rf_simplified, matching Stage 2.",
    )

    ap.add_argument("--vol-z-max", type=float, default=1.5, help="Default highL-lowz cut helper metadata only.")
    ap.add_argument("--vol-upper-at-zmin", type=float, default=45.8, help="Default highL-lowz envelope metadata only.")
    ap.add_argument("--vol-upper-slope", type=float, default=0.80, help="Default highL-lowz envelope metadata only.")
    ap.add_argument("--sel-z-min", type=float, default=1.2, help="Default lowL-highz cut helper metadata only.")
    ap.add_argument("--sel-lower-at-sel-zmin", type=float, default=44.4, help="Default lowL-highz envelope metadata only.")
    ap.add_argument("--sel-lower-slope", type=float, default=0.85, help="Default lowL-highz envelope metadata only.")
    ap.add_argument(
        "--scenarios-json",
        type=str,
        default=None,
        help="Optional scenario JSON to store as analysis metadata only; no scenario files are written.",
    )

    return ap


def _generate_truth_observables(
    *,
    rng: np.random.Generator,
    z: np.ndarray,
    logl3000: np.ndarray,
    h0: float,
    om0: float,
    w0: float,
    wa: float,
    n_steps_mu: int,
    a_band: float,
    k_corr: np.ndarray,
    log10_lambda_ratio: np.ndarray,
    d_pivot: float,
    d_slope_logl: float,
    alpha_d: float,
    logl_pivot: float,
    sigma_log10d_intr: float,
) -> dict[str, np.ndarray]:
    """Map the injected population to window magnitudes and variability-rate truth.
    
    Set m_win = a_band + distance_modulus + K - 2.5*log10(L3000). The luminosity
    and wavelength relation predicts the mean of log10(D); independent Gaussian
    scatter with standard deviation sigma_log10d_intr (dex) gives each source's
    injected rate. D is in mag^2 per rest-frame day. L3000 follows the SED scaling
    convention in variability_utils.compute_M_rest_logL3000_linear_coeff."""
    mu = _distance_modulus(
        z=z,
        h0=h0,
        om0=om0,
        w0=w0,
        wa=wa,
        n_steps_mu=n_steps_mu,
    )

    m_win_true = float(a_band) + mu + np.asarray(k_corr, dtype=np.float64) - 2.5 * np.asarray(logl3000, dtype=np.float64)
    log10_d_true_mean = (
        float(d_pivot)
        + float(d_slope_logl) * (np.asarray(logl3000, dtype=np.float64) - float(logl_pivot))
        - float(alpha_d) * np.asarray(log10_lambda_ratio, dtype=np.float64)
    )
    log10_d_true = log10_d_true_mean + rng.normal(0.0, sigma_log10d_intr, size=z.shape[0])

    return {
        "mu": np.asarray(mu, dtype=np.float64),
        "m_win_true": np.asarray(m_win_true, dtype=np.float64),
        "log10_d_true_mean": np.asarray(log10_d_true_mean, dtype=np.float64),
        "log10_d_true": np.asarray(log10_d_true, dtype=np.float64),
    }


def _write_parent_npz(
    *,
    out_path: Path,
    source_id: np.ndarray,
    z: np.ndarray,
    logl3000: np.ndarray,
    mu: np.ndarray,
    m_win_true: np.ndarray,
    log10_d_true_mean: np.ndarray,
    log10_d_true: np.ndarray,
    aux_terms: dict[str, np.ndarray | float | str],
    truth_scalars: dict[str, float],
) -> None:
    """Write per-source injected truth and the shared population parameters.
    
    Source arrays retain their common order. Logarithm bases are explicit in keys:
    true_log10_d is in dex; true_log_sigma_int is the natural log of the scatter
    in log10(D). The schema identifier is read by prepare_closure_simulations.py."""
    np.savez(
        out_path,
        source_id=np.asarray(source_id, dtype=np.int64),
        z=np.asarray(z, dtype=np.float64),
        logl3000=np.asarray(logl3000, dtype=np.float64),
        true_m_win=np.asarray(m_win_true, dtype=np.float64),
        true_distmod=np.asarray(mu, dtype=np.float64),
        true_log10_d_mean=np.asarray(log10_d_true_mean, dtype=np.float64),
        true_log10_d=np.asarray(log10_d_true, dtype=np.float64),
        true_k_corr=np.asarray(aux_terms["k_corr"], dtype=np.float64),
        true_lambda_rf_eff=np.asarray(aux_terms["lambda_rf_eff"], dtype=np.float64),
        true_log10_lambda_ratio=np.asarray(aux_terms["log10_lambda_ratio"], dtype=np.float64),
        true_a_band=np.asarray([float(aux_terms["a_band"])], dtype=np.float64),
        true_lambda_0=np.asarray([float(truth_scalars["lambda_0"])], dtype=np.float64),
        true_H0=np.asarray([float(truth_scalars["H0"])], dtype=np.float64),
        true_Om0=np.asarray([float(truth_scalars["Om0"])], dtype=np.float64),
        true_w0=np.asarray([float(truth_scalars["w0"])], dtype=np.float64),
        true_wa=np.asarray([float(truth_scalars["wa"])], dtype=np.float64),
        true_D_pivot=np.asarray([float(truth_scalars["D_pivot"])], dtype=np.float64),
        true_d_D=np.asarray([float(truth_scalars["d_D"])], dtype=np.float64),
        true_alpha_D=np.asarray([float(truth_scalars["alpha_D"])], dtype=np.float64),
        true_log_sigma_int=np.asarray([float(truth_scalars["log_sigma_int"])], dtype=np.float64),
        true_sigma_int=np.asarray([float(truth_scalars["sigma_int"])], dtype=np.float64),
        true_logL3000_pivot=np.asarray([float(truth_scalars["logL3000_pivot"])], dtype=np.float64),
        true_lambda_rf_fiducial_logl3000=np.asarray(
            [float(truth_scalars["lambda_rf_fiducial_logl3000"])], dtype=np.float64
        ),
        true_stage1_band=np.asarray([str(aux_terms["stage1_band"])], dtype="U32"),
        true_passband=np.asarray([str(aux_terms["passband"])], dtype="U32"),
        parent_schema=np.asarray([PARENT_SCHEMA], dtype="U64"),
        mock_kind=np.asarray([PARENT_SCHEMA], dtype="U64"),
    )


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / str(args.output_name)
    summary_path = out_dir / str(args.summary_name)

    rng = np.random.default_rng(args.seed)
    source_id_all, z_all, logl_all = _sample_full_box(
        rng=rng,
        n_objects=args.n_objects,
        z_min=args.z_min,
        z_max=args.z_max,
        logl_min=args.logl_min,
        logl_max=args.logl_max,
        z_sampling=args.z_sampling,
        h0=args.H0,
        om0=args.Om0,
        w0=args.w0,
        wa=args.wa,
        n_steps_mu=args.n_steps_mu,
    )

    n_objects = int(source_id_all.size)
    chunk_size = max(1, min(int(args.chunk_size), n_objects))
    sed_terms = _prepare_sed_band_terms(
        stage1_band=args.band,
        sed_grid=args.sed_grid,
        passband_file=args.passband_file,
        ztf_filter_dir=args.ztf_filter_dir,
        n_integration_wavelength=args.n_integration_wavelength,
        chunk_size=chunk_size,
        z=z_all,
        lambda_rf_fiducial_logl3000=args.lambda_rf_fiducial_logl3000,
    )

    sigma_int = float(np.exp(args.log_sigma_int))

    mu_all = np.empty(n_objects, dtype=np.float64)
    true_mwin_all = np.empty(n_objects, dtype=np.float64)
    true_log10d_mean_all = np.empty(n_objects, dtype=np.float64)
    true_log10d_all = np.empty(n_objects, dtype=np.float64)
    k_corr_all = np.asarray(sed_terms["k_corr"], dtype=np.float64)
    log10_lambda_ratio_all = np.asarray(sed_terms["log10_lambda_ratio"], dtype=np.float64)

    for start in range(0, n_objects, chunk_size):
        stop = min(start + chunk_size, n_objects)
        obs_chunk = _generate_truth_observables(
            rng=rng,
            z=z_all[start:stop],
            logl3000=logl_all[start:stop],
            h0=args.H0,
            om0=args.Om0,
            w0=args.w0,
            wa=args.wa,
            n_steps_mu=args.n_steps_mu,
            a_band=float(sed_terms["a_band"]),
            k_corr=k_corr_all[start:stop],
            log10_lambda_ratio=log10_lambda_ratio_all[start:stop],
            d_pivot=args.D_pivot,
            d_slope_logl=args.d_D,
            alpha_d=args.alpha_D,
            logl_pivot=args.logL_pivot,
            sigma_log10d_intr=sigma_int,
        )
        mu_all[start:stop] = obs_chunk["mu"]
        true_mwin_all[start:stop] = obs_chunk["m_win_true"]
        true_log10d_mean_all[start:stop] = obs_chunk["log10_d_true_mean"]
        true_log10d_all[start:stop] = obs_chunk["log10_d_true"]

    truth_scalars = {
        "H0": float(args.H0),
        "Om0": float(args.Om0),
        "w0": float(args.w0),
        "wa": float(args.wa),
        "D_pivot": float(args.D_pivot),
        "d_D": float(args.d_D),
        "alpha_D": float(args.alpha_D),
        "log_sigma_int": float(args.log_sigma_int),
        "sigma_int": float(sigma_int),
        "logL3000_pivot": float(args.logL_pivot),
        "lambda_0": float(LAMBDA_0),
        "lambda_rf_fiducial_logl3000": float(args.lambda_rf_fiducial_logl3000),
    }

    _write_parent_npz(
        out_path=out_npz,
        source_id=source_id_all,
        z=z_all,
        logl3000=logl_all,
        mu=mu_all,
        m_win_true=true_mwin_all,
        log10_d_true_mean=true_log10d_mean_all,
        log10_d_true=true_log10d_all,
        aux_terms=sed_terms,
        truth_scalars=truth_scalars,
    )

    scenarios_payload = _load_scenarios(args.scenarios_json) if args.scenarios_json else _default_analysis_scenarios()
    summary = {
        "generator": "generate_closure_population.py",
        "parent_schema": PARENT_SCHEMA,
        "created_files": {
            "parent_npz": str(out_npz),
            "summary_json": str(summary_path),
        },
        "deliberately_absent_fields": [
            "median_m_win",
            "median_log10_d",
            "median_ln_D",
            "true_log10_d_obs",
            "free_flatlnD__lnD__{band}__m_win",
            "free_flatlnD__lnD__{band}__ln_D",
        ],
        "included_truth_fields": [
            "source_id",
            "z",
            "logl3000",
            "true_m_win",
            "true_distmod",
            "true_log10_d_mean",
            "true_log10_d",
            "true_k_corr",
            "true_lambda_rf_eff",
            "true_log10_lambda_ratio",
        ],
        "seed": int(args.seed),
        "band": str(args.band),
        "passband": str(_map_stage1_band_to_passband(args.band)),
        "z_sampling": str(args.z_sampling),
        "input_box": {
            "n_objects": int(args.n_objects),
            "z_min": float(args.z_min),
            "z_max": float(args.z_max),
            "logl_min": float(args.logl_min),
            "logl_max": float(args.logl_max),
        },
        "truth": {
            **truth_scalars,
            "a_band": float(sed_terms["a_band"]),
            "m_win_scatter": "none; true_m_win is deterministic from z, logl3000, cosmology, SED, and K-correction",
            "log10_d_scatter": "true_log10_d = true_log10_d_mean + Normal(0, true_sigma_int)",
        },
        "sed_setup": {
            "sed_grid": str(Path(args.sed_grid).expanduser()),
            "passband_file": str(Path(args.passband_file).expanduser()),
            "ztf_filter_dir": str(Path(args.ztf_filter_dir).expanduser()),
            "n_integration_wavelength": int(args.n_integration_wavelength),
        },
        "analysis_defaults": {
            "chunk_size": int(chunk_size),
            "scenarios": scenarios_payload,
            "cutouts": {
                "volume_corner": {
                    "applies_for_z_le": float(args.vol_z_max),
                    "upper_logL_at_z_min": float(args.vol_upper_at_zmin),
                    "slope_per_z": float(args.vol_upper_slope),
                },
                "selection_corner": {
                    "applies_for_z_ge": float(args.sel_z_min),
                    "lower_logL_at_sel_z_min": float(args.sel_lower_at_sel_zmin),
                    "slope_per_z": float(args.sel_lower_slope),
                },
            },
        },
        "parent": {
            "path": str(out_npz),
            "n_objects": int(n_objects),
            "z_range": [float(np.min(z_all)), float(np.max(z_all))],
            "logl3000_range": [float(np.min(logl_all)), float(np.max(logl_all))],
            "true_m_win_range": [float(np.min(true_mwin_all)), float(np.max(true_mwin_all))],
            "true_log10_d_mean_range": [float(np.min(true_log10d_mean_all)), float(np.max(true_log10d_mean_all))],
            "true_log10_d_range": [float(np.min(true_log10d_all)), float(np.max(true_log10d_all))],
            "lambda_rf_eff_range": [
                float(np.min(np.asarray(sed_terms["lambda_rf_eff"], dtype=np.float64))),
                float(np.max(np.asarray(sed_terms["lambda_rf_eff"], dtype=np.float64))),
            ],
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 78)
    print("Full-closure truth-only parent mock generation complete")
    print("=" * 78)
    print(f"Output directory: {out_dir}")
    print(f"Parent schema: {PARENT_SCHEMA}")
    print(f"Parent NPZ: {out_npz}")
    print(f"Summary: {summary_path}")
    print(f"N objects: {n_objects}")
    print(f"Chunk size: {chunk_size}")


if __name__ == "__main__":
    main()
