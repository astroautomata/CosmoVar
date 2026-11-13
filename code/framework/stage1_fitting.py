#!/usr/bin/env python
"""
stage1_fitting.py
Fit Kelly et al. (2009) DRW (first-order OU) + jitter GPs to individual AGN light curves in magnitude space.

Kernel: RealTerm(a=σ², c=1/τ) + white noise jitter = first-order Ornstein-Uhlenbeck process
  Autocorrelation: k(Δt) = σ² exp(-|Δt|/τ) + jitter²·δ(Δt)

IMPORTANT: Times are converted to REST-FRAME using the input redshift:
  t_rest = (t_obs - t0) / (1 + z)
so the fitted τ is directly comparable to the true rest-frame τ used in the mocks.

MULTI-BAND SUPPORT:
  - Use --bands to specify the input band names (e.g., "Gaia_G" or "Gaia_G,Gaia_BP,Gaia_RP")
  - Each band is fitted independently
  - Output arrays retain the band suffix, e.g. posterior_ln_D_Gaia_G for Gaia G.

Input:
  - Light-curve FITS file, with optional injected-truth columns for mocks:
      source_id
      redsh
      true_log10_D_{band}, true_log10_sigma2_{band}, true_log10_tau_{band} (for validation)
      {band}_transit_time
      {band}_mag
      {band}_mag_err

Output:
  - NPZ file with arrays (per band):
      source_id                    shape (N_fit,)
      redsh                        shape (N_fit,)
      bands_fitted                 list of bands
        posterior_m_{band}           shape (N_fit, nsamples)       # DRW process mean (long-term center)
      posterior_ln_D_{band}        shape (N_fit, nsamples)       # ln(D), D = σ²/τ
      posterior_ln_tau_{band}      shape (N_fit, nsamples)       # ln(τ) [days, rest-frame]
      posterior_ln_jitter_{band}   shape (N_fit, nsamples)       # ln(jitter) [mag]
        posterior_m_win_{band}       shape (N_fit, nsamples)       # if --mwin-mode trap/both
        posterior_m_win_cont_{band}  shape (N_fit, nsamples)       # if --mwin-mode cont/both
      true_log10_D_{band}          shape (N_fit,)  [if available]
      true_log10_sigma2_{band}     shape (N_fit,)  [if available]
      true_log10_tau_{band}        shape (N_fit,)  [if available]
        true_mean_mag_{band}         shape (N_fit,)  [if available; injected process mean]
        true_m_win                   shape (N_fit,)  [if available; trapezoid latent window mean]
        true_m_win_cont              shape (N_fit,)  [if available; continuous latent window mean draw]

Terminology contract (mock runs)
--------------------------------
There are three distinct magnitude-level quantities:

1) Process mean (scalar per object)
    - Conceptual symbol: m0
    - Stage-1 fitted parameter: posterior_m_{band}
    - Mock truth column: true_mean_mag_{band} (or true_mean_mag in single-band files)
    - Correct recovery target: posterior_m_{band} -> true_mean_mag_{band}

2) Latent realization (vector over epochs)
    - Internal generator variable: often named m_true
    - Not a scalar fit target by itself

3) Latent window-mean summaries (scalar per object)
    - Stage-1 derived parameter: posterior_m_win_{band} (when --mwin-mode is trap/both)
    - Stage-1 derived parameter: posterior_m_win_cont_{band} (when --mwin-mode is cont/both)
    - Mock truth: true_m_win when provided by the generator
    - Mock truth: true_m_win_cont when provided by the generator
    - Correct recovery target: posterior_m_win_{band} -> true_m_win
    - Correct recovery target: posterior_m_win_cont_{band} -> true_m_win_cont

Important comparison rule:
  - Do not compare posterior_m_win to the full latent path array (m_true): scalar vs vector mismatch.
  - Do not use true_m_win as the truth target for posterior_m: they are only equal in expectation,
     not object-by-object.

Stored DRW parameter conventions:
  - posterior_ln_D, posterior_ln_tau and posterior_ln_jitter store natural logs.
  - posterior_m and posterior_m_win draws/conditional means remain in magnitudes.
  - posterior_m_win_var* arrays store variances in mag^2.
  - true_log10_* columns store base-10 logs; true_ln_* columns store natural logs.
  - Convert a stored ln(x) to log10(x) by dividing by ln(10).

Sampling diagnostics:
  - diag_iat_ln_D_{band} is the IAT of ln_D in the ordered post-burn chain.
  - diag_neff_ln_D_{band} is n_post / IAT, not the exported draw count.
  - Derived ln_D diagnostics are reporting-only for amplitude parameterizations.
  - NaN denotes an unavailable estimate; these fields do not certify convergence.
"""

import argparse
import os
import time

import numpy as np
from astropy.table import Table

from framework.drw_fitting import run_drw_mcmc_jitter

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

SEED_SCHEME = "input-row-band-v1"

# Canonical flat-box prior bounds matching drw_fitting.py (free_flatlnD model).
# ln_D bounds derived from: ln_sigma2 ∈ [-20, 20], ln_tau ∈ [-6, 15]
#   → ln_D_min = -20 - 15 = -35,  ln_D_max = 20 - (-6) = 26
PRIOR_M_MIN, PRIOR_M_MAX = 10.0, 25.0
PRIOR_LN_D_MIN, PRIOR_LN_D_MAX = -35.0, 26.0
PRIOR_LN_TAU_MIN, PRIOR_LN_TAU_MAX = -6.0, 15.0
PRIOR_LN_JITTER_MIN, PRIOR_LN_JITTER_MAX = np.log(1e-8), np.log(1.0)
# lnsigma2_lntau_flat model: flat box on (ln_sigma2, ln_tau).
# Spans the same extreme ln_D values as lnD_lntau_flat. At these default
# bounds the marginal ln_D density is trapezoidal, rather than uniform.
PRIOR_LN_SIGMA2_MIN, PRIOR_LN_SIGMA2_MAX = -20.0, 20.0
# lnsigma_lntau_flat: flat box on (ln_sigma, ln_tau).
# ln_sigma in [-10, 10] maps to ln_sigma2 in [-20, 20], so at these defaults
# both parameterisations induce the same ln_D density and support [-35, 26].
PRIOR_LN_SIGMA_MIN, PRIOR_LN_SIGMA_MAX = -10.0, 10.0


# Known sampling-model labels. A label fully identifies the parameterization + prior type.
# Add entries here when new models are implemented.
SAMPLING_MODEL_REGISTRY = {
    "lnD_lntau_flat": {
        "sampled_params": ["m", "ln_D", "ln_tau", "ln_jitter"],
        "parameterization": "ln_D = ln(sigma^2 / tau),  ln_tau = ln(tau / days, rest-frame)",
        "prior_type": "flat_box",
        "description": (
            "Flat box prior directly on ln_D and ln_tau. "
            "Equivalent to free_flatlnD in drw_fitting.py."
        ),
    },
    "lnsigma2_lntau_flat": {
        "sampled_params": ["m", "ln_sigma2", "ln_tau", "ln_jitter"],
        "parameterization": "ln_sigma2 = ln(sigma^2),  ln_tau = ln(tau / days, rest-frame)",
        "prior_type": "flat_box",
        "description": (
            "Flat box prior on ln_sigma2 and ln_tau. At the default bounds the marginal "
            "ln_D prior is trapezoidal on [-35,26]. "
            "Output stored as ln_D for consistency."
        ),
    },
    "lnsigma_lntau_flat": {
        "sampled_params": ["m", "ln_sigma", "ln_tau", "ln_jitter"],
        "parameterization": "ln_sigma = ln(sigma),  ln_tau = ln(tau / days, rest-frame)",
        "prior_type": "flat_box",
        "description": (
            "Flat box prior on ln_sigma and ln_tau (Jeffreys-scale prior on sigma). "
            "ln_D = 2*ln_sigma - ln_tau. Default bounds [-10,10] correspond to ln_sigma2 "
            "bounds [-20,20], giving the same default marginal ln_D prior as lnsigma2_lntau_flat. "
            "Overridden bounds change this support. Output stored as ln_D for consistency."
        ),
    },
    "lnD_fixed_tau": {
        "sampled_params": ["m", "ln_D", "ln_jitter"],
        "parameterization": "ln_D = ln(sigma^2 / tau),  tau fixed (not sampled)",
        "prior_type": "flat_box",
        "description": (
            "3-parameter fit: tau is fixed externally (global value or per-source from FITS). "
            "Samples (m, ln_D, ln_jitter) only. Shows how well ln_D is recovered when tau "
            "is known exactly."
        ),
    },
}


def fit_single_object(
    t_obs,
    m,
    merr,
    z,
    log_sigma2_bounds,   # not used; see the module prior bounds
    log_tau_bounds,      # not used; see the module prior bounds
    nwalkers=32,
    nburn=200,
    nsteps=400,
    nsamples_output=500,
    rng=None,
    adaptive=False,
    target_thinned=500,
    step_chunk=500,
    max_steps=6000,
    sampling_model="lnD_lntau_flat",
    tau_fixed=None,
    mwin_mode="none",
):
    """
    Run an emcee sampler for a single lightcurve.

    sampling_model controls the parameterization:
      "lnD_lntau_flat"      — (m, ln_D, ln_tau, ln_jitter), flat box prior
      "lnsigma2_lntau_flat" — (m, ln_sigma2, ln_tau, ln_jitter); output as ln_D = ln_sigma2 - ln_tau
      "lnsigma_lntau_flat"  — (m, ln_sigma, ln_tau, ln_jitter); output as ln_D = 2*ln_sigma - ln_tau
      "lnD_fixed_tau"       — (m, ln_D, ln_jitter), tau fixed via tau_fixed argument

    tau_fixed : float or None
        Rest-frame tau in days. Required when sampling_model=="lnD_fixed_tau".
        Ignored for all other models.

    Parameters
    ----------
    t_obs, m, merr : arrays
        Observer-frame time (days), magnitude, magnitude error.
    z : float
        Redshift. Used to convert to rest-frame: t_rest = (t_obs - t0) / (1 + z).
    log_sigma2_bounds, log_tau_bounds : tuple
        These arguments are not used. Set priors with the CLI --prior-* options or module constants.
    nwalkers, nburn, nsteps : int
        emcee settings. nsteps is only used when adaptive=False.
    nsamples_output : int
        Number of posterior samples to return per object.
    rng : np.random.Generator or None
    adaptive : bool
        If True, run in step_chunk increments until IAT-thinned samples >= target_thinned
        or max_steps is reached. Mirrors the adaptive mode in drw_fitting.py.
    target_thinned : int
        Target candidate count after the export-stride selection (adaptive mode only).
    step_chunk : int
        Steps per adaptive iteration.
    max_steps : int
        Hard cap on production steps (adaptive mode only).

    Returns
    -------
    samples_m, samples_ln_D, samples_ln_tau, samples_ln_jitter : ndarray, shape (nsamples_output,)
    diag : dict
        Per-object MCMC diagnostics: accept_mean, accept_min, accept_max,
        logprob_nonfinite_frac, iat (per param), neff (per param),
        nsteps_used, thin_stride, n_thinned_available, n_thinned_used, autocorr_ok.
    """
    if rng is None:
        rng = np.random.default_rng()

    mwin_mode = str(mwin_mode).lower().strip()
    if mwin_mode not in {"none", "trap", "cont", "both"}:
        raise ValueError("mwin_mode must be one of: none, trap, cont, both")

    if sampling_model == "lnsigma2_lntau_flat":
        parameterization = "lnsigma2"
        lnD_prior = "induced"
    elif sampling_model == "lnsigma_lntau_flat":
        parameterization = "lnsigma"
        lnD_prior = "induced"
    elif sampling_model == "lnD_fixed_tau":
        parameterization = "lnD"
        lnD_prior = "flat"
        if tau_fixed is None or tau_fixed <= 0:
            raise ValueError("sampling_model='lnD_fixed_tau' requires tau_fixed > 0")
    else:
        parameterization = "lnD"
        lnD_prior = "flat"

    random_seed = int(rng.integers(0, 2**63 - 1))
    res = run_drw_mcmc_jitter(
        t_obs,
        m,
        merr,
        z,
        parameterization=parameterization,
        tau_fixed=tau_fixed,
        min_points=10,
        nwalkers=nwalkers,
        nburn=nburn,
        nsteps=nsteps,
        nsamples_output=nsamples_output,
        random_seed=random_seed,
        lnD_prior=lnD_prior,
        ln_tau_bounds=(PRIOR_LN_TAU_MIN, PRIOR_LN_TAU_MAX),
        lnD_bounds=(PRIOR_LN_D_MIN, PRIOR_LN_D_MAX),
        ln_sigma2_bounds=(PRIOR_LN_SIGMA2_MIN, PRIOR_LN_SIGMA2_MAX),
        ln_sigma_bounds=(PRIOR_LN_SIGMA_MIN, PRIOR_LN_SIGMA_MAX),
        mwin_mode=mwin_mode,
        adaptive=adaptive,
        target_thinned=target_thinned,
        step_chunk=step_chunk,
        max_steps=max_steps,
        return_chain=False,
    )
    if res is None:
        raise RuntimeError("Centralized fitter returned no samples.")

    samples = np.asarray(res["samples"], dtype=float)
    param_names = list(res["param_names"])
    diag = dict(res.get("diag", {}))
    baseline_rest = float(res.get("baseline_rest", np.nan))
    npts = int(res.get("t_rest_npts", 0))

    def _col(name, default=np.nan):
        if name in param_names:
            return np.asarray(samples[:, param_names.index(name)], dtype=float)
        return np.full(nsamples_output, default, dtype=float)

    s_m = _col("m")
    s_ln_jitter = _col("ln_jitter")
    s_ln_D = _col("ln_D")
    s_ln_tau = _col("ln_tau")
    s_m_win = _col("m_win")
    s_m_win_mu = _col("m_win_mu")
    s_m_win_var = _col("m_win_var")
    s_m_win_cont = _col("m_win_cont")
    s_m_win_mu_cont = _col("m_win_mu_cont")
    s_m_win_var_cont = _col("m_win_var_cont")

    if "ln_D" not in param_names:
        if "ln_sigma2" in param_names and "ln_tau" in param_names:
            s_ln_D = _col("ln_sigma2") - _col("ln_tau")
        elif "ln_sigma" in param_names and "ln_tau" in param_names:
            s_ln_D = 2.0 * _col("ln_sigma") - _col("ln_tau")
        else:
            s_ln_D = np.full(nsamples_output, np.nan, dtype=float)

    if "ln_tau" not in param_names and tau_fixed is not None:
        s_ln_tau = np.full(nsamples_output, np.log(float(tau_fixed)), dtype=float)

    diag["nsteps_used"] = int(diag.get("nsteps", 0))
    diag["baseline_rest"] = baseline_rest
    diag["npts"] = npts
    if mwin_mode in {"trap", "both"}:
        diag["m_win"] = s_m_win
        diag["m_win_mu"] = s_m_win_mu
        diag["m_win_var"] = s_m_win_var
    if mwin_mode in {"cont", "both"}:
        diag["m_win_cont"] = s_m_win_cont
        diag["m_win_mu_cont"] = s_m_win_mu_cont
        diag["m_win_var_cont"] = s_m_win_var_cont
    return s_m, s_ln_D, s_ln_tau, s_ln_jitter, diag


def fit_object_wrapper(
    i,
    sid,
    z,
    t,
    m,
    merr,
    log_sigma2_bounds,  # not used; see the module prior bounds
    log_tau_bounds,     # not used; see the module prior bounds
    nwalkers,
    nburn,
    nsteps,
    nsamples_output,
    rng,
    progress_file=None,
    adaptive=False,
    target_thinned=500,
    step_chunk=500,
    max_steps=6000,
    sampling_model="lnD_lntau_flat",
    tau_fixed=None,
    mwin_mode="none",
):
    """
    Wrapper for parallelization: fits a single object and returns results + diagnostics.

    Returns
    -------
    result : dict with keys:
        index, success, m, ln_D, ln_tau, ln_jitter,
        [m_win, m_win_mu, m_win_var, m_win_cont, m_win_mu_cont, m_win_var_cont,] diag, error
    """
    if sampling_model == "lnsigma2_lntau_flat":
        param1_name = "ln_sigma2"
    elif sampling_model == "lnsigma_lntau_flat":
        param1_name = "ln_sigma"
    else:
        param1_name = "ln_D"
    nan_diag = {
        "nsteps_used": 0, "nwalkers": nwalkers, "n_post": 0,
        "accept_mean": np.nan, "accept_min": np.nan, "accept_max": np.nan,
        "logprob_nonfinite_frac": np.nan,
        "iat": {k: np.nan for k in ("m", param1_name, "ln_tau", "ln_jitter")},
        "neff": {k: np.nan for k in ("m", param1_name, "ln_tau", "ln_jitter")},
        "autocorr_ok": False,
        "thin_stride": np.nan, "n_thinned_available": -1, "n_thinned_used": -1,
    }
    result = {
        'index': i, 'success': False,
        'm':        np.full(nsamples_output, np.nan),
        'ln_D':     np.full(nsamples_output, np.nan),
        'ln_tau':   np.full(nsamples_output, np.nan),
        'ln_jitter':np.full(nsamples_output, np.nan),
        'm_win':     np.full(nsamples_output, np.nan),
        'm_win_mu':  np.full(nsamples_output, np.nan),
        'm_win_var': np.full(nsamples_output, np.nan),
        'm_win_cont':     np.full(nsamples_output, np.nan),
        'm_win_mu_cont':  np.full(nsamples_output, np.nan),
        'm_win_var_cont': np.full(nsamples_output, np.nan),
        'diag': nan_diag, 'error': None,
    }

    try:
        s_m, s_ln_D, s_ln_tau, s_ln_jitter, diag = fit_single_object(
            t, m, merr, z,
            log_sigma2_bounds, log_tau_bounds,
            nwalkers=nwalkers, nburn=nburn, nsteps=nsteps,
            nsamples_output=nsamples_output, rng=rng,
            adaptive=adaptive, target_thinned=target_thinned,
            step_chunk=step_chunk, max_steps=max_steps,
            sampling_model=sampling_model,
            tau_fixed=tau_fixed,
            mwin_mode=mwin_mode,
        )
        result.update({
            'm': s_m, 'ln_D': s_ln_D, 'ln_tau': s_ln_tau,
            'ln_jitter': s_ln_jitter, 'diag': diag, 'success': True,
            'm_win':     diag.pop('m_win',     np.full(nsamples_output, np.nan)),
            'm_win_mu':  diag.pop('m_win_mu',  np.full(nsamples_output, np.nan)),
            'm_win_var': diag.pop('m_win_var', np.full(nsamples_output, np.nan)),
            'm_win_cont':     diag.pop('m_win_cont',     np.full(nsamples_output, np.nan)),
            'm_win_mu_cont':  diag.pop('m_win_mu_cont',  np.full(nsamples_output, np.nan)),
            'm_win_var_cont': diag.pop('m_win_var_cont', np.full(nsamples_output, np.nan)),
        })
    except Exception as e:
        result['error'] = str(e)

    if progress_file:
        try:
            with open(progress_file, 'a') as f:
                f.write(f"[{i+1:5d}] idx={i} sid={sid}: "
                        f"{'OK' if result['success'] else 'FAIL'}\n")
                f.flush()
        except Exception:
            pass

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fit DRW + jitter GPs to mock Gaia QSO lightcurves (multi-band support)."
    )
    parser.add_argument(
        "--input-fits",
        type=str,
        default="data/mock_gaia_qso_lightcurves.fits",
        help="Input FITS with mocks (default: data/mock_gaia_qso_lightcurves.fits)",
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        default="data/mock_gaia_qso_drw_posteriors.npz",
        help="Output NPZ file for posterior samples (default: data/mock_gaia_qso_drw_posteriors.npz)",
    )
    parser.add_argument(
        "--bands",
        type=str,
        default=None,
        help="Comma-separated bands to fit: 'g', 'g,bp', 'g,bp,rp', etc. "
             "If not specified, auto-detects from FITS file columns.",
    )
    parser.add_argument(
        "--redshift-catalog",
        type=str,
        default=None,
        help="Path to a FITS file with columns 'source_id' and a redshift column "
             "('Z', 'z', or 'redsh'). Used when the input FITS lacks a redshift column "
             "(e.g. photometry-only FITS). Rows are matched by source_id.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=0,
        help="Maximum number of objects to fit. 0 or negative = fit all available (default: 0).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index in the input table (default: 0). Used for parallel chunk processing.",
    )
    parser.add_argument(
        "--nwalkers",
        type=int,
        default=32,
        help="Number of emcee walkers (default: 32).",
    )
    parser.add_argument(
        "--nburn",
        type=int,
        default=200,
        help="Number of burn-in steps (default: 200).",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=400,
        help="Number of production steps (default: 400).",
    )
    parser.add_argument(
        "--nsamples-output",
        type=int,
        default=500,
        help="Number of posterior samples per object to save (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base seed for deterministic fits by input row and band (default: 123).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs. -1 = all cores, 1 = sequential (default: 1).",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10000,
        help=(
            "Save a checkpoint every this many completed tasks so the run can be resumed "
            "after a node failure or walltime kill. 0 disables checkpointing. "
            "Checkpoints are written to <output-npz-stem>_checkpoints/checkpoint_current.npz "
            "and auto-detected on restart. The folder is deleted on successful completion. "
            "Default: 10000."
        ),
    )

    # Adaptive MCMC
    parser.add_argument(
        "--adaptive", action="store_true", default=False,
        help="Run production steps adaptively until IAT-thinned samples >= target-thinned "
             "(or max-steps reached). Mirrors drw_fitting.py adaptive mode.",
    )
    parser.add_argument(
        "--target-thinned", type=int, default=500,
        help="Target candidate count after the export-stride selection (adaptive mode). Default: 500.",
    )
    parser.add_argument(
        "--step-chunk", type=int, default=500,
        help="Steps per adaptive iteration (adaptive mode). Default: 500.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=6000,
        help="Hard cap on production steps (adaptive mode). Default: 6000.",
    )
    parser.add_argument(
        "--mwin-mode",
        choices=("none", "trap", "cont", "both"),
        default="both",
        help=(
            "Which posterior latent window-mean magnitude(s) to derive for each MCMC draw: "
            "'none' computes no m_win posteriors; 'trap' computes only the trapezoid/"
            "quadrature version; 'cont' computes only the continuous-integral version; "
            "'both' computes both. Default: both."
        ),
    )

    # Sampling model + prior overrides
    parser.add_argument(
        "--sampling-model",
        type=str,
        default="lnD_lntau_flat",
        help=(
            "Label identifying the sampling parameterization and prior type. "
            "Stored verbatim in the metadata JSON for reproducibility. "
            "Models: lnD_lntau_flat, lnsigma2_lntau_flat, lnsigma_lntau_flat, lnD_fixed_tau. "
            "Default: lnD_lntau_flat"
        ),
    )
    parser.add_argument(
        "--prior-lnD-min", type=float, default=None,
        help="Override lower bound of flat ln_D prior (default: canonical -35.0).",
    )
    parser.add_argument(
        "--prior-lnD-max", type=float, default=None,
        help="Override upper bound of flat ln_D prior (default: canonical 26.0).",
    )
    parser.add_argument(
        "--prior-lntau-min", type=float, default=None,
        help="Override lower bound of ln_tau prior (default: canonical -6.0).",
    )
    parser.add_argument(
        "--prior-lntau-max", type=float, default=None,
        help="Override upper bound of ln_tau prior (default: canonical 15.0).",
    )
    parser.add_argument(
        "--prior-lnsigma2-min", type=float, default=None,
        help="Override lower bound of ln_sigma2 prior for lnsigma2_lntau_flat (default: -20.0).",
    )
    parser.add_argument(
        "--prior-lnsigma2-max", type=float, default=None,
        help="Override upper bound of ln_sigma2 prior for lnsigma2_lntau_flat (default: 20.0).",
    )
    parser.add_argument(
        "--prior-lnsigma-min", type=float, default=None,
        help="Override lower bound of ln_sigma prior for lnsigma_lntau_flat (default: -10.0).",
    )
    parser.add_argument(
        "--prior-lnsigma-max", type=float, default=None,
        help="Override upper bound of ln_sigma prior for lnsigma_lntau_flat (default: 10.0).",
    )

    # Fixed-tau options (for lnD_fixed_tau sampling model)
    parser.add_argument(
        "--fixed-tau", type=float, default=None,
        help=(
            "Option A: fix tau to this value (rest-frame days) for ALL sources. "
            "Requires --sampling-model lnD_fixed_tau."
        ),
    )
    parser.add_argument(
        "--fixed-tau-column", type=str, default=None,
        help=(
            "Option B: name of FITS column to read per-source tau from. "
            "Use --fixed-tau-column-is-log10 if the column is in log10(tau/days) "
            "(e.g. 'true_log10_tau'); otherwise it is treated as ln(tau/days)."
        ),
    )
    parser.add_argument(
        "--fixed-tau-column-is-log10", action="store_true", default=False,
        help="If set, the --fixed-tau-column values are log10(tau/days) and will be converted.",
    )

    args = parser.parse_args()

    # Apply prior bound overrides (modifies module-level constants passed to the DRW prior helpers).
    global PRIOR_LN_D_MIN, PRIOR_LN_D_MAX, PRIOR_LN_TAU_MIN, PRIOR_LN_TAU_MAX
    global PRIOR_LN_SIGMA2_MIN, PRIOR_LN_SIGMA2_MAX, PRIOR_LN_SIGMA_MIN, PRIOR_LN_SIGMA_MAX
    if args.prior_lnD_min is not None:
        PRIOR_LN_D_MIN = args.prior_lnD_min
    if args.prior_lnD_max is not None:
        PRIOR_LN_D_MAX = args.prior_lnD_max
    if args.prior_lntau_min is not None:
        PRIOR_LN_TAU_MIN = args.prior_lntau_min
    if args.prior_lntau_max is not None:
        PRIOR_LN_TAU_MAX = args.prior_lntau_max
    if args.prior_lnsigma2_min is not None:
        PRIOR_LN_SIGMA2_MIN = args.prior_lnsigma2_min
    if args.prior_lnsigma2_max is not None:
        PRIOR_LN_SIGMA2_MAX = args.prior_lnsigma2_max
    if args.prior_lnsigma_min is not None:
        PRIOR_LN_SIGMA_MIN = args.prior_lnsigma_min
    if args.prior_lnsigma_max is not None:
        PRIOR_LN_SIGMA_MAX = args.prior_lnsigma_max

    if args.seed < 0:
        parser.error("--seed must be non-negative")

    input_path = os.path.expanduser(args.input_fits)
    print("Reading mocks from:", input_path)
    tbl = Table.read(input_path)

    # =========================================================================
    # IN-MEMORY COLUMN NORMALISATION — raw Gaia flag-cleaned format
    # =========================================================================
    # The flag-cleaned real-data FITS uses g_transit_* column names and stores
    # errors as flux errors.  Normalise to the Gaia_G_* names expected below
    # without writing a duplicate file.
    #
    # Magnitude error from Pogson scale: G = -2.5 log10(f) + C
    #   => sigma_G = (2.5 / ln 10) * sigma_f / f
    if "g_transit_time" in tbl.colnames and "Gaia_G_transit_time" not in tbl.colnames:
        from astropy.table import Column
        tbl.rename_column("g_transit_time", "Gaia_G_transit_time")

        if "g_transit_mag" in tbl.colnames:
            # Raw Gaia flag-cleaned format: magnitudes stored as g_transit_mag,
            # errors must be computed from flux columns.
            print("Detected raw flag-cleaned column format — normalising to Gaia_G_* in memory.")
            tbl.rename_column("g_transit_mag", "Gaia_G_mag")
            _factor = 2.5 / np.log(10.0)
            _flux  = tbl["g_transit_flux"]
            _ferr  = tbl["g_transit_flux_error"]
            _merr  = np.empty(len(tbl), dtype=object)
            for _i in range(len(tbl)):
                _f  = np.asarray(_flux[_i],  dtype=float)
                _fe = np.asarray(_ferr[_i],  dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    _merr[_i] = _factor * (_fe / _f)
            tbl.add_column(Column(_merr, name="Gaia_G_mag_err"))
            print(f"  Converted flux errors to mag errors for {len(tbl):,} sources.")
        elif "g_mag" in tbl.colnames:
            # Magnitude-error format: mag and mag_error are supplied directly.
            print("Detected magnitude-error column format — normalising to Gaia_G_* in memory.")
            tbl.rename_column("g_mag", "Gaia_G_mag")
            if "g_mag_error" in tbl.colnames:
                tbl.rename_column("g_mag_error", "Gaia_G_mag_err")
            print(f"  Renamed g_mag/g_mag_error → Gaia_G_mag/Gaia_G_mag_err for {len(tbl):,} sources.")
        else:
            raise KeyError(
                "g_transit_time is present but neither g_transit_mag nor g_mag found. "
                f"Available columns: {tbl.colnames}"
            )

    if "z" in tbl.colnames and "redsh" not in tbl.colnames:
        tbl.rename_column("z", "redsh")
        print("  Renamed column 'z' -> 'redsh'.")

    # =========================================================================
    # OPTIONAL REDSHIFT MERGE — for FITS files that lack a redshift column
    # =========================================================================
    if "redsh" not in tbl.colnames:
        if args.redshift_catalog is None:
            raise KeyError(
                "Input FITS has no 'redsh' or 'z' column. "
                "Provide a redshift catalog via --redshift-catalog."
            )
        print(f"Merging redshifts from: {args.redshift_catalog}")
        from astropy.table import Column, join as astropy_join
        zcat = Table.read(os.path.expanduser(args.redshift_catalog))
        # Normalise the redshift column name in the catalog
        for _zname in ("redsh", "z", "Z"):
            if _zname in zcat.colnames and _zname != "redsh":
                zcat.rename_column(_zname, "redsh")
                break
        if "redsh" not in zcat.colnames:
            raise KeyError(
                f"--redshift-catalog '{args.redshift_catalog}' has no recognised redshift column "
                f"('Z', 'z', 'redsh'). Found: {zcat.colnames}"
            )
        # Keep only source_id + redsh (avoid column conflicts on join)
        zcat = zcat["source_id", "redsh"]
        n_before = len(tbl)
        tbl = astropy_join(tbl, zcat, keys="source_id", join_type="left")
        n_matched = int(np.sum(~np.isnan(tbl["redsh"].filled(np.nan) if hasattr(tbl["redsh"], "filled") else tbl["redsh"])))
        print(f"  Merged redshifts: {n_matched:,}/{n_before:,} sources matched.")

    # =========================================================================
    # AUTO-DETECT OR PARSE BANDS
    # =========================================================================
    # ZTF bands: 'g', 'r' (lowercase)
    # Gaia bands: 'Gaia_G', 'Gaia_BP', 'Gaia_RP'
    valid_bands = ['g', 'r', 'Gaia_G', 'Gaia_BP', 'Gaia_RP']

    if args.bands is None:
        # Auto-detect bands from FITS columns
        bands_to_fit = []
        for band in valid_bands:
            if f"{band}_transit_time" in tbl.colnames:
                bands_to_fit.append(band)
        if not bands_to_fit:
            raise ValueError("No band columns found in FITS file. Expected {band}_transit_time columns.")
        print(f"Auto-detected bands from FITS: {bands_to_fit}")
    else:
        # Parse user-specified bands (case-sensitive)
        bands_to_fit = [b.strip() for b in args.bands.split(',')]
        for b in bands_to_fit:
            if b not in valid_bands:
                raise ValueError(f"Invalid band '{b}'. Must be one of: {valid_bands}")
            if f"{b}_transit_time" not in tbl.colnames:
                raise KeyError(f"Band '{b}' requested but column '{b}_transit_time' not in FITS file.")
        print(f"Bands to fit (user-specified): {bands_to_fit}")

    n_all = len(tbl)
    start_idx = args.start_index
    if start_idx < 0:
        parser.error("--start-index must be non-negative")

    def task_rng(index, band):
        # Use the input row and a fixed band index so chunking and worker order share the same seed.
        return np.random.default_rng([args.seed, start_idx + index, valid_bands.index(band)])

    # Handle start_index beyond data range - exit gracefully
    if start_idx >= n_all:
        print(f"Total objects in file: {n_all}")
        print(f"[SKIP] start_index={start_idx} >= total rows={n_all}. Nothing to fit.")
        print("This chunk is empty - exiting successfully.")
        return

    # Handle max_objects: 0 or negative means "all remaining", otherwise cap at available
    if args.max_objects <= 0:
        # Process all objects from start_idx to end
        end_idx = n_all
    else:
        # Process up to max_objects, but don't exceed available data
        end_idx = min(start_idx + args.max_objects, n_all)

    n_fit = end_idx - start_idx
    n_bands = len(bands_to_fit)
    n_total_tasks = n_fit * n_bands

    print(f"Total objects in file: {n_all}")
    if args.max_objects <= 0:
        print(f"max_objects={args.max_objects} → fitting ALL available objects from start_index")
    elif start_idx + args.max_objects > n_all:
        print(f"max_objects={args.max_objects} exceeds available data → capped to {n_fit}")
    print(f"Processing objects [{start_idx}, {end_idx}) → {n_fit} objects")
    print(f"Fitting {n_fit} objects × {n_bands} bands = {n_total_tasks} total fitting tasks")

    # Prior bounds: kept for API compatibility, actual priors are passed to drw_fitting.py
    log_sigma2_bounds = (np.log(1e-8), 4.0)
    log_tau_bounds = (np.log(0.1), 14.0)

    source_ids = np.array(tbl["source_id"][start_idx:end_idx])
    redsh = np.array(tbl["redsh"][start_idx:end_idx], dtype=float)

    # ---------------------------------------------------------------- fixed-tau resolution
    # tau_fixed_arr: per-source rest-frame tau in days, or None for free-tau models.
    tau_fixed_arr = None
    if args.sampling_model == "lnD_fixed_tau":
        if args.fixed_tau is not None and args.fixed_tau_column is not None:
            raise ValueError("Specify either --fixed-tau or --fixed-tau-column, not both.")
        if args.fixed_tau is not None:
            # Option A: same tau for every source
            tau_fixed_arr = np.full(n_fit, float(args.fixed_tau))
            print(f"Fixed-tau mode (Option A): tau = {args.fixed_tau:.1f} days (rest-frame) for all sources")
        elif args.fixed_tau_column is not None:
            if args.fixed_tau_column not in tbl.colnames:
                raise ValueError(f"--fixed-tau-column '{args.fixed_tau_column}' not found in FITS. "
                                  f"Available: {tbl.colnames}")
            col_vals = np.array(tbl[args.fixed_tau_column][start_idx:end_idx], dtype=float)
            if args.fixed_tau_column_is_log10:
                tau_fixed_arr = 10.0 ** col_vals
                print(f"Fixed-tau mode (Option B): reading log10(tau) from column '{args.fixed_tau_column}'")
            else:
                tau_fixed_arr = np.exp(col_vals)
                print(f"Fixed-tau mode (Option B): reading ln(tau) from column '{args.fixed_tau_column}'")
            print(f"  tau range: [{tau_fixed_arr.min():.1f}, {tau_fixed_arr.max():.1f}] days (rest-frame)")
        else:
            raise ValueError("sampling_model='lnD_fixed_tau' requires --fixed-tau or --fixed-tau-column.")
    elif args.fixed_tau is not None or args.fixed_tau_column is not None:
        print("[WARN] --fixed-tau / --fixed-tau-column ignored (only used with --sampling-model lnD_fixed_tau)")

    nsamp_out = args.nsamples_output
    want_mwin_trap = args.mwin_mode in {"trap", "both"}
    want_mwin_cont = args.mwin_mode in {"cont", "both"}

    # Pre-load all band data
    band_data = {}
    for band in bands_to_fit:
        band_data[band] = {
            'times': np.array(tbl[f"{band}_transit_time"][start_idx:end_idx]),
            'mags': np.array(tbl[f"{band}_mag"][start_idx:end_idx]),
            'merr': np.array(tbl[f"{band}_mag_err"][start_idx:end_idx]),
        }

    # Initialize per-band posterior + diagnostic storage
    posteriors = {}
    diagnostics = {}
    for band in bands_to_fit:
        posteriors[band] = {
            'm':        np.full((n_fit, nsamp_out), np.nan, dtype=float),
            'ln_D':     np.full((n_fit, nsamp_out), np.nan, dtype=float),
            'ln_tau':   np.full((n_fit, nsamp_out), np.nan, dtype=float),
            'ln_jitter':np.full((n_fit, nsamp_out), np.nan, dtype=float),
        }
        if want_mwin_trap:
            posteriors[band].update({
                'm_win':     np.full((n_fit, nsamp_out), np.nan, dtype=float),
                'm_win_mu':  np.full((n_fit, nsamp_out), np.nan, dtype=float),
                'm_win_var': np.full((n_fit, nsamp_out), np.nan, dtype=float),
            })
        if want_mwin_cont:
            posteriors[band].update({
                'm_win_cont':     np.full((n_fit, nsamp_out), np.nan, dtype=float),
                'm_win_mu_cont':  np.full((n_fit, nsamp_out), np.nan, dtype=float),
                'm_win_var_cont': np.full((n_fit, nsamp_out), np.nan, dtype=float),
            })
        diagnostics[band] = {
            'baseline_rest':            np.full(n_fit, np.nan),
            'npts':                     np.full(n_fit, -1, dtype=int),
            'nsteps_used':              np.zeros(n_fit, dtype=int),
            'nwalkers':                 np.full(n_fit, -1, dtype=int),
            'accept_mean':              np.full(n_fit, np.nan),
            'accept_min':               np.full(n_fit, np.nan),
            'accept_max':               np.full(n_fit, np.nan),
            'logprob_nonfinite_frac':   np.full(n_fit, np.nan),
            'logprob_nonfinite_count':  np.full(n_fit, -1, dtype=int),
            'iat_ln_D':                 np.full(n_fit, np.nan),
            'iat_ln_tau':               np.full(n_fit, np.nan),
            'iat_ln_jitter':            np.full(n_fit, np.nan),
            'iat_m':                    np.full(n_fit, np.nan),
            'iat_ln_sigma2':            np.full(n_fit, np.nan),
            'iat_ln_sigma':             np.full(n_fit, np.nan),
            'neff_ln_D':                np.full(n_fit, np.nan),
            'neff_ln_tau':              np.full(n_fit, np.nan),
            'neff_ln_jitter':           np.full(n_fit, np.nan),
            'neff_m':                   np.full(n_fit, np.nan),
            'neff_ln_sigma2':           np.full(n_fit, np.nan),
            'neff_ln_sigma':            np.full(n_fit, np.nan),
            'thin_stride':              np.full(n_fit, np.nan),
            'n_thinned_available':      np.full(n_fit, -1, dtype=int),
            'n_thinned_used':           np.full(n_fit, -1, dtype=int),
            'autocorr_ok':              np.zeros(n_fit, dtype=bool),
        }

    def _store_diag(band, i, diag):
        d = diagnostics[band]
        d['baseline_rest'][i]          = diag.get('baseline_rest', np.nan)
        d['npts'][i]                   = diag.get('npts', -1)
        d['nsteps_used'][i]            = diag.get('nsteps_used', 0)
        d['nwalkers'][i]               = diag.get('nwalkers', -1)
        d['accept_mean'][i]            = diag.get('accept_mean', np.nan)
        d['accept_min'][i]             = diag.get('accept_min',  np.nan)
        d['accept_max'][i]             = diag.get('accept_max',  np.nan)
        d['logprob_nonfinite_frac'][i] = diag.get('logprob_nonfinite_frac', np.nan)
        d['logprob_nonfinite_count'][i]= diag.get('logprob_nonfinite_count', -1)
        _iat = diag.get('iat', {})
        d['iat_ln_D'][i]               = _iat.get('ln_D', np.nan)
        d['iat_ln_tau'][i]             = _iat.get('ln_tau', np.nan)
        d['iat_ln_jitter'][i]          = _iat.get('ln_jitter', np.nan)
        d['iat_m'][i]                  = _iat.get('m', np.nan)
        d['iat_ln_sigma2'][i]          = _iat.get('ln_sigma2', np.nan)
        d['iat_ln_sigma'][i]           = _iat.get('ln_sigma', np.nan)
        _neff = diag.get('neff', {})
        d['neff_ln_D'][i]              = _neff.get('ln_D', np.nan)
        d['neff_ln_tau'][i]            = _neff.get('ln_tau', np.nan)
        d['neff_ln_jitter'][i]         = _neff.get('ln_jitter', np.nan)
        d['neff_m'][i]                 = _neff.get('m', np.nan)
        d['neff_ln_sigma2'][i]         = _neff.get('ln_sigma2', np.nan)
        d['neff_ln_sigma'][i]          = _neff.get('ln_sigma', np.nan)
        d['thin_stride'][i]            = diag.get('thin_stride', np.nan)
        d['n_thinned_available'][i]    = diag.get('n_thinned_available', -1)
        d['n_thinned_used'][i]         = diag.get('n_thinned_used', -1)
        d['autocorr_ok'][i]            = bool(diag.get('autocorr_ok', False))

    use_parallel = args.n_jobs != 1 and HAS_JOBLIB
    if args.n_jobs != 1 and not HAS_JOBLIB:
        print("[WARN] joblib not installed; falling back to sequential fitting.")
        use_parallel = False

    # =========================================================================
    # CHECKPOINT SETUP
    # =========================================================================
    output_path_expanded = os.path.expanduser(args.output_npz)
    _output_stem = output_path_expanded[:-4] if output_path_expanded.endswith('.npz') else output_path_expanded
    checkpoint_dir  = _output_stem + '_checkpoints'
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_current.npz')
    ckpt_interval = args.checkpoint_interval  # 0 = disabled

    # completed[band] is a bool array of shape (n_fit,): True = already fitted
    completed = {band: np.zeros(n_fit, dtype=bool) for band in bands_to_fit}

    if ckpt_interval > 0 and os.path.exists(checkpoint_path):
        print(f"\nFound checkpoint: {checkpoint_path}")
        ckpt = np.load(checkpoint_path, allow_pickle=True)
        if (
            "_seed_scheme" not in ckpt
            or ckpt["_seed_scheme"].item() != SEED_SCHEME
            or ckpt["_seed"].item() != args.seed
            or ckpt["_start_index"].item() != start_idx
            or not np.array_equal(ckpt["source_id"], source_ids)
        ):
            raise ValueError(
                f"Checkpoint {checkpoint_path} uses a different seed scheme, seed, or input rows. "
                "Use a new --output-npz to start a separate fit."
            )
        for band in bands_to_fit:
            mask_key = f'_completed_{band}'
            if mask_key in ckpt:
                completed[band][:] = ckpt[mask_key]
            # Restore posteriors
            for key in list(posteriors[band].keys()):
                ck = f'posterior_{key}_{band}' if key in ('m', 'ln_D', 'ln_tau', 'ln_jitter') \
                     else f'posterior_{key}_{band}'
                if ck in ckpt:
                    posteriors[band][key][:] = ckpt[ck]
            # Restore diagnostics
            for key in list(diagnostics[band].keys()):
                ck = f'diag_{key}_{band}'
                if ck in ckpt:
                    diagnostics[band][key][:] = ckpt[ck]
        n_done = int(np.sum(completed[bands_to_fit[0]]))
        print(f"  Resuming: {n_done}/{n_fit} objects already completed, {n_fit - n_done} remaining.")

    def _save_checkpoint():
        ckpt_kw = {
            'source_id': source_ids, 'redsh': redsh,
            '_seed_scheme': SEED_SCHEME, '_seed': args.seed, '_start_index': start_idx,
        }
        for band in bands_to_fit:
            ckpt_kw[f'_completed_{band}'] = completed[band]
            ckpt_kw[f'posterior_m_{band}']         = posteriors[band]['m']
            ckpt_kw[f'posterior_ln_D_{band}']      = posteriors[band]['ln_D']
            ckpt_kw[f'posterior_ln_tau_{band}']    = posteriors[band]['ln_tau']
            ckpt_kw[f'posterior_ln_jitter_{band}'] = posteriors[band]['ln_jitter']
            if want_mwin_trap:
                ckpt_kw[f'posterior_m_win_{band}']     = posteriors[band]['m_win']
                ckpt_kw[f'posterior_m_win_mu_{band}']  = posteriors[band]['m_win_mu']
                ckpt_kw[f'posterior_m_win_var_{band}'] = posteriors[band]['m_win_var']
            if want_mwin_cont:
                ckpt_kw[f'posterior_m_win_cont_{band}']     = posteriors[band]['m_win_cont']
                ckpt_kw[f'posterior_m_win_mu_cont_{band}']  = posteriors[band]['m_win_mu_cont']
                ckpt_kw[f'posterior_m_win_var_cont_{band}'] = posteriors[band]['m_win_var_cont']
            for key, arr in diagnostics[band].items():
                ckpt_kw[f'diag_{key}_{band}'] = arr
        os.makedirs(checkpoint_dir, exist_ok=True)
        np.savez(checkpoint_path, **ckpt_kw)
        n_done = int(np.sum(completed[bands_to_fit[0]]))
        print(f"  [checkpoint] {n_done}/{n_fit} objects saved -> {checkpoint_path}", flush=True)

    total_start_time = time.time()

    print(f"\n{'='*70}")
    print(f"FITTING {n_bands} BANDS: {', '.join(b.upper() for b in bands_to_fit)}")
    if args.adaptive:
        print(f"Adaptive mode: step_chunk={args.step_chunk}, target_thinned={args.target_thinned}, max_steps={args.max_steps}")
    else:
        print(f"Fixed mode: nburn={args.nburn}, nsteps={args.nsteps}")
    print(f"m_win mode: {args.mwin_mode}")
    if ckpt_interval > 0:
        print(f"Checkpointing every {ckpt_interval} tasks -> {checkpoint_path}")
    print('='*70)

    # Build task list, skipping any already completed from checkpoint
    tasks = [
        (i, band)
        for i in range(n_fit)
        for band in bands_to_fit
        if not completed[band][i]
    ]
    n_remaining = len(tasks)
    n_already_done = n_total_tasks - n_remaining
    if n_already_done > 0:
        print(f"Skipping {n_already_done} already-completed tasks from checkpoint.")

    # Pre-count successes from already-completed checkpoint data
    n_success_per_band = {
        band: int(np.sum(completed[band]))
        for band in bands_to_fit
    }

    if use_parallel:
        print(f"Using parallel fitting with n_jobs={args.n_jobs}")
        progress_file = os.path.join(os.path.dirname(output_path_expanded), "fit_progress_all_bands.log")
        print(f"Progress will be logged to: {progress_file}")
        try:
            open(progress_file, 'a').close()
        except Exception:
            pass

        # Split tasks into sub-batches so we can checkpoint between them.
        # Sub-batch size: checkpoint_interval tasks (or all tasks if checkpointing disabled).
        batch_size = ckpt_interval if ckpt_interval > 0 else len(tasks)
        if batch_size == 0:
            batch_size = len(tasks) or 1
        n_batches = max(1, (len(tasks) + batch_size - 1) // batch_size)

        n_failed_printed = 0
        tasks_done_total = n_already_done

        for batch_idx in range(n_batches):
            batch = tasks[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            if not batch:
                continue

            batch_results = Parallel(n_jobs=args.n_jobs, verbose=0)(
                delayed(fit_object_wrapper)(
                    i,
                    int(source_ids[i]),
                    float(redsh[i]),
                    np.asarray(band_data[band]['times'][i], dtype=float),
                    np.asarray(band_data[band]['mags'][i], dtype=float),
                    np.asarray(band_data[band]['merr'][i], dtype=float),
                    log_sigma2_bounds, log_tau_bounds,
                    nwalkers=args.nwalkers,
                    nburn=args.nburn,
                    nsteps=args.nsteps,
                    nsamples_output=nsamp_out,
                    rng=task_rng(i, band),
                    progress_file=progress_file,
                    adaptive=args.adaptive,
                    target_thinned=args.target_thinned,
                    step_chunk=args.step_chunk,
                    max_steps=args.max_steps,
                    sampling_model=args.sampling_model,
                    tau_fixed=float(tau_fixed_arr[i]) if tau_fixed_arr is not None else None,
                    mwin_mode=args.mwin_mode,
                )
                for i, band in batch
            )

            for (i, band), result in zip(batch, batch_results):
                if result['success']:
                    posteriors[band]['m'][i, :]         = result['m']
                    posteriors[band]['ln_D'][i, :]      = result['ln_D']
                    posteriors[band]['ln_tau'][i, :]    = result['ln_tau']
                    posteriors[band]['ln_jitter'][i, :] = result['ln_jitter']
                    if want_mwin_trap:
                        posteriors[band]['m_win'][i, :]     = result['m_win']
                        posteriors[band]['m_win_mu'][i, :]  = result['m_win_mu']
                        posteriors[band]['m_win_var'][i, :] = result['m_win_var']
                    if want_mwin_cont:
                        posteriors[band]['m_win_cont'][i, :]     = result['m_win_cont']
                        posteriors[band]['m_win_mu_cont'][i, :]  = result['m_win_mu_cont']
                        posteriors[band]['m_win_var_cont'][i, :] = result['m_win_var_cont']
                    _store_diag(band, i, result['diag'])
                    n_success_per_band[band] += 1
                    completed[band][i] = True
                else:
                    if n_failed_printed < 10:
                        print(f"[WARN] Failed obj {i} band {band}: {result['error']}")
                        n_failed_printed += 1

            tasks_done_total += len(batch)
            print(f"Progress: {tasks_done_total}/{n_total_tasks} tasks completed")
            if ckpt_interval > 0:
                _save_checkpoint()

    else:
        print("Using sequential fitting (n_jobs=1)")

        for task_idx, (i, band) in enumerate(tasks):
            sid = int(source_ids[i])
            z_i = float(redsh[i])
            t   = np.asarray(band_data[band]['times'][i], dtype=float)
            m_  = np.asarray(band_data[band]['mags'][i], dtype=float)
            merr_ = np.asarray(band_data[band]['merr'][i], dtype=float)

            try:
                s_m, s_ln_D, s_ln_tau, s_ln_jitter, diag = fit_single_object(
                    t, m_, merr_, z_i,
                    log_sigma2_bounds, log_tau_bounds,
                    nwalkers=args.nwalkers, nburn=args.nburn, nsteps=args.nsteps,
                    nsamples_output=nsamp_out, rng=task_rng(i, band),
                    adaptive=args.adaptive, target_thinned=args.target_thinned,
                    step_chunk=args.step_chunk, max_steps=args.max_steps,
                    sampling_model=args.sampling_model,
                    tau_fixed=float(tau_fixed_arr[i]) if tau_fixed_arr is not None else None,
                    mwin_mode=args.mwin_mode,
                )
                posteriors[band]['m'][i, :]         = s_m
                posteriors[band]['ln_D'][i, :]      = s_ln_D
                posteriors[band]['ln_tau'][i, :]    = s_ln_tau
                posteriors[band]['ln_jitter'][i, :] = s_ln_jitter
                if want_mwin_trap:
                    posteriors[band]['m_win'][i, :]     = diag.pop('m_win',     np.full(nsamp_out, np.nan))
                    posteriors[band]['m_win_mu'][i, :]  = diag.pop('m_win_mu',  np.full(nsamp_out, np.nan))
                    posteriors[band]['m_win_var'][i, :] = diag.pop('m_win_var', np.full(nsamp_out, np.nan))
                if want_mwin_cont:
                    posteriors[band]['m_win_cont'][i, :]     = diag.pop('m_win_cont',     np.full(nsamp_out, np.nan))
                    posteriors[band]['m_win_mu_cont'][i, :]  = diag.pop('m_win_mu_cont',  np.full(nsamp_out, np.nan))
                    posteriors[band]['m_win_var_cont'][i, :] = diag.pop('m_win_var_cont', np.full(nsamp_out, np.nan))
                _store_diag(band, i, diag)
                n_success_per_band[band] += 1
                completed[band][i] = True
            except Exception as e:
                if task_idx < 10:
                    print(f"[WARN] Failed source_id={sid} band={band}: {e}")

            tasks_done = n_already_done + task_idx + 1
            if (task_idx + 1) % 100 == 0 or task_idx == n_remaining - 1:
                print(f"Progress: {tasks_done}/{n_total_tasks} tasks completed")
            if ckpt_interval > 0 and (task_idx + 1) % ckpt_interval == 0:
                _save_checkpoint()

    total_elapsed = time.time() - total_start_time

    print(f"\n{'='*70}")
    print(f"FITTING COMPLETE")
    print('='*70)
    for band in bands_to_fit:
        n_ok = n_success_per_band[band]
        if n_ok > 0:
            d = diagnostics[band]
            ok = d['autocorr_ok']
            med_steps = int(np.median(d['nsteps_used'][d['nsteps_used'] > 0])) if np.any(d['nsteps_used'] > 0) else 0
            med_accept = float(np.nanmedian(d['accept_mean']))
            med_iat_D  = float(np.nanmedian(d['iat_ln_D']))
            med_neff_D = float(np.nanmedian(d['neff_ln_D']))
            frac_conv  = float(np.mean(ok)) * 100
            print(f"  {band.upper()}: {n_ok}/{n_fit} successful")
            print(f"    median steps={med_steps}, median accept={med_accept:.2f}, "
                  f"median IAT(ln_D)={med_iat_D:.1f}, median N_eff(ln_D)={med_neff_D:.0f}, "
                  f"IAT converged={frac_conv:.0f}%")
        else:
            print(f"  {band.upper()}: {n_ok}/{n_fit} successful")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/n_total_tasks:.3f} s/task)")
    print('='*70)

    output_path = os.path.expanduser(args.output_npz)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Writing posterior samples to:", output_path)

    # Prepare savez kwargs with per-band arrays
    savez_kwargs = {
        "source_id": source_ids,
        "redsh": redsh,
        "bands_fitted": np.array(bands_to_fit),
    }

    # Store per-source fixed tau if used (lnD_fixed_tau model, option A or B)
    if tau_fixed_arr is not None:
        savez_kwargs["tau_fixed_days"] = tau_fixed_arr

    # Add per-band posterior samples + diagnostics
    for band in bands_to_fit:
        savez_kwargs[f"posterior_m_{band}"]         = posteriors[band]['m']
        savez_kwargs[f"posterior_ln_D_{band}"]      = posteriors[band]['ln_D']
        savez_kwargs[f"posterior_ln_tau_{band}"]    = posteriors[band]['ln_tau']
        savez_kwargs[f"posterior_ln_jitter_{band}"] = posteriors[band]['ln_jitter']
        if want_mwin_trap:
            savez_kwargs[f"posterior_m_win_{band}"]     = posteriors[band]['m_win']
            savez_kwargs[f"posterior_m_win_mu_{band}"]  = posteriors[band]['m_win_mu']
            savez_kwargs[f"posterior_m_win_var_{band}"] = posteriors[band]['m_win_var']
        if want_mwin_cont:
            savez_kwargs[f"posterior_m_win_cont_{band}"]     = posteriors[band]['m_win_cont']
            savez_kwargs[f"posterior_m_win_mu_cont_{band}"]  = posteriors[band]['m_win_mu_cont']
            savez_kwargs[f"posterior_m_win_var_cont_{band}"] = posteriors[band]['m_win_var_cont']
        # Per-source diagnostics — use prefix diag_ for easy filtering
        for key, arr in diagnostics[band].items():
            savez_kwargs[f"diag_{key}_{band}"] = arr

    # Copy per-band true parameters from FITS to NPZ (new multi-band format)
    for band in bands_to_fit:
        for param in ['true_log10_D', 'true_log10_sigma2', 'true_log10_tau', 'true_ln_D', 'true_ln_tau',
                      'true_mean_mag', 'lambda_rf', 'M_rest', 'K_corr']:
            col_name = f"{param}_{band}"
            if col_name in tbl.colnames:
                savez_kwargs[col_name] = np.array(tbl[col_name][start_idx:end_idx], dtype=float)

    # Copy shared columns (not per-band)
    for col in ['logL3000', 'M_abs_bol', 'distance_modulus']:
        if col in tbl.colnames:
            savez_kwargs[col] = np.array(tbl[col][start_idx:end_idx], dtype=float)

    # Copy scalar window-mean truth columns whenever a mock generator provides them.
    for col in ['true_m_win', 'true_m_win_cont', 'true_m_win_cont_mu', 'true_m_win_cont_var']:
        if col in tbl.colnames and col not in savez_kwargs:
            savez_kwargs[col] = np.array(tbl[col][start_idx:end_idx], dtype=float)

    # Backward compatibility: also check for old single-band column names
    if len(bands_to_fit) == 1 and bands_to_fit[0] == 'g':
        old_cols = ['true_log10_sigma', 'true_log10_tau', 'true_log10_D', 'true_ln_D',
                    'true_ln_tau', 'true_mean_mag']
        for col in old_cols:
            if col in tbl.colnames and col not in savez_kwargs:
                savez_kwargs[col] = np.array(tbl[col][start_idx:end_idx], dtype=float)

    # Recovery-mock format: FITS has true_log10_sigma / true_log10_tau / true_mean_mag
    # without band suffix (recorded by the recovery-grid generator). Derive/copy
    # per-band truth columns for scalar posterior-recovery comparisons.
    if 'true_log10_sigma' in tbl.colnames and 'true_log10_tau' in tbl.colnames:
        _ln10 = np.log(10.0)
        _s = np.array(tbl['true_log10_sigma'][start_idx:end_idx], dtype=float)
        _t = np.array(tbl['true_log10_tau'][start_idx:end_idx], dtype=float)
        _m0 = None
        if 'true_mean_mag' in tbl.colnames:
            _m0 = np.array(tbl['true_mean_mag'][start_idx:end_idx], dtype=float)
        for band in bands_to_fit:
            if f'true_log10_D_{band}' not in savez_kwargs:
                savez_kwargs[f'true_log10_sigma_{band}'] = _s
                savez_kwargs[f'true_log10_tau_{band}']   = _t
                savez_kwargs[f'true_log10_D_{band}']     = 2.0 * _s - _t
                savez_kwargs[f'true_ln_D_{band}']        = (2.0 * _s - _t) * _ln10
                savez_kwargs[f'true_ln_tau_{band}']      = _t * _ln10
            if _m0 is not None and f'true_mean_mag_{band}' not in savez_kwargs:
                savez_kwargs[f'true_mean_mag_{band}'] = _m0
        if 'true_mean_mag' in tbl.colnames and 'true_mean_mag' not in savez_kwargs:
            savez_kwargs['true_mean_mag'] = np.array(tbl['true_mean_mag'][start_idx:end_idx], dtype=float)
        if 'true_m_win' in tbl.colnames and 'true_m_win' not in savez_kwargs:
            savez_kwargs['true_m_win'] = np.array(tbl['true_m_win'][start_idx:end_idx], dtype=float)
        for col in ['true_m_win_cont', 'true_m_win_cont_mu', 'true_m_win_cont_var']:
            if col in tbl.colnames and col not in savez_kwargs:
                savez_kwargs[col] = np.array(tbl[col][start_idx:end_idx], dtype=float)
        # Also copy grid metadata for recovery analysis
        for col in ['grid_i', 'grid_j', 'repeat_k', 'template_source_id']:
            if col in tbl.colnames and col not in savez_kwargs:
                savez_kwargs[col] = np.array(tbl[col][start_idx:end_idx])

    np.savez(output_path, **savez_kwargs)
    print("Done.")

    if ckpt_interval > 0 and os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
        print(f"Checkpoint folder deleted: {checkpoint_dir}")
    print(f"\nBands fitted: {bands_to_fit}")
    print("IMPORTANT: posterior_ln_D, posterior_ln_tau, and posterior_ln_jitter are in NATURAL LOG (ln).")
    print("Convert to log₁₀ for comparison: log₁₀(x) = ln(x) / ln(10) ≈ ln(x) / 2.303")

    # ============================================================================
    # SAVE FITTING METADATA
    # ============================================================================
    import json
    from datetime import datetime

    metadata = {
        # Fitting timestamp
        "fitting_timestamp": datetime.now().isoformat(),

        # Input/Output files
        "input_fits": args.input_fits,
        "output_npz": args.output_npz,

        # Bands
        "bands_fitted": bands_to_fit,
        "n_bands": len(bands_to_fit),

        # Fitting settings
        "start_index": start_idx,
        "end_index": end_idx,
        "max_objects_requested": args.max_objects,
        "n_objects_fitted": n_fit,
        "n_objects_successful_per_band": {band: int(n_success_per_band[band]) for band in bands_to_fit},
        "nwalkers": args.nwalkers,
        "nburn": args.nburn,
        "nsteps": args.nsteps,
        "nsamples_output": args.nsamples_output,
        "seed": args.seed,
        "seed_scheme": SEED_SCHEME,
        "n_jobs": args.n_jobs,
        "parallel_mode": use_parallel,
        "adaptive": args.adaptive,
        "target_thinned": args.target_thinned,
        "step_chunk": args.step_chunk,
        "max_steps": args.max_steps,
        "mwin_mode": args.mwin_mode,
        "diagnostics_summary": {
            band: {
                "median_nsteps_used":    int(np.median(diagnostics[band]['nsteps_used'][diagnostics[band]['nsteps_used'] > 0])) if np.any(diagnostics[band]['nsteps_used'] > 0) else 0,
                "median_accept_mean":    float(np.nanmedian(diagnostics[band]['accept_mean'])),
                "median_iat_ln_D":       float(np.nanmedian(diagnostics[band]['iat_ln_D'])),
                "median_iat_ln_tau":     float(np.nanmedian(diagnostics[band]['iat_ln_tau'])),
                "median_neff_ln_D":      float(np.nanmedian(diagnostics[band]['neff_ln_D'])),
                "median_neff_ln_tau":    float(np.nanmedian(diagnostics[band]['neff_ln_tau'])),
                "frac_autocorr_ok":      float(np.mean(diagnostics[band]['autocorr_ok'])),
            }
            for band in bands_to_fit
        },

        # Sampling model: parameterization + prior type, fully identifying the inference setup.
        "sampling_model": {
            "label": args.sampling_model,
            **SAMPLING_MODEL_REGISTRY.get(args.sampling_model, {
                "sampled_params": ["m", "ln_D", "ln_tau", "ln_jitter"],
                "parameterization": "user-defined",
                "prior_type": "user-defined",
                "description": "Unrecognised label — check SAMPLING_MODEL_REGISTRY in stage1_fitting.py.",
            }),
            "prior_bounds_used": {
                "m":         [float(PRIOR_M_MIN),          float(PRIOR_M_MAX)],
                "ln_D":      [float(PRIOR_LN_D_MIN),       float(PRIOR_LN_D_MAX)],
                "ln_sigma2": [float(PRIOR_LN_SIGMA2_MIN),  float(PRIOR_LN_SIGMA2_MAX)],
                "ln_sigma":  [float(PRIOR_LN_SIGMA_MIN),   float(PRIOR_LN_SIGMA_MAX)],
                "ln_tau":    [float(PRIOR_LN_TAU_MIN),     float(PRIOR_LN_TAU_MAX)],
                "ln_jitter": [float(PRIOR_LN_JITTER_MIN),  float(PRIOR_LN_JITTER_MAX)],
            },
        },

        # Output format
        "output_format": {
            "posterior_variables_per_band": [
                "posterior_m_{band}",
                "posterior_ln_D_{band}",
                "posterior_ln_tau_{band}",
                "posterior_ln_jitter_{band}",
            ] + ([
                "posterior_m_win_{band}",
                "posterior_m_win_mu_{band}",
                "posterior_m_win_var_{band}",
            ] if want_mwin_trap else []) + ([
                "posterior_m_win_cont_{band}",
                "posterior_m_win_mu_cont_{band}",
                "posterior_m_win_var_cont_{band}",
            ] if want_mwin_cont else []),
            "logarithm_base": "posterior_ln_*: natural log; posterior_m* draws/means: mag; posterior_m_win_var*: mag^2",
            "conversion_to_log10": "log10(x) = ln(x) / ln(10) ≈ ln(x) / 2.303",
            "shape": {
                "n_objects": n_fit,
                "n_samples_per_object": args.nsamples_output
            }
        },

        # Performance
        "performance": {
            "total_time_seconds": float(total_elapsed),
            "time_per_object_per_band_seconds": float(total_elapsed / (n_fit * len(bands_to_fit)))
        }
    }

    # Save metadata to JSON file
    metadata_path = output_path.replace('.npz', '_metadata.json')
    print(f"\nWriting fitting metadata to: {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved")

if __name__ == "__main__":
    main()
