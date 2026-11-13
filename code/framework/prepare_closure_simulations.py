#!/usr/bin/env python3
"""Prepare an end-to-end Stage-2 closure sample.

This script builds an explicit real-template <-> mock-truth pair table, draws
a declared nuisance tau population, computes the implied DRW sigma/SF_inf
values, simulates Gaia G light curves on the matched real cadences, and writes
the truth/manifest/PDF outputs used to inspect the closure sample before
Stage-1 fitting.
"""

from __future__ import annotations

from framework import PROJECT_ROOT, DATA_ROOT

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from framework.photometry import real_logl3000_from_mwin
from astropy.io import fits
from astropy.table import Table

from scipy.spatial import cKDTree

from framework.drw_fitting import (
    drw_window_integral_covariances,
    sample_true_mwin_cont,
    trapezoid_time_weights,
)
from plotting.diagnostics import (
    _write_selected_page,
)


DEFAULT_MOCK_DIR = Path(
    str(DATA_ROOT / 'results/stage2_mwin_chains/degeneracy_tests_mock/mock3')
)
# ── Real templates from a pre-processed MCMC cache ──────────────────────────
# A pre-processed NPZ containing source summaries and per-band posterior arrays.
# Fields expected: source_id, z, logl3000, and per-band posterior arrays
# with keys matching *__{band}__m_win / *__{band}__ln_D.
# Used by _load_real_targets.
DEFAULT_REAL_CHAINS_NPZ = Path(
    str(DATA_ROOT / 'results/mcmc_filtered_realdata_loo_gaia_g.npz')
)
# ── Real templates from Stage-1 posteriors ──────────────────────────────────
# A raw Stage-1 posteriors NPZ (e.g. posterior_samples.npz).
# Fields: source_id, redsh, logL3000, posterior_m_win_cont_Gaia_G,
#         posterior_ln_D_Gaia_G, posterior_ln_tau_Gaia_G, diag_npts_Gaia_G.
# Used by _load_real_targets_from_stage1_posteriors.
# Pass via --real-stage1-posteriors-npz; overrides --real-chains-npz when set.
DEFAULT_REAL_STAGE1_POSTERIORS_NPZ: Path | None = None  # None = disabled by default
DEFAULT_QUALITY_RUN_DIR = Path(
    str(DATA_ROOT / 'results/cosmology_closure/prior_5/stage2')
)
# ── Cadence FITS files ────────────────────────────────────────────────────────
# Flag-cleaned epochs before variability-flag rejection.
DEFAULT_CADENCE_FITS_FLAGCLEAN = Path(
    str(DATA_ROOT / 'external_data/gaia_g_cadence_pool.fits')
)
# Variability-flag-masked light curves for --real-stage1-posteriors-npz.
DEFAULT_CADENCE_FITS_VARREJECT = Path(
    str(DATA_ROOT / 'external_data/observed_gaia_light_curves.fits')
)
# Default cadence pool for the pre-processed MCMC cache input.
DEFAULT_CADENCE_FITS = DEFAULT_CADENCE_FITS_FLAGCLEAN

PARENT_SCHEMA_STAGE2_MWIN = "stage2_mwin_parent"
PARENT_SCHEMA_FULLCLOSURE = "fullclosure_truth_parent"


def _load_real_chains_cache(chains_npz: Path) -> dict:
    chains = np.load(chains_npz, allow_pickle=True)
    band = str(np.asarray(chains["bands_fit"]).flat[0])
    m_win_key = next(k for k in chains.files if k.endswith(f"__{band}__m_win") and "__diag__" not in k)
    ln_d_key = next(k for k in chains.files if k.endswith(f"__{band}__ln_D") and "__diag__" not in k)
    return {
        "band": band,
        "z": np.asarray(chains["z"], dtype=np.float64),
        "logl3000": np.asarray(chains["logl3000"], dtype=np.float64),
        "source_id": np.asarray(chains["source_id"], dtype=np.int64),
        "mwin_med": np.median(np.asarray(chains[m_win_key], dtype=np.float64), axis=1).astype(np.float64),
        "log10d_med": np.median(np.asarray(chains[ln_d_key], dtype=np.float64) / np.log(10.0), axis=1).astype(np.float64),
    }


def _weighted_standardize(
    x_target: np.ndarray,
    y_target: np.ndarray,
    z_target: np.ndarray,
    x_cand: np.ndarray,
    y_cand: np.ndarray,
    z_cand: np.ndarray,
    *,
    weights: tuple[float, float, float] = (1.0, 1.0, 2.5),
) -> tuple[np.ndarray, np.ndarray]:
    """Put matching features on common robust scales estimated from the targets.
    
    Use the target medians and central 68-percent half-widths, with scale fallbacks
    for constant features. Multiplication by sqrt(weights) sets their relative
    contributions to the subsequent squared Euclidean matching distance."""
    def _robust_center_scale(arr: np.ndarray) -> tuple[float, float]:
        center = float(np.nanmedian(arr))
        q16, q84 = np.nanpercentile(arr, [16.0, 84.0])
        scale = float(0.5 * (q84 - q16))
        if not np.isfinite(scale) or scale <= 1.0e-8:
            scale = float(np.nanstd(arr))
        if not np.isfinite(scale) or scale <= 1.0e-8:
            scale = 1.0
        return center, scale

    cx, sx = _robust_center_scale(np.asarray(x_target, dtype=np.float64))
    cy, sy = _robust_center_scale(np.asarray(y_target, dtype=np.float64))
    cz, sz = _robust_center_scale(np.asarray(z_target, dtype=np.float64))
    w = np.sqrt(np.asarray(weights, dtype=np.float64))
    target_feat = np.column_stack(
        [
            w[0] * ((np.asarray(x_target, dtype=np.float64) - cx) / sx),
            w[1] * ((np.asarray(y_target, dtype=np.float64) - cy) / sy),
            w[2] * ((np.asarray(z_target, dtype=np.float64) - cz) / sz),
        ]
    )
    cand_feat = np.column_stack(
        [
            w[0] * ((np.asarray(x_cand, dtype=np.float64) - cx) / sx),
            w[1] * ((np.asarray(y_cand, dtype=np.float64) - cy) / sy),
            w[2] * ((np.asarray(z_cand, dtype=np.float64) - cz) / sz),
        ]
    )
    return target_feat, cand_feat


def _fast_greedy_unique_match(
    target_feat: np.ndarray,
    cand_feat: np.ndarray,
    *,
    k_initial: int = 64,
    k_step: int = 64,
    k_max: int = 2048,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Assign distinct candidates to targets with a greedy nearest-neighbour search.
    
    Visit targets with the largest initial shortlist distances first, reserving
    each assigned candidate. Expand shortlists for remaining targets up to k_max.
    Return candidate indices (-1 when unmatched) and distance/count summaries.
    This is a greedy assignment, not a global minimum-cost matching algorithm."""
    n_target = int(target_feat.shape[0])
    n_cand = int(cand_feat.shape[0])
    if n_target == 0 or n_cand == 0:
        return np.full(n_target, -1, dtype=np.int64), {
            "shortlist_k_initial": int(k_initial),
            "shortlist_k_max": int(k_max),
            "n_batch_expansions": 0,
            "n_unmatched": int(n_target),
            "mean_assignment_distance": np.nan,
            "p90_assignment_distance": np.nan,
            "hardest_first_distance_p90": np.nan,
        }

    tree = cKDTree(cand_feat)
    k0 = max(1, min(int(k_initial), n_cand))
    d0, i0 = tree.query(target_feat, k=k0, workers=-1)
    if k0 == 1:
        d0 = d0[:, None]
        i0 = i0[:, None]

    hardness = np.where(np.isfinite(d0[:, -1]), d0[:, -1], np.inf)
    order = np.argsort(-hardness, kind="mergesort")
    assigned = np.full(n_target, -1, dtype=np.int64)
    assign_dist = np.full(n_target, np.nan, dtype=np.float64)
    used = np.zeros(n_cand, dtype=bool)

    for ridx in order:
        row_i = i0[ridx]
        row_d = d0[ridx]
        avail = np.isfinite(row_d) & ~used[row_i]
        hits = np.nonzero(avail)[0]
        if hits.size > 0:
            j = int(hits[0])
            c = int(row_i[j])
            used[c] = True
            assigned[ridx] = c
            assign_dist[ridx] = float(row_d[j])

    unmatched = np.where(assigned < 0)[0]
    k_cur = k0
    n_batch_expansions = 0
    while unmatched.size > 0 and k_cur < k_max and k_cur < n_cand:
        k_cur = min(int(k_cur + k_step), int(k_max), n_cand)
        n_batch_expansions += 1
        d_b, i_b = tree.query(target_feat[unmatched], k=k_cur, workers=-1)
        if k_cur == 1:
            d_b = d_b[:, None]
            i_b = i_b[:, None]
        still_unmatched: list[int] = []
        for j_local, ridx in enumerate(unmatched):
            row_i_b = i_b[j_local]
            row_d_b = d_b[j_local]
            avail = np.isfinite(row_d_b) & ~used[row_i_b]
            hits = np.nonzero(avail)[0]
            if hits.size > 0:
                j = int(hits[0])
                c = int(row_i_b[j])
                used[c] = True
                assigned[ridx] = c
                assign_dist[ridx] = float(row_d_b[j])
            else:
                still_unmatched.append(int(ridx))
        unmatched = np.asarray(still_unmatched, dtype=np.int64)

    matched = np.isfinite(assign_dist)
    return assigned, {
        "shortlist_k_initial": int(k0),
        "shortlist_k_max": int(min(n_cand, int(k_max))),
        "n_batch_expansions": int(n_batch_expansions),
        "n_unmatched": int(np.sum(assigned < 0)),
        "mean_assignment_distance": float(np.nanmean(assign_dist)) if np.any(matched) else np.nan,
        "p90_assignment_distance": float(np.nanpercentile(assign_dist[matched], 90.0)) if np.any(matched) else np.nan,
        "hardest_first_distance_p90": float(np.nanpercentile(hardness[np.isfinite(hardness)], 90.0))
        if np.any(np.isfinite(hardness))
        else np.nan,
    }


def _sha256_int_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(arr, dtype=np.int64).tobytes()).hexdigest()


def _finite_percentiles(x: np.ndarray) -> dict[str, float]:
    vals = np.asarray(x, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {}
    qs = [0.5, 1.0, 5.0, 16.0, 50.0, 84.0, 95.0, 99.0, 99.5]
    out = {f"p{q:g}": float(v) for q, v in zip(qs, np.percentile(vals, qs))}
    out["mean"] = float(np.mean(vals))
    out["std"] = float(np.std(vals))
    return out


def _json_number(x: object) -> int | float:
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    raise TypeError(f"Cannot convert {type(x)!r} to a JSON number")


def _draw_truncated_log10_tau(
    *,
    rng: np.random.Generator,
    n: int,
    mu: float,
    sigma: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    out = np.empty(int(n), dtype=np.float64)
    filled = 0
    batch = max(4096, int(2 * n))
    while filled < n:
        draw = rng.normal(float(mu), float(sigma), size=batch)
        draw = draw[(draw >= lo) & (draw <= hi)]
        if draw.size == 0:
            continue
        take = min(int(draw.size), int(n - filled))
        out[filled : filled + take] = draw[:take]
        filled += take
    return out


def _simulate_ou(times_rf: np.ndarray, *, sigma: float, tau: float, mean_mag: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate an OU/DRW path at rest-frame epochs, returned in input order."""
    t = np.asarray(times_rf, dtype=np.float64)
    if t.size == 0:
        return np.array([], dtype=np.float64)

    order = np.argsort(t)
    t_s = t[order]
    m_s = np.empty(t_s.size, dtype=np.float64)
    m_s[0] = float(mean_mag) + float(sigma) * rng.standard_normal()
    for k in range(1, t_s.size):
        dt = max(float(t_s[k] - t_s[k - 1]), 0.0)
        rho = np.exp(-dt / float(tau))
        cond_std = float(sigma) * np.sqrt(max(1.0 - rho * rho, 0.0))
        m_s[k] = float(mean_mag) + rho * (m_s[k - 1] - float(mean_mag)) + cond_std * rng.standard_normal()

    out = np.empty_like(m_s)
    out[order] = m_s
    return out


def _simulate_ou_conditioned_on_mwin(
    times_rf: np.ndarray,
    *,
    sigma: float,
    tau: float,
    mean_mag: float,
    mwin_target: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw latent OU epochs conditioned on a continuous-window magnitude.
    
    Jointly draw an unconditional path and its continuous window mean, then apply
    the Gaussian shift Cov(path, mean)/Var(mean) times the target-minus-drawn mean.
    For degenerate covariances, the fallback matches the trapezoid window mean by
    a constant shift. Times and tau are in rest-frame days; sigma is in magnitudes."""
    m_uncond = _simulate_ou(times_rf, sigma=sigma, tau=tau, mean_mag=mean_mag, rng=rng)
    t = np.asarray(times_rf, dtype=np.float64)
    if t.size <= 1:
        return np.full_like(m_uncond, float(mwin_target), dtype=np.float64)

    sigma2 = float(sigma) * float(sigma)
    mwin_draw, _, _ = sample_true_mwin_cont(t, m_uncond, mean_mag, sigma2=sigma2, tau=tau, rng=rng)
    c, v = drw_window_integral_covariances(t, sigma2=sigma2, tau=tau)
    if np.isfinite(mwin_draw) and np.isfinite(v) and v > 0.0 and np.all(np.isfinite(c)):
        return m_uncond + (c / float(v)) * (float(mwin_target) - float(mwin_draw))

    w = trapezoid_time_weights(t)
    trap = float(np.dot(w, m_uncond))
    return m_uncond + (float(mwin_target) - trap)


def _build_empirical_gaia_error_curve(templates: list[dict[str, Any]], *, bin_width: float = 0.1) -> dict[str, np.ndarray]:
    mags = np.concatenate([np.asarray(t["g_mag"], dtype=np.float64) for t in templates])
    errs = np.concatenate([np.asarray(t["g_mag_err"], dtype=np.float64) for t in templates])
    ok = np.isfinite(mags) & np.isfinite(errs) & (errs > 0.0)
    mags = mags[ok]
    errs = errs[ok]
    if mags.size < 100:
        return {
            "centers": np.asarray([15.0, 20.0], dtype=np.float64),
            "log_err": np.log(np.asarray([0.02, 0.02], dtype=np.float64)),
        }

    lo = float(np.floor(np.nanpercentile(mags, 0.1) / bin_width) * bin_width)
    hi = float(np.ceil(np.nanpercentile(mags, 99.9) / bin_width) * bin_width)
    edges = np.arange(lo, hi + bin_width, bin_width, dtype=np.float64)
    if edges.size < 4:
        edges = np.linspace(float(np.nanmin(mags)), float(np.nanmax(mags)), 12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(mags, edges) - 1
    med = np.full(centers.size, np.nan, dtype=np.float64)
    for k in range(centers.size):
        vals = errs[idx == k]
        if vals.size >= 20:
            med[k] = float(np.nanmedian(vals))

    good = np.isfinite(med) & (med > 0.0)
    if np.count_nonzero(good) < 2:
        val = float(np.nanmedian(errs))
        med[:] = val if np.isfinite(val) and val > 0.0 else 0.02
    else:
        med[~good] = np.interp(centers[~good], centers[good], med[good])
    med = np.maximum.accumulate(np.clip(med, 1.0e-4, 1.0))
    return {"centers": centers, "log_err": np.log(med)}


def _eval_empirical_gaia_error_curve(curve: dict[str, np.ndarray], mag: np.ndarray) -> np.ndarray:
    centers = np.asarray(curve["centers"], dtype=np.float64)
    log_err = np.asarray(curve["log_err"], dtype=np.float64)
    out = np.exp(np.interp(np.asarray(mag, dtype=np.float64), centers, log_err, left=log_err[0], right=log_err[-1]))
    return np.clip(out, 1.0e-4, 1.0)


def _clean_gaia_template_row(row: Any, *, z_for_rest_frame: float, min_epochs: int) -> dict[str, Any]:
    """Clean and sort Gaia epochs, using the mock redshift for rest-frame intervals.
    
    Keep observed dates and template photometric uncertainties; the returned
    baseline and intervals use z_for_rest_frame rather than the template redshift.
    Raise ValueError when fewer than min_epochs valid points remain."""
    t = np.asarray(row["g_transit_time"], dtype=np.float64)
    f = np.asarray(row["g_transit_flux"], dtype=np.float64)
    fe = np.asarray(row["g_transit_flux_error"], dtype=np.float64)
    mag = np.asarray(row["g_transit_mag"], dtype=np.float64)

    good = (
        np.isfinite(t)
        & np.isfinite(f)
        & np.isfinite(fe)
        & np.isfinite(mag)
        & (f > 0.0)
        & (fe > 0.0)
        & (mag > 0.0)
    )
    if int(np.sum(good)) < int(min_epochs):
        raise ValueError(f"Template source_id={int(row['source_id'])} has fewer than {min_epochs} clean epochs")

    t = t[good]
    f = f[good]
    fe = fe[good]
    mag = mag[good]
    order = np.argsort(t)
    t = t[order]
    f = f[order]
    fe = fe[order]
    mag = mag[order]

    with np.errstate(divide="ignore", invalid="ignore"):
        mag_err = 1.0857362047581294 * (fe / f)
    bad = ~np.isfinite(mag_err) | (mag_err <= 0.0)
    if bad.all():
        mag_err[:] = 0.02
    elif bad.any():
        mag_err[bad] = float(np.nanmedian(mag_err[~bad]))
    mag_err = np.clip(mag_err, 1.0e-4, 1.0)

    t0 = float(np.nanmin(t))
    z = float(z_for_rest_frame)
    t_rf = (t - t0) / (1.0 + z)
    baseline_rest = float(np.nanmax(t_rf) - np.nanmin(t_rf)) if t_rf.size else np.nan
    mean_dt_rest = baseline_rest / max(int(t_rf.size) - 1, 1) if np.isfinite(baseline_rest) else np.nan
    return {
        "template_source_id": int(row["source_id"]),
        "template_z": float(row["z"]),
        "g_time": t.astype(np.float64),
        "t_rf": t_rf.astype(np.float64),
        "g_mag": mag.astype(np.float64),
        "g_mag_err": mag_err.astype(np.float64),
        "template_median_mag": float(np.nanmedian(mag)),
        "template_median_mag_err": float(np.nanmedian(mag_err)),
        "npts": int(t_rf.size),
        "baseline_rest": float(baseline_rest),
        "mean_dt_rest": float(mean_dt_rest),
    }


def _load_gaia_templates_by_source_id(
    *,
    cadence_fits: Path,
    source_ids: np.ndarray,
    mock_z: np.ndarray,
    min_epochs: int,
) -> list[dict[str, Any]]:
    requested = np.asarray(source_ids, dtype=np.int64)
    z_for_rf = np.asarray(mock_z, dtype=np.float64)
    if requested.shape != z_for_rf.shape:
        raise ValueError("source_ids and mock_z must have the same shape")

    print(f"Loading {requested.size:,} exact Gaia cadence templates from {cadence_fits} ...", flush=True)
    with fits.open(cadence_fits, memmap=True, lazy_load_hdus=True) as hdul:
        data = hdul[1].data
        all_ids = np.asarray(data["source_id"], dtype=np.int64)
        order = np.argsort(all_ids, kind="mergesort")
        sorted_ids = all_ids[order]

        templates: list[dict[str, Any]] = []
        missing: list[int] = []
        for sid, z in zip(requested, z_for_rf):
            pos = int(np.searchsorted(sorted_ids, int(sid)))
            if pos >= sorted_ids.size or int(sorted_ids[pos]) != int(sid):
                missing.append(int(sid))
                continue
            row_idx = int(order[pos])
            templates.append(_clean_gaia_template_row(data[row_idx], z_for_rest_frame=float(z), min_epochs=min_epochs))

    if missing:
        preview = ", ".join(str(x) for x in missing[:10])
        raise RuntimeError(f"{len(missing)} requested real template source IDs were not found in {cadence_fits}: {preview}")
    return templates


def _simulate_closure_light_curves(
    *,
    out_fits: Path,
    templates: list[dict[str, Any]],
    real_source_id: np.ndarray,
    mock_source_id: np.ndarray,
    mock_z: np.ndarray,
    mock_logl3000: np.ndarray,
    mock_mwin_target_true: np.ndarray,
    mock_log10_d_true: np.ndarray,
    tau_days: np.ndarray,
    sigma_drw_mag: np.ndarray,
    rng_lc: np.random.Generator,
    rng_mwin_cont: np.random.Generator,
    raw_mag_max: float | None = None,
    raw_snr_min: float | None = None,
    raw_min_epochs: int | None = None,
    n_keep: int | None = None,
    anchor_mode: str = "stationary_m0",
) -> dict[str, np.ndarray]:
    """Simulate light curves on matched cadences and write their selected FITS rows.
    
    Use D = sigma^2/tau and the supplied nuisance damping times. Conditional modes
    anchor the continuous window mean to the injected target; the stochastic-m0
    mode first draws the process mean around it using the unconditional window
    variance. In stationary_m0 mode the process mean is fixed and the realised
    window mean is drawn conditional on the latent epochs.
    
    Template magnitude errors are reused unchanged as independent Gaussian noise.
    Apply the optional raw magnitude/SNR/epoch cuts to the simulated observations,
    then retain the first n_keep passing rows if requested. Store both the process
    mean and the realised continuous/trapezoid window summaries as distinct truth
    quantities, together with source/template IDs and the observed epoch vectors."""
    n = int(len(templates))
    if n != int(mock_source_id.size):
        raise ValueError("templates and mock arrays have inconsistent lengths")
    if anchor_mode not in {"stationary_m0", "conditional_mwin", "conditional_mwin_stochastic_m0"}:
        raise ValueError(f"Unknown anchor_mode={anchor_mode!r}")

    true_log10_sigma = np.log10(np.asarray(sigma_drw_mag, dtype=np.float64))
    true_log10_sigma2 = 2.0 * true_log10_sigma
    true_log10_tau = np.log10(np.asarray(tau_days, dtype=np.float64))
    true_ln_tau = np.log(np.asarray(tau_days, dtype=np.float64))
    true_ln_d = np.asarray(mock_log10_d_true, dtype=np.float64) * np.log(10.0)

    rows_m0_gen = np.empty(n, dtype=np.float64)
    rows_m0_minus_mwin_cont = np.empty(n, dtype=np.float64)
    rows_mwin_cont_uncond_var = np.empty(n, dtype=np.float64)
    rows_true_mwin_trap = np.empty(n, dtype=np.float64)
    rows_true_mwin_cont = np.empty(n, dtype=np.float64)
    rows_true_mwin_cont_mu = np.empty(n, dtype=np.float64)
    rows_true_mwin_cont_var = np.empty(n, dtype=np.float64)
    rows_baseline_rf = np.empty(n, dtype=np.float64)
    rows_mean_dt_rf = np.empty(n, dtype=np.float64)
    rows_npts = np.empty(n, dtype=np.int32)
    rows_median_mag_err = np.empty(n, dtype=np.float64)
    rows_mock_median_mag_err = np.empty(n, dtype=np.float64)
    rows_error_scale_med = np.empty(n, dtype=np.float64)
    rows_template_z = np.empty(n, dtype=np.float64)
    rows_template_median_mag = np.empty(n, dtype=np.float64)
    rows_gaia_g_mag_med = np.empty(n, dtype=np.float64)
    rows_gaia_g_snr = np.empty(n, dtype=np.float64)
    rows_gaia_g_n_epochs = np.empty(n, dtype=np.int32)

    rows_times = np.empty(n, dtype=object)
    rows_mags_true = np.empty(n, dtype=object)
    rows_mags = np.empty(n, dtype=object)
    rows_mag_errs = np.empty(n, dtype=object)

    print(f"Simulating {n:,} Gaia G closure light curves ...", flush=True)
    for i, tmpl in enumerate(templates):
        mwin_target = float(mock_mwin_target_true[i])
        sigma = float(sigma_drw_mag[i])
        sigma2 = float(sigma * sigma)
        tau = float(tau_days[i])
        t_rf = np.asarray(tmpl["t_rf"], dtype=np.float64)
        template_mag_err = np.asarray(tmpl["g_mag_err"], dtype=np.float64)
        _, v_win_uncond = drw_window_integral_covariances(t_rf, sigma2=sigma2, tau=tau)
        if not np.isfinite(v_win_uncond) or v_win_uncond < 0.0:
            v_win_uncond = 0.0

        if anchor_mode == "conditional_mwin_stochastic_m0":
            mean_mag = mwin_target + float(rng_mwin_cont.normal(scale=np.sqrt(v_win_uncond)))
            m_true = _simulate_ou_conditioned_on_mwin(
                t_rf,
                sigma=sigma,
                tau=tau,
                mean_mag=mean_mag,
                mwin_target=mwin_target,
                rng=rng_lc,
            )
        elif anchor_mode == "conditional_mwin":
            mean_mag = mwin_target
            m_true = _simulate_ou_conditioned_on_mwin(
                t_rf,
                sigma=sigma,
                tau=tau,
                mean_mag=mean_mag,
                mwin_target=mwin_target,
                rng=rng_lc,
            )
        else:
            mean_mag = mwin_target
            m_true = _simulate_ou(t_rf, sigma=sigma, tau=tau, mean_mag=mean_mag, rng=rng_lc)

        mag_err = template_mag_err
        scale = np.ones_like(mag_err, dtype=np.float64)
        m_obs = m_true + rng_lc.normal(scale=mag_err)
        raw_good = np.isfinite(m_obs) & np.isfinite(mag_err) & (mag_err > 0.0)
        raw_n = int(np.count_nonzero(raw_good))
        if raw_n > 0:
            raw_mag = m_obs[raw_good]
            raw_err = mag_err[raw_good]
            raw_mag_med = float(np.median(raw_mag))
            if raw_n >= 2:
                raw_signal = float(np.percentile(raw_mag, 95.0) - np.percentile(raw_mag, 5.0))
                raw_noise = float(np.median(raw_err))
                raw_snr = raw_signal / raw_noise if raw_noise > 0.0 else np.nan
            else:
                raw_snr = np.nan
        else:
            raw_mag_med = np.nan
            raw_snr = np.nan

        w_rf = trapezoid_time_weights(t_rf)
        true_mwin_trap = float(np.dot(w_rf, m_true))
        if anchor_mode in {"conditional_mwin", "conditional_mwin_stochastic_m0"}:
            true_mwin_cont = mwin_target
            true_mwin_cont_mu = mwin_target
            true_mwin_cont_var = 0.0
        else:
            true_mwin_cont, true_mwin_cont_mu, true_mwin_cont_var = sample_true_mwin_cont(
                t_rf,
                m_true,
                mean_mag,
                sigma2=float(sigma * sigma),
                tau=tau,
                rng=rng_mwin_cont,
            )

        rows_true_mwin_trap[i] = true_mwin_trap
        rows_true_mwin_cont[i] = true_mwin_cont
        rows_true_mwin_cont_mu[i] = true_mwin_cont_mu
        rows_true_mwin_cont_var[i] = true_mwin_cont_var
        rows_m0_gen[i] = mean_mag
        rows_m0_minus_mwin_cont[i] = mean_mag - true_mwin_cont
        rows_mwin_cont_uncond_var[i] = v_win_uncond
        rows_baseline_rf[i] = float(tmpl["baseline_rest"])
        rows_mean_dt_rf[i] = float(tmpl["mean_dt_rest"])
        rows_npts[i] = int(tmpl["npts"])
        rows_median_mag_err[i] = float(tmpl["template_median_mag_err"])
        rows_mock_median_mag_err[i] = float(np.nanmedian(mag_err))
        rows_error_scale_med[i] = float(np.nanmedian(scale))
        rows_template_z[i] = float(tmpl["template_z"])
        rows_template_median_mag[i] = float(tmpl["template_median_mag"])
        rows_gaia_g_mag_med[i] = raw_mag_med
        rows_gaia_g_snr[i] = raw_snr
        rows_gaia_g_n_epochs[i] = raw_n
        rows_times[i] = np.asarray(tmpl["g_time"], dtype=np.float64)
        rows_mags_true[i] = m_true.astype(np.float64)
        rows_mags[i] = m_obs.astype(np.float64)
        rows_mag_errs[i] = mag_err.astype(np.float64)

    selection_mask = np.ones(n, dtype=bool)
    n_after_raw_mag = n
    n_after_raw_snr = n
    n_after_raw_epochs = n
    if raw_mag_max is not None:
        selection_mask &= np.isfinite(rows_gaia_g_mag_med) & (rows_gaia_g_mag_med < float(raw_mag_max))
        n_after_raw_mag = int(np.sum(selection_mask))
    if raw_snr_min is not None:
        selection_mask &= np.isfinite(rows_gaia_g_snr) & (rows_gaia_g_snr > float(raw_snr_min))
        n_after_raw_snr = int(np.sum(selection_mask))
    if raw_min_epochs is not None:
        selection_mask &= np.isfinite(rows_gaia_g_n_epochs) & (rows_gaia_g_n_epochs >= int(raw_min_epochs))
        n_after_raw_epochs = int(np.sum(selection_mask))

    selected_idx = np.flatnonzero(selection_mask)
    n_after_raw_cuts = int(selected_idx.size)
    if n_keep is not None:
        n_keep_int = int(n_keep)
        if selected_idx.size < n_keep_int:
            raise RuntimeError(
                f"Only {selected_idx.size:,} / {n:,} simulated closure candidates pass the mock raw cuts, "
                f"but n_keep={n_keep_int:,}. Increase --candidate-oversample-factor."
            )
        selected_idx = selected_idx[:n_keep_int]

    s = selected_idx
    tbl = Table()
    tbl["source_id"] = np.asarray(mock_source_id, dtype=np.int64)[s]
    tbl["redsh"] = np.asarray(mock_z, dtype=np.float64)[s]
    tbl["logl3000"] = np.asarray(mock_logl3000, dtype=np.float64)[s]
    tbl["true_log10_sigma"] = true_log10_sigma[s]
    tbl["true_log10_sigma2"] = true_log10_sigma2[s]
    tbl["true_log10_tau"] = true_log10_tau[s]
    tbl["true_log10_D"] = np.asarray(mock_log10_d_true, dtype=np.float64)[s]
    tbl["true_ln_tau"] = true_ln_tau[s]
    tbl["true_ln_D"] = true_ln_d[s]
    tbl["true_m0"] = rows_m0_gen[s]
    tbl["true_mean_mag"] = rows_m0_gen[s]
    tbl["true_m0_minus_m_win_cont"] = rows_m0_minus_mwin_cont[s]
    tbl["true_m_win"] = rows_true_mwin_trap[s]
    tbl["true_m_win_cont"] = rows_true_mwin_cont[s]
    tbl["true_m_win_cont_mu"] = rows_true_mwin_cont_mu[s]
    tbl["true_m_win_cont_var"] = rows_true_mwin_cont_var[s]
    tbl["true_m_win_cont_uncond_var"] = rows_mwin_cont_uncond_var[s]
    tbl["template_source_id"] = np.asarray(real_source_id, dtype=np.int64)[s]
    tbl["template_z"] = rows_template_z[s]
    tbl["template_median_mag"] = rows_template_median_mag[s]
    tbl["true_baseline_rest"] = rows_baseline_rf[s]
    tbl["true_mean_dt_rest"] = rows_mean_dt_rf[s]
    tbl["true_npts"] = rows_npts[s]
    tbl["template_median_mag_err"] = rows_median_mag_err[s]
    tbl["mock_median_mag_err"] = rows_mock_median_mag_err[s]
    tbl["error_scale_median"] = rows_error_scale_med[s]
    tbl["gaia_g_mag_med"] = rows_gaia_g_mag_med[s]
    tbl["gaia_g_snr"] = rows_gaia_g_snr[s]
    tbl["gaia_g_n_epochs"] = rows_gaia_g_n_epochs[s]
    tbl["candidate_index_before_raw_cut"] = s.astype(np.int32)
    tbl["Gaia_G_transit_time"] = rows_times[s]
    tbl["Gaia_G_mag_true"] = rows_mags_true[s]
    tbl["Gaia_G_mag"] = rows_mags[s]
    tbl["Gaia_G_mag_err"] = rows_mag_errs[s]

    out_fits.parent.mkdir(parents=True, exist_ok=True)
    tbl.write(out_fits, overwrite=True)
    return {
        "selected_candidate_indices": s.astype(np.int64),
        "n_candidates_simulated": np.asarray([n], dtype=np.int64),
        "n_after_mock_raw_mag": np.asarray([n_after_raw_mag], dtype=np.int64),
        "n_after_mock_raw_snr": np.asarray([n_after_raw_snr], dtype=np.int64),
        "n_after_mock_raw_epochs": np.asarray([n_after_raw_epochs], dtype=np.int64),
        "n_after_mock_raw_cuts": np.asarray([n_after_raw_cuts], dtype=np.int64),
        "mock_m0_gen_true": rows_m0_gen[s],
        "mock_m0_minus_mwin_cont_true": rows_m0_minus_mwin_cont[s],
        "mock_m_win_trap_true": rows_true_mwin_trap[s],
        "mock_m_win_cont_true": rows_true_mwin_cont[s],
        "mock_m_win_cont_mu": rows_true_mwin_cont_mu[s],
        "mock_m_win_cont_var": rows_true_mwin_cont_var[s],
        "mock_m_win_cont_uncond_var": rows_mwin_cont_uncond_var[s],
        "baseline_rest_days": rows_baseline_rf[s],
        "mean_dt_rest_days": rows_mean_dt_rf[s],
        "npts": rows_npts[s],
        "median_mag_err": rows_median_mag_err[s],
        "mock_median_mag_err": rows_mock_median_mag_err[s],
        "error_scale_median": rows_error_scale_med[s],
        "template_z": rows_template_z[s],
        "template_median_mag": rows_template_median_mag[s],
        "gaia_g_mag_med": rows_gaia_g_mag_med[s],
        "gaia_g_snr": rows_gaia_g_snr[s],
        "gaia_g_n_epochs": rows_gaia_g_n_epochs[s],
    }


def _load_real_targets(
    *,
    real_chains_npz: Path,
    quality_cuts_run_dir: Path,
) -> dict[str, np.ndarray]:
    cache = _load_real_chains_cache(real_chains_npz)
    with np.load(quality_cuts_run_dir / 'selected_source_ids.npz', allow_pickle=False) as used:
        sid_used = np.asarray(used["source_id_selected"], dtype=np.int64)

    mask = np.isin(np.asarray(cache["source_id"], dtype=np.int64), sid_used)
    for key in ("z", "logl3000", "mwin_med", "log10d_med"):
        mask &= np.isfinite(np.asarray(cache[key], dtype=np.float64))

    return {
        "source_id": np.asarray(cache["source_id"], dtype=np.int64)[mask],
        "z": np.asarray(cache["z"], dtype=np.float64)[mask],
        "logl3000_temple": np.asarray(cache["logl3000"], dtype=np.float64)[mask],
        "m_win": np.asarray(cache["mwin_med"], dtype=np.float64)[mask],
        "log10d": np.asarray(cache["log10d_med"], dtype=np.float64)[mask],
        "band": np.asarray([str(cache["band"])]),
    }


def _load_real_targets_from_stage1_posteriors(
    *,
    stage1_posteriors_npz: Path,
    real_used_source_ids_npz: Path | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    mwin_cont_max: float | None = None,
    tau_med_min_days: float | None = None,
    pool_min_epochs: int | None = None,
    logl_min: float | None = None,
    logl_max: float | None = None,
) -> dict[str, np.ndarray]:
    """Load real templates from the Stage-1 posterior arrays.

    The --real-stage1-posteriors-npz input contains source_id, redsh,
    logL3000, posterior_m_win_cont_Gaia_G, posterior_ln_D_Gaia_G,
    posterior_ln_tau_Gaia_G and diag_npts_Gaia_G. Posterior arrays have
    shape (N, K), and diag_npts_Gaia_G supplies one epoch count per source.

    When real_used_source_ids_npz is supplied, sources are selected and
    ordered to match its manifest before applying the optional pool cuts.
    Pool cuts can also be applied without a source-selection manifest.
    """
    print(f"Loading real template pool from Stage-1 posteriors NPZ: {stage1_posteriors_npz}", flush=True)
    manifest_source_ids: np.ndarray | None = None
    if real_used_source_ids_npz is not None:
        print(f"  Applying exact real source manifest: {real_used_source_ids_npz}", flush=True)
        with np.load(real_used_source_ids_npz, allow_pickle=False) as used:
            manifest_source_ids = np.asarray(used["source_id_selected"], dtype=np.int64)
        if manifest_source_ids.ndim != 1:
            raise ValueError(
                f"{real_used_source_ids_npz} source_id_selected must be 1D; "
                f"got shape={manifest_source_ids.shape}"
            )
        if np.unique(manifest_source_ids).size != manifest_source_ids.size:
            raise ValueError(f"{real_used_source_ids_npz} contains duplicate source_id_selected values")

    with np.load(stage1_posteriors_npz, allow_pickle=False) as d:
        source_id = np.asarray(d["source_id"], dtype=np.int64)
        z = np.asarray(d["redsh"], dtype=np.float64)
        logl3000 = np.asarray(d["logL3000"], dtype=np.float64)
        npts = np.asarray(d["diag_npts_Gaia_G"], dtype=np.int32)
        if manifest_source_ids is not None:
            order = np.argsort(source_id, kind="mergesort")
            sorted_source_id = source_id[order]
            positions = np.searchsorted(sorted_source_id, manifest_source_ids)
            in_bounds = positions < sorted_source_id.size
            found = np.zeros(manifest_source_ids.shape, dtype=bool)
            found[in_bounds] = sorted_source_id[positions[in_bounds]] == manifest_source_ids[in_bounds]
            if not np.all(found):
                missing = manifest_source_ids[~found]
                preview = ", ".join(str(int(x)) for x in missing[:10])
                suffix = " ..." if missing.size > 10 else ""
                raise ValueError(
                    f"{missing.size:,} source IDs from {real_used_source_ids_npz} are missing from "
                    f"{stage1_posteriors_npz}. First missing: {preview}{suffix}"
                )
            row_idx = order[positions]
            source_id = source_id[row_idx]
            z = z[row_idx]
            logl3000 = logl3000[row_idx]
            npts = npts[row_idx]
            print(f"  Manifest matched {source_id.size:,} Stage-1 rows in manifest order.", flush=True)
        else:
            row_idx = None
        def _posterior_median(key: str) -> np.ndarray:
            draws = np.asarray(d[key])
            if row_idx is not None:
                draws = draws[row_idx]
            return np.nanmedian(np.asarray(draws, dtype=np.float64), axis=1)

        # Use nanmedian throughout. Some chains have a minority of NaN draws:
        # the affected source can still have valid posteriors from its remaining
        # finite draws, matching the diagnostic scripts' selection convention.
        mwin_cont_med = _posterior_median("posterior_m_win_cont_Gaia_G")
        log10d_med = _posterior_median("posterior_ln_D_Gaia_G") / np.log(10.0)
        tau_med = np.exp(_posterior_median("posterior_ln_tau_Gaia_G"))

    mask = np.isfinite(z) & np.isfinite(logl3000) & np.isfinite(mwin_cont_med) & np.isfinite(log10d_med)

    cut_log: list[str] = []
    if z_min is not None:
        mask &= z >= float(z_min)
        cut_log.append(f"z>={z_min}")
    if z_max is not None:
        mask &= z <= float(z_max)
        cut_log.append(f"z<={z_max}")
    if mwin_cont_max is not None:
        mask &= mwin_cont_med <= float(mwin_cont_max)
        cut_log.append(f"mwin_cont_med<={mwin_cont_max}")
    if tau_med_min_days is not None:
        mask &= np.isfinite(tau_med) & (tau_med >= float(tau_med_min_days))
        cut_log.append(f"tau_med>={tau_med_min_days}days")
    if pool_min_epochs is not None:
        mask &= npts >= int(pool_min_epochs)
        cut_log.append(f"npts>={pool_min_epochs}")
    if logl_min is not None:
        mask &= logl3000 >= float(logl_min)
        cut_log.append(f"logl>={logl_min}")
    if logl_max is not None:
        mask &= logl3000 <= float(logl_max)
        cut_log.append(f"logl<={logl_max}")

    n_total = int(source_id.size)
    n_pass = int(np.sum(mask))
    cuts_str = ", ".join(cut_log) if cut_log else "none"
    selection_kind = "manifest + pool quality cuts" if manifest_source_ids is not None else "pool quality cuts"
    print(f"  {selection_kind} ({cuts_str}): {n_pass:,} / {n_total:,} pass", flush=True)

    return {
        "source_id": source_id[mask],
        "z": z[mask],
        "logl3000_temple": logl3000[mask],
        "m_win": mwin_cont_med[mask],
        "log10d": log10d_med[mask],
        "band": np.asarray(["gaia_g"]),
    }


def _choose_pilot_targets(
    targets: dict[str, np.ndarray],
    *,
    n_pilot: int,
    seed: int,
) -> np.ndarray:
    n_available = int(np.asarray(targets["source_id"]).size)
    if n_pilot > n_available:
        raise ValueError(f"Requested n_pilot={n_pilot}, but only {n_available} targets are available")
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(n_available, size=int(n_pilot), replace=False)
    # Sort by source ID for stable files while keeping the random selection.
    chosen = chosen[np.argsort(np.asarray(targets["source_id"], dtype=np.int64)[chosen], kind="mergesort")]
    return chosen.astype(np.int64, copy=False)


def _require_npz_keys(npz: np.lib.npyio.NpzFile, *, keys: tuple[str, ...], path: Path, schema: str) -> None:
    missing = [key for key in keys if key not in npz.files]
    if missing:
        raise KeyError(
            f"{path} does not match --mock-parent-schema {schema!r}; "
            f"missing required field(s): {', '.join(missing)}"
        )


def _read_single_string(npz: np.lib.npyio.NpzFile, *, key: str, path: Path, schema: str) -> str:
    _require_npz_keys(npz, keys=(key,), path=path, schema=schema)
    arr = np.asarray(npz[key])
    if arr.size != 1:
        raise ValueError(f"{path} field {key!r} must contain exactly one string for schema {schema!r}.")
    return str(arr.reshape(-1)[0])


def _load_mock_candidates(
    *,
    mock_parent_npz: Path,
    quality_cuts_run_dir: Path,
    apply_parent_mag_cut: bool,
    mock_parent_schema: str,
) -> dict[str, np.ndarray]:
    """Load injected candidates and apply the configured pre-simulation cuts.
    
    The declared parent schema determines whether matching uses injected truth or
    supplied posterior medians. Select finite rows, configured redshift/luminosity
    and inclusive matching-rate bounds, and the optional parent-magnitude bound.
    For the full-closure schema the matching rate is injected true log10(D).
    Return all arrays in the same selected source order."""
    with (quality_cuts_run_dir / 'inference_settings.json').open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cuts_submit = cfg.get("cuts_submit", {})
    cuts_stage2 = cfg.get("cuts_stage2", {})

    with np.load(mock_parent_npz, allow_pickle=False) as parent:
        common_keys = (
            "source_id",
            "z",
            "logl3000",
            "true_m_win",
            "true_log10_d",
            "true_log10_d_mean",
            "true_lambda_rf_eff",
            "true_log10_lambda_ratio",
        )
        _require_npz_keys(parent, keys=common_keys, path=mock_parent_npz, schema=mock_parent_schema)

        source_id = np.asarray(parent["source_id"], dtype=np.int64)
        z = np.asarray(parent["z"], dtype=np.float64)
        logl = np.asarray(parent["logl3000"], dtype=np.float64)
        true_mwin = np.asarray(parent["true_m_win"], dtype=np.float64)
        true_log10d = np.asarray(parent["true_log10_d"], dtype=np.float64)
        true_log10d_mean = np.asarray(parent["true_log10_d_mean"], dtype=np.float64)
        true_lambda_rf_eff = np.asarray(parent["true_lambda_rf_eff"], dtype=np.float64)
        true_log10_lambda_ratio = np.asarray(parent["true_log10_lambda_ratio"], dtype=np.float64)

        if mock_parent_schema == PARENT_SCHEMA_STAGE2_MWIN:
            _require_npz_keys(
                parent,
                keys=("median_m_win", "median_log10_d"),
                path=mock_parent_npz,
                schema=mock_parent_schema,
            )
            parent_match_mwin = np.asarray(parent["median_m_win"], dtype=np.float64)
            parent_match_log10d = np.asarray(parent["median_log10_d"], dtype=np.float64)
        elif mock_parent_schema == PARENT_SCHEMA_FULLCLOSURE:
            parent_schema = _read_single_string(
                parent,
                key="parent_schema",
                path=mock_parent_npz,
                schema=mock_parent_schema,
            )
            if parent_schema != PARENT_SCHEMA_FULLCLOSURE:
                raise ValueError(
                    f"{mock_parent_npz} declares parent_schema={parent_schema!r}, "
                    f"but --mock-parent-schema={mock_parent_schema!r} was requested."
                )
            parent_match_mwin = true_mwin
            parent_match_log10d = true_log10d
        else:
            raise ValueError(
                f"Unknown --mock-parent-schema={mock_parent_schema!r}; expected "
                f"{PARENT_SCHEMA_STAGE2_MWIN!r} or {PARENT_SCHEMA_FULLCLOSURE!r}."
            )

    expected_n = int(source_id.size)
    for name, arr in (
        ("z", z),
        ("logl3000", logl),
        ("true_m_win", true_mwin),
        ("true_log10_d", true_log10d),
        ("true_log10_d_mean", true_log10d_mean),
        ("true_lambda_rf_eff", true_lambda_rf_eff),
        ("true_log10_lambda_ratio", true_log10_lambda_ratio),
        ("parent_match_m_win", parent_match_mwin),
        ("parent_match_log10_d", parent_match_log10d),
    ):
        if int(np.asarray(arr).shape[0]) != expected_n:
            raise ValueError(
                f"{mock_parent_npz} field {name!r} has length {np.asarray(arr).shape[0]}, "
                f"but source_id has length {expected_n}."
            )

    mask = (
        np.isfinite(z)
        & np.isfinite(logl)
        & np.isfinite(parent_match_mwin)
        & np.isfinite(parent_match_log10d)
        & np.isfinite(true_mwin)
        & np.isfinite(true_log10d)
    )

    for cfg_key, col, op in (
        ("z_min", z, ">="),
        ("z_max", z, "<="),
        ("logl_min", logl, ">="),
        ("logl_max", logl, "<="),
    ):
        val = cuts_submit.get(cfg_key)
        if val is None:
            continue
        mask &= col >= float(val) if op == ">=" else col <= float(val)

    for cfg_key, col, op in (
        ("d50_min", parent_match_log10d, ">="),
        ("d50_max", parent_match_log10d, "<="),
    ):
        val = cuts_stage2.get(cfg_key)
        if val is None:
            continue
        mask &= col >= float(val) if op == ">=" else col <= float(val)

    mag_max = cuts_submit.get("mag_max")
    if apply_parent_mag_cut and mag_max is not None:
        mask &= parent_match_mwin <= float(mag_max)

    return {
        "source_id": source_id[mask],
        "z": z[mask],
        "logl3000": logl[mask],
        "parent_match_m_win": parent_match_mwin[mask],
        "parent_match_log10_d": parent_match_log10d[mask],
        "true_m_win": true_mwin[mask],
        "true_log10_d": true_log10d[mask],
        "true_log10_d_mean": true_log10d_mean[mask],
        "true_lambda_rf_eff": true_lambda_rf_eff[mask],
        "true_log10_lambda_ratio": true_log10_lambda_ratio[mask],
        "candidate_mask_n": np.asarray([int(np.sum(mask))], dtype=np.int64),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a Stage-2 closure sample with simulated Gaia G light curves.")
    parser.add_argument("--mock-dir", type=Path, default=DEFAULT_MOCK_DIR)
    parser.add_argument("--mock-parent-npz", type=Path, default=None)
    parser.add_argument(
        "--mock-parent-schema",
        choices=(PARENT_SCHEMA_STAGE2_MWIN, PARENT_SCHEMA_FULLCLOSURE),
        default=PARENT_SCHEMA_STAGE2_MWIN,
        help=(
            "Declare the exact mock parent schema. "
            "stage2_mwin_parent requires median_m_win/median_log10_d and uses them for matching. "
            "fullclosure_truth_parent requires parent_schema=fullclosure_truth_parent and uses "
            "true_m_win/true_log10_d for matching. No automatic fallback is attempted."
        ),
    )
    parser.add_argument("--quality-cuts-run-dir", type=Path, default=DEFAULT_QUALITY_RUN_DIR)

    # ── Real-template source: two mutually exclusive paths ───────────────────
    # Pre-processed MCMC cache input: --real-chains-npz
    #   Pre-processed MCMC cache NPZ (mcmc_filtered_realdata_loo_gaia_g.npz).
    #   Used by _load_real_targets + _load_real_chains_cache.
    #   Source pool is filtered by selected_source_ids.npz in --quality-cuts-run-dir.
    parser.add_argument(
        "--real-chains-npz",
        type=Path,
        default=DEFAULT_REAL_CHAINS_NPZ,
        help=(
            "Pre-processed MCMC cache NPZ. "
            "Ignored when --real-stage1-posteriors-npz is set."
        ),
    )
    # Stage-1 posterior input: --real-stage1-posteriors-npz
    #   Raw Stage-1 posteriors NPZ (e.g. posterior_samples.npz).
    #   Used by _load_real_targets_from_stage1_posteriors.
    #   Pool quality cuts applied via --pool-* arguments below. Optionally, an
    #   exact source manifest can be supplied via --real-used-source-ids-npz.
    #   When this is set, --real-chains-npz is ignored.
    parser.add_argument(
        "--real-stage1-posteriors-npz",
        type=Path,
        default=None,
        help=(
            "Stage-1 posteriors NPZ with fields: "
            "source_id, redsh, logL3000, posterior_m_win_cont_Gaia_G, "
            "posterior_ln_D_Gaia_G, posterior_ln_tau_Gaia_G, diag_npts_Gaia_G. "
            "When set, --real-chains-npz is ignored and pool quality cuts are "
            "controlled by --pool-* arguments. Add --real-used-source-ids-npz "
            "to reproduce an exact source list from a previous Stage-2 run."
        ),
    )
    parser.add_argument(
        "--real-used-source-ids-npz",
        type=Path,
        default=None,
        help=(
            "Optional manifest with source_id_selected. Only used with "
            "--real-stage1-posteriors-npz. When provided, the Stage-1 table is "
            "filtered and ordered to this exact source list before any --pool-* cuts."
        ),
    )
    # Pool quality cuts — only active when --real-stage1-posteriors-npz is used.
    # These filter which real sources enter the template pool before matching.
    parser.add_argument("--pool-z-min", type=float, default=None, help="[pool cut] Minimum redshift.")
    parser.add_argument("--pool-z-max", type=float, default=None, help="[pool cut] Maximum redshift.")
    parser.add_argument("--pool-mwin-cont-max", type=float, default=None, help="[pool cut] Maximum median m_win_cont.")
    parser.add_argument("--pool-tau-med-min-days", type=float, default=None, help="[pool cut] Minimum median tau (rest-frame days, from posterior_ln_tau_Gaia_G).")
    parser.add_argument("--pool-min-epochs", type=int, default=None, help="[pool cut] Minimum epoch count (diag_npts_Gaia_G).")
    parser.add_argument("--pool-logl-min", type=float, default=None, help="[pool cut] Minimum log10 L3000.")
    parser.add_argument("--pool-logl-max", type=float, default=None, help="[pool cut] Maximum log10 L3000.")
    # ────────────────────────────────────────────────────────────────────────

    # Cadence FITS: use DEFAULT_CADENCE_FITS_VARREJECT with --real-stage1-posteriors-npz.
    parser.add_argument("--cadence-fits", type=Path, default=DEFAULT_CADENCE_FITS)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--output-fits", type=Path, default=None)
    parser.add_argument(
        "--pair-tau-only",
        action="store_true",
        help=(
            "Stop after exact real-template/mock-parent matching and tau/sigma assignment. "
            "Writes pair/truth, manifest, summary, and diagnostic PDF, but does not load cadences "
            "or simulate light curves."
        ),
    )
    parser.add_argument("--sample-label", default='cadence_matched_sample')
    parser.add_argument("--n-pilot", type=int, default=5000, help="Requested final sample size unless --all-targets is set.")
    parser.add_argument(
        "--all-targets",
        action="store_true",
        help="Use every real source selected by --quality-cuts-run-dir as a cadence/noise target.",
    )
    parser.add_argument(
        "--candidate-oversample-factor",
        type=float,
        default=1.2,
        help="Generate this many matched light-curve candidates per requested final object before mock raw cuts.",
    )
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--selection-seed", type=int, default=20260514)
    parser.add_argument("--tau-seed", type=int, default=20260515)
    parser.add_argument("--lightcurve-seed", type=int, default=20260516)
    parser.add_argument("--mwin-cont-seed", type=int, default=20260517)
    parser.add_argument("--tau-log10-mu", type=float, default=float(np.log10(500.0)))
    parser.add_argument("--tau-log10-sigma", type=float, default=0.35)
    parser.add_argument("--tau-min-days", type=float, default=30.0)
    parser.add_argument("--tau-max-days", type=float, default=3000.0)
    parser.add_argument("--match-k-initial", type=int, default=64)
    parser.add_argument("--match-k-step", type=int, default=64)
    parser.add_argument("--match-k-max", type=int, default=2048)
    parser.add_argument(
        "--require-all-matched",
        action="store_true",
        help="Raise an error instead of dropping targets when the unique mock matcher leaves any target unmatched.",
    )
    parser.add_argument("--scenario-name", default="fullboxonly")
    parser.add_argument("--output-pdf", type=Path, default=None)
    parser.add_argument(
        "--anchor-mode",
        choices=("stationary_m0", "conditional_mwin", "conditional_mwin_stochastic_m0"),
        default="conditional_mwin_stochastic_m0",
        help=(
            "stationary_m0 draws an ordinary DRW around m0; conditional_mwin is the 2a sanity check "
            "with m0=mwin; conditional_mwin_stochastic_m0 is Closure 2b, drawing m0-mwin from the "
            "DRW finite-window variance and conditioning the latent DRW on the injected continuous m_win."
        ),
    )
    parser.add_argument(
        "--mock-raw-mag-max",
        type=float,
        default=None,
        help="Mock raw median Gaia-G magnitude cut. Defaults to cuts_submit.mag_max from --quality-cuts-run-dir.",
    )
    parser.add_argument(
        "--no-mock-raw-mag-cut",
        action="store_true",
        help="Disable the post-simulation raw median Gaia-G magnitude cut.",
    )
    parser.add_argument(
        "--keep-all-passing-raw-cuts",
        action="store_true",
        help="After simulating candidates, keep all objects passing mock raw cuts instead of downselecting to --n-pilot.",
    )
    parser.add_argument(
        "--apply-parent-median-mag-cut",
        action="store_true",
        help=(
            "Also apply the parent matching-magnitude <= mag_max cut before light-curve simulation. "
            "For stage2_mwin_parent this column is median_m_win; for fullclosure_truth_parent it is true_m_win."
        ),
    )
    parser.add_argument(
        "--apply-mock-raw-snr-epochs-cuts",
        action="store_true",
        help="Also apply the real qcuts S/N and n_epoch cuts to the simulated Gaia-G light curves.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sample_label = str(args.sample_label)

    mock_dir = args.mock_dir.expanduser().resolve()
    mock_parent_npz = (
        args.mock_parent_npz.expanduser().resolve()
        if args.mock_parent_npz is not None
        else mock_dir / "stage2_mock_parent.npz"
    )
    quality_cuts_run_dir = args.quality_cuts_run_dir.expanduser().resolve()
    real_chains_npz = args.real_chains_npz.expanduser().resolve()
    cadence_fits = args.cadence_fits.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else mock_dir / "closure_tests" / sample_label
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fits = (
        args.output_fits.expanduser().resolve()
        if args.output_fits is not None
        else out_dir / f"{sample_label}_mock_lightcurves.fits"
    )
    with (quality_cuts_run_dir / 'inference_settings.json').open("r", encoding="utf-8") as f:
        quality_cfg = json.load(f)
    cuts_submit = quality_cfg.get("cuts_submit", {})
    mock_raw_mag_max = None
    if not bool(args.no_mock_raw_mag_cut):
        mock_raw_mag_max = (
            float(args.mock_raw_mag_max)
            if args.mock_raw_mag_max is not None
            else (float(cuts_submit["mag_max"]) if cuts_submit.get("mag_max") is not None else None)
        )
    mock_raw_snr_min = None
    mock_raw_min_epochs = None
    if bool(args.apply_mock_raw_snr_epochs_cuts):
        if cuts_submit.get("snr_min") is not None:
            mock_raw_snr_min = float(cuts_submit["snr_min"])
        if cuts_submit.get("min_epochs_gaia") is not None:
            mock_raw_min_epochs = int(cuts_submit["min_epochs_gaia"])
    apply_mock_raw_cuts = (
        mock_raw_mag_max is not None
        or mock_raw_snr_min is not None
        or mock_raw_min_epochs is not None
    )

    print("Loading real target medians ...", flush=True)
    # ── Select Stage-1 posterior arrays or a pre-processed MCMC cache ─────────
    # Stage-1 posterior arrays are selected by --real-stage1-posteriors-npz.
    #   Reads raw Stage-1 posteriors directly; pool cuts applied via --pool-* args.
    #   If --real-used-source-ids-npz is set, that exact manifest is applied first.
    #   --real-chains-npz and selected_source_ids.npz from quality_cuts_run_dir are ignored.
    # Otherwise, use the pre-processed MCMC cache.
    #   Reads the pre-processed MCMC cache NPZ and filters by selected_source_ids.npz.
    if args.real_stage1_posteriors_npz is not None:
        targets = _load_real_targets_from_stage1_posteriors(
            stage1_posteriors_npz=args.real_stage1_posteriors_npz.expanduser().resolve(),
            real_used_source_ids_npz=(
                None
                if args.real_used_source_ids_npz is None
                else args.real_used_source_ids_npz.expanduser().resolve()
            ),
            z_min=args.pool_z_min,
            z_max=args.pool_z_max,
            mwin_cont_max=args.pool_mwin_cont_max,
            tau_med_min_days=args.pool_tau_med_min_days,
            pool_min_epochs=args.pool_min_epochs,
            logl_min=args.pool_logl_min,
            logl_max=args.pool_logl_max,
        )
    else:
        targets = _load_real_targets(real_chains_npz=real_chains_npz, quality_cuts_run_dir=quality_cuts_run_dir)
    # ─────────────────────────────────────────────────────────────────────────
    n_real_targets_available = int(np.asarray(targets["source_id"]).size)
    oversample_factor = max(1.0, float(args.candidate_oversample_factor))
    if bool(args.all_targets):
        requested_n_pilot = n_real_targets_available
        n_candidate = n_real_targets_available
        target_choice = np.arange(n_real_targets_available, dtype=np.int64)
    else:
        requested_n_pilot = int(args.n_pilot)
        n_candidate = (
            int(np.ceil(requested_n_pilot * oversample_factor))
            if apply_mock_raw_cuts and not bool(args.keep_all_passing_raw_cuts)
            else requested_n_pilot
        )
        target_choice = _choose_pilot_targets(targets, n_pilot=n_candidate, seed=int(args.selection_seed))
    print(
        "Closure sample request: "
        f"label={sample_label}, target_mode={'all' if args.all_targets else 'random_subset'}, "
        f"requested_final={requested_n_pilot:,}, candidates={n_candidate:,}, "
        f"mock_raw_mag_max={mock_raw_mag_max}, "
        f"mock_raw_snr_min={mock_raw_snr_min}, mock_raw_min_epochs={mock_raw_min_epochs}, "
        f"keep_all_passing_raw_cuts={bool(args.keep_all_passing_raw_cuts)}",
        flush=True,
    )
    pilot_targets = {key: np.asarray(val)[target_choice] if np.asarray(val).ndim > 0 and np.asarray(val).shape[0] == targets["source_id"].size else np.asarray(val) for key, val in targets.items()}

    print("Loading mock candidate truth arrays ...", flush=True)
    candidates = _load_mock_candidates(
        mock_parent_npz=mock_parent_npz,
        quality_cuts_run_dir=quality_cuts_run_dir,
        apply_parent_mag_cut=bool(args.apply_parent_median_mag_cut),
        mock_parent_schema=str(args.mock_parent_schema),
    )

    print("Matching closure real targets to mock candidates ...", flush=True)
    target_feat, cand_feat = _weighted_standardize(
        np.asarray(pilot_targets["z"], dtype=np.float64),
        np.asarray(pilot_targets["logl3000_temple"], dtype=np.float64),
        np.asarray(pilot_targets["m_win"], dtype=np.float64),
        np.asarray(candidates["z"], dtype=np.float64),
        np.asarray(candidates["logl3000"], dtype=np.float64),
        np.asarray(candidates["parent_match_m_win"], dtype=np.float64),
        weights=(1.0, 1.0, 2.5),
    )
    assigned_idx, match_diag = _fast_greedy_unique_match(
        target_feat,
        cand_feat,
        k_initial=int(args.match_k_initial),
        k_step=int(args.match_k_step),
        k_max=int(args.match_k_max),
    )
    good = assigned_idx >= 0
    if not np.all(good):
        msg = f"{int(np.sum(~good))} targets unmatched after unique mock matching"
        if bool(args.require_all_matched):
            raise RuntimeError(
                f"{msg}; refusing to drop real templates because --require-all-matched was set. "
                f"Try increasing --match-k-max or relaxing the target sample."
            )
        print(f"Warning: {msg}; keeping matched targets only.", flush=True)
        pilot_targets = {key: np.asarray(val)[good] if np.asarray(val).ndim > 0 and np.asarray(val).shape[0] == good.size else np.asarray(val) for key, val in pilot_targets.items()}
        assigned_idx = assigned_idx[good]

    pilot_mock = {key: np.asarray(val)[assigned_idx] if np.asarray(val).ndim > 0 and np.asarray(val).shape[0] == candidates["source_id"].size else np.asarray(val) for key, val in candidates.items()}
    n_candidates_matched = int(np.asarray(pilot_mock["source_id"]).size)

    rng_tau = np.random.default_rng(int(args.tau_seed))
    log10_tau = _draw_truncated_log10_tau(
        rng=rng_tau,
        n=n_candidates_matched,
        mu=float(args.tau_log10_mu),
        sigma=float(args.tau_log10_sigma),
        lo=float(np.log10(args.tau_min_days)),
        hi=float(np.log10(args.tau_max_days)),
    )
    tau_days = np.power(10.0, log10_tau)
    log10_d_true = np.asarray(pilot_mock["true_log10_d"], dtype=np.float64)
    d_true = np.power(10.0, log10_d_true)
    sigma_drw_mag = np.sqrt(d_true * tau_days)
    sf_inf_mag = np.sqrt(2.0) * sigma_drw_mag
    log10_sigma = np.log10(sigma_drw_mag)
    log10_sf_inf = np.log10(sf_inf_mag)

    real_logl_from_mwin = real_logl3000_from_mwin(
        z=np.asarray(pilot_targets["z"], dtype=np.float64),
        mwin_med=np.asarray(pilot_targets["m_win"], dtype=np.float64),
        band="gaia_g",
    )
    mock_match_mwin_column = "median_m_win" if str(args.mock_parent_schema) == PARENT_SCHEMA_STAGE2_MWIN else "true_m_win"
    mock_match_log10d_column = (
        "median_log10_d" if str(args.mock_parent_schema) == PARENT_SCHEMA_STAGE2_MWIN else "true_log10_d"
    )

    if bool(args.pair_tau_only):
        n_pilot = n_candidates_matched
        pair_npz = out_dir / f"{sample_label}_pairs_and_truth.npz"
        pair_payload = {
            "real_source_id": np.asarray(pilot_targets["source_id"], dtype=np.int64),
            "real_z": np.asarray(pilot_targets["z"], dtype=np.float64),
            "real_logl3000_temple": np.asarray(pilot_targets["logl3000_temple"], dtype=np.float64),
            "real_logl3000_from_mwin": np.asarray(real_logl_from_mwin, dtype=np.float64),
            "real_m_win_med": np.asarray(pilot_targets["m_win"], dtype=np.float64),
            "real_log10_d_med": np.asarray(pilot_targets["log10d"], dtype=np.float64),
            "mock_parent_schema": np.asarray([str(args.mock_parent_schema)], dtype="U64"),
            "mock_source_id": np.asarray(pilot_mock["source_id"], dtype=np.int64),
            "mock_z": np.asarray(pilot_mock["z"], dtype=np.float64),
            "mock_logl3000": np.asarray(pilot_mock["logl3000"], dtype=np.float64),
            "mock_m_win_parent_true": np.asarray(pilot_mock["true_m_win"], dtype=np.float64),
            "mock_m_win_target_true": np.asarray(pilot_mock["true_m_win"], dtype=np.float64),
            "mock_log10_d_true": np.asarray(pilot_mock["true_log10_d"], dtype=np.float64),
            "mock_log10_d_mean": np.asarray(pilot_mock["true_log10_d_mean"], dtype=np.float64),
            "mock_parent_match_m_win": np.asarray(pilot_mock["parent_match_m_win"], dtype=np.float64),
            "mock_parent_match_log10_d": np.asarray(pilot_mock["parent_match_log10_d"], dtype=np.float64),
            "mock_lambda_rf_eff": np.asarray(pilot_mock["true_lambda_rf_eff"], dtype=np.float64),
            "mock_log10_lambda_ratio": np.asarray(pilot_mock["true_log10_lambda_ratio"], dtype=np.float64),
            "tau_rest_days": tau_days,
            "log10_tau_rest_days": log10_tau,
            "sigma_drw_mag": sigma_drw_mag,
            "sf_inf_mag": sf_inf_mag,
            "log10_sigma_drw_mag": log10_sigma,
            "log10_sf_inf_mag": log10_sf_inf,
            "selection_seed": np.asarray([int(args.selection_seed)], dtype=np.int64),
            "tau_seed": np.asarray([int(args.tau_seed)], dtype=np.int64),
        }
        if str(args.mock_parent_schema) == PARENT_SCHEMA_STAGE2_MWIN:
            pair_payload["mock_m_win_med"] = np.asarray(pilot_mock["parent_match_m_win"], dtype=np.float64)
            pair_payload["mock_log10_d_med"] = np.asarray(pilot_mock["parent_match_log10_d"], dtype=np.float64)
        np.savez_compressed(pair_npz, **pair_payload)

        manifest_npz = out_dir / f"selected_source_ids_{sample_label}.npz"
        np.savez_compressed(
            manifest_npz,
            source_id_selected=np.asarray(pilot_mock["source_id"], dtype=np.int64),
            real_source_id_template=np.asarray(pilot_targets["source_id"], dtype=np.int64),
            n_selected=np.asarray(n_pilot, dtype=np.int64),
            source_id_sha256=np.asarray(_sha256_int_array(np.asarray(pilot_mock["source_id"], dtype=np.int64))),
            real_source_id_sha256=np.asarray(_sha256_int_array(np.asarray(pilot_targets["source_id"], dtype=np.int64))),
        )

        real_temple_row = {
            "z": np.asarray(pilot_targets["z"], dtype=np.float64),
            "logl3000": np.asarray(pilot_targets["logl3000_temple"], dtype=np.float64),
            "log10d": np.asarray(pilot_targets["log10d"], dtype=np.float64),
            "m_win": np.asarray(pilot_targets["m_win"], dtype=np.float64),
            "label": "Real templates",
            "value_kind": "med",
            "logl_axis_label": r"$\log_{10}(L_{3000}\ [\mathrm{Temple{+}2021}])$",
        }
        real_mwin_row = dict(real_temple_row)
        real_mwin_row["label"] = r"Real templates from $m_{\rm win}$"
        real_mwin_row["logl3000"] = np.asarray(real_logl_from_mwin, dtype=np.float64)
        real_mwin_row["logl_axis_label"] = r"$\log_{10}(L_{3000}\ [\mathrm{from}\ m_{\rm win}])$"
        mock_parent_row = {
            "z": np.asarray(pilot_mock["z"], dtype=np.float64),
            "logl3000": np.asarray(pilot_mock["logl3000"], dtype=np.float64),
            "log10d": log10_d_true,
            "m_win": np.asarray(pilot_mock["true_m_win"], dtype=np.float64),
            "label": r"Matched mock parent truth",
            "value_kind": "true",
            "mwin_axis_label": r"$m_{\rm win}^{\rm true}$",
        }
        out_pdf = (
            args.output_pdf.expanduser().resolve()
            if args.output_pdf is not None
            else out_dir / f"{sample_label}_selected_3x4_temple_mwin_mock_truth.pdf"
        )
        print("Writing pair/tau diagnostic PDF ...", flush=True)
        _write_selected_page(
            out_pdf=out_pdf,
            rows_for_limits=[real_temple_row, real_mwin_row, mock_parent_row],
            rows_to_plot=[real_temple_row, real_mwin_row, mock_parent_row],
            scenario_name=str(args.scenario_name),
            title_suffix=f" | pair/tau only | sample={n_pilot:,} | real={quality_cuts_run_dir.name}",
            show_hist_row=True,
        )

        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "script": "prepare_closure_simulations.py",
            "stage_reached": "pair_tau_only",
            "sample_label": sample_label,
            "out_dir": str(out_dir),
            "pair_npz": str(pair_npz),
            "manifest_npz": str(manifest_npz),
            "lightcurve_fits": None,
            "planned_lightcurve_fits": str(out_fits),
            "diagnostics_pdf": str(out_pdf),
            "mock_parent_npz": str(mock_parent_npz),
            "mock_parent_schema": str(args.mock_parent_schema),
            "real_chains_npz": str(real_chains_npz),
            "real_stage1_posteriors_npz": (
                None
                if args.real_stage1_posteriors_npz is None
                else str(args.real_stage1_posteriors_npz.expanduser().resolve())
            ),
            "real_used_source_ids_npz": (
                None
                if args.real_used_source_ids_npz is None
                else str(args.real_used_source_ids_npz.expanduser().resolve())
            ),
            "cadence_fits": None,
            "quality_cuts_run_dir": str(quality_cuts_run_dir),
            "n_pilot_requested": requested_n_pilot,
            "n_pilot": n_pilot,
            "n_candidate_requested": n_candidate,
            "n_candidates_matched_before_mock_raw_cuts": n_candidates_matched,
            "n_real_targets_available": n_real_targets_available,
            "n_mock_candidates_after_cuts": int(candidates["candidate_mask_n"][0]),
            "target_mode": "all" if bool(args.all_targets) else "random_subset",
            "selection_seed": int(args.selection_seed),
            "tau_seed": int(args.tau_seed),
            "real_target_selection": {
                "mode": "stage1_posteriors" if args.real_stage1_posteriors_npz is not None else "legacy_cache",
                "exact_manifest_applied": args.real_used_source_ids_npz is not None,
                "pool_cuts": {
                    "z_min": None if args.pool_z_min is None else float(args.pool_z_min),
                    "z_max": None if args.pool_z_max is None else float(args.pool_z_max),
                    "mwin_cont_max": None if args.pool_mwin_cont_max is None else float(args.pool_mwin_cont_max),
                    "tau_med_min_days": None if args.pool_tau_med_min_days is None else float(args.pool_tau_med_min_days),
                    "min_epochs": None if args.pool_min_epochs is None else int(args.pool_min_epochs),
                    "logl_min": None if args.pool_logl_min is None else float(args.pool_logl_min),
                    "logl_max": None if args.pool_logl_max is None else float(args.pool_logl_max),
                },
            },
            "lightcurve_simulation": {
                "stage_reached": "not run; --pair-tau-only stopped before cadence loading and light-curve simulation",
                "lightcurves_written": False,
            },
            "tau_model": {
                "distribution": "truncated_normal_log10_tau_rest_days",
                "log10_mu": float(args.tau_log10_mu),
                "log10_sigma": float(args.tau_log10_sigma),
                "tau_min_days": float(args.tau_min_days),
                "tau_max_days": float(args.tau_max_days),
                "seed": int(args.tau_seed),
            },
            "matching": {
                "mode": "soft_nn_unique",
                "weights": {"z": 1.0, "logl3000": 1.0, "m_win": 2.5},
                "mock_m_win_column": mock_match_mwin_column,
                "mock_log10_d_cut_column": mock_match_log10d_column,
                **{k: _json_number(v) for k, v in match_diag.items()},
            },
            "hashes": {
                "mock_source_id_sha256": _sha256_int_array(np.asarray(pilot_mock["source_id"], dtype=np.int64)),
                "real_source_id_sha256": _sha256_int_array(np.asarray(pilot_targets["source_id"], dtype=np.int64)),
            },
            "percentiles": {
                "mock_log10_d_true": _finite_percentiles(log10_d_true),
                "mock_log10_d_mean": _finite_percentiles(np.asarray(pilot_mock["true_log10_d_mean"], dtype=np.float64)),
                "log10_tau_rest_days": _finite_percentiles(log10_tau),
                "tau_rest_days": _finite_percentiles(tau_days),
                "sigma_drw_mag": _finite_percentiles(sigma_drw_mag),
                "sf_inf_mag": _finite_percentiles(sf_inf_mag),
                "log10_sigma_drw_mag": _finite_percentiles(log10_sigma),
                "log10_sf_inf_mag": _finite_percentiles(log10_sf_inf),
                "mock_z": _finite_percentiles(np.asarray(pilot_mock["z"], dtype=np.float64)),
                "mock_logl3000": _finite_percentiles(np.asarray(pilot_mock["logl3000"], dtype=np.float64)),
                "mock_m_win_parent_true": _finite_percentiles(np.asarray(pilot_mock["true_m_win"], dtype=np.float64)),
                "real_z": _finite_percentiles(np.asarray(pilot_targets["z"], dtype=np.float64)),
                "real_logl3000_temple": _finite_percentiles(np.asarray(pilot_targets["logl3000_temple"], dtype=np.float64)),
                "real_m_win_med": _finite_percentiles(np.asarray(pilot_targets["m_win"], dtype=np.float64)),
            },
            "fractions": {
                "sigma_gt_0p2": float(np.mean(sigma_drw_mag > 0.2)),
                "sigma_gt_0p5": float(np.mean(sigma_drw_mag > 0.5)),
                "sigma_gt_1p0": float(np.mean(sigma_drw_mag > 1.0)),
                "sf_inf_gt_0p3": float(np.mean(sf_inf_mag > 0.3)),
                "sf_inf_gt_1p0": float(np.mean(sf_inf_mag > 1.0)),
            },
        }
        summary_path = out_dir / f"{sample_label}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(f"Wrote pair/tau truth: {pair_npz}")
        print(f"Wrote manifest: {manifest_npz}")
        print(f"Wrote summary: {summary_path}")
        print(f"Wrote PDF: {out_pdf}")
        print(json.dumps(summary["percentiles"]["mock_log10_d_true"], indent=2))
        return

    templates = _load_gaia_templates_by_source_id(
        cadence_fits=cadence_fits,
        source_ids=np.asarray(pilot_targets["source_id"], dtype=np.int64),
        mock_z=np.asarray(pilot_mock["z"], dtype=np.float64),
        min_epochs=int(args.min_epochs),
    )
    lc_truth = _simulate_closure_light_curves(
        out_fits=out_fits,
        templates=templates,
        real_source_id=np.asarray(pilot_targets["source_id"], dtype=np.int64),
        mock_source_id=np.asarray(pilot_mock["source_id"], dtype=np.int64),
        mock_z=np.asarray(pilot_mock["z"], dtype=np.float64),
        mock_logl3000=np.asarray(pilot_mock["logl3000"], dtype=np.float64),
        mock_mwin_target_true=np.asarray(pilot_mock["true_m_win"], dtype=np.float64),
        mock_log10_d_true=log10_d_true,
        tau_days=tau_days,
        sigma_drw_mag=sigma_drw_mag,
        rng_lc=np.random.default_rng(int(args.lightcurve_seed)),
        rng_mwin_cont=np.random.default_rng(int(args.mwin_cont_seed)),
        raw_mag_max=mock_raw_mag_max,
        raw_snr_min=mock_raw_snr_min,
        raw_min_epochs=mock_raw_min_epochs,
        n_keep=(
            requested_n_pilot
            if apply_mock_raw_cuts and not bool(args.keep_all_passing_raw_cuts)
            else None
        ),
        anchor_mode=str(args.anchor_mode),
    )
    selected_candidate_idx = np.asarray(lc_truth["selected_candidate_indices"], dtype=np.int64)
    n_pilot = int(selected_candidate_idx.size)
    pilot_targets = {
        key: np.asarray(val)[selected_candidate_idx]
        if np.asarray(val).ndim > 0 and np.asarray(val).shape[0] == n_candidates_matched
        else np.asarray(val)
        for key, val in pilot_targets.items()
    }
    pilot_mock = {
        key: np.asarray(val)[selected_candidate_idx]
        if np.asarray(val).ndim > 0 and np.asarray(val).shape[0] == n_candidates_matched
        else np.asarray(val)
        for key, val in pilot_mock.items()
    }
    real_logl_from_mwin = np.asarray(real_logl_from_mwin, dtype=np.float64)[selected_candidate_idx]
    log10_tau = log10_tau[selected_candidate_idx]
    tau_days = tau_days[selected_candidate_idx]
    log10_d_true = log10_d_true[selected_candidate_idx]
    sigma_drw_mag = sigma_drw_mag[selected_candidate_idx]
    sf_inf_mag = sf_inf_mag[selected_candidate_idx]
    log10_sigma = log10_sigma[selected_candidate_idx]
    log10_sf_inf = log10_sf_inf[selected_candidate_idx]

    pair_npz = out_dir / f"{sample_label}_pairs_and_truth.npz"
    pair_payload = {
        "real_source_id": np.asarray(pilot_targets["source_id"], dtype=np.int64),
        "real_z": np.asarray(pilot_targets["z"], dtype=np.float64),
        "real_logl3000_temple": np.asarray(pilot_targets["logl3000_temple"], dtype=np.float64),
        "real_logl3000_from_mwin": np.asarray(real_logl_from_mwin, dtype=np.float64),
        "real_m_win_med": np.asarray(pilot_targets["m_win"], dtype=np.float64),
        "real_log10_d_med": np.asarray(pilot_targets["log10d"], dtype=np.float64),
        "mock_parent_schema": np.asarray([str(args.mock_parent_schema)], dtype="U64"),
        "mock_source_id": np.asarray(pilot_mock["source_id"], dtype=np.int64),
        "mock_z": np.asarray(pilot_mock["z"], dtype=np.float64),
        "mock_logl3000": np.asarray(pilot_mock["logl3000"], dtype=np.float64),
        "mock_m0_true": np.asarray(lc_truth["mock_m0_gen_true"], dtype=np.float64),
        "mock_m0_gen_true": np.asarray(lc_truth["mock_m0_gen_true"], dtype=np.float64),
        "mock_m0_minus_mwin_cont_true": np.asarray(lc_truth["mock_m0_minus_mwin_cont_true"], dtype=np.float64),
        "mock_m_win_parent_true": np.asarray(pilot_mock["true_m_win"], dtype=np.float64),
        "mock_m_win_true": np.asarray(lc_truth["mock_m_win_cont_true"], dtype=np.float64),
        "mock_m_win_target_true": np.asarray(lc_truth["mock_m_win_cont_true"], dtype=np.float64),
        "mock_m_win_trap_true": np.asarray(lc_truth["mock_m_win_trap_true"], dtype=np.float64),
        "mock_m_win_cont_true": np.asarray(lc_truth["mock_m_win_cont_true"], dtype=np.float64),
        "mock_m_win_cont_mu": np.asarray(lc_truth["mock_m_win_cont_mu"], dtype=np.float64),
        "mock_m_win_cont_var": np.asarray(lc_truth["mock_m_win_cont_var"], dtype=np.float64),
        "mock_m_win_cont_uncond_var": np.asarray(lc_truth["mock_m_win_cont_uncond_var"], dtype=np.float64),
        "mock_log10_d_true": np.asarray(pilot_mock["true_log10_d"], dtype=np.float64),
        "mock_log10_d_mean": np.asarray(pilot_mock["true_log10_d_mean"], dtype=np.float64),
        "mock_parent_match_m_win": np.asarray(pilot_mock["parent_match_m_win"], dtype=np.float64),
        "mock_parent_match_log10_d": np.asarray(pilot_mock["parent_match_log10_d"], dtype=np.float64),
        "mock_lambda_rf_eff": np.asarray(pilot_mock["true_lambda_rf_eff"], dtype=np.float64),
        "mock_log10_lambda_ratio": np.asarray(pilot_mock["true_log10_lambda_ratio"], dtype=np.float64),
        "baseline_rest_days": np.asarray(lc_truth["baseline_rest_days"], dtype=np.float64),
        "mean_dt_rest_days": np.asarray(lc_truth["mean_dt_rest_days"], dtype=np.float64),
        "npts": np.asarray(lc_truth["npts"], dtype=np.int32),
        "median_mag_err": np.asarray(lc_truth["median_mag_err"], dtype=np.float64),
        "mock_median_mag_err": np.asarray(lc_truth["mock_median_mag_err"], dtype=np.float64),
        "error_scale_median": np.asarray(lc_truth["error_scale_median"], dtype=np.float64),
        "template_z": np.asarray(lc_truth["template_z"], dtype=np.float64),
        "template_median_mag": np.asarray(lc_truth["template_median_mag"], dtype=np.float64),
        "gaia_g_mag_med": np.asarray(lc_truth["gaia_g_mag_med"], dtype=np.float64),
        "gaia_g_snr": np.asarray(lc_truth["gaia_g_snr"], dtype=np.float64),
        "gaia_g_n_epochs": np.asarray(lc_truth["gaia_g_n_epochs"], dtype=np.int32),
        "selected_candidate_index_before_raw_cut": selected_candidate_idx,
        "tau_rest_days": tau_days,
        "log10_tau_rest_days": log10_tau,
        "sigma_drw_mag": sigma_drw_mag,
        "sf_inf_mag": sf_inf_mag,
        "log10_sigma_drw_mag": log10_sigma,
        "log10_sf_inf_mag": log10_sf_inf,
        "selection_seed": np.asarray([int(args.selection_seed)], dtype=np.int64),
        "tau_seed": np.asarray([int(args.tau_seed)], dtype=np.int64),
        "lightcurve_seed": np.asarray([int(args.lightcurve_seed)], dtype=np.int64),
        "mwin_cont_seed": np.asarray([int(args.mwin_cont_seed)], dtype=np.int64),
    }
    if str(args.mock_parent_schema) == PARENT_SCHEMA_STAGE2_MWIN:
        pair_payload["mock_m_win_med"] = np.asarray(pilot_mock["parent_match_m_win"], dtype=np.float64)
        pair_payload["mock_log10_d_med"] = np.asarray(pilot_mock["parent_match_log10_d"], dtype=np.float64)
    np.savez_compressed(pair_npz, **pair_payload)

    manifest_npz = out_dir / f"selected_source_ids_{sample_label}.npz"
    np.savez_compressed(
        manifest_npz,
        source_id_selected=np.asarray(pilot_mock["source_id"], dtype=np.int64),
        real_source_id_template=np.asarray(pilot_targets["source_id"], dtype=np.int64),
        n_selected=np.asarray(n_pilot, dtype=np.int64),
        source_id_sha256=np.asarray(_sha256_int_array(np.asarray(pilot_mock["source_id"], dtype=np.int64))),
        real_source_id_sha256=np.asarray(_sha256_int_array(np.asarray(pilot_targets["source_id"], dtype=np.int64))),
    )

    diagnostic_base_rows: list[dict[str, Any]] = [
        {
            "z": np.asarray(pilot_targets["z"], dtype=np.float64),
            "logl3000": np.asarray(pilot_targets["logl3000_temple"], dtype=np.float64),
            "log10d": np.asarray(pilot_targets["log10d"], dtype=np.float64),
            "m_win": np.asarray(pilot_targets["m_win"], dtype=np.float64),
            "label": "Real template targets",
            "value_kind": "med",
        },
        {
            "z": np.asarray(pilot_mock["z"], dtype=np.float64),
            "logl3000": np.asarray(pilot_mock["logl3000"], dtype=np.float64),
            "log10d": log10_d_true,
            "m_win": np.asarray(lc_truth["mock_m_win_cont_true"], dtype=np.float64),
            "label": r"Mock LC $m_{\rm win,cont}^{\rm true}$",
            "value_kind": "true",
            "mwin_axis_label": r"$m_{\rm win,cont}^{\rm true}$",
        },
    ]
    real_temple_row = dict(diagnostic_base_rows[0])
    real_temple_row["label"] = "Real templates"
    real_temple_row["logl_axis_label"] = r"$\log_{10}(L_{3000}\ [\mathrm{Temple{+}2021}])$"
    real_mwin_row = dict(diagnostic_base_rows[0])
    real_mwin_row["label"] = r"Real templates from $m_{\rm win}$"
    real_mwin_row["logl3000"] = np.asarray(real_logl_from_mwin, dtype=np.float64)
    real_mwin_row["logl_axis_label"] = r"$\log_{10}(L_{3000}\ [\mathrm{from}\ m_{\rm win}])$"
    mock_mwin_cont_row = dict(diagnostic_base_rows[1])
    mock_mwin_cont_row["label"] = r"Mock LC $m_{\rm win,cont}^{\rm true}$"

    out_pdf = (
        args.output_pdf.expanduser().resolve()
        if args.output_pdf is not None
        else out_dir / f"{sample_label}_selected_3x4_temple_mwin_mock_truth.pdf"
    )
    print("Writing closure diagnostics PDF ...", flush=True)
    _write_selected_page(
        out_pdf=out_pdf,
        rows_for_limits=[real_temple_row, real_mwin_row, mock_mwin_cont_row],
        rows_to_plot=[real_temple_row, real_mwin_row, mock_mwin_cont_row],
        scenario_name=str(args.scenario_name),
        title_suffix=f" | sample={n_pilot:,} | real={quality_cuts_run_dir.name}",
        show_hist_row=True,
    )

    mock_match_mwin_column = "median_m_win" if str(args.mock_parent_schema) == PARENT_SCHEMA_STAGE2_MWIN else "true_m_win"
    mock_match_log10d_column = (
        "median_log10_d" if str(args.mock_parent_schema) == PARENT_SCHEMA_STAGE2_MWIN else "true_log10_d"
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "prepare_closure_simulations.py",
        "sample_label": sample_label,
        "out_dir": str(out_dir),
        "pair_npz": str(pair_npz),
        "manifest_npz": str(manifest_npz),
        "lightcurve_fits": str(out_fits),
        "diagnostics_pdf": str(out_pdf),
        "mock_parent_npz": str(mock_parent_npz),
        "mock_parent_schema": str(args.mock_parent_schema),
        "real_chains_npz": str(real_chains_npz),
        "real_stage1_posteriors_npz": (
            None
            if args.real_stage1_posteriors_npz is None
            else str(args.real_stage1_posteriors_npz.expanduser().resolve())
        ),
        "real_used_source_ids_npz": (
            None
            if args.real_used_source_ids_npz is None
            else str(args.real_used_source_ids_npz.expanduser().resolve())
        ),
        "cadence_fits": str(cadence_fits),
        "quality_cuts_run_dir": str(quality_cuts_run_dir),
        "n_pilot_requested": requested_n_pilot,
        "n_pilot": n_pilot,
        "n_candidate_requested": n_candidate,
        "n_candidates_matched_before_mock_raw_cuts": n_candidates_matched,
        "n_real_targets_available": n_real_targets_available,
        "n_mock_candidates_after_cuts": int(candidates["candidate_mask_n"][0]),
        "target_mode": "all" if bool(args.all_targets) else "random_subset",
        "selection_seed": int(args.selection_seed),
        "lightcurve_seed": int(args.lightcurve_seed),
        "mwin_cont_seed": int(args.mwin_cont_seed),
        "anchor_mode": str(args.anchor_mode),
        "real_target_selection": {
            "mode": "stage1_posteriors" if args.real_stage1_posteriors_npz is not None else "legacy_cache",
            "exact_manifest_applied": args.real_used_source_ids_npz is not None,
            "pool_cuts": {
                "z_min": None if args.pool_z_min is None else float(args.pool_z_min),
                "z_max": None if args.pool_z_max is None else float(args.pool_z_max),
                "mwin_cont_max": None if args.pool_mwin_cont_max is None else float(args.pool_mwin_cont_max),
                "tau_med_min_days": None if args.pool_tau_med_min_days is None else float(args.pool_tau_med_min_days),
                "min_epochs": None if args.pool_min_epochs is None else int(args.pool_min_epochs),
                "logl_min": None if args.pool_logl_min is None else float(args.pool_logl_min),
                "logl_max": None if args.pool_logl_max is None else float(args.pool_logl_max),
            },
        },
        "lightcurve_simulation": {
            "stage_reached": "through step 5: paired templates, tau/sigma decomposition, and simulated Gaia light curves",
            "cadence_rule": "one-to-one: each mock object uses the observed Gaia epochs and epoch errors of its matched real target",
            "rest_frame_rule": "observed Gaia times are converted to rest-frame intervals using the mock redshift stored in redsh",
            "anchor_mode": str(args.anchor_mode),
            "error_model": "borrowed_template_epoch_errors",
            "m0_truth": "mock_m0_true is the nuisance stationary mean used by the light-curve generator; in Closure 2b it is drawn around the injected Stage-2 m_win coordinate",
            "mwin_trap_truth": "true_m_win in the FITS, trapezoid-weighted latent noiseless finite-window mean",
            "mwin_cont_truth": "true_m_win_cont in the FITS; for conditional_mwin and conditional_mwin_stochastic_m0 this is fixed to the injected Stage-2 m_win coordinate",
            "min_epochs": int(args.min_epochs),
        },
        "mock_raw_selection": {
            "rule": "simulate Gaia_G_mag first, then apply the same style of raw epoch-median magnitude cut used for the real sample",
            "enabled": bool(apply_mock_raw_cuts),
            "candidate_oversample_factor": float(args.candidate_oversample_factor),
            "keep_all_passing_raw_cuts": bool(args.keep_all_passing_raw_cuts),
            "parent_median_mag_cut_applied_before_simulation": bool(args.apply_parent_median_mag_cut),
            "parent_mag_cut_column": mock_match_mwin_column,
            "gaia_g_mag_med_max": None if mock_raw_mag_max is None else float(mock_raw_mag_max),
            "gaia_g_snr_min": None if mock_raw_snr_min is None else float(mock_raw_snr_min),
            "gaia_g_n_epochs_min": None if mock_raw_min_epochs is None else int(mock_raw_min_epochs),
            "n_candidates_simulated": int(np.asarray(lc_truth["n_candidates_simulated"])[0]),
            "n_after_mock_raw_mag": int(np.asarray(lc_truth["n_after_mock_raw_mag"])[0]),
            "n_after_mock_raw_snr": int(np.asarray(lc_truth["n_after_mock_raw_snr"])[0]),
            "n_after_mock_raw_epochs": int(np.asarray(lc_truth["n_after_mock_raw_epochs"])[0]),
            "n_after_mock_raw_cuts_before_downselect": int(np.asarray(lc_truth["n_after_mock_raw_cuts"])[0]),
            "n_kept": int(n_pilot),
            "mag_cut_operator": "<",
        },
        "tau_model": {
            "distribution": "truncated_normal_log10_tau_rest_days",
            "log10_mu": float(args.tau_log10_mu),
            "log10_sigma": float(args.tau_log10_sigma),
            "tau_min_days": float(args.tau_min_days),
            "tau_max_days": float(args.tau_max_days),
            "seed": int(args.tau_seed),
        },
        "matching": {
            "mode": "soft_nn_unique",
            "weights": {"z": 1.0, "logl3000": 1.0, "m_win": 2.5},
            "mock_m_win_column": mock_match_mwin_column,
            "mock_log10_d_cut_column": mock_match_log10d_column,
            **{k: _json_number(v) for k, v in match_diag.items()},
        },
        "hashes": {
            "mock_source_id_sha256": _sha256_int_array(np.asarray(pilot_mock["source_id"], dtype=np.int64)),
            "real_source_id_sha256": _sha256_int_array(np.asarray(pilot_targets["source_id"], dtype=np.int64)),
        },
        "percentiles": {
            "mock_log10_d_true": _finite_percentiles(log10_d_true),
            "mock_log10_d_mean": _finite_percentiles(np.asarray(pilot_mock["true_log10_d_mean"], dtype=np.float64)),
            "log10_tau_rest_days": _finite_percentiles(log10_tau),
            "tau_rest_days": _finite_percentiles(tau_days),
            "sigma_drw_mag": _finite_percentiles(sigma_drw_mag),
            "sf_inf_mag": _finite_percentiles(sf_inf_mag),
            "log10_sigma_drw_mag": _finite_percentiles(log10_sigma),
            "log10_sf_inf_mag": _finite_percentiles(log10_sf_inf),
            "mock_z": _finite_percentiles(np.asarray(pilot_mock["z"], dtype=np.float64)),
            "mock_logl3000": _finite_percentiles(np.asarray(pilot_mock["logl3000"], dtype=np.float64)),
            "mock_m0_true": _finite_percentiles(np.asarray(lc_truth["mock_m0_gen_true"], dtype=np.float64)),
            "mock_m0_gen_true": _finite_percentiles(np.asarray(lc_truth["mock_m0_gen_true"], dtype=np.float64)),
            "mock_m0_minus_mwin_cont_true": _finite_percentiles(
                np.asarray(lc_truth["mock_m0_minus_mwin_cont_true"], dtype=np.float64)
            ),
            "mock_m_win_parent_true": _finite_percentiles(np.asarray(pilot_mock["true_m_win"], dtype=np.float64)),
            "mock_m_win_trap_true": _finite_percentiles(np.asarray(lc_truth["mock_m_win_trap_true"], dtype=np.float64)),
            "mock_m_win_cont_true": _finite_percentiles(np.asarray(lc_truth["mock_m_win_cont_true"], dtype=np.float64)),
            "mock_m_win_cont_uncond_var": _finite_percentiles(
                np.asarray(lc_truth["mock_m_win_cont_uncond_var"], dtype=np.float64)
            ),
            "mock_m_win_cont_minus_m0": _finite_percentiles(
                np.asarray(lc_truth["mock_m_win_cont_true"], dtype=np.float64)
                - np.asarray(lc_truth["mock_m0_gen_true"], dtype=np.float64)
            ),
            "mock_m_win_trap_minus_m0": _finite_percentiles(
                np.asarray(lc_truth["mock_m_win_trap_true"], dtype=np.float64)
                - np.asarray(lc_truth["mock_m0_gen_true"], dtype=np.float64)
            ),
            "baseline_rest_days": _finite_percentiles(np.asarray(lc_truth["baseline_rest_days"], dtype=np.float64)),
            "npts": _finite_percentiles(np.asarray(lc_truth["npts"], dtype=np.float64)),
            "median_mag_err": _finite_percentiles(np.asarray(lc_truth["median_mag_err"], dtype=np.float64)),
            "mock_median_mag_err": _finite_percentiles(np.asarray(lc_truth["mock_median_mag_err"], dtype=np.float64)),
            "error_scale_median": _finite_percentiles(np.asarray(lc_truth["error_scale_median"], dtype=np.float64)),
            "gaia_g_mag_med": _finite_percentiles(np.asarray(lc_truth["gaia_g_mag_med"], dtype=np.float64)),
            "gaia_g_snr": _finite_percentiles(np.asarray(lc_truth["gaia_g_snr"], dtype=np.float64)),
            "gaia_g_n_epochs": _finite_percentiles(np.asarray(lc_truth["gaia_g_n_epochs"], dtype=np.float64)),
            "real_z": _finite_percentiles(np.asarray(pilot_targets["z"], dtype=np.float64)),
            "real_logl3000_temple": _finite_percentiles(np.asarray(pilot_targets["logl3000_temple"], dtype=np.float64)),
            "real_m_win_med": _finite_percentiles(np.asarray(pilot_targets["m_win"], dtype=np.float64)),
        },
        "fractions": {
            "sigma_gt_0p2": float(np.mean(sigma_drw_mag > 0.2)),
            "sigma_gt_0p5": float(np.mean(sigma_drw_mag > 0.5)),
            "sigma_gt_1p0": float(np.mean(sigma_drw_mag > 1.0)),
            "sf_inf_gt_0p3": float(np.mean(sf_inf_mag > 0.3)),
            "sf_inf_gt_1p0": float(np.mean(sf_inf_mag > 1.0)),
            "gaia_g_mag_med_ge_20": float(np.mean(np.asarray(lc_truth["gaia_g_mag_med"], dtype=np.float64) >= 20.0)),
            "mock_m_win_cont_true_gt_20": float(np.mean(np.asarray(lc_truth["mock_m_win_cont_true"], dtype=np.float64) > 20.0)),
        },
    }
    summary_path = out_dir / f"{sample_label}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote pairs/truth: {pair_npz}")
    print(f"Wrote manifest: {manifest_npz}")
    print(f"Wrote light curves: {out_fits}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote PDF: {out_pdf}")
    print(json.dumps(summary["percentiles"]["mock_log10_d_true"], indent=2))


if __name__ == "__main__":
    main()
