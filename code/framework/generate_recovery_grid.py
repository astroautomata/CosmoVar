#!/usr/bin/env python
"""
generate_recovery_grid.py

Generate a literature-motivated Stage-1 DRW recovery mock.

The recovery-grid construction uses two sampling choices:

1. The injected grid is rectangular in the primitive DRW parameters
   (tau, sigma), rather than a broad log-uniform sigma/tau stress grid.
   Defaults are chosen from optical AGN DRW literature:

       tau_days  = 5, 20, 80, 300, 1000, 3000, 10000
       sigma_mag = 0.03, 0.06, 0.10, 0.14, 0.20, 0.30

2. Real Gaia cadence templates are selected once in bins of median apparent
   magnitude and epoch count, then reused for every (tau, sigma) cell. This
   keeps the m0/N_epoch/cadence/noise distribution matched across the entire
   injected parameter grid.

   The default cadence pool applies N_epoch >= 10, with N_epoch = 10 included.
   This removes only a negligible tail in the current Gaia sample and keeps the
   template stratification balanced.
   The default template stratification is:

       median Gaia G edges = 19.0, 19.8, 20.3
       N_epoch edges       = 29, 40, 52

   With 50 templates per cell, this gives 800 real Gaia cadence/noise
   templates and 33,600 mock light curves for the default 7 x 6 DRW grid.

The output FITS file is an input to stage1_fitting.py. Its key columns are:

    source_id
    redsh
    true_log10_sigma
    true_log10_tau
    true_log10_sigma2
    true_log10_D
    true_m0 / true_mean_mag
    true_m_win
    true_m_win_cont / true_m_win_cont_mu / true_m_win_cont_var
    true_baseline_rest
    true_npts
    Gaia_G_transit_time
    Gaia_G_mag_true
    Gaia_G_mag
    Gaia_G_mag_err
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Iterable

import numpy as np
from astropy.io import fits
from astropy.table import Table

from framework.drw_fitting import sample_true_mwin_cont, trapezoid_time_weights


DEFAULT_TAU_DAYS = (5.0, 20.0, 80.0, 300.0, 1000.0, 3000.0, 10000.0)
DEFAULT_SIGMA_MAG = (0.03, 0.06, 0.10, 0.14, 0.20, 0.30)
DEFAULT_M0_EDGES = (19.0, 19.8, 20.3)
DEFAULT_NPTS_EDGES = (29.0, 40.0, 52.0)
DEFAULT_MIN_EPOCHS = 10


def _parse_float_list(text: str | None, default: Iterable[float]) -> np.ndarray:
    if text is None or str(text).strip() == "":
        vals = list(default)
    else:
        vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"Invalid numeric list: {text!r}")
    return arr


def _bin_index(value: float, edges: np.ndarray) -> int:
    """Return 0..len(edges) bin index for open-ended bins around edges."""
    return int(np.searchsorted(edges, float(value), side="right"))


def simulate_ou(times_rf: np.ndarray, sigma: float, tau: float, mean_mag: float, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate a stationary OU/DRW process at rest-frame times.

    sigma is the stationary standard deviation in magnitudes, and tau is the
    damping time-scale in rest-frame days.
    """
    t = np.asarray(times_rf, dtype=float)
    if t.size == 0:
        return np.array([], dtype=float)

    order = np.argsort(t)
    t_s = t[order]
    m_s = np.empty(t_s.size, dtype=float)

    m_s[0] = mean_mag + float(sigma) * rng.standard_normal()
    for k in range(1, t_s.size):
        dt = float(t_s[k] - t_s[k - 1])
        rho = np.exp(-dt / float(tau))
        cond_std = float(sigma) * np.sqrt(max(1.0 - rho * rho, 0.0))
        m_s[k] = mean_mag + rho * (m_s[k - 1] - mean_mag) + cond_std * rng.standard_normal()

    out = np.empty_like(m_s)
    out[order] = m_s
    return out


def load_cadence_pool(input_fits: str, *, min_epochs: int, max_templates: int | None = None) -> list[dict]:
    """
    Load real Gaia G cadence/noise templates using the same epoch mask as v1.
    """
    pool: list[dict] = []
    print(f"Opening {input_fits} ...")
    with fits.open(input_fits, memmap=True) as hdul:
        data = hdul[1].data
        print(f"  Rows in file: {len(data):,}")
        for i in range(len(data)):
            sid = int(data["source_id"][i])
            z = float(data["z"][i])
            if not np.isfinite(z) or z < 0.0:
                continue

            t = np.asarray(data["g_transit_time"][i], dtype=float)
            f = np.asarray(data["g_transit_flux"][i], dtype=float)
            fe = np.asarray(data["g_transit_flux_error"][i], dtype=float)
            mag = np.asarray(data["g_transit_mag"][i], dtype=float)

            good = (
                np.isfinite(t)
                & np.isfinite(f)
                & np.isfinite(fe)
                & np.isfinite(mag)
                & (f > 0.0)
                & (fe > 0.0)
                & (mag > 0.0)
            )
            if int(good.sum()) < int(min_epochs):
                continue

            t = t[good]
            f = f[good]
            fe = fe[good]
            mag = mag[good]

            order = np.argsort(t)
            t   = t[order]
            f   = f[order]
            fe  = fe[order]
            mag = mag[order]

            baseline_rest = (float(np.max(t)) - float(np.min(t))) / (1.0 + z)
            if not np.isfinite(baseline_rest) or baseline_rest <= 0.0:
                continue

            with np.errstate(divide="ignore", invalid="ignore"):
                mag_err = 1.0857362047581294 * (fe / f)
            bad = ~np.isfinite(mag_err) | (mag_err <= 0.0)
            if bad.all():
                mag_err[:] = 0.02
            elif bad.any():
                mag_err[bad] = float(np.median(mag_err[~bad]))
            mag_err = np.clip(mag_err, 1e-4, 1.0)

            t0 = float(np.min(t))
            t_rf = (t - t0) / (1.0 + z)
            npts = int(t_rf.size)
            mean_dt_rest = baseline_rest / max(npts - 1, 1)

            pool.append(
                {
                    "source_id": sid,
                    "z": z,
                    "g_time": t.astype(np.float64),
                    "t_rf": t_rf.astype(np.float64),
                    "g_mag_err": mag_err.astype(np.float64),
                    "median_mag": float(np.median(mag)),
                    "npts": npts,
                    "baseline_rest": float(baseline_rest),
                    "mean_dt_rest": float(mean_dt_rest),
                    "median_mag_err": float(np.median(mag_err)),
                }
            )
            if max_templates is not None and len(pool) >= int(max_templates):
                break
    return pool


def select_stratified_templates(
    pool: list[dict],
    *,
    templates_per_bin: int,
    m0_edges: np.ndarray,
    npts_edges: np.ndarray,
    rng: np.random.Generator,
    replace: bool,
    allow_sparse_bins: bool,
) -> tuple[list[dict], dict]:
    """
    Select a matched template set from open-ended m0 x N_epoch bins.
    """
    n_m_bins = m0_edges.size + 1
    n_n_bins = npts_edges.size + 1
    bins: list[list[list[int]]] = [[[] for _ in range(n_n_bins)] for _ in range(n_m_bins)]

    for idx, tmpl in enumerate(pool):
        mi = _bin_index(tmpl["median_mag"], m0_edges)
        ni = _bin_index(tmpl["npts"], npts_edges)
        bins[mi][ni].append(idx)

    counts = np.zeros((n_m_bins, n_n_bins), dtype=int)
    selected: list[dict] = []
    selected_counts = np.zeros_like(counts)
    for mi in range(n_m_bins):
        for ni in range(n_n_bins):
            candidates = np.asarray(bins[mi][ni], dtype=int)
            counts[mi, ni] = int(candidates.size)
            if candidates.size == 0:
                if allow_sparse_bins:
                    continue
                raise RuntimeError(
                    f"Bin (m0_bin={mi}, npts_bin={ni}) has no templates. "
                    "Use wider bins or leave sparse bins enabled."
                )
            take_n = int(templates_per_bin)
            choose_replace = bool(replace)
            if candidates.size < templates_per_bin and not replace and allow_sparse_bins:
                take_n = int(candidates.size)
            elif candidates.size < templates_per_bin and not replace:
                raise RuntimeError(
                    f"Bin (m0_bin={mi}, npts_bin={ni}) has only {candidates.size} templates; "
                    f"need {templates_per_bin}. Lower --templates-per-bin, pass --replace, "
                    "or leave sparse bins enabled."
                )
            chosen = rng.choice(candidates, size=take_n, replace=choose_replace)
            for rank, pool_idx in enumerate(chosen):
                item = dict(pool[int(pool_idx)])
                item["m0_bin"] = int(mi)
                item["npts_bin"] = int(ni)
                item["template_rank_in_bin"] = int(rank)
                selected.append(item)
            selected_counts[mi, ni] = int(chosen.size)

    meta = {
        "m0_edges": m0_edges.tolist(),
        "npts_edges": npts_edges.tolist(),
        "counts_available": counts.tolist(),
        "counts_selected": selected_counts.tolist(),
        "templates_per_bin": int(templates_per_bin),
        "replace": bool(replace),
        "allow_sparse_bins": bool(allow_sparse_bins),
    }
    return selected, meta


def _print_preflight(pool: list[dict], selected: list[dict], tau_grid: np.ndarray, selection_meta: dict) -> None:
    counts = np.asarray(selection_meta["counts_available"], dtype=int)
    print()
    print("Stratified template availability")
    print("  m0 edges:   ", selection_meta["m0_edges"])
    print("  npts edges: ", selection_meta["npts_edges"])
    print("  counts rows=m0 bins, cols=npts bins:")
    print(counts)
    print("  selected counts rows=m0 bins, cols=npts bins:")
    print(np.asarray(selection_meta["counts_selected"], dtype=int))
    print(f"  minimum available bin count: {int(np.min(counts)):,}")
    print(f"  templates selected per bin:  {selection_meta['templates_per_bin']:,}")
    if selection_meta.get("allow_sparse_bins", False):
        sparse = (counts < int(selection_meta["templates_per_bin"])) & (counts > 0)
        empty = counts == 0
        if np.any(sparse) or np.any(empty):
            print("  sparse/empty bins are diagnostic only; no replacement was used.")

    b_all = np.asarray([x["baseline_rest"] for x in selected], dtype=float)
    n_all = np.asarray([x["npts"] for x in selected], dtype=float)
    m_all = np.asarray([x["median_mag"] for x in selected], dtype=float)
    print()
    print(f"Selected matched templates: {len(selected):,}")
    for name, arr in [("m0", m_all), ("npts", n_all), ("baseline_rest", b_all)]:
        print(
            f"  {name:13s} p5/p16/p50/p84/p95 = "
            + np.array2string(np.nanpercentile(arr, [5, 16, 50, 84, 95]), precision=3, separator=", ")
        )
    print()
    print("log10(T_rf/tau_true) coverage for selected templates")
    for tau in tau_grid:
        x = np.log10(b_all / float(tau))
        print(
            f"  tau={tau:8.1f} d: p5/p16/p50/p84/p95 = "
            + np.array2string(np.nanpercentile(x, [5, 16, 50, 84, 95]), precision=3, separator=", ")
        )
    x_all = np.concatenate([np.log10(b_all / float(tau)) for tau in tau_grid])
    print(
        "  combined:      p1/p5/p16/p50/p84/p95/p99 = "
        + np.array2string(np.nanpercentile(x_all, [1, 5, 16, 50, 84, 95, 99]), precision=3, separator=", ")
    )
    print(f"  combined min/max = {np.nanmin(x_all):.3f}, {np.nanmax(x_all):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a literature-motivated, stratified Stage-1 DRW recovery mock.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-fits", required=True, help="Real Gaia G light-curve FITS used as cadence/noise pool.")
    parser.add_argument("--output-fits", default=None, help="Output mock FITS. Required unless --dry-run is set.")
    parser.add_argument("--output-meta", default=None, help="Output JSON metadata path. Default: <output-fits>.meta.json")
    parser.add_argument("--tau-days", default=None, help="Comma-separated injected tau grid in rest-frame days.")
    parser.add_argument("--sigma-mag", default=None, help="Comma-separated injected stationary sigma grid in mag.")
    parser.add_argument("--templates-per-bin", type=int, default=50, help="Templates selected per m0 x N_epoch bin.")
    parser.add_argument("--m0-bin-edges", default=None, help="Comma-separated m0 bin edges; open-ended bins are used.")
    parser.add_argument("--npts-bin-edges", default=None, help="Comma-separated N_epoch bin edges; open-ended bins are used.")
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=DEFAULT_MIN_EPOCHS,
        help="Minimum usable epochs for cadence templates. Sources with exactly this count are included.",
    )
    parser.add_argument("--max-templates", type=int, default=None, help="Optional cap on loaded templates for debugging.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for template selection, DRW, and noise.")
    parser.add_argument("--replace", action="store_true", help="Allow replacement within template bins if needed.")
    parser.add_argument(
        "--require-full-bins",
        action="store_true",
        help="Fail if any m0 x N_epoch bin has fewer than --templates-per-bin templates.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only load/select templates and print coverage; do not simulate/write.")
    args = parser.parse_args()

    if not args.dry_run and not args.output_fits:
        raise ValueError("--output-fits is required unless --dry-run is set.")
    if args.templates_per_bin <= 0:
        raise ValueError("--templates-per-bin must be positive.")
    if args.min_epochs < 2:
        raise ValueError("--min-epochs must be at least 2 so each template has a positive baseline.")

    tau_grid = _parse_float_list(args.tau_days, DEFAULT_TAU_DAYS)
    sigma_grid = _parse_float_list(args.sigma_mag, DEFAULT_SIGMA_MAG)
    m0_edges = np.sort(_parse_float_list(args.m0_bin_edges, DEFAULT_M0_EDGES))
    npts_edges = np.sort(_parse_float_list(args.npts_bin_edges, DEFAULT_NPTS_EDGES))
    if np.any(tau_grid <= 0.0) or np.any(sigma_grid <= 0.0):
        raise ValueError("All tau and sigma values must be positive.")

    rng = np.random.default_rng(int(args.seed))
    rng_mwin_cont = np.random.default_rng(np.random.SeedSequence([int(args.seed), 447291]))
    pool = load_cadence_pool(args.input_fits, min_epochs=int(args.min_epochs), max_templates=args.max_templates)
    if not pool:
        raise RuntimeError("No usable templates found.")
    print(f"Templates passing mask: {len(pool):,}")

    selected, selection_meta = select_stratified_templates(
        pool,
        templates_per_bin=int(args.templates_per_bin),
        m0_edges=m0_edges,
        npts_edges=npts_edges,
        rng=rng,
        replace=bool(args.replace),
        allow_sparse_bins=not bool(args.require_full_bins),
    )
    _print_preflight(pool, selected, tau_grid, selection_meta)

    n_total = int(len(selected) * tau_grid.size * sigma_grid.size)
    print()
    print("Injected grid")
    print("  tau_days:  ", tau_grid.tolist())
    print("  sigma_mag: ", sigma_grid.tolist())
    print(f"  total mock light curves: {len(selected):,} templates x {tau_grid.size} tau x {sigma_grid.size} sigma = {n_total:,}")

    if args.dry_run:
        print("Dry run complete; no FITS written.")
        return

    rows_source_id = np.empty(n_total, dtype=np.int64)
    rows_redsh = np.empty(n_total, dtype=np.float64)
    rows_true_log10_sigma = np.empty(n_total, dtype=np.float64)
    rows_true_log10_sigma2 = np.empty(n_total, dtype=np.float64)
    rows_true_log10_tau = np.empty(n_total, dtype=np.float64)
    rows_true_log10_D = np.empty(n_total, dtype=np.float64)
    rows_true_ln_tau = np.empty(n_total, dtype=np.float64)
    rows_true_ln_D = np.empty(n_total, dtype=np.float64)
    rows_grid_i = np.empty(n_total, dtype=np.int32)
    rows_grid_j = np.empty(n_total, dtype=np.int32)
    rows_repeat_k = np.empty(n_total, dtype=np.int32)
    rows_template_sid = np.empty(n_total, dtype=np.int64)
    rows_m0_bin = np.empty(n_total, dtype=np.int16)
    rows_npts_bin = np.empty(n_total, dtype=np.int16)
    rows_template_rank = np.empty(n_total, dtype=np.int32)
    rows_true_m0 = np.empty(n_total, dtype=np.float64)
    rows_true_mean_mag = np.empty(n_total, dtype=np.float64)
    rows_true_mwin = np.empty(n_total, dtype=np.float64)
    rows_true_mwin_cont = np.empty(n_total, dtype=np.float64)
    rows_true_mwin_cont_mu = np.empty(n_total, dtype=np.float64)
    rows_true_mwin_cont_var = np.empty(n_total, dtype=np.float64)
    rows_true_baseline_rf = np.empty(n_total, dtype=np.float64)
    rows_true_mean_dt_rf = np.empty(n_total, dtype=np.float64)
    rows_true_npts = np.empty(n_total, dtype=np.int32)
    rows_median_mag_err = np.empty(n_total, dtype=np.float64)

    rows_times = np.empty(n_total, dtype=object)
    rows_mags_true = np.empty(n_total, dtype=object)
    rows_mags = np.empty(n_total, dtype=object)
    rows_mag_errs = np.empty(n_total, dtype=object)

    row = 0
    ln10 = np.log(10.0)
    print()
    print("Simulating mock light curves")
    for j, tau in enumerate(tau_grid):
        for i, sigma in enumerate(sigma_grid):
            log10_sigma = float(np.log10(sigma))
            log10_sigma2 = float(2.0 * log10_sigma)
            log10_tau = float(np.log10(tau))
            log10_D = float(log10_sigma2 - log10_tau)
            for k, tmpl in enumerate(selected):
                m_true = simulate_ou(
                    tmpl["t_rf"],
                    sigma=float(sigma),
                    tau=float(tau),
                    mean_mag=float(tmpl["median_mag"]),
                    rng=rng,
                )
                w_rf = trapezoid_time_weights(tmpl["t_rf"])
                true_mwin = float(np.dot(w_rf, m_true))
                true_mwin_cont, true_mwin_cont_mu, true_mwin_cont_var = sample_true_mwin_cont(
                    tmpl["t_rf"],
                    m_true,
                    float(tmpl["median_mag"]),
                    sigma2=float(sigma * sigma),
                    tau=float(tau),
                    rng=rng_mwin_cont,
                )
                m_obs = m_true + rng.normal(scale=tmpl["g_mag_err"])

                rows_source_id[row] = row + 1
                rows_redsh[row] = float(tmpl["z"])
                rows_true_log10_sigma[row] = log10_sigma
                rows_true_log10_sigma2[row] = log10_sigma2
                rows_true_log10_tau[row] = log10_tau
                rows_true_log10_D[row] = log10_D
                rows_true_ln_tau[row] = log10_tau * ln10
                rows_true_ln_D[row] = log10_D * ln10
                rows_grid_i[row] = i
                rows_grid_j[row] = j
                rows_repeat_k[row] = k
                rows_template_sid[row] = int(tmpl["source_id"])
                rows_m0_bin[row] = int(tmpl["m0_bin"])
                rows_npts_bin[row] = int(tmpl["npts_bin"])
                rows_template_rank[row] = int(tmpl["template_rank_in_bin"])
                rows_true_m0[row] = float(tmpl["median_mag"])
                rows_true_mean_mag[row] = float(tmpl["median_mag"])
                rows_true_mwin[row] = true_mwin
                rows_true_mwin_cont[row] = true_mwin_cont
                rows_true_mwin_cont_mu[row] = true_mwin_cont_mu
                rows_true_mwin_cont_var[row] = true_mwin_cont_var
                rows_true_baseline_rf[row] = float(tmpl["baseline_rest"])
                rows_true_mean_dt_rf[row] = float(tmpl["mean_dt_rest"])
                rows_true_npts[row] = int(tmpl["npts"])
                rows_median_mag_err[row] = float(tmpl["median_mag_err"])
                rows_times[row] = tmpl["g_time"].astype(np.float64)
                rows_mags_true[row] = m_true.astype(np.float64)
                rows_mags[row] = m_obs.astype(np.float64)
                rows_mag_errs[row] = tmpl["g_mag_err"].astype(np.float64)
                row += 1

            print(
                f"  tau[{j}]={tau:.1f} d, sigma[{i}]={sigma:.3f} mag "
                f"-> log10D={log10_D:+.3f} [{row:,}/{n_total:,}]",
                flush=True,
            )

    tbl = Table()
    tbl["source_id"] = rows_source_id
    tbl["redsh"] = rows_redsh
    tbl["true_log10_sigma"] = rows_true_log10_sigma
    tbl["true_log10_sigma2"] = rows_true_log10_sigma2
    tbl["true_log10_tau"] = rows_true_log10_tau
    tbl["true_log10_D"] = rows_true_log10_D
    tbl["true_ln_tau"] = rows_true_ln_tau
    tbl["true_ln_D"] = rows_true_ln_D
    tbl["grid_i"] = rows_grid_i
    tbl["grid_j"] = rows_grid_j
    tbl["repeat_k"] = rows_repeat_k
    tbl["template_source_id"] = rows_template_sid
    tbl["template_m0_bin"] = rows_m0_bin
    tbl["template_npts_bin"] = rows_npts_bin
    tbl["template_rank_in_bin"] = rows_template_rank
    tbl["true_m0"] = rows_true_m0
    tbl["true_mean_mag"] = rows_true_mean_mag
    tbl["true_m_win"] = rows_true_mwin
    tbl["true_m_win_cont"] = rows_true_mwin_cont
    tbl["true_m_win_cont_mu"] = rows_true_mwin_cont_mu
    tbl["true_m_win_cont_var"] = rows_true_mwin_cont_var
    tbl["true_baseline_rest"] = rows_true_baseline_rf
    tbl["true_mean_dt_rest"] = rows_true_mean_dt_rf
    tbl["true_npts"] = rows_true_npts
    tbl["template_median_mag_err"] = rows_median_mag_err
    tbl["Gaia_G_transit_time"] = rows_times
    tbl["Gaia_G_mag_true"] = rows_mags_true
    tbl["Gaia_G_mag"] = rows_mags
    tbl["Gaia_G_mag_err"] = rows_mag_errs

    out_path = os.path.abspath(os.path.expanduser(str(args.output_fits)))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tbl.write(out_path, overwrite=True)
    print(f"Written {n_total:,} rows -> {out_path}")

    meta_path = args.output_meta or (out_path + ".meta.json")
    meta = {
        "created_at": datetime.utcnow().isoformat(),
        "script": "generate_recovery_grid.py",
        "input_fits": os.path.abspath(args.input_fits),
        "output_fits": out_path,
        "seed": int(args.seed),
        "min_epochs": int(args.min_epochs),
        "grid": {
            "tau_days": tau_grid.tolist(),
            "sigma_mag": sigma_grid.tolist(),
            "n_tau": int(tau_grid.size),
            "n_sigma": int(sigma_grid.size),
            "n_parameter_cells": int(tau_grid.size * sigma_grid.size),
            "parameterization": "primitive DRW grid in tau_days and stationary sigma_mag",
            "derived_D": "D = sigma^2 / tau",
            "sf_infinity": "SF_inf = sqrt(2) * sigma",
        },
        "template_selection": selection_meta,
        "output": {
            "n_templates_selected": int(len(selected)),
            "n_total": int(n_total),
        },
        "truth_columns": {
            "true_log10_sigma": "log10(stationary sigma / mag)",
            "true_log10_sigma2": "log10(stationary sigma^2 / mag^2)",
            "true_log10_tau": "log10(tau / rest-frame day)",
            "true_log10_D": "log10(sigma^2 / tau)",
            "true_m_win": "trapezoid-weighted latent noiseless finite-window mean magnitude",
            "true_m_win_cont": "conditional draw of continuous latent finite-window mean magnitude",
            "true_m_win_cont_mu": "conditional mean of true_m_win_cont given noiseless latent epoch values",
            "true_m_win_cont_var": "conditional variance of true_m_win_cont given noiseless latent epoch values",
        },
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Metadata -> {meta_path}")


if __name__ == "__main__":
    main()
