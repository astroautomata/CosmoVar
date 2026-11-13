#!/usr/bin/env python
"""Prepare Gaia G source summaries and quality selections for Figure 2."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from astropy.io import fits

from framework.catalog_utils import DEFAULT_CATALOG, DEFAULT_CROSSMATCH
from framework import DATA_ROOT

BAND_LC_COLS = {"gaia_g": ("g_transit_time", "g_mag", "g_mag_error")}
GAIA_G_MAG_ALIASES = ("g_mag", "g_transit_mag")
GAIA_G_MAGERR_ALIASES = ("g_mag_error",)
GAIA_G_FLUX_ALIASES = ("g_transit_flux",)
GAIA_G_FLUXERR_ALIASES = ("g_transit_flux_error",)



def _file_mtime(path: str) -> Optional[float]:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _summary_stats_from_lc_file(
    lc_file: str,
    time_col: str,
    mag_col: str,
    magerr_col: str,
    min_pts_for_stats: int,
    band: Optional[str] = None,
    gaia_time_zero: float = 55197.0,
) -> Dict[str, np.ndarray]:
    """
    Compute per-source summary stats used for quality cuts:
      - n_epochs: number of valid epochs (finite t, mag, magerr>0)
      - mag_med: median mag across valid epochs (if enough points)
      - snr: (p95 - p5) / median(magerr) across valid epochs (if enough points)
      - baseline_obs: max(t)-min(t) across valid epochs (if >= 2 points)
      - tmin_obs/tmax_obs: min/max valid observed time (Gaia shifted by gaia_time_zero)
    """
    with fits.open(lc_file, memmap=True) as hdul:
        data = hdul[1].data
        cols = set(hdul[1].columns.names)

        # Support Gaia schema differences:
        # - Magnitude-error files have g_mag / g_mag_error
        # - Flag-cleaned files typically have g_transit_mag and flux/flux_error
        use_flux_errors = False
        flux_col = None
        fluxerr_col = None
        resolved_time_col = time_col
        resolved_mag_col = mag_col
        resolved_magerr_col = magerr_col
        if band == "gaia_g":
            if resolved_mag_col not in cols:
                for cand in GAIA_G_MAG_ALIASES:
                    if cand in cols:
                        resolved_mag_col = cand
                        break

            if resolved_magerr_col in cols:
                pass
            else:
                for cand in GAIA_G_MAGERR_ALIASES:
                    if cand in cols:
                        resolved_magerr_col = cand
                        break
                else:
                    # Fallback: derive mag errors from flux/flux_err if available.
                    for fc in GAIA_G_FLUX_ALIASES:
                        if fc in cols:
                            flux_col = fc
                            break
                    for fec in GAIA_G_FLUXERR_ALIASES:
                        if fec in cols:
                            fluxerr_col = fec
                            break
                    if flux_col is not None and fluxerr_col is not None:
                        use_flux_errors = True
                        resolved_magerr_col = None

        missing = []
        if resolved_time_col not in cols:
            missing.append(resolved_time_col)
        if resolved_mag_col not in cols:
            missing.append(resolved_mag_col)
        if not use_flux_errors and (resolved_magerr_col not in cols):
            missing.append(resolved_magerr_col)
        if missing:
            raise KeyError(
                f"Missing required columns in {lc_file}: {missing}. "
                f"Available columns: {sorted(cols)}"
            )

        source_id = np.asarray(data["source_id"], dtype=np.int64)
        times = data[resolved_time_col]
        mags = data[resolved_mag_col]
        errs = data[resolved_magerr_col] if not use_flux_errors else None
        fluxes = data[flux_col] if use_flux_errors else None
        flux_errs = data[fluxerr_col] if use_flux_errors else None

        n = len(source_id)
        n_epochs = np.zeros(n, dtype=np.int32)
        snr = np.full(n, np.nan, dtype=np.float64)
        mag_med = np.full(n, np.nan, dtype=np.float64)
        baseline_obs = np.full(n, np.nan, dtype=np.float64)
        tmin_obs = np.full(n, np.nan, dtype=np.float64)
        tmax_obs = np.full(n, np.nan, dtype=np.float64)

        for i in range(n):
            t = np.asarray(times[i], dtype=float)
            m = np.asarray(mags[i], dtype=float)
            if use_flux_errors:
                f = np.asarray(fluxes[i], dtype=float)
                fe = np.asarray(flux_errs[i], dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    e = (2.5 / np.log(10.0)) * (fe / f)
            else:
                e = np.asarray(errs[i], dtype=float)
            valid = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
            n_valid = int(np.count_nonzero(valid))
            n_epochs[i] = n_valid
            if n_valid >= 2:
                tv = t[valid]
                if band == "gaia_g":
                    tv = np.asarray(tv, dtype=np.float64) + float(gaia_time_zero)
                if tv.size:
                    tmin_obs[i] = float(np.min(tv))
                    tmax_obs[i] = float(np.max(tv))
                    baseline_obs[i] = float(np.max(tv) - np.min(tv))
            elif n_valid == 1:
                tv = t[valid]
                if band == "gaia_g":
                    tv = np.asarray(tv, dtype=np.float64) + float(gaia_time_zero)
                if tv.size:
                    tmin_obs[i] = float(tv[0])
                    tmax_obs[i] = float(tv[0])
            if n_valid >= int(min_pts_for_stats):
                mv = m[valid]
                ev = e[valid]
                if mv.size:
                    mag_med[i] = float(np.median(mv))
                    signal = np.percentile(mv, 95) - np.percentile(mv, 5)
                    # Apply the structure-function quality-cut definition.
                    noise = float(np.median(ev)) if ev.size else np.nan
                    snr[i] = signal / noise if noise > 0 else np.nan

    order = np.argsort(source_id)
    return {
        "source_id": source_id[order],
        "n_epochs": n_epochs[order],
        "snr": snr[order],
        "mag_med": mag_med[order],
        "baseline_obs": baseline_obs[order],
        "tmin_obs": tmin_obs[order],
        "tmax_obs": tmax_obs[order],
    }


def _union_coverage_and_largest_gap(tmins: np.ndarray, tmaxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-source union coverage length and largest internal gap across intervals.
    Inputs shape [N,B] with NaN for missing intervals.
    Returns:
      - coverage_obs_nogap [N]
      - largest_gap_obs [N]
    """
    tmins = np.asarray(tmins, dtype=np.float64)
    tmaxs = np.asarray(tmaxs, dtype=np.float64)
    if tmins.shape != tmaxs.shape:
        raise ValueError("tmins/tmaxs shape mismatch")
    n_obj, n_band = tmins.shape
    cov = np.full(n_obj, np.nan, dtype=np.float64)
    gap = np.full(n_obj, np.nan, dtype=np.float64)
    for i in range(n_obj):
        m = np.isfinite(tmins[i]) & np.isfinite(tmaxs[i]) & (tmaxs[i] >= tmins[i])
        if not np.any(m):
            continue
        starts = np.asarray(tmins[i, m], dtype=np.float64)
        ends = np.asarray(tmaxs[i, m], dtype=np.float64)
        order = np.argsort(starts)
        starts = starts[order]
        ends = ends[order]
        cur_s = starts[0]
        cur_e = ends[0]
        total = 0.0
        max_gap = 0.0
        for s, e in zip(starts[1:], ends[1:]):
            if s <= cur_e:
                if e > cur_e:
                    cur_e = e
            else:
                total += float(cur_e - cur_s)
                max_gap = max(max_gap, float(s - cur_e))
                cur_s, cur_e = float(s), float(e)
        total += float(cur_e - cur_s)
        cov[i] = total
        gap[i] = max_gap
    return cov, gap


def _intersect_sorted_ids(arrays: Iterable[np.ndarray]) -> np.ndarray:
    arrays = list(arrays)
    if not arrays:
        return np.array([], dtype=np.int64)
    out = np.asarray(arrays[0], dtype=np.int64)
    for a in arrays[1:]:
        out = np.intersect1d(out, np.asarray(a, dtype=np.int64), assume_unique=False)
        if out.size == 0:
            break
    return out


def _align_by_id(sorted_source_id: np.ndarray, sorted_values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_source_id, ids)
    # ids come from intersections, but keep a safety check.
    ok = (idx >= 0) & (idx < sorted_source_id.size) & (sorted_source_id[idx] == ids)
    if not np.all(ok):
        raise RuntimeError("Internal alignment error: some ids not found in source_id array")
    return sorted_values[idx]


def _load_z_logl3000(catalog_file: str, crossmatch_file: str):
    from framework.catalog_utils import load_catalog_z_logl3000

    return load_catalog_z_logl3000(catalog_file, crossmatch_file=crossmatch_file)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands", choices=["gaia_g"], default="gaia_g")
    parser.add_argument("--lc-gaia-g", default=str(DATA_ROOT / "external_data/observed_gaia_light_curves.fits"))
    parser.add_argument("--catalog-file", default=DEFAULT_CATALOG)
    parser.add_argument("--crossmatch-file", default=DEFAULT_CROSSMATCH)
    parser.add_argument("--snr-min", type=float, default=0.0)
    parser.add_argument("--mag-max", type=float, default=99.0)
    parser.add_argument("--min-epochs-gaia", type=int, default=1)
    parser.add_argument("--min-points-snr", type=int, default=0)
    parser.add_argument("--gaia-time-zero", type=float, default=55197.0)
    parser.add_argument("--output", default=str(DATA_ROOT / "results/observed_sample/gaia_sdss_sample.npz"))
    return parser.parse_args()



def main():
    args = parse_args()
    bands = ["gaia_g"]

    lc_paths = {"gaia_g": args.lc_gaia_g}
    required = [lc_paths[b] for b in bands]
    for path in required + [args.catalog_file]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"Bands: {bands}")
    print("Computing per-band summary stats for quality cuts...")
    stats_by_band: Dict[str, Dict[str, np.ndarray]] = {}
    for b in bands:
        tcol, mcol, ecol = BAND_LC_COLS[b]
        print(f"  {b}: {lc_paths[b]}")
        stats_by_band[b] = _summary_stats_from_lc_file(
            lc_paths[b],
            time_col=tcol,
            mag_col=mcol,
            magerr_col=ecol,
            min_pts_for_stats=int(args.min_points_snr),
            band=b,
            gaia_time_zero=float(args.gaia_time_zero),
        )

    candidate_source_id = _intersect_sorted_ids([stats_by_band[b]["source_id"] for b in bands])
    print(f"Candidates present in all requested bands: {candidate_source_id.size:,}")
    if candidate_source_id.size == 0:
        raise SystemExit("No candidates found for the requested bands.")

    # Build aligned per-band arrays for the candidate order (core bands).
    per_band_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for b in bands:
        sid = stats_by_band[b]["source_id"]
        per_band_arrays[b] = {
            "n_epochs": _align_by_id(sid, stats_by_band[b]["n_epochs"], candidate_source_id),
            "snr": _align_by_id(sid, stats_by_band[b]["snr"], candidate_source_id),
            "mag_med": _align_by_id(sid, stats_by_band[b]["mag_med"], candidate_source_id),
            "baseline_obs": _align_by_id(sid, stats_by_band[b]["baseline_obs"], candidate_source_id),
            "tmin_obs": _align_by_id(sid, stats_by_band[b]["tmin_obs"], candidate_source_id),
            "tmax_obs": _align_by_id(sid, stats_by_band[b]["tmax_obs"], candidate_source_id),
        }

    # Apply cuts for all available bands; global pass_source_id stays defined by requested core bands.
    pass_masks = []
    for b, stats in per_band_arrays.items():
        ne = np.asarray(stats["n_epochs"], dtype=np.float64)
        snr = np.asarray(stats["snr"], dtype=np.float64)
        mag = np.asarray(stats["mag_med"], dtype=np.float64)
        min_epochs = int(args.min_epochs_gaia)
        pass_b = (
            (ne >= min_epochs)
            & np.isfinite(mag)
            & (mag <= float(args.mag_max))
            & np.isfinite(snr)
            & (snr >= float(args.snr_min))
        )
        stats["pass"] = pass_b.astype(np.int8)
        if b in bands:
            pass_masks.append(pass_b)

    pass_mask = np.logical_and.reduce(pass_masks) if pass_masks else np.zeros(candidate_source_id.size, dtype=bool)
    pass_source_id = candidate_source_id[pass_mask]
    print(f"Passed quality cuts: {pass_source_id.size:,} / {candidate_source_id.size:,}")

    def _has_band(band: str) -> np.ndarray:
        if band not in per_band_arrays:
            return np.zeros(candidate_source_id.size, dtype=bool)
        return np.asarray(per_band_arrays[band]["n_epochs"], dtype=np.int32) > 0

    def _pass_band(band: str) -> np.ndarray:
        if band not in per_band_arrays:
            return np.zeros(candidate_source_id.size, dtype=bool)
        return np.asarray(per_band_arrays[band]["pass"], dtype=np.int8) > 0

    has_gaia = _has_band("gaia_g")
    has_ztf = _has_band("ztf_g") & _has_band("ztf_r")
    has_ps1 = _has_band("ps1_g") & _has_band("ps1_r") & _has_band("ps1_i")
    has_sdss = _has_band("sdss_g") & _has_band("sdss_r") & _has_band("sdss_i")

    pass_gaia = _pass_band("gaia_g")
    pass_ztf = _pass_band("ztf_g") & _pass_band("ztf_r")
    pass_ps1 = _pass_band("ps1_g") & _pass_band("ps1_r") & _pass_band("ps1_i")
    pass_sdss = _pass_band("sdss_g") & _pass_band("sdss_r") & _pass_band("sdss_i")

    scenario_masks = {
        "gaia_only": (has_gaia, pass_gaia),
        "gaia_ztf": (has_gaia & has_ztf, pass_gaia & pass_ztf),
        "gaia_ztf_ps1": (has_gaia & has_ztf & has_ps1, pass_gaia & pass_ztf & pass_ps1),
        "gaia_ztf_ps1_sdss": (
            has_gaia & has_ztf & has_ps1 & has_sdss,
            pass_gaia & pass_ztf & pass_ps1 & pass_sdss,
        ),
    }

    print("Loading z and logL3000 from catalog...")
    z_dict, logl3000_dict = _load_z_logl3000(args.catalog_file, crossmatch_file=args.crossmatch_file)
    candidate_z = np.array([z_dict.get(int(sid), np.nan) for sid in candidate_source_id], dtype=float)
    candidate_logl3000 = np.array([logl3000_dict.get(int(sid), np.nan) for sid in candidate_source_id], dtype=float)
    pass_z = candidate_z[pass_mask]
    pass_logl3000 = candidate_logl3000[pass_mask]

    # Manifest + outputs
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "bands": bands,
        "inputs": {
            "catalog_file": args.catalog_file,
            "crossmatch_file": args.crossmatch_file,
            "lc_files": {b: lc_paths[b] for b in bands},
        },
        "inputs_mtime": {
            "catalog_file": _file_mtime(args.catalog_file),
            "crossmatch_file": _file_mtime(args.crossmatch_file),
            "lc_files": {b: _file_mtime(lc_paths[b]) for b in bands},
        },
        "cuts": {
            "snr_min": float(args.snr_min),
            "mag_max": float(args.mag_max),
            "min_epochs_gaia": int(args.min_epochs_gaia),
            "min_points_snr": int(args.min_points_snr),
            "bands_apply_cuts": bands,
            "gaia_time_zero": float(args.gaia_time_zero),
        },
        "counts": {
            "n_candidates": int(candidate_source_id.size),
            "n_pass": int(pass_source_id.size),
            "n_gaia_only": int(np.sum(scenario_masks["gaia_only"][0])),
            "n_gaia_ztf": int(np.sum(scenario_masks["gaia_ztf"][0])),
            "n_gaia_ztf_ps1": int(np.sum(scenario_masks["gaia_ztf_ps1"][0])),
            "n_gaia_ztf_ps1_sdss": int(np.sum(scenario_masks["gaia_ztf_ps1_sdss"][0])),
        },
        "outputs": {
            "npz": args.output,
        },
    }

    save = {
        "bands": np.array(bands, dtype=object),
        "candidate_source_id": candidate_source_id.astype(np.int64),
        "pass_source_id": pass_source_id.astype(np.int64),
        "pass_mask_in_candidate": pass_mask.astype(np.int8),
        "candidate_z": candidate_z.astype(np.float64),
        "candidate_logl3000": candidate_logl3000.astype(np.float64),
        "pass_z": pass_z.astype(np.float64),
        "pass_logl3000": pass_logl3000.astype(np.float64),
        "manifest_json": json.dumps(manifest, indent=2, sort_keys=True),
        "mask_gaia_only_present": np.asarray(scenario_masks["gaia_only"][0], dtype=np.int8),
        "mask_gaia_only_pass": np.asarray(scenario_masks["gaia_only"][1], dtype=np.int8),
        "mask_gaia_ztf_present": np.asarray(scenario_masks["gaia_ztf"][0], dtype=np.int8),
        "mask_gaia_ztf_pass": np.asarray(scenario_masks["gaia_ztf"][1], dtype=np.int8),
        "mask_gaia_ztf_ps1_present": np.asarray(scenario_masks["gaia_ztf_ps1"][0], dtype=np.int8),
        "mask_gaia_ztf_ps1_pass": np.asarray(scenario_masks["gaia_ztf_ps1"][1], dtype=np.int8),
        "mask_gaia_ztf_ps1_sdss_present": np.asarray(scenario_masks["gaia_ztf_ps1_sdss"][0], dtype=np.int8),
        "mask_gaia_ztf_ps1_sdss_pass": np.asarray(scenario_masks["gaia_ztf_ps1_sdss"][1], dtype=np.int8),
    }

    # Exact combined timing diagnostics across all available per-band windows.
    n_cand = int(candidate_source_id.size)
    all_time_bands = list(per_band_arrays.keys())
    tmin_mat = np.full((n_cand, len(all_time_bands)), np.nan, dtype=np.float64)
    tmax_mat = np.full((n_cand, len(all_time_bands)), np.nan, dtype=np.float64)
    for bi, b in enumerate(all_time_bands):
        tmin_mat[:, bi] = np.asarray(per_band_arrays[b]["tmin_obs"], dtype=np.float64)
        tmax_mat[:, bi] = np.asarray(per_band_arrays[b]["tmax_obs"], dtype=np.float64)
    valid_any = np.any(np.isfinite(tmin_mat) & np.isfinite(tmax_mat), axis=1)
    combined_tmin_obs = np.full(n_cand, np.nan, dtype=np.float64)
    combined_tmax_obs = np.full(n_cand, np.nan, dtype=np.float64)
    if np.any(valid_any):
        tmp_min = np.where(np.isfinite(tmin_mat), tmin_mat, np.inf)
        tmp_max = np.where(np.isfinite(tmax_mat), tmax_mat, -np.inf)
        combined_tmin_obs[valid_any] = np.min(tmp_min[valid_any], axis=1)
        combined_tmax_obs[valid_any] = np.max(tmp_max[valid_any], axis=1)
    combined_baseline_obs = np.where(
        np.isfinite(combined_tmin_obs) & np.isfinite(combined_tmax_obs),
        combined_tmax_obs - combined_tmin_obs,
        np.nan,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        combined_baseline_rest = combined_baseline_obs / (1.0 + candidate_z.astype(np.float64))
    combined_coverage_obs_nogap, combined_largest_gap_obs = _union_coverage_and_largest_gap(tmin_mat, tmax_mat)
    with np.errstate(divide="ignore", invalid="ignore"):
        combined_coverage_rest_nogap = combined_coverage_obs_nogap / (1.0 + candidate_z.astype(np.float64))
        combined_largest_gap_rest = combined_largest_gap_obs / (1.0 + candidate_z.astype(np.float64))

    # Survey-family windows: combine ZTF (g+r) versus Gaia for inter-survey gap metrics.
    ztf_idx = [i for i, b in enumerate(all_time_bands) if b.startswith("ztf_")]
    gaia_idx = [i for i, b in enumerate(all_time_bands) if b == "gaia_g"]
    ztf_tmin_obs = np.full(n_cand, np.nan, dtype=np.float64)
    ztf_tmax_obs = np.full(n_cand, np.nan, dtype=np.float64)
    gaia_tmin_obs = np.full(n_cand, np.nan, dtype=np.float64)
    gaia_tmax_obs = np.full(n_cand, np.nan, dtype=np.float64)
    if ztf_idx:
        ztf_min_mat = tmin_mat[:, ztf_idx]
        ztf_max_mat = tmax_mat[:, ztf_idx]
        m_ztf = np.any(np.isfinite(ztf_min_mat) & np.isfinite(ztf_max_mat), axis=1)
        if np.any(m_ztf):
            ztf_tmin_obs[m_ztf] = np.min(np.where(np.isfinite(ztf_min_mat[m_ztf]), ztf_min_mat[m_ztf], np.inf), axis=1)
            ztf_tmax_obs[m_ztf] = np.max(np.where(np.isfinite(ztf_max_mat[m_ztf]), ztf_max_mat[m_ztf], -np.inf), axis=1)
    if gaia_idx:
        gi = gaia_idx[0]
        gaia_tmin_obs = np.asarray(tmin_mat[:, gi], dtype=np.float64)
        gaia_tmax_obs = np.asarray(tmax_mat[:, gi], dtype=np.float64)
    gap_obs_ztf_gaia = np.full(n_cand, np.nan, dtype=np.float64)
    gap_obs_gaia_minus_ztf = np.full(n_cand, np.nan, dtype=np.float64)
    m_gap = np.isfinite(ztf_tmin_obs) & np.isfinite(ztf_tmax_obs) & np.isfinite(gaia_tmin_obs) & np.isfinite(gaia_tmax_obs)
    if np.any(m_gap):
        gap_obs_gaia_minus_ztf[m_gap] = gaia_tmin_obs[m_gap] - ztf_tmax_obs[m_gap]
        gap_obs_ztf_minus_gaia = ztf_tmin_obs[m_gap] - gaia_tmax_obs[m_gap]
        gap_obs_ztf_gaia[m_gap] = np.maximum(0.0, np.maximum(gap_obs_gaia_minus_ztf[m_gap], gap_obs_ztf_minus_gaia))
    with np.errstate(divide="ignore", invalid="ignore"):
        gap_rest_ztf_gaia = gap_obs_ztf_gaia / (1.0 + candidate_z.astype(np.float64))
        gap_rest_gaia_minus_ztf = gap_obs_gaia_minus_ztf / (1.0 + candidate_z.astype(np.float64))
    for b in per_band_arrays.keys():
        # Per-band summaries are aligned to candidate_source_id.
        for key in ("n_epochs", "snr", "mag_med", "baseline_obs", "pass", "tmin_obs", "tmax_obs"):
            save[f"{b}_{key}"] = per_band_arrays[b][key]
        with np.errstate(divide="ignore", invalid="ignore"):
            save[f"{b}_baseline_rest"] = per_band_arrays[b]["baseline_obs"] / (1.0 + candidate_z.astype(np.float64))
    save["combined_tmin_obs_allbands"] = combined_tmin_obs
    save["combined_tmax_obs_allbands"] = combined_tmax_obs
    save["combined_baseline_obs_allbands"] = combined_baseline_obs
    save["combined_baseline_rest_allbands"] = combined_baseline_rest
    save["combined_coverage_obs_allbands_nogap"] = combined_coverage_obs_nogap
    save["combined_coverage_rest_allbands_nogap"] = combined_coverage_rest_nogap
    save["largest_gap_obs_allbands"] = combined_largest_gap_obs
    save["largest_gap_rest_allbands"] = combined_largest_gap_rest
    save["ztf_tmin_obs"] = ztf_tmin_obs
    save["ztf_tmax_obs"] = ztf_tmax_obs
    save["gaia_tmin_obs"] = gaia_tmin_obs
    save["gaia_tmax_obs"] = gaia_tmax_obs
    save["gap_obs_ztf_gaia"] = gap_obs_ztf_gaia
    save["gap_rest_ztf_gaia"] = gap_rest_ztf_gaia
    save["gap_obs_gaia_minus_ztf"] = gap_obs_gaia_minus_ztf
    save["gap_rest_gaia_minus_ztf"] = gap_rest_gaia_minus_ztf

    print(f"Writing NPZ: {args.output}")
    np.savez(args.output, **save)


if __name__ == "__main__":
    main()
