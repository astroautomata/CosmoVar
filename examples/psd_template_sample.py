#!/usr/bin/env python3
"""Select Gaia templates for the Section 5.1 PSD tests.

- representative: draw a random sample from the eligible Gaia sources.
- common_support: sample redshift–luminosity cells within the luminosity
  range shared by all redshift bins.
- Assign injected variability rates and damping times, keeping them
  consistent for sources appearing in both samples.
- Save the template manifest as NPZ and CSV, with settings in JSON.

Inputs are the observed Gaia light curves and Prior 1/5 Stage 1 fits.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
import warnings
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
MAG_ERR_FACTOR = 2.5 / np.log(10.0)


def _load_analysis_lookup(path):
    with np.load(path) as d, warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sid, z, ell = d["source_id"], d["redsh"], d["logL3000"]
        mwin = np.nanmedian(d["posterior_m_win_cont_Gaia_G"], axis=1)
    good = np.isfinite(z) & np.isfinite(ell) & np.isfinite(mwin) & (mwin <= 20.0)
    return {
        int(s): (float(zz), float(ll), float(mm))
        for s, zz, ll, mm in zip(sid[good], z[good], ell[good], mwin[good])
    }


def _load_empirical_d(path):
    with np.load(path) as d, warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ids = d["source_id"]
        rates = np.nanmedian(d["posterior_ln_D_Gaia_G"], axis=1) / np.log(10.0)
    return {int(s): float(v) for s, v in zip(ids, rates) if np.isfinite(v)}


def load_final_parent(
    lc_fits: Path,
    selection_npz: Path,
    prior5_cache: Path,
    clagn_fits: Path,
    z_min: float,
    z_max: float,
    min_epochs: int,
    max_rows: int | None = None,
) -> dict[str, np.ndarray]:
    analysis = _load_analysis_lookup(selection_npz)
    empirical_d = _load_empirical_d(prior5_cache)
    excluded = (set(np.asarray(Table.read(clagn_fits)["source_id"], dtype=np.int64).tolist())
                if clagn_fits is not None else set())

    rows: dict[str, list[Any]] = {
        "source_id": [], "z": [], "ell": [], "mwin": [], "g_med": [],
        "n_epoch": [], "baseline_rf": [], "median_mag_err": [], "log10D_empirical_prior5": [],
    }
    with fits.open(lc_fits, memmap=True) as hdul:
        data = hdul[1].data
        n_rows = len(data) if max_rows is None else min(len(data), int(max_rows))
        for i in range(n_rows):
            sid = int(data["source_id"][i])
            if sid in excluded or sid not in analysis:
                continue
            z_lookup, ell, mwin = analysis[sid]
            z = float(data["z"][i])
            if not (np.isfinite(z) and z_min <= z <= z_max):
                continue
            if abs(z - z_lookup) > 1e-8:
                raise RuntimeError(f"redshift mismatch for source {sid}: FITS={z}, selection={z_lookup}")
            t = np.asarray(data["g_transit_time"][i], dtype=float)
            fl = np.asarray(data["g_transit_flux"][i], dtype=float)
            fe = np.asarray(data["g_transit_flux_error"][i], dtype=float)
            mag = np.asarray(data["g_transit_mag"][i], dtype=float)
            good = np.isfinite(t) & np.isfinite(fl) & np.isfinite(fe) & np.isfinite(mag)
            good &= (fl > 0.0) & (fe > 0.0) & (mag > 0.0)
            if "variability_flag_g_reject" in data.dtype.names:
                good &= np.asarray(data["variability_flag_g_reject"][i]) == 0
            if int(good.sum()) < min_epochs:
                continue
            t, fl, fe, mag = t[good], fl[good], fe[good], mag[good]
            order = np.argsort(t)
            t, fl, fe, mag = t[order], fl[order], fe[order], mag[order]
            merr = np.clip(MAG_ERR_FACTOR * fe / fl, 1e-4, 1.0)
            rows["source_id"].append(sid)
            rows["z"].append(z)
            rows["ell"].append(ell)
            rows["mwin"].append(mwin)
            rows["g_med"].append(float(np.median(mag)))
            rows["n_epoch"].append(int(t.size))
            rows["baseline_rf"].append(float(np.ptp(t) / (1.0 + z)))
            rows["median_mag_err"].append(float(np.median(merr)))
            rows["log10D_empirical_prior5"].append(empirical_d.get(sid, np.nan))

    out = {key: np.asarray(value) for key, value in rows.items()}
    out["source_id"] = out["source_id"].astype(np.int64)
    out["n_epoch"] = out["n_epoch"].astype(np.int32)
    if out["source_id"].size == 0:
        raise RuntimeError("final manuscript parent is empty")
    return out


def _row_subset(parent: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[idx] for key, value in parent.items()}


def select_representative(
    parent: dict[str, np.ndarray], n: int, rng: np.random.Generator,
    g_edges: np.ndarray, n_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    total = int(parent["source_id"].size)
    n = min(int(n), total)
    idx = np.sort(rng.choice(total, size=n, replace=False))
    out = _row_subset(parent, idx)
    out["selection_probability"] = np.full(n, n / total, dtype=float)
    out["selection_weight"] = np.full(n, total / n, dtype=float)
    out["z_bin"] = np.full(n, -1, dtype=np.int16)
    out["ell_bin"] = np.full(n, -1, dtype=np.int16)
    out["g_bin"] = np.digitize(out["g_med"], g_edges).astype(np.int16)
    out["n_bin"] = np.digitize(out["n_epoch"], n_edges).astype(np.int16)
    return out


def common_luminosity_support(parent: dict[str, np.ndarray], z_edges: np.ndarray, q: float) -> tuple[float, float, list[dict[str, float]]]:
    details: list[dict[str, float]] = []
    lows, highs = [], []
    z = parent["z"]
    ell = parent["ell"]
    for iz, (lo, hi) in enumerate(zip(z_edges[:-1], z_edges[1:])):
        use = (z >= lo) & ((z < hi) if iz < len(z_edges) - 2 else (z <= hi))
        x = ell[use]
        if x.size == 0:
            raise RuntimeError(f"empty redshift bin {lo}..{hi}")
        low = float(np.quantile(x, q))
        high = float(np.quantile(x, 1.0 - q))
        lows.append(low)
        highs.append(high)
        details.append({"z_low": float(lo), "z_high": float(hi), "n": int(x.size), "ell_q_low": low, "ell_q_high": high})
    # Keep the luminosity range shared by all redshift bins.
    common_low, common_high = max(lows), min(highs)
    if not common_low < common_high:
        raise RuntimeError(f"no common luminosity support at quantile {q}: {common_low} >= {common_high}")
    return common_low, common_high, details


def _allocate_quality_quotas(min_counts: np.ndarray, target: int) -> np.ndarray:
    capacity = int(min_counts.sum())
    if capacity < target:
        raise ValueError(f"quality-stratified capacity {capacity} is below requested {target}")
    raw = target * min_counts.astype(float) / capacity
    quota = np.minimum(np.floor(raw).astype(int), min_counts)
    remainder = int(target - quota.sum())
    priority = np.argsort(-(raw - quota).ravel())
    for flat in priority:
        if remainder == 0:
            break
        cell = np.unravel_index(int(flat), quota.shape)
        if quota[cell] < min_counts[cell]:
            quota[cell] += 1
            remainder -= 1
    if remainder:
        raise RuntimeError(f"could not allocate {remainder} quality-stratified slots")
    return quota


def select_common_support(
    parent: dict[str, np.ndarray],
    z_edges: np.ndarray,
    ell_edges: np.ndarray,
    g_edges: np.ndarray,
    n_edges: np.ndarray,
    n_per_z_ell: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    z, ell = parent["z"], parent["ell"]
    in_support = (ell >= ell_edges[0]) & (ell <= ell_edges[-1])
    zb = np.digitize(z, z_edges[1:-1])
    lb = np.digitize(ell, ell_edges[1:-1])
    gb = np.digitize(parent["g_med"], g_edges)
    nb = np.digitize(parent["n_epoch"], n_edges)
    nz, nl, ng, nn = len(z_edges) - 1, len(ell_edges) - 1, len(g_edges) + 1, len(n_edges) + 1
    chosen: list[int] = []
    probabilities: list[float] = []
    cell_meta: list[dict[str, Any]] = []

    for il in range(nl):
        available = np.zeros((nz, ng, nn), dtype=int)
        indices: dict[tuple[int, int, int], np.ndarray] = {}
        for iz in range(nz):
            for ig in range(ng):
                for jn in range(nn):
                    idx = np.flatnonzero(in_support & (zb == iz) & (lb == il) & (gb == ig) & (nb == jn))
                    indices[(iz, ig, jn)] = idx
                    available[iz, ig, jn] = idx.size
        quotas = np.zeros_like(available)
        for iz in range(nz):
            effective_target = min(int(n_per_z_ell), int(available[iz].sum()))
            if effective_target <= 0:
                raise RuntimeError(f"empty z x luminosity cell: z bin {iz}, luminosity bin {il}")
            # Sample brightness/epoch bins in proportion to their available sources.
            quota = _allocate_quality_quotas(available[iz], effective_target)
            quotas[iz] = quota
            for ig in range(ng):
                for jn in range(nn):
                    take = int(quota[ig, jn])
                    if take == 0:
                        continue
                    candidates = indices[(iz, ig, jn)]
                    picked = rng.choice(candidates, size=take, replace=False)
                    chosen.extend(int(x) for x in picked)
                    probabilities.extend([take / candidates.size] * take)
        cell_meta.append({
            "ell_bin": il,
            "ell_low": float(ell_edges[il]),
            "ell_high": float(ell_edges[il + 1]),
            "available_by_z_g_n": available.tolist(),
            "proportional_quota_by_z_g_n": quotas.tolist(),
            "requested_per_z": int(n_per_z_ell),
            "available_per_z": available.sum(axis=(1, 2)).tolist(),
            "selected_per_z": quotas.sum(axis=(1, 2)).tolist(),
        })

    idx = np.asarray(chosen, dtype=int)
    order = np.lexsort((parent["source_id"][idx], ell[idx], z[idx]))
    idx = idx[order]
    prob = np.asarray(probabilities, dtype=float)[order]
    out = _row_subset(parent, idx)
    out["selection_probability"] = prob
    out["selection_weight"] = 1.0 / prob
    out["z_bin"] = np.digitize(out["z"], z_edges[1:-1]).astype(np.int16)
    out["ell_bin"] = np.digitize(out["ell"], ell_edges[1:-1]).astype(np.int16)
    out["g_bin"] = np.digitize(out["g_med"], g_edges).astype(np.int16)
    out["n_bin"] = np.digitize(out["n_epoch"], n_edges).astype(np.int16)
    meta = {"cells": cell_meta, "n_selected": int(idx.size)}
    return out, meta


def _stable_source_draws(source_ids: np.ndarray, seed: int, sigma: float) -> dict[int, float]:
    out: dict[int, float] = {}
    for sid in np.unique(np.asarray(source_ids, dtype=np.int64)):
        ss = np.random.SeedSequence([int(seed), int(sid & np.int64(0xFFFFFFFF)), int(sid >> np.int64(32))])
        out[int(sid)] = float(np.random.default_rng(ss).normal(0.0, sigma))
    return out


def _stable_tau_draws(source_ids: np.ndarray, seed: int, mu: float, sigma: float, low: float, high: float) -> dict[int, float]:
    out: dict[int, float] = {}
    for sid in np.unique(np.asarray(source_ids, dtype=np.int64)):
        ss = np.random.SeedSequence([int(seed), 99173, int(sid & np.int64(0xFFFFFFFF)), int(sid >> np.int64(32))])
        rng = np.random.default_rng(ss)
        while True:
            tau = float(10.0 ** rng.normal(mu, sigma))
            if low < tau < high:
                out[int(sid)] = tau
                break
    return out


def attach_population_truth(
    cohorts: dict[str, dict[str, np.ndarray]], args: argparse.Namespace,
) -> dict[str, Any]:
    union_sid = np.unique(np.concatenate([c["source_id"] for c in cohorts.values()]))
    sid_to_z: dict[int, float] = {}
    for cohort in cohorts.values():
        sid_to_z.update({int(s): float(z) for s, z in zip(cohort["source_id"], cohort["z"])})
    z_union = np.asarray([sid_to_z[int(s)] for s in union_sid], dtype=float)

    from framework.generate_closure_population import _prepare_sed_band_terms

    sed = _prepare_sed_band_terms(
        stage1_band="gaia_g", sed_grid=str(args.sed_grid), passband_file=str(args.passband_file),
        ztf_filter_dir=str(ROOT / "assets"), n_integration_wavelength=args.n_integration_wavelength, chunk_size=25000,
        z=z_union, lambda_rf_fiducial_logl3000=45.0,
    )
    lambda_by_sid = {int(s): float(v) for s, v in zip(union_sid, sed["lambda_rf_eff"])}
    ratio_by_sid = {int(s): float(v) for s, v in zip(union_sid, sed["log10_lambda_ratio"])}
    # Shared sources receive the same injected parameters in both samples.
    eps_by_sid = _stable_source_draws(union_sid, args.seed + 2001, args.sigma_int)
    tau_by_sid = _stable_tau_draws(union_sid, args.seed + 3001, np.log10(750.0), 0.35, 30.0, 3000.0)
    for cohort in cohorts.values():
        ratio = np.asarray([ratio_by_sid[int(s)] for s in cohort["source_id"]], dtype=float)
        eps = np.asarray([eps_by_sid[int(s)] for s in cohort["source_id"]], dtype=float)
        mean = args.D_pivot + args.d_D * (cohort["ell"] - args.ell_pivot) - args.alpha_D * ratio
        cohort["lambda_rf_eff"] = np.asarray([lambda_by_sid[int(s)] for s in cohort["source_id"]], dtype=float)
        cohort["log10_lambda_ratio"] = ratio
        cohort["log10D_population_mean"] = mean
        cohort["log10D_population"] = mean + eps
        cohort["tau_true"] = np.asarray([tau_by_sid[int(s)] for s in cohort["source_id"]], dtype=float)
    return {
        "D_pivot": args.D_pivot, "d_D": args.d_D, "alpha_D": args.alpha_D,
        "ell_pivot": args.ell_pivot, "sigma_int": args.sigma_int,
        "tau_distribution": "log10(tau/day) Normal(log10(750),0.35), truncated to 30..3000 days",
        "sed_grid": str(args.sed_grid), "passband_file": str(args.passband_file),
    }


def write_manifest(cohorts: dict[str, dict[str, np.ndarray]], out_dir: Path, metadata: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_id", "z", "ell", "mwin", "g_med", "n_epoch", "baseline_rf", "median_mag_err",
        "z_bin", "ell_bin", "g_bin", "n_bin", "selection_probability", "selection_weight",
        "lambda_rf_eff", "log10_lambda_ratio", "log10D_population_mean", "log10D_population",
        "log10D_empirical_prior5", "tau_true",
    ]
    packed: dict[str, list[np.ndarray]] = {field: [] for field in fields}
    cohort_col: list[np.ndarray] = []
    for name, cohort in cohorts.items():
        cohort_col.append(np.full(cohort["source_id"].size, name))
        for field in fields:
            packed[field].append(np.asarray(cohort[field]))
    arrays = {field: np.concatenate(parts) for field, parts in packed.items()}
    arrays["cohort"] = np.concatenate(cohort_col)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez(out_dir / "psd_template_manifest.npz", **arrays)

    with (out_dir / "psd_template_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cohort", *fields])
        for i in range(arrays["cohort"].size):
            writer.writerow([arrays["cohort"][i], *[arrays[field][i] for field in fields]])
    (out_dir / "psd_template_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--lc-fits", type=Path, required=True)
    ap.add_argument("--observed-prior1", type=Path, required=True)
    ap.add_argument("--observed-prior5", type=Path, required=True)
    ap.add_argument("--veto-fits", type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--representative-n", type=int, default=8200)
    ap.add_argument("--n-per-z-luminosity", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20261030)
    args = ap.parse_args()
    parent = load_final_parent(
        args.lc_fits, args.observed_prior1, args.observed_prior5,
        args.veto_fits, .5, 3.5, 10,
    )
    z_edges = np.asarray((.5, 1.1, 1.7, 2.3, 2.9, 3.5))
    g_edges = np.asarray((19., 19.8, 20.3))
    n_edges = np.asarray((29., 40., 52.))
    low, high, _ = common_luminosity_support(parent, z_edges, .0025)
    ell_edges = np.linspace(low, high, 3)
    common, meta = select_common_support(
        parent, z_edges, ell_edges, g_edges, n_edges,
        args.n_per_z_luminosity, np.random.default_rng(args.seed + 11),
    )
    representative = select_representative(
        parent, args.representative_n, np.random.default_rng(args.seed + 5),
        g_edges, n_edges,
    )
    cohorts = {"representative": representative, "common_support": common}
    args.sed_grid = ROOT / "assets/sed/quasar_continuum_and_lines.npz"
    args.passband_file = ROOT / "assets/passbands/gaia_g.dat"
    args.n_integration_wavelength = 1000
    args.D_pivot = -4.49
    args.d_D = -.85
    args.alpha_D = 2.51
    args.ell_pivot = 45.5
    args.sigma_int = .12
    truth = attach_population_truth(cohorts, args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "random_seed": args.seed, "common_support": meta,
        "injected_population_relation": truth,
        "cohorts": {k: {"n": len(v["source_id"])} for k, v in cohorts.items()},
    }
    write_manifest(cohorts, args.output_dir, metadata)


if __name__ == "__main__":
    main()
