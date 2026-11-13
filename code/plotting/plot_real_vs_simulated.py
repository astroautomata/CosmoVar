#!/usr/bin/env python3
"""Compare observed and simulated Stage-1 fits for Figure 4.

Summarise the retained sources in redshift and luminosity using posterior
window magnitudes and variability rates. Luminosities use the fixed plotting
cosmology in photometry.py; Stage-2 cosmological inference is a separate fit.
The optional bootstrap modes propagate source sampling and posterior draws
into the binned curves."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from plotting.diagnostics import (
    DEFAULT_CLOSURE_DIR,
    DEFAULT_RUN_NAME,
    DEFAULT_SAMPLE_LABEL,
    _default_pair_npz,
    _default_stage1_npz,
    _load_pair_rows,
    _load_stage1_posterior_row,
)
from plotting.diagnostics import _write_selected_page


LN10 = float(np.log(10.0))
POSTPROP_BOOTSTRAP_CACHE_VERSION = 1


def _default_output_pdf(closure_dir: Path, sample_label: str) -> Path:
    return closure_dir / f"{sample_label}_selected_2x4_real_mwin_vs_mock_stage1post_presentation.pdf"


def _parse_columns(text: str) -> tuple[int, ...]:
    cols = tuple(int(part.strip()) for part in str(text).split(",") if part.strip())
    allowed = {0, 1, 2, 3}
    if not cols or any(col not in allowed for col in cols):
        raise argparse.ArgumentTypeError("columns must be a comma-separated subset of 0,1,2,3")
    return cols


def _parse_column_cmaps(text: str) -> dict[int, str]:
    cmaps: dict[int, str] = {}
    allowed = {0, 1, 2, 3}
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise argparse.ArgumentTypeError(
                "column colormaps must use COL:CMAP entries, e.g. 0:plasma,1:plasma"
            )
        col_text, cmap = part.split(":", 1)
        col = int(col_text.strip())
        cmap = cmap.strip()
        if col not in allowed or not cmap:
            raise argparse.ArgumentTypeError(
                "column colormaps must be a comma-separated subset of 0:CMAP,1:CMAP,2:CMAP,3:CMAP"
            )
        cmaps[col] = cmap
    return cmaps


def _parse_float_range(text: str) -> tuple[float, float]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be two comma-separated values, e.g. 0.5,3.5")
    lo, hi = (float(parts[0]), float(parts[1]))
    if hi <= lo:
        raise argparse.ArgumentTypeError("range upper bound must be larger than lower bound")
    return (lo, hi)


def _append_label(existing: str | None, addition: str) -> str:
    if existing:
        return f"{existing}\n{addition}"
    return addition


def _apply_row_mask(row: dict, mask: np.ndarray) -> None:
    n = int(mask.size)
    for key, value in list(row.items()):
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == n:
            row[key] = arr[mask]


def _apply_posterior_quality_cuts(
    row: dict,
    *,
    label: str,
    tau_min_days: float | None,
    mwin_max: float | None,
    z_min: float | None,
    z_max: float | None,
    log10d_halfwidth_max: float | None,
    annotate: bool = True,
) -> None:
    if tau_min_days is None and mwin_max is None and z_min is None and z_max is None and log10d_halfwidth_max is None:
        return

    n_total = int(np.asarray(row["z"]).size)
    mask = np.ones(n_total, dtype=bool)
    label_bits: list[str] = []

    if tau_min_days is not None:
        if "log10tau" not in row:
            raise KeyError(f"{label}: requested tau cut but row has no log10tau field")
        tau_min_log10 = float(np.log10(float(tau_min_days)))
        log10tau = np.asarray(row["log10tau"], dtype=np.float64)
        mask &= np.isfinite(log10tau) & (log10tau >= tau_min_log10)
        label_bits.append(rf"$\tau_{{50}}\ge{float(tau_min_days):g}$ d")

    if mwin_max is not None:
        if "m_win" not in row:
            raise KeyError(f"{label}: requested m_win cut but row has no m_win field")
        mwin = np.asarray(row["m_win"], dtype=np.float64)
        mask &= np.isfinite(mwin) & (mwin <= float(mwin_max))
        label_bits.append(rf"$m_{{50}}\le{float(mwin_max):g}$")

    if z_min is not None or z_max is not None:
        z = np.asarray(row["z"], dtype=np.float64)
        mask &= np.isfinite(z)
        if z_min is not None:
            mask &= z >= float(z_min)
        if z_max is not None:
            mask &= z <= float(z_max)
        if z_min is not None and z_max is not None:
            label_bits.append(rf"${float(z_min):g}\le z\le{float(z_max):g}$")
        elif z_min is not None:
            label_bits.append(rf"$z\ge{float(z_min):g}$")
        else:
            label_bits.append(rf"$z\le{float(z_max):g}$")

    if log10d_halfwidth_max is not None:
        if "log10d_halfwidth" not in row:
            raise KeyError(
                f"{label}: requested log10D halfwidth cut but row has no log10d_halfwidth field"
            )
        halfwidth = np.asarray(row["log10d_halfwidth"], dtype=np.float64)
        mask &= np.isfinite(halfwidth) & (halfwidth <= float(log10d_halfwidth_max))
        label_bits.append(rf"$\Delta_{{68}}\log_{{10}}D/2\le{float(log10d_halfwidth_max):g}$")

    n_kept = int(np.count_nonzero(mask))
    print(
        f"Applying posterior cuts to {label}: {n_kept:,}/{n_total:,} kept",
        flush=True,
    )
    _apply_row_mask(row, mask)
    if annotate:
        cut_text = "post. cuts: " + ", ".join(label_bits) + f"\nkept: {n_kept:,}/{n_total:,}"
        row["cut_label"] = _append_label(row.get("cut_label"), cut_text)


def _source_id_index(*, source_ids: np.ndarray, reference_source_ids: np.ndarray, name: str) -> np.ndarray | None:
    source_ids = np.asarray(source_ids, dtype=np.int64).reshape(-1)
    reference_source_ids = np.asarray(reference_source_ids, dtype=np.int64).reshape(-1)
    if np.array_equal(source_ids, reference_source_ids):
        return None
    order = np.argsort(source_ids)
    sorted_ids = source_ids[order]
    loc = np.searchsorted(sorted_ids, reference_source_ids)
    found = loc < sorted_ids.size
    found[found] &= sorted_ids[loc[found]] == reference_source_ids[found]
    if not np.all(found):
        missing = reference_source_ids[~found][:10]
        raise ValueError(f"{name}: missing {np.count_nonzero(~found)} source IDs; first missing={missing}")
    return order[loc]


def _postprop_edges(
    *,
    z_range: tuple[float, float],
    z_bin_width: float,
    logl_xlim: tuple[float, float],
    n_l_bins: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    z_lo, z_hi = map(float, z_range)
    dz = float(z_bin_width)
    if z_hi <= z_lo:
        raise ValueError(f"invalid redshift range: {z_range}")
    if dz <= 0.0:
        raise ValueError(f"invalid redshift bin width: {z_bin_width}")
    n_z_bins = int(np.ceil((z_hi - z_lo) / dz))
    z_edges = z_lo + dz * np.arange(n_z_bins + 1, dtype=np.float64)
    if z_edges[-1] < z_hi:
        z_edges = np.append(z_edges, z_hi)
    else:
        z_edges[-1] = z_hi
    logl_edges = np.linspace(float(logl_xlim[0]), float(logl_xlim[1]), int(n_l_bins) + 1)
    return z_edges, logl_edges


def _load_postprop_bootstrap_cache(
    *,
    cache_npz: Path | None,
    stage1_npz: Path,
    reference_source_ids: np.ndarray,
    band: str,
    z_edges: np.ndarray,
    logl_edges: np.ndarray,
    eta_ylim: tuple[float, float],
    n_bootstrap: int,
    min_count: int,
    seed: int,
) -> dict[str, np.ndarray] | None:
    if cache_npz is None or not cache_npz.exists():
        return None
    try:
        stat = stage1_npz.stat()
        with np.load(cache_npz, allow_pickle=False) as cache:
            required = {
                "cache_version",
                "stage1_npz",
                "stage1_size",
                "stage1_mtime_ns",
                "reference_source_id",
                "band",
                "z_edges",
                "logl_edges",
                "eta_ylim",
                "n_bootstrap",
                "min_count",
                "seed",
                "q16",
                "q50",
                "q84",
                "count",
            }
            missing = sorted(required - set(cache.files))
            if missing:
                raise ValueError(f"missing keys {missing}")
            if int(cache["cache_version"]) != POSTPROP_BOOTSTRAP_CACHE_VERSION:
                raise ValueError("cache version mismatch")
            if str(np.asarray(cache["stage1_npz"]).item()) != str(stage1_npz.resolve()):
                raise ValueError("stage1 path mismatch")
            if int(cache["stage1_size"]) != int(stat.st_size):
                raise ValueError("stage1 file size changed")
            if int(cache["stage1_mtime_ns"]) != int(stat.st_mtime_ns):
                raise ValueError("stage1 file mtime changed")
            if str(np.asarray(cache["band"]).item()) != str(band):
                raise ValueError("band mismatch")
            if not np.array_equal(np.asarray(cache["reference_source_id"], dtype=np.int64), reference_source_ids):
                raise ValueError("reference source IDs changed")
            if not np.allclose(np.asarray(cache["z_edges"], dtype=float), z_edges, rtol=0.0, atol=0.0):
                raise ValueError("z edges changed")
            if not np.allclose(np.asarray(cache["logl_edges"], dtype=float), logl_edges, rtol=0.0, atol=0.0):
                raise ValueError("logL edges changed")
            if not np.allclose(np.asarray(cache["eta_ylim"], dtype=float), eta_ylim, rtol=0.0, atol=0.0):
                raise ValueError("eta limits changed")
            if int(cache["n_bootstrap"]) != int(n_bootstrap):
                raise ValueError("bootstrap count changed")
            if int(cache["min_count"]) != int(min_count):
                raise ValueError("minimum count changed")
            if int(cache["seed"]) != int(seed):
                raise ValueError("seed changed")
            print(f"Using cached posterior-propagated bootstrap medians: {cache_npz}", flush=True)
            return {
                "q16": np.asarray(cache["q16"], dtype=np.float64),
                "q50": np.asarray(cache["q50"], dtype=np.float64),
                "q84": np.asarray(cache["q84"], dtype=np.float64),
                "count": np.asarray(cache["count"], dtype=np.int64),
            }
    except Exception as exc:
        print(f"[cache] Ignoring stale/unusable postprop bootstrap cache {cache_npz}: {exc}", flush=True)
        return None


def _write_postprop_bootstrap_cache(
    *,
    cache_npz: Path | None,
    stage1_npz: Path,
    reference_source_ids: np.ndarray,
    band: str,
    z_edges: np.ndarray,
    logl_edges: np.ndarray,
    eta_ylim: tuple[float, float],
    n_bootstrap: int,
    min_count: int,
    seed: int,
    result: dict[str, np.ndarray],
) -> None:
    if cache_npz is None:
        return
    try:
        stat = stage1_npz.stat()
        cache_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_npz,
            cache_version=np.asarray(POSTPROP_BOOTSTRAP_CACHE_VERSION, dtype=np.int16),
            stage1_npz=np.asarray(str(stage1_npz.resolve())),
            stage1_size=np.asarray(int(stat.st_size), dtype=np.int64),
            stage1_mtime_ns=np.asarray(int(stat.st_mtime_ns), dtype=np.int64),
            reference_source_id=np.asarray(reference_source_ids, dtype=np.int64),
            band=np.asarray(str(band)),
            z_edges=np.asarray(z_edges, dtype=np.float64),
            logl_edges=np.asarray(logl_edges, dtype=np.float64),
            eta_ylim=np.asarray(eta_ylim, dtype=np.float64),
            n_bootstrap=np.asarray(int(n_bootstrap), dtype=np.int64),
            min_count=np.asarray(int(min_count), dtype=np.int64),
            seed=np.asarray(int(seed), dtype=np.int64),
            q16=np.asarray(result["q16"], dtype=np.float64),
            q50=np.asarray(result["q50"], dtype=np.float64),
            q84=np.asarray(result["q84"], dtype=np.float64),
            count=np.asarray(result["count"], dtype=np.int64),
        )
        print(f"Cached posterior-propagated bootstrap medians: {cache_npz}", flush=True)
    except Exception as exc:
        print(f"[cache] Could not write postprop bootstrap cache {cache_npz}: {exc}", flush=True)


def _compute_postprop_bootstrap(
    *,
    row: dict,
    stage1_npz: Path,
    reference_source_ids: np.ndarray,
    band: str,
    z_edges: np.ndarray,
    logl_edges: np.ndarray,
    eta_ylim: tuple[float, float],
    n_bootstrap: int,
    min_count: int,
    seed: int,
    label: str,
) -> dict[str, np.ndarray]:
    """Bootstrap binned median log10(D), keeping each source's luminosity bin fixed.
    
    Within each visible bin, resample sources and pick one posterior rate draw for
    each occurrence. Return the 16th/50th/84th percentiles of the bootstrap medians.
    This mode propagates rate uncertainty while using the row's fixed luminosities."""
    key_ln_d = f"posterior_ln_D_{band}"
    reference_source_ids = np.asarray(reference_source_ids, dtype=np.int64).reshape(-1)
    z = np.asarray(row["z"], dtype=np.float64)
    logl = np.asarray(row["logl3000"], dtype=np.float64)
    eta_med = np.asarray(row["log10d"], dtype=np.float64)
    if z.shape[0] != reference_source_ids.size:
        raise ValueError(f"{label}: row/source_id length mismatch")

    n_z_bins = int(z_edges.size - 1)
    n_l_bins = int(logl_edges.size - 1)
    shape = (n_z_bins, n_l_bins)
    q16 = np.full(shape, np.nan, dtype=np.float64)
    q50 = np.full(shape, np.nan, dtype=np.float64)
    q84 = np.full(shape, np.nan, dtype=np.float64)
    count = np.zeros(shape, dtype=np.int64)

    visible = np.isfinite(z) & np.isfinite(logl) & np.isfinite(eta_med)
    visible &= (logl >= logl_edges[0]) & (logl <= logl_edges[-1])
    visible &= (eta_med >= float(eta_ylim[0])) & (eta_med <= float(eta_ylim[1]))
    visible_idx = np.where(visible)[0]
    if visible_idx.size == 0:
        return {"q16": q16, "q50": q50, "q84": q84, "count": count}

    print(
        f"Loading {label} posterior eta draws for postprop bootstrap: {stage1_npz}",
        flush=True,
    )
    with np.load(stage1_npz, allow_pickle=False) as post:
        keys = set(post.files)
        if key_ln_d not in keys or "source_id" not in keys:
            raise KeyError(f"{label}: Stage-1 NPZ missing {key_ln_d!r} or 'source_id'")
        post_source_id = np.asarray(post["source_id"], dtype=np.int64)
        row_idx = _source_id_index(
            source_ids=post_source_id,
            reference_source_ids=reference_source_ids,
            name=f"{label} {key_ln_d}",
        )
        ln_d_all = np.asarray(post[key_ln_d])
        if row_idx is None:
            log10d_draws = ln_d_all.astype(np.float32)
            del ln_d_all
            log10d_draws /= np.float32(LN10)
        else:
            selected = ln_d_all[row_idx]
            del ln_d_all
            log10d_draws = selected.astype(np.float32)
            del selected
            log10d_draws /= np.float32(LN10)

    n_draws = int(log10d_draws.shape[1])
    rng = np.random.default_rng(int(seed))
    n_done = 0
    for iz in range(n_z_bins):
        z_lo, z_hi = z_edges[iz], z_edges[iz + 1]
        z_mask = (z[visible_idx] >= z_lo) & (
            z[visible_idx] < z_hi if iz < n_z_bins - 1 else z[visible_idx] <= z_hi
        )
        z_idx = visible_idx[z_mask]
        if z_idx.size == 0:
            continue
        for il in range(n_l_bins):
            l_lo, l_hi = logl_edges[il], logl_edges[il + 1]
            l_mask = (logl[z_idx] >= l_lo) & (
                logl[z_idx] < l_hi if il < n_l_bins - 1 else logl[z_idx] <= l_hi
            )
            idx = z_idx[l_mask]
            n = int(idx.size)
            count[iz, il] = n
            if n < int(min_count):
                continue
            source_pick = rng.integers(0, n, size=(int(n_bootstrap), n), dtype=np.int64)
            draw_pick = rng.integers(0, n_draws, size=(int(n_bootstrap), n), dtype=np.int64)
            vals = log10d_draws[idx[source_pick], draw_pick]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
                med_boot = np.nanmedian(vals, axis=1)
                q16[iz, il], q50[iz, il], q84[iz, il] = np.nanpercentile(
                    med_boot,
                    [16.0, 50.0, 84.0],
                )
            n_done += 1
    print(
        f"Computed {label} postprop bootstrap intervals for {n_done} bins "
        f"(R={int(n_bootstrap)}, min_count={int(min_count)})",
        flush=True,
    )
    return {"q16": q16, "q50": q50, "q84": q84, "count": count}


def _compute_postprop_paired_bootstrap(
    *,
    row: dict,
    stage1_npz: Path,
    reference_source_ids: np.ndarray,
    band: str,
    z_edges: np.ndarray,
    logl_edges: np.ndarray,
    eta_ylim: tuple[float, float],
    n_bootstrap: int,
    min_count: int,
    seed: int,
    label: str,
) -> dict[str, np.ndarray]:
    """Bootstrap binned medians using paired magnitude and log10(D) draws.
    
    Resample sources from the finite redshift pool and use the same draw index for
    m_win and D in each occurrence. Convert the magnitude draw to luminosity with
    the row's fixed distance/SED offset, then reapply visibility cuts and rebin.
    Return bootstrap-median percentiles and bin counts; count_base describes the
    selection made from per-source medians before posterior-draw propagation."""
    key_ln_d = f"posterior_ln_D_{band}"
    key_mwin = f"posterior_m_win_cont_{band}"
    reference_source_ids = np.asarray(reference_source_ids, dtype=np.int64).reshape(-1)
    z = np.asarray(row["z"], dtype=np.float32)
    logl_med = np.asarray(row["logl3000"], dtype=np.float32)
    mwin_med = np.asarray(row["m_win"], dtype=np.float32)
    eta_med = np.asarray(row["log10d"], dtype=np.float32)
    if z.shape[0] != reference_source_ids.size:
        raise ValueError(f"{label}: row/source_id length mismatch")

    n_z_bins = int(z_edges.size - 1)
    n_l_bins = int(logl_edges.size - 1)
    n_bins = n_z_bins * n_l_bins
    shape = (n_z_bins, n_l_bins)
    count_base = np.zeros(shape, dtype=np.int64)
    med_boot = np.full((int(n_bootstrap), n_bins), np.nan, dtype=np.float32)
    count_boot = np.zeros((int(n_bootstrap), n_bins), dtype=np.int32)

    offset = logl_med + mwin_med / np.float32(2.5)
    finite_pool = np.isfinite(z) & np.isfinite(logl_med) & np.isfinite(mwin_med) & np.isfinite(eta_med) & np.isfinite(offset)
    finite_pool &= (z >= float(z_edges[0])) & (z <= float(z_edges[-1]))
    pool_idx = np.where(finite_pool)[0]
    if pool_idx.size == 0:
        return {
            "q16": np.full(shape, np.nan, dtype=np.float64),
            "q50": np.full(shape, np.nan, dtype=np.float64),
            "q84": np.full(shape, np.nan, dtype=np.float64),
            "count": count_base,
        }

    base_visible = finite_pool.copy()
    base_visible &= (logl_med >= float(logl_edges[0])) & (logl_med <= float(logl_edges[-1]))
    base_visible &= (eta_med >= float(eta_ylim[0])) & (eta_med <= float(eta_ylim[1]))
    base_idx = np.where(base_visible)[0]
    if base_idx.size:
        base_iz = np.searchsorted(z_edges, z[base_idx].astype(np.float64), side="right") - 1
        base_il = np.searchsorted(logl_edges, logl_med[base_idx].astype(np.float64), side="right") - 1
        base_iz = np.clip(base_iz, 0, n_z_bins - 1)
        base_il = np.clip(base_il, 0, n_l_bins - 1)
        flat = base_iz * n_l_bins + base_il
        count_base = np.bincount(flat, minlength=n_bins).reshape(shape).astype(np.int64)

    print(
        f"Loading {label} paired posterior eta/m_win draws for postprop bootstrap: {stage1_npz}",
        flush=True,
    )
    with np.load(stage1_npz, allow_pickle=False) as post:
        keys = set(post.files)
        if key_ln_d not in keys or key_mwin not in keys or "source_id" not in keys:
            raise KeyError(f"{label}: Stage-1 NPZ missing {key_ln_d!r}, {key_mwin!r}, or 'source_id'")
        post_source_id = np.asarray(post["source_id"], dtype=np.int64)
        row_idx = _source_id_index(
            source_ids=post_source_id,
            reference_source_ids=reference_source_ids,
            name=f"{label} paired posterior draws",
        )

        ln_d_all = np.asarray(post[key_ln_d])
        if row_idx is None:
            log10d_draws = ln_d_all.astype(np.float32)
            del ln_d_all
        else:
            selected = ln_d_all[row_idx]
            del ln_d_all
            log10d_draws = selected.astype(np.float32)
            del selected
        log10d_draws /= np.float32(LN10)

        mwin_all = np.asarray(post[key_mwin])
        if row_idx is None:
            mwin_draws = mwin_all.astype(np.float32)
            del mwin_all
        else:
            selected = mwin_all[row_idx]
            del mwin_all
            mwin_draws = selected.astype(np.float32)
            del selected

    n_pool = int(pool_idx.size)
    n_draws = int(min(log10d_draws.shape[1], mwin_draws.shape[1]))
    rng = np.random.default_rng(int(seed))
    z_edges64 = np.asarray(z_edges, dtype=np.float64)
    logl_edges64 = np.asarray(logl_edges, dtype=np.float64)
    eta_lo, eta_hi = map(float, eta_ylim)

    for r in range(int(n_bootstrap)):
        source_pick = pool_idx[rng.integers(0, n_pool, size=n_pool, dtype=np.int64)]
        draw_pick = rng.integers(0, n_draws, size=n_pool, dtype=np.int64)
        eta_draw = log10d_draws[source_pick, draw_pick]
        ell_draw = offset[source_pick] - mwin_draws[source_pick, draw_pick] / np.float32(2.5)
        z_draw = z[source_pick]
        ok = np.isfinite(eta_draw) & np.isfinite(ell_draw) & np.isfinite(z_draw)
        ok &= (eta_draw >= eta_lo) & (eta_draw <= eta_hi)
        ok &= (ell_draw >= float(logl_edges[0])) & (ell_draw <= float(logl_edges[-1]))
        ok &= (z_draw >= float(z_edges[0])) & (z_draw <= float(z_edges[-1]))
        if not np.any(ok):
            continue

        eta_ok = eta_draw[ok]
        ell_ok = ell_draw[ok]
        z_ok = z_draw[ok]
        iz = np.searchsorted(z_edges64, z_ok.astype(np.float64), side="right") - 1
        il = np.searchsorted(logl_edges64, ell_ok.astype(np.float64), side="right") - 1
        iz = np.clip(iz, 0, n_z_bins - 1)
        il = np.clip(il, 0, n_l_bins - 1)
        flat = iz * n_l_bins + il
        order = np.argsort(flat)
        flat_s = flat[order]
        eta_s = eta_ok[order]
        unique, start, cnt = np.unique(flat_s, return_index=True, return_counts=True)
        count_boot[r, unique] = cnt.astype(np.int32)
        for b, st, ct in zip(unique.tolist(), start.tolist(), cnt.tolist()):
            if int(ct) < int(min_count):
                continue
            med_boot[r, int(b)] = float(np.median(eta_s[st : st + ct]))

    valid_counts = np.count_nonzero(np.isfinite(med_boot), axis=0)
    q16_flat = np.full(n_bins, np.nan, dtype=np.float64)
    q50_flat = np.full(n_bins, np.nan, dtype=np.float64)
    q84_flat = np.full(n_bins, np.nan, dtype=np.float64)
    ok_bins = valid_counts >= max(10, int(0.1 * int(n_bootstrap)))
    if np.any(ok_bins):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
            q16_flat[ok_bins], q50_flat[ok_bins], q84_flat[ok_bins] = np.nanpercentile(
                med_boot[:, ok_bins].astype(np.float64),
                [16.0, 50.0, 84.0],
                axis=0,
            )

    print(
        f"Computed {label} paired postprop bootstrap intervals for {int(np.count_nonzero(ok_bins))} bins "
        f"(R={int(n_bootstrap)}, min_count={int(min_count)}, pool={n_pool:,})",
        flush=True,
    )
    return {
        "q16": q16_flat.reshape(shape),
        "q50": q50_flat.reshape(shape),
        "q84": q84_flat.reshape(shape),
        "count": np.nanmedian(count_boot.astype(np.float64), axis=0).reshape(shape).astype(np.int64),
        "count_base": count_base,
    }


def _get_postprop_bootstrap(
    *,
    row: dict,
    stage1_npz: Path,
    cache_npz: Path | None,
    band: str,
    z_range: tuple[float, float],
    z_bin_width: float,
    logl_xlim: tuple[float, float],
    eta_ylim: tuple[float, float],
    n_bootstrap: int,
    min_count: int,
    seed: int,
    label: str,
    paired: bool = False,
) -> dict[str, np.ndarray]:
    reference_source_ids = np.asarray(row["source_id"], dtype=np.int64)
    z_edges, logl_edges = _postprop_edges(
        z_range=z_range,
        z_bin_width=z_bin_width,
        logl_xlim=logl_xlim,
    )
    cached = _load_postprop_bootstrap_cache(
        cache_npz=cache_npz,
        stage1_npz=stage1_npz,
        reference_source_ids=reference_source_ids,
        band=band,
        z_edges=z_edges,
        logl_edges=logl_edges,
        eta_ylim=eta_ylim,
        n_bootstrap=n_bootstrap,
        min_count=min_count,
        seed=seed,
    )
    if cached is not None:
        return cached
    if paired:
        result = _compute_postprop_paired_bootstrap(
            row=row,
            stage1_npz=stage1_npz,
            reference_source_ids=reference_source_ids,
            band=band,
            z_edges=z_edges,
            logl_edges=logl_edges,
            eta_ylim=eta_ylim,
            n_bootstrap=n_bootstrap,
            min_count=min_count,
            seed=seed,
            label=label,
        )
    else:
        result = _compute_postprop_bootstrap(
            row=row,
            stage1_npz=stage1_npz,
            reference_source_ids=reference_source_ids,
            band=band,
            z_edges=z_edges,
            logl_edges=logl_edges,
            eta_ylim=eta_ylim,
            n_bootstrap=n_bootstrap,
            min_count=min_count,
            seed=seed,
            label=label,
        )
    _write_postprop_bootstrap_cache(
        cache_npz=cache_npz,
        stage1_npz=stage1_npz,
        reference_source_ids=reference_source_ids,
        band=band,
        z_edges=z_edges,
        logl_edges=logl_edges,
        eta_ylim=eta_ylim,
        n_bootstrap=n_bootstrap,
        min_count=min_count,
        seed=seed,
        result=result,
    )
    return result


def _pair_cut_labels(pair_npz: Path) -> tuple[str, str]:
    """Summarize the source-selection cuts represented by this paired sample."""
    with np.load(pair_npz, allow_pickle=False) as pair:
        real_z = np.asarray(pair["real_z"], dtype=np.float64)
        real_logl = np.asarray(pair["real_logl3000_from_mwin"], dtype=np.float64)
        real_mwin = np.asarray(pair["real_m_win_med"], dtype=np.float64)
        mock_z = np.asarray(pair["mock_z"], dtype=np.float64)
        mock_logl = np.asarray(pair["mock_logl3000"], dtype=np.float64)
        mock_mwin = np.asarray(pair["mock_m_win_cont_true"], dtype=np.float64)

    n_pair = int(real_z.size)
    real_mask = (
        np.isfinite(real_z)
        & np.isfinite(real_logl)
        & np.isfinite(real_mwin)
        & (real_z >= 0.5)
        & (real_z <= 3.5)
        & (real_mwin <= 20.0)
    )
    mock_truth_mask = (
        np.isfinite(mock_z)
        & np.isfinite(mock_logl)
        & np.isfinite(mock_mwin)
        & (mock_z >= 0.5)
        & (mock_z <= 3.5)
        & (mock_mwin <= 20.0)
    )

    real_label = (
        r"source cuts: $z=[0.5,3.5]$, $m\le20$, Gaia $n\ge10$" "\n"
        f"real pass: {int(np.count_nonzero(real_mask)):,}/{n_pair:,}"
    )
    mock_label = (
        "matched to real-cut IDs; no mock mag cut\n"
        rf"mock truth $m\le20$: {int(np.count_nonzero(mock_truth_mask)):,}/{n_pair:,}"
    )
    return real_label, mock_label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a compact full-closure presentation PDF using only row 2 "
            "(real from m_win) and row 4 (mock Stage-1 posterior medians)."
        )
    )
    parser.add_argument("--closure-dir", type=Path, default=DEFAULT_CLOSURE_DIR)
    parser.add_argument("--sample-label", default=DEFAULT_SAMPLE_LABEL)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--pair-npz", type=Path, default=None)
    parser.add_argument("--stage1-posteriors-npz", type=Path, default=None)
    parser.add_argument("--output-pdf", type=Path, default=None)
    parser.add_argument("--band", default="Gaia_G")
    parser.add_argument("--scenario-name", default="fullboxonly")
    parser.add_argument(
        "--selected-columns",
        type=_parse_columns,
        default=(0, 1, 2, 3),
        help=(
            "Comma-separated panel columns to draw. Default keeps all four: "
            "0=binned D-L, 1=D-L hex, 2=z-L hex, 3=z-mwin hex."
        ),
    )
    parser.add_argument(
        "--not-presentation",
        action="store_true",
        help="Use the regular plotting typography instead of the presentation style.",
    )
    parser.add_argument(
        "--show-hist-row",
        action="store_true",
        help="Append the compact distribution row below the two comparison rows.",
    )
    parser.add_argument(
        "--paper-row-labels",
        action="store_true",
        help=(
            "Use the row labels from the Stage-2 mock paper diagnostic page: "
            "'Real, with quality cuts' and 'Mock, run-matched'."
        ),
    )
    parser.add_argument(
        "--plain-logl-axis-labels",
        action="store_true",
        help="Use the same generic log10(L3000) x-axis label for both plotted rows.",
    )
    parser.add_argument(
        "--compact-ell-axis-labels",
        action="store_true",
        help="Use compact ell labels for log10(L3000) axes.",
    )
    parser.add_argument(
        "--mwin-med-axis-label",
        action="store_true",
        help="Use m_50 as the fourth-column y-axis label.",
    )
    parser.add_argument(
        "--compact-z-axis-label",
        action="store_true",
        help="Use z instead of 'redshift z' on z-axis labels.",
    )
    parser.add_argument(
        "--hide-n-out-labels",
        action="store_true",
        help="Do not include N_out in the N label boxes.",
    )
    parser.add_argument(
        "--n-label-font-scale",
        type=float,
        default=1.0,
        help="Multiplicative font-size scale for N label boxes.",
    )
    parser.add_argument(
        "--axis-label-font-scale",
        type=float,
        default=1.0,
        help="Multiplicative font-size scale for axis labels.",
    )
    parser.add_argument(
        "--tick-label-font-scale",
        type=float,
        default=1.0,
        help="Multiplicative font-size scale for axis tick labels and colorbar tick labels.",
    )
    parser.add_argument(
        "--colorbar-tick-pad",
        type=float,
        default=None,
        help="Optional padding between colorbar ticks and tick labels, in points.",
    )
    parser.add_argument(
        "--colorbar-label-pad",
        type=float,
        default=None,
        help="Optional padding between each colorbar and its axis label, in points.",
    )
    parser.add_argument(
        "--column-cmaps",
        type=_parse_column_cmaps,
        default=None,
        help=(
            "Optional per-column colormap overrides as COL:CMAP entries, "
            "e.g. 0:plasma,1:plasma,2:viridis,3:viridis."
        ),
    )
    parser.add_argument(
        "--d-panel-z-range",
        type=_parse_float_range,
        default=None,
        help=(
            "Optional fixed redshift range for the first-column z bins and "
            "the second-column z color scale, e.g. 0.5,3.5."
        ),
    )
    parser.add_argument(
        "--d-panel-z-bin-width",
        type=float,
        default=None,
        help="Optional first-column redshift-bin width, e.g. 0.2.",
    )
    parser.add_argument(
        "--d-binned-panel-xlim",
        type=_parse_float_range,
        default=None,
        help="Optional x-limits for the first binned eta-vs-luminosity column, e.g. 44.25,46.75.",
    )
    parser.add_argument(
        "--d-binned-panel-ylim",
        type=_parse_float_range,
        default=None,
        help="Optional y-limits for the first binned eta-vs-luminosity column, e.g. -5.35,-3.25.",
    )
    parser.add_argument(
        "--d-binned-line-width",
        type=float,
        default=None,
        help="Optional line width for the first-column binned eta-vs-luminosity curves.",
    )
    parser.add_argument(
        "--d-binned-uncert-style",
        choices=("none", "errorbar", "shade", "postprop", "postprop_paired"),
        default="none",
        help=(
            "Optional uncertainty overlay for the first-column binned curves. "
            "'errorbar' and 'shade' show the per-bin 16th-84th percentile scatter; "
            "'postprop' shows posterior-propagated bootstrap uncertainty on the median with fixed bins; "
            "'postprop_paired' also resamples paired m_win draws and re-bins luminosity per replicate."
        ),
    )
    parser.add_argument(
        "--d-binned-postprop-n-bootstrap",
        type=int,
        default=300,
        help="Number of posterior-propagated bootstrap replicates per first-column bin.",
    )
    parser.add_argument(
        "--d-binned-postprop-min-count",
        type=int,
        default=10,
        help="Minimum source count in a first-column bin before drawing postprop error bars.",
    )
    parser.add_argument(
        "--d-binned-postprop-seed",
        type=int,
        default=20260625,
        help="Random seed for posterior-propagated bootstrap intervals.",
    )
    parser.add_argument(
        "--mock-d-binned-postprop-cache-npz",
        type=Path,
        default=None,
        help="Optional cache path for mock first-column posterior-propagated bootstrap intervals.",
    )
    parser.add_argument(
        "--real-d-binned-postprop-cache-npz",
        type=Path,
        default=None,
        help="Optional cache path for real first-column posterior-propagated bootstrap intervals.",
    )
    parser.add_argument(
        "--d-binned-n-label-xlim",
        type=_parse_float_range,
        default=None,
        help=(
            "Optional x-limits used only for the first-column N label. "
            "Defaults to the plotted first-column x-limits."
        ),
    )
    parser.add_argument(
        "--d-binned-n-label-ylim",
        type=_parse_float_range,
        default=None,
        help=(
            "Optional y-limits used only for the first-column N label. "
            "Use this to keep old broad-count labels while cropping the plotted eta range."
        ),
    )
    parser.add_argument(
        "--mwin-panel-ylim",
        type=_parse_float_range,
        default=None,
        help="Optional y-limits for the fourth m_win-vs-z column, e.g. 14,20.",
    )
    parser.add_argument(
        "--logl-axis-min",
        type=float,
        default=None,
        help="Optional lower bound for log10(L3000) axes in the per-source eta and z panels.",
    )
    parser.add_argument(
        "--row-label-x",
        type=float,
        default=None,
        help="Optional row-label x coordinate in first-axis fraction units; more negative moves labels left.",
    )
    parser.add_argument(
        "--row-label-y",
        type=float,
        default=None,
        help="Optional row-label y coordinate in first-axis fraction units; 0.5 is centered.",
    )
    parser.add_argument(
        "--real-prior-label",
        default=None,
        help=(
            "Optional label drawn under the N count in the real-data row "
            "(identifies the Stage-1 prior used for the real fits)."
        ),
    )
    parser.add_argument(
        "--mock-prior-label",
        default=None,
        help=(
            "Optional label drawn under the N count in the mock-data row "
            "(identifies the Stage-1 prior used for the mock fits)."
        ),
    )
    parser.add_argument(
        "--real-stage1-posteriors-npz",
        type=Path,
        default=None,
        help=(
            "Optional Stage-1 posteriors NPZ to use for the real-data row. "
            "When provided, m_win/log10D medians are recomputed from this NPZ "
            "aligned to the pair NPZ's real_source_id (same paired sources as "
            "the default real row), instead of using the pre-computed "
            "real_*_med columns from the pair NPZ."
        ),
    )
    parser.add_argument(
        "--stage1-summary-cache-npz",
        type=Path,
        default=None,
        help=(
            "Optional cache for the mock Stage-1 posterior summaries used by this plot. "
            "Defaults to a file next to --output-pdf."
        ),
    )
    parser.add_argument(
        "--real-stage1-summary-cache-npz",
        type=Path,
        default=None,
        help=(
            "Optional cache for the real-data Stage-1 posterior summaries used by this plot. "
            "Defaults to a file next to --output-pdf when --real-stage1-posteriors-npz is used."
        ),
    )
    parser.add_argument(
        "--no-stage1-summary-cache",
        action="store_true",
        help="Disable reading/writing Stage-1 posterior summary caches.",
    )
    parser.add_argument(
        "--real-n-draws",
        type=int,
        default=None,
        help=(
            "Restrict the real-data posterior medians to the first N MCMC draws "
            "per source. Use to match the mock's draw budget (e.g. --real-n-draws 500 "
            "when the mock posterior has 500 draws/source). Only meaningful with "
            "--real-stage1-posteriors-npz."
        ),
    )
    parser.add_argument(
        "--show-cut-info",
        action="store_true",
        help=(
            "Draw a compact cut summary in the first column. The source cuts are "
            "read from the paired sample: the real side is the real quality-cut "
            "sample; the mock side is matched to those IDs."
        ),
    )
    parser.add_argument(
        "--mock-truth-mwin-max",
        type=float,
        default=None,
        help=(
            "Filter the mock row to sources with mock_m_win_cont_true <= this "
            "value (matches the real-data m_win quality cut). Has no effect "
            "unless set."
        ),
    )
    parser.add_argument(
        "--add-mock-uncut-row",
        action="store_true",
        help=(
            "Add a third row showing the mock posterior medians for ALL paired "
            "sources (no truth m_win cut), so the effect of "
            "--mock-truth-mwin-max can be compared side-by-side."
        ),
    )
    parser.add_argument(
        "--posterior-tau-min-days",
        type=float,
        default=None,
        help=(
            "Optional posterior quality cut applied to plotted Stage-1 rows: "
            "keep sources with posterior median tau >= this many days."
        ),
    )
    parser.add_argument(
        "--posterior-mwin-max",
        type=float,
        default=None,
        help=(
            "Optional posterior quality cut applied to plotted Stage-1 rows: "
            "keep sources with posterior median m_win <= this value."
        ),
    )
    parser.add_argument(
        "--posterior-z-min",
        type=float,
        default=None,
        help="Optional plotted-row cut: keep sources with z >= this value.",
    )
    parser.add_argument(
        "--posterior-z-max",
        type=float,
        default=None,
        help="Optional plotted-row cut: keep sources with z <= this value.",
    )
    parser.add_argument(
        "--posterior-log10d-halfwidth-max",
        type=float,
        default=None,
        help=(
            "Optional posterior quality cut applied to plotted Stage-1 rows: "
            "keep sources with 0.5*(q84-q16) of log10D <= this value."
        ),
    )
    parser.add_argument(
        "--hide-posterior-cut-labels",
        action="store_true",
        help="Apply posterior quality cuts without drawing their text in the N label boxes.",
    )
    parser.add_argument("--export-plot-data", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    closure_dir = args.closure_dir.expanduser().resolve()
    sample_label = str(args.sample_label)
    run_name = str(args.run_name)
    pair_npz = (
        args.pair_npz.expanduser().resolve()
        if args.pair_npz is not None
        else _default_pair_npz(closure_dir, sample_label)
    )
    stage1_npz = (
        args.stage1_posteriors_npz.expanduser().resolve()
        if args.stage1_posteriors_npz is not None
        else _default_stage1_npz(closure_dir, run_name)
    )
    out_pdf = (
        args.output_pdf.expanduser().resolve()
        if args.output_pdf is not None
        else _default_output_pdf(closure_dir, sample_label)
    )
    use_stage1_summary_cache = not bool(args.no_stage1_summary_cache)
    stage1_summary_cache_npz = None
    real_stage1_summary_cache_npz = None
    if use_stage1_summary_cache:
        stage1_summary_cache_npz = (
            args.stage1_summary_cache_npz.expanduser().resolve()
            if args.stage1_summary_cache_npz is not None
            else out_pdf.with_name(f"{out_pdf.stem}_mock_stage1_summary_cache.npz")
        )
        real_stage1_summary_cache_npz = (
            args.real_stage1_summary_cache_npz.expanduser().resolve()
            if args.real_stage1_summary_cache_npz is not None
            else out_pdf.with_name(f"{out_pdf.stem}_real_stage1_summary_cache.npz")
        )

    if not pair_npz.exists():
        raise FileNotFoundError(str(pair_npz))
    if not stage1_npz.exists():
        raise FileNotFoundError(str(stage1_npz))

    print(f"Loading pair/truth rows: {pair_npz}", flush=True)
    base_rows, mock_source_id = _load_pair_rows(pair_npz)
    mock_z = np.asarray(base_rows[2]["z"], dtype=np.float64)

    print(f"Loading Stage-1 posterior medians: {stage1_npz}", flush=True)
    posterior_row = _load_stage1_posterior_row(
        stage1_npz=stage1_npz,
        reference_source_ids=mock_source_id,
        reference_z=mock_z,
        band=str(args.band),
        summary_cache_npz=stage1_summary_cache_npz,
    )
    posterior_row["source_id"] = np.asarray(mock_source_id, dtype=np.int64)

    if args.real_stage1_posteriors_npz is not None:
        real_stage1_npz = args.real_stage1_posteriors_npz.expanduser().resolve()
        if not real_stage1_npz.exists():
            raise FileNotFoundError(str(real_stage1_npz))
        with np.load(pair_npz, allow_pickle=False) as pair:
            real_source_id = np.asarray(pair["real_source_id"], dtype=np.int64)
            real_z = np.asarray(pair["real_z"], dtype=np.float64)
        n_draws_msg = (
            f" (first {int(args.real_n_draws):,} draws/source)"
            if args.real_n_draws is not None
            else ""
        )
        print(
            f"Recomputing real-data row from {real_stage1_npz}{n_draws_msg} "
            f"aligned to {real_source_id.size:,} paired real_source_id values",
            flush=True,
        )
        real_mwin_row = _load_stage1_posterior_row(
            stage1_npz=real_stage1_npz,
            reference_source_ids=real_source_id,
            reference_z=real_z,
            band=str(args.band),
            n_draws_limit=(
                int(args.real_n_draws) if args.real_n_draws is not None else None
            ),
            summary_cache_npz=real_stage1_summary_cache_npz,
        )
        real_mwin_row["label"] = "Real data"
        real_mwin_row["source_id"] = np.asarray(real_source_id, dtype=np.int64)
    else:
        with np.load(pair_npz, allow_pickle=False) as pair:
            real_source_id = np.asarray(pair["real_source_id"], dtype=np.int64)
        real_mwin_row = dict(base_rows[1])
        real_mwin_row["label"] = "Real data"
        real_mwin_row["source_id"] = np.asarray(real_source_id, dtype=np.int64)

    rows_for_limits = [*base_rows, posterior_row, real_mwin_row]

    if args.real_prior_label is not None:
        real_mwin_row["prior_label"] = str(args.real_prior_label)
    posterior_row = dict(posterior_row)
    posterior_row["label"] = "Mock data"
    if args.mock_prior_label is not None:
        posterior_row["prior_label"] = str(args.mock_prior_label)

    if bool(args.paper_row_labels):
        real_mwin_row["label"] = "Real Data"
        posterior_row["label"] = "Mock Data"

    if bool(args.plain_logl_axis_labels):
        plain_logl_label = r"$\log_{10}(L_{3000})$"
        real_mwin_row["logl_axis_label"] = plain_logl_label
        posterior_row["logl_axis_label"] = plain_logl_label

    if args.d_binned_line_width is not None:
        real_mwin_row["d_binned_line_width"] = float(args.d_binned_line_width)
        posterior_row["d_binned_line_width"] = float(args.d_binned_line_width)
    if str(args.d_binned_uncert_style) != "none":
        real_mwin_row["d_binned_uncert_style"] = str(args.d_binned_uncert_style)
        posterior_row["d_binned_uncert_style"] = str(args.d_binned_uncert_style)

    if bool(args.show_cut_info):
        real_cut_label, mock_cut_label = _pair_cut_labels(pair_npz)
        real_mwin_row["cut_label"] = real_cut_label
        posterior_row["cut_label"] = mock_cut_label

    rows_to_plot = [real_mwin_row, posterior_row]

    if args.mock_truth_mwin_max is not None or args.add_mock_uncut_row:
        with np.load(pair_npz, allow_pickle=False) as pair:
            mock_mwin_true = np.asarray(pair["mock_m_win_cont_true"], dtype=np.float64)
        if mock_mwin_true.size != int(np.asarray(posterior_row["z"]).size):
            raise ValueError(
                "mock_m_win_cont_true length does not match the mock posterior row; "
                "expected the pair NPZ to share the mock sample with --stage1-posteriors-npz"
            )

    if args.mock_truth_mwin_max is not None:
        mwin_max = float(args.mock_truth_mwin_max)
        truth_mask = np.isfinite(mock_mwin_true) & (mock_mwin_true <= mwin_max)
        n_kept = int(np.count_nonzero(truth_mask))
        n_total = int(truth_mask.size)
        print(
            f"Applying mock truth m_win_cont <= {mwin_max}: "
            f"{n_kept:,}/{n_total:,} kept",
            flush=True,
        )
        _apply_row_mask(posterior_row, truth_mask)
        posterior_row["label"] = f"Mock data (truth $m\\le{mwin_max:g}$)"

    if args.add_mock_uncut_row:
        with np.load(pair_npz, allow_pickle=False) as pair:
            n_total_mock = int(np.asarray(pair["mock_source_id"]).size)
        uncut_z = np.asarray(base_rows[2]["z"], dtype=np.float64)
        full_posterior_row = _load_stage1_posterior_row(
            stage1_npz=stage1_npz,
            reference_source_ids=mock_source_id,
            reference_z=uncut_z,
            band=str(args.band),
            summary_cache_npz=stage1_summary_cache_npz,
        )
        full_posterior_row["label"] = "Mock data (no truth cut)"
        if args.mock_prior_label is not None:
            full_posterior_row["prior_label"] = str(args.mock_prior_label)
        if bool(args.show_cut_info):
            _, mock_cut_label = _pair_cut_labels(pair_npz)
            full_posterior_row["cut_label"] = (
                f"matched to real-cut IDs; no mock truth m cut\n"
                f"$N$={n_total_mock:,}"
            )
        rows_to_plot.append(full_posterior_row)
        rows_for_limits.append(full_posterior_row)

    for row in rows_to_plot:
        _apply_posterior_quality_cuts(
            row,
            label=str(row.get("label", "row")),
            tau_min_days=(
                float(args.posterior_tau_min_days)
                if args.posterior_tau_min_days is not None
                else None
            ),
            mwin_max=(
                float(args.posterior_mwin_max)
                if args.posterior_mwin_max is not None
                else None
            ),
            z_min=(
                float(args.posterior_z_min)
                if args.posterior_z_min is not None
                else None
            ),
            z_max=(
                float(args.posterior_z_max)
                if args.posterior_z_max is not None
                else None
            ),
            log10d_halfwidth_max=(
                float(args.posterior_log10d_halfwidth_max)
                if args.posterior_log10d_halfwidth_max is not None
                else None
            ),
            annotate=not bool(args.hide_posterior_cut_labels),
        )

    if str(args.d_binned_uncert_style) in {"postprop", "postprop_paired"}:
        if args.d_panel_z_range is None:
            raise ValueError("--d-binned-uncert-style postprop requires --d-panel-z-range")
        if args.d_panel_z_bin_width is None:
            raise ValueError("--d-binned-uncert-style postprop requires --d-panel-z-bin-width")
        if args.d_binned_panel_xlim is None:
            raise ValueError("--d-binned-uncert-style postprop requires --d-binned-panel-xlim")
        if args.d_binned_panel_ylim is None:
            raise ValueError("--d-binned-uncert-style postprop requires --d-binned-panel-ylim")
        if args.real_stage1_posteriors_npz is None:
            raise ValueError("--d-binned-uncert-style postprop requires --real-stage1-posteriors-npz")
        n_boot = int(args.d_binned_postprop_n_bootstrap)
        min_count = int(args.d_binned_postprop_min_count)
        if n_boot <= 0:
            raise ValueError("--d-binned-postprop-n-bootstrap must be positive")
        if min_count <= 0:
            raise ValueError("--d-binned-postprop-min-count must be positive")
        seed = int(args.d_binned_postprop_seed)
        paired_postprop = str(args.d_binned_uncert_style) == "postprop_paired"
        mock_postprop_cache = (
            args.mock_d_binned_postprop_cache_npz.expanduser().resolve()
            if args.mock_d_binned_postprop_cache_npz is not None
            else out_pdf.with_name(f"{out_pdf.stem}_mock_postprop_bootstrap_cache.npz")
        )
        real_postprop_cache = (
            args.real_d_binned_postprop_cache_npz.expanduser().resolve()
            if args.real_d_binned_postprop_cache_npz is not None
            else out_pdf.with_name(f"{out_pdf.stem}_real_postprop_bootstrap_cache.npz")
        )
        real_mwin_row["d_binned_postprop"] = _get_postprop_bootstrap(
            row=real_mwin_row,
            stage1_npz=args.real_stage1_posteriors_npz.expanduser().resolve(),
            cache_npz=real_postprop_cache,
            band=str(args.band),
            z_range=tuple(args.d_panel_z_range),
            z_bin_width=float(args.d_panel_z_bin_width),
            logl_xlim=tuple(args.d_binned_panel_xlim),
            eta_ylim=tuple(args.d_binned_panel_ylim),
            n_bootstrap=n_boot,
            min_count=min_count,
            seed=seed + 1,
            label="real",
            paired=paired_postprop,
        )
        posterior_row["d_binned_postprop"] = _get_postprop_bootstrap(
            row=posterior_row,
            stage1_npz=stage1_npz,
            cache_npz=mock_postprop_cache,
            band=str(args.band),
            z_range=tuple(args.d_panel_z_range),
            z_bin_width=float(args.d_panel_z_bin_width),
            logl_xlim=tuple(args.d_binned_panel_xlim),
            eta_ylim=tuple(args.d_binned_panel_ylim),
            n_bootstrap=n_boot,
            min_count=min_count,
            seed=seed + 2,
            label="mock",
            paired=paired_postprop,
        )

    plot_kwargs = dict(
        rows_for_limits=rows_for_limits,
        rows_to_plot=rows_to_plot,
        scenario_name=str(args.scenario_name),
        title_suffix=f" | sample={mock_source_id.size:,} | row2 vs row4",
        selected_columns=tuple(args.selected_columns),
        presentation=not bool(args.not_presentation),
        show_hist_row=bool(args.show_hist_row),
        n_label_columns=(tuple(args.selected_columns)[0],),
        show_n_out=not bool(args.hide_n_out_labels),
        n_label_font_scale=float(args.n_label_font_scale),
        axis_label_font_scale=float(args.axis_label_font_scale),
        tick_label_font_scale=float(args.tick_label_font_scale),
        colorbar_tick_pad=args.colorbar_tick_pad,
        colorbar_label_pad=args.colorbar_label_pad,
        compact_logl_label=bool(args.compact_ell_axis_labels),
        compact_z_label=bool(args.compact_z_axis_label),
        mwin_med_axis_label=bool(args.mwin_med_axis_label),
        column_cmaps=args.column_cmaps,
        d_panel_z_range=args.d_panel_z_range,
        d_panel_z_bin_width=args.d_panel_z_bin_width,
        d_binned_panel_xlim=args.d_binned_panel_xlim,
        d_binned_panel_ylim=args.d_binned_panel_ylim,
        d_binned_n_label_xlim=args.d_binned_n_label_xlim,
        d_binned_n_label_ylim=args.d_binned_n_label_ylim,
        d_binned_uncert_style=str(args.d_binned_uncert_style),
        mwin_panel_ylim=args.mwin_panel_ylim,
        logl_axis_min=args.logl_axis_min,
        row_label_x_override=args.row_label_x,
        row_label_y_override=args.row_label_y,
    )
    if args.export_plot_data is not None:
        from plotting.plot_data import save_snapshot
        save_snapshot(args.export_plot_data, 'figure_04_observed_and_simulated', plot_kwargs)
        return
    print(f"Writing 2x4 real-vs-mock presentation PDF: {out_pdf}", flush=True)
    _write_selected_page(out_pdf=out_pdf, **plot_kwargs)
    print(f"Written: {out_pdf}", flush=True)


if __name__ == "__main__":
    main()
