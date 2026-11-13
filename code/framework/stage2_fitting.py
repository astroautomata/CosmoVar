#!/usr/bin/env python
"""Fit the population and cosmological model to joint Stage-1 posterior draws.

Each source supplies K paired window-mean magnitudes and variability rates.
Stored ln(D) is converted to eta = log10(D), with D in mag^2 per rest-frame day;
magnitudes remain unlogged. Source IDs align the draws and catalogue entries.
Catalogue luminosities define selection cuts; the likelihood infers luminosity
from each magnitude draw and the trial distance modulus instead.

The likelihood uses a per-source mean of population densities over the paired
draws, evaluated with logsumexp. For the supported flat-box reference priors,
the conditional density of eta at fixed local nuisance parameters is constant
on its finite support, so its parameter-independent factor is omitted. The
paper describes the finite-support approximation. Other reference priors
require checking the conditional density ratio and support explicitly.

The output NPZ stores parameter dictionaries as samples (pooled draws) and
samples_grouped (chains, draws), selected source IDs, draw-selection metadata
and sampler metrics. One run_config_json record describes the entire fit."""
import argparse
import json
import os
import tempfile
import warnings
from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from framework import variability_utils as su

LAMBDA_0 = 3000.0  # Å


@dataclass(frozen=True)
class CutSummary:
    n_total: int
    n_after_z: int
    n_after_logl: int
    n_after_d50: int
    n_after_max_objects: int


def _count_percentiles(counts: np.ndarray) -> dict:
    counts = np.asarray(counts)
    if counts.size == 0:
        return {}
    percentiles = [0, 1, 5, 10, 50, 90, 95, 99, 100]
    labels = ["min", "p01", "p05", "p10", "p50", "p90", "p95", "p99", "max"]
    values = np.percentile(counts, percentiles)
    return {label: int(round(float(value))) for label, value in zip(labels, values)}


def _paired_finite_draw_subset(
    m_win: np.ndarray,
    ln_d: np.ndarray,
    k_idx: np.ndarray,
    min_finite_draws: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Keep finite joint draws and return matching magnitude and log10(D) arrays.
    
    After selecting k_idx, discard sources with fewer than min_finite_draws pairs.
    Keep the first K finite pairs in index order, with K the smallest retained
    pair count. Pair positions can differ
    between sources, but each magnitude retains its corresponding D draw. Return
    arrays of shape (N_kept, K), the source mask and draw-count metadata."""
    if min_finite_draws <= 0:
        raise ValueError("--min-finite-draws must be >= 1")
    if min_finite_draws > len(k_idx):
        raise ValueError(
            f"--min-finite-draws={min_finite_draws} exceeds available draws after stride/K: {len(k_idx)}"
        )

    m_sel = m_win[:, k_idx]
    log10_d_sel = ln_d[:, k_idx] / np.log(10.0)
    paired_finite = np.isfinite(m_sel) & np.isfinite(log10_d_sel)
    finite_counts = paired_finite.sum(axis=1)
    keep = finite_counts >= min_finite_draws
    n_before = int(m_sel.shape[0])
    n_keep = int(keep.sum())
    if n_keep == 0:
        raise ValueError(
            "No sources have enough finite paired posterior draws after applying "
            f"--min-finite-draws={min_finite_draws}"
        )

    k_use = int(min(len(k_idx), finite_counts[keep].min()))
    m_keep = m_sel[keep]
    d_keep = log10_d_sel[keep]
    paired_keep = paired_finite[keep]

    m_clean = np.empty((n_keep, k_use), dtype=np.float64)
    d_clean = np.empty((n_keep, k_use), dtype=np.float64)
    for i in range(n_keep):
        good = np.flatnonzero(paired_keep[i])[:k_use]
        m_clean[i, :] = m_keep[i, good]
        d_clean[i, :] = d_keep[i, good]

    if not np.isfinite(m_clean).all() or not np.isfinite(d_clean).all():
        raise ValueError("Internal error: non-finite values remain after finite-paired-draw filtering")

    metadata = {
        "n_sources_before_min_finite_draws": n_before,
        "n_sources_after_min_finite_draws": n_keep,
        "n_dropped_by_min_finite_draws": n_before - n_keep,
        "min_finite_draws": int(min_finite_draws),
        "K_requested_after_stride": int(len(k_idx)),
        "K_use": k_use,
        "finite_paired_draw_count_percentiles": _count_percentiles(finite_counts),
    }
    return m_clean, d_clean, keep, metadata


def _load_source_id_manifest(path: Path) -> np.ndarray:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Source-ID manifest not found: {path}")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as f:
            for key in ("source_id_selected", "source_id", "source_ids"):
                if key in f.files:
                    return np.asarray(f[key], dtype=np.int64)
            raise KeyError(f"No source-id array found in {path}; expected source_id_selected/source_id/source_ids")
    return np.loadtxt(path, dtype=np.int64, ndmin=1)


def _source_id_membership_mask(source_id: np.ndarray, selected_source_id: np.ndarray) -> np.ndarray:
    selected = np.unique(np.asarray(selected_source_id, dtype=np.int64))
    sid = np.asarray(source_id, dtype=np.int64)
    pos = np.searchsorted(selected, sid)
    in_range = pos < selected.size
    out = np.zeros(sid.shape, dtype=bool)
    out[in_range] = selected[pos[in_range]] == sid[in_range]
    return out


def setup_jax_devices():
    devices = jax.devices()
    n_devices = len(devices)
    device_type = devices[0].platform if devices else "unknown"

    print("\n" + "=" * 70)
    print("JAX DEVICE CONFIGURATION")
    print("=" * 70)
    print(f"Number of devices: {n_devices}")
    print(f"Device type: {device_type}")
    for i, dev in enumerate(devices):
        print(f"  Device {i}: {dev}")
    print("=" * 70 + "\n")

    if n_devices > 1:
        return devices, "parallel"
    return devices, "vectorized"


def _ensure_writable_output_file(path_str: str, *, what: str) -> Path:
    path = Path(path_str).expanduser()
    parent = path.parent if str(path.parent) else Path(".")

    try:
        parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise SystemExit(
            f"{what} directory is not writable: {parent}\n"
            f"Requested {what} path: {path}\n"
            "Tip: if you used an absolute path like `/filename.npz`, that targets the filesystem root.\n"
            "Use a path under your workspace, e.g. `--output results/stage2_posterior.npz`."
        ) from e

    try:
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".writetest", dir=parent)
        os.close(fd)
        os.unlink(tmp)
    except PermissionError as e:
        raise SystemExit(
            f"{what} directory is not writable: {parent}\n"
            f"Requested {what} path: {path}\n"
            "Tip: choose a location under `/scratch/...` or a relative path like `results/...`."
        ) from e

    return path


def _ensure_writable_dir(path: Path, *, what: str) -> Path:
    path = Path(path).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise SystemExit(
            f"{what} directory is not writable: {path}\n"
            "Tip: choose a location under `/scratch/...` or a relative path like `results/...`."
        ) from e

    try:
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".writetest", dir=path)
        os.close(fd)
        os.unlink(tmp)
    except PermissionError as e:
        raise SystemExit(
            f"{what} directory is not writable: {path}\n"
            "Tip: choose a location under `/scratch/...` or a relative path like `results/...`."
        ) from e

    return path


def _resolve_output_path(output_arg: str) -> Path:
    """Resolve the posterior filename from a file or directory argument."""
    out = Path(output_arg).expanduser()
    if out.suffix.lower() == ".npz":
        _ensure_writable_dir(out.parent, what="Output")
        return out

    output_dir = _ensure_writable_dir(out, what="Output")
    return output_dir / f"{output_dir.name or 'stage2'}.npz"


def _atomic_savez(path: Path, **kwargs):
    parent = path.parent if str(path.parent) else Path(".")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=parent,
            prefix=f".{path.name}.",
            suffix=".tmp.npz",
        ) as f:
            tmp_path = Path(f.name)
        np.savez(tmp_path, **kwargs)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _infer_key(npz: np.lib.npyio.NpzFile, suffix: str, band: str) -> str:
    candidates = [k for k in npz.keys() if k.endswith(suffix) and (band in k)]
    if len(candidates) > 1:
        candidates_no_diag = [k for k in candidates if "__diag__" not in k]
        if len(candidates_no_diag) == 1:
            return candidates_no_diag[0]
        candidates = candidates_no_diag or candidates
        exact = [k for k in candidates if k == f"free_flatlnD__lnD__{band}{suffix}"]
        if len(exact) == 1:
            return exact[0]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise KeyError(f"Could not find key ending with '{suffix}' containing '{band}'. Available keys: {list(npz.keys())}")
    raise KeyError(f"Multiple candidate keys for '{suffix}' with band '{band}': {candidates}")

def _map_stage1_band_to_passband(stage1_band: str) -> str:
    """
    Map Stage-1 band labels (e.g. gaia_g, ztf_r) to stage_utils passband names.

    If the input is already a known passband name, return it unchanged.
    """
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
    if bl in mapping:
        return mapping[bl]
    # Fall back to assuming the user already passed a stage_utils passband name.
    return b

def _extract_stage1_band_from_key(key: str) -> str | None:
    """
    Extract Stage-1 band label from a Stage-1 array key, if present.

    Example keys:
      - free_flatlnD__lnD__gaia_g__m_win
      - free_flatlnD__lnD__gaia_g__ln_D
    """
    s = str(key)
    token = "__lnD__"
    if token not in s:
        return None
    tail = s.split(token, 1)[1]
    band = tail.split("__", 1)[0].strip()
    return band or None


def _auto_detect_stage1_band_and_keys(
    dat: np.lib.npyio.NpzFile,
    *,
    band_arg: str | None,
    m_win_key_arg: str | None,
    ln_d_key_arg: str | None,
) -> tuple[str, str, str]:
    """
    Return (stage1_band, m_win_key, ln_d_key).

    Preference order:
      1) both key overrides provided
      2) band provided (use suffix+band inference)
      3) auto-detect unique candidates from NPZ keys
    """
    keys = list(dat.keys())

    if m_win_key_arg and ln_d_key_arg:
        if m_win_key_arg not in dat:
            raise KeyError(f"--m-win-key not found in chains NPZ: {m_win_key_arg}.")
        if ln_d_key_arg not in dat:
            raise KeyError(f"--lnD-key not found in chains NPZ: {ln_d_key_arg}.")
        stage1_band = (
            _extract_stage1_band_from_key(m_win_key_arg)
            or _extract_stage1_band_from_key(ln_d_key_arg)
            or (str(band_arg).strip() if band_arg else "")
        )
        if not stage1_band:
            raise SystemExit(
                "Could not infer Stage-1 band label from provided keys. "
                "Pass --band explicitly (e.g. gaia_g / ztf_r)."
            )
        return stage1_band, m_win_key_arg, ln_d_key_arg

    if band_arg and str(band_arg).strip():
        stage1_band = str(band_arg).strip()
        m_win_key = m_win_key_arg or _infer_key(dat, "__m_win", stage1_band)
        ln_d_key = ln_d_key_arg or _infer_key(dat, "__ln_D", stage1_band)
        return stage1_band, m_win_key, ln_d_key

    # Auto-detect only if there is exactly one m_win and one ln_D in the file.
    m_win_candidates = [k for k in keys if str(k).endswith("__m_win")]
    ln_d_candidates = [k for k in keys if str(k).endswith("__ln_D")]
    if len(m_win_candidates) != 1 or len(ln_d_candidates) != 1:
        raise SystemExit(
            "Could not auto-detect a unique Stage-1 band from the chains NPZ.\n"
            "Pass --band (e.g. gaia_g / ztf_r) or pass both --m-win-key and --lnD-key.\n"
            f"Found m_win candidates ({len(m_win_candidates)}): {m_win_candidates}\n"
            f"Found ln_D candidates ({len(ln_d_candidates)}): {ln_d_candidates}"
        )
    m_win_key = m_win_candidates[0]
    ln_d_key = ln_d_candidates[0]
    stage1_band = _extract_stage1_band_from_key(m_win_key) or _extract_stage1_band_from_key(ln_d_key)
    if not stage1_band:
        raise SystemExit(
            "Auto-detected m_win/ln_D keys but could not parse the band label from them. "
            "Pass --band explicitly."
        )
    return stage1_band, m_win_key, ln_d_key


def _mock_catalogue_logl3000(pair_npz: Path, source_id: np.ndarray, redshift: np.ndarray) -> np.ndarray:
    """Align the closure catalogue by source ID."""
    with np.load(pair_npz, allow_pickle=False) as pair:
        ids = np.asarray(pair["mock_source_id"])
        z = np.asarray(pair["mock_z"], dtype=np.float64)
        logl = np.asarray(pair["mock_logl3000"], dtype=np.float64)
    if ids.ndim != 1 or z.shape != ids.shape or logl.shape != ids.shape:
        raise ValueError("Pair/truth catalogue arrays must be aligned one-dimensional arrays")
    if not np.issubdtype(ids.dtype, np.integer) or np.unique(ids).size != ids.size:
        raise ValueError("Pair/truth source IDs must be unique integers")
    order = np.argsort(ids)
    positions = np.searchsorted(ids[order], source_id)
    valid = positions < ids.size
    valid[valid] &= ids[order[positions[valid]]] == source_id[valid]
    if not np.all(valid):
        raise ValueError("Pair/truth catalogue is missing Stage 1 source IDs")
    rows = order[positions]
    if not np.array_equal(redshift, z[rows], equal_nan=True):
        raise ValueError("Pair/truth redshifts disagree with the Stage 1 source IDs")
    return logl[rows]


def load_mwin_chains(
    chains_npz: Path,
    band: str,
    draw_stride: int,
    draw_offset: int,
    k_keep: int | None,
    z_min: float,
    z_max: float,
    logl3000_min: float,
    logl3000_max: float,
    d50_min: float,
    d50_max: float,
    max_objects: int | None,
    *,
    source_id_key: str = "source_id",
    z_key: str = "z",
    logl3000_key: str = "logl3000",
    m_win_key: str | None = None,
    ln_d_key: str | None = None,
    nan_policy: str = "strict",
    min_finite_draws: int | None = None,
    source_id_manifest: Path | None = None,
    pair_npz: Path | None = None,
):
    """Load, select and align joint Stage-1 summaries for the Stage-2 likelihood.
    
    Input magnitude and ln(D) arrays have shape (N, K). Catalogue luminosities are
    read directly or joined from pair_npz by source ID; they are used for selection.
    Apply finite-value handling, an optional source-ID manifest, redshift and
    catalogue-luminosity cuts, then cuts on the full-row median log10(D). The lower
    D bound is inclusive and the upper bound is exclusive. Select posterior columns
    by offset/stride/K, retain finite pairs if requested, and apply max_objects last.
    
    Return IDs, redshifts, catalogue luminosities, paired magnitudes and log10(D)
    draws, plus selection metadata. Magnitudes and rates always share their row and
    draw ordering; strict mode rejects any row containing non-finite input draws."""
    dat = np.load(chains_npz, allow_pickle=False)

    if m_win_key is None:
        m_win_key = _infer_key(dat, "__m_win", band)
    if ln_d_key is None:
        ln_d_key = _infer_key(dat, "__ln_D", band)

    source_id = np.asarray(dat[source_id_key])
    z = np.asarray(dat[z_key], dtype=np.float64)
    if logl3000_key in dat.files:
        logl3000_cat = np.asarray(dat[logl3000_key], dtype=np.float64)
        catalogue_source = {"path": str(chains_npz), "key": logl3000_key}
    elif pair_npz is not None:
        logl3000_cat = _mock_catalogue_logl3000(Path(pair_npz), source_id, z)
        catalogue_source = {"path": str(pair_npz), "key": "mock_logl3000", "alignment": "source_id"}
    else:
        raise KeyError(f"Missing {logl3000_key!r} in {chains_npz}; for closure-test posteriors, supply --pair-npz with the matching pair/truth catalogue")
    m_win = np.asarray(dat[m_win_key], dtype=np.float64)
    ln_d = np.asarray(dat[ln_d_key], dtype=np.float64)

    if m_win.ndim != 2 or ln_d.ndim != 2:
        raise ValueError(f"Expected 2D arrays for m_win and ln_D, got {m_win.shape=} {ln_d.shape=}")
    if m_win.shape != ln_d.shape:
        raise ValueError(f"m_win and ln_D shapes must match, got {m_win.shape=} {ln_d.shape=}")
    if len(z) != m_win.shape[0] or len(logl3000_cat) != m_win.shape[0]:
        raise ValueError("Catalog arrays must have same length as posterior first dimension")

    n_total = int(m_win.shape[0])
    k_total = int(m_win.shape[1])

    if nan_policy not in {"strict", "finite-paired-draws"}:
        raise ValueError(f"Unknown nan_policy={nan_policy!r}")

    raw_nonfinite = {
        "policy": nan_policy,
        "n_sources_loaded": n_total,
        "m_win_sources_with_any_nonfinite": int((~np.isfinite(m_win).all(axis=1)).sum()),
        "ln_D_sources_with_any_nonfinite": int((~np.isfinite(ln_d).all(axis=1)).sum()),
        "either_sources_with_any_nonfinite": int(((~np.isfinite(m_win).all(axis=1)) | (~np.isfinite(ln_d).all(axis=1))).sum()),
        "m_win_nonfinite_draws_total": int((~np.isfinite(m_win)).sum()),
        "ln_D_nonfinite_draws_total": int((~np.isfinite(ln_d)).sum()),
    }

    if nan_policy == "strict":
        mask = np.isfinite(z) & np.isfinite(logl3000_cat) & np.isfinite(m_win).all(axis=1) & np.isfinite(ln_d).all(axis=1)
        if not np.all(mask):
            n_bad = int((~mask).sum())
            print(f"Filtering {n_bad} rows with non-finite values across required arrays")
            z = z[mask]
            logl3000_cat = logl3000_cat[mask]
            m_win = m_win[mask]
            ln_d = ln_d[mask]
            source_id = source_id[mask]
    else:
        mask = np.isfinite(z) & np.isfinite(logl3000_cat)
        if not np.all(mask):
            n_bad = int((~mask).sum())
            print(f"Filtering {n_bad} rows with non-finite z/logL3000 values")
            z = z[mask]
            logl3000_cat = logl3000_cat[mask]
            m_win = m_win[mask]
            ln_d = ln_d[mask]
            source_id = source_id[mask]

    manifest_md = None
    if source_id_manifest is not None:
        selected_source_id = _load_source_id_manifest(Path(source_id_manifest))
        mask_manifest = _source_id_membership_mask(source_id, selected_source_id)
        n_before_manifest = int(source_id.shape[0])
        if not np.any(mask_manifest):
            raise ValueError(f"Source-ID manifest selected zero rows: {source_id_manifest}")
        z = z[mask_manifest]
        logl3000_cat = logl3000_cat[mask_manifest]
        m_win = m_win[mask_manifest]
        ln_d = ln_d[mask_manifest]
        source_id = source_id[mask_manifest]
        manifest_md = {
            "path": str(Path(source_id_manifest)),
            "n_manifest": int(np.unique(selected_source_id).size),
            "n_before_manifest": n_before_manifest,
            "n_after_manifest": int(source_id.shape[0]),
        }

    # Cuts: z then logl3000 then d50
    mask_z = (z >= z_min) & (z <= z_max)
    z = z[mask_z]
    logl3000_cat = logl3000_cat[mask_z]
    m_win = m_win[mask_z]
    ln_d = ln_d[mask_z]
    source_id = source_id[mask_z]
    n_after_z = int(z.shape[0])

    mask_l = (logl3000_cat >= logl3000_min) & (logl3000_cat <= logl3000_max)
    z = z[mask_l]
    logl3000_cat = logl3000_cat[mask_l]
    m_win = m_win[mask_l]
    ln_d = ln_d[mask_l]
    source_id = source_id[mask_l]
    n_after_logl = int(z.shape[0])

    log10_d_all = ln_d / np.log(10.0)
    if nan_policy == "finite-paired-draws":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
            d50 = np.nanmedian(log10_d_all, axis=1)
    else:
        d50 = np.median(log10_d_all, axis=1)
    mask_d50 = (d50 >= d50_min) & (d50 < d50_max)
    z = z[mask_d50]
    logl3000_cat = logl3000_cat[mask_d50]
    m_win = m_win[mask_d50]
    ln_d = ln_d[mask_d50]
    source_id = source_id[mask_d50]
    n_after_d50 = int(z.shape[0])

    # Subsample posterior draws (preserve draw-by-draw correlation between m_win and D)
    if draw_stride <= 0:
        raise ValueError("--draw-stride must be >= 1")
    if draw_offset < 0:
        raise ValueError("--draw-offset must be >= 0")
    if draw_offset >= k_total:
        raise ValueError(f"--draw-offset={draw_offset} exceeds available draws: {k_total}")
    k_idx = np.arange(draw_offset, k_total, draw_stride, dtype=np.int64)
    if k_keep is not None:
        if k_keep <= 0:
            raise ValueError("--K must be >= 1")
        if k_keep > len(k_idx):
            raise ValueError(
                f"--K={k_keep} exceeds available draws after offset/stride: {len(k_idx)}"
            )
        k_idx = k_idx[:k_keep]

    nan_handling = dict(raw_nonfinite)
    nan_handling.update({
        "K_total": k_total,
        "draw_stride": int(draw_stride),
        "draw_offset": int(draw_offset),
        "K_requested_after_stride": int(len(k_idx)),
    })

    if nan_policy == "finite-paired-draws":
        if min_finite_draws is None:
            min_finite_draws = 1
        m_win, log10_d, mask_draws, draw_md = _paired_finite_draw_subset(
            m_win=m_win,
            ln_d=ln_d,
            k_idx=k_idx,
            min_finite_draws=int(min_finite_draws),
        )
        z = z[mask_draws]
        logl3000_cat = logl3000_cat[mask_draws]
        source_id = source_id[mask_draws]
        nan_handling.update(draw_md)
        print("NaN policy: finite-paired-draws")
        print(
            f"  sources with any non-finite m_win/ln_D draws: "
            f"{nan_handling['either_sources_with_any_nonfinite']:,}"
        )
        print(f"  non-finite draws total: m_win={nan_handling['m_win_nonfinite_draws_total']:,}, ln_D={nan_handling['ln_D_nonfinite_draws_total']:,}")
        print(f"  finite paired draw count percentiles: {nan_handling['finite_paired_draw_count_percentiles']}")
        print(
            f"  dropped by --min-finite-draws={int(min_finite_draws)}: "
            f"{nan_handling['n_dropped_by_min_finite_draws']:,}"
        )
        print(f"  final K_use={nan_handling['K_use']}")
    else:
        m_win = m_win[:, k_idx]
        log10_d = (ln_d[:, k_idx]) / np.log(10.0)
        nan_handling.update({
            "min_finite_draws": None,
            "K_use": int(m_win.shape[1]),
            "n_dropped_by_min_finite_draws": 0,
        })

    # Apply max_objects last, deterministically (keep first N)
    if max_objects is not None:
        if max_objects <= 0:
            raise ValueError("--max-objects must be >= 1 (or omit it)")
        n_use = min(int(max_objects), int(z.shape[0]))
        z = z[:n_use]
        logl3000_cat = logl3000_cat[:n_use]
        m_win = m_win[:n_use]
        log10_d = log10_d[:n_use]
        source_id = source_id[:n_use]
    n_after_max = int(z.shape[0])

    cut_summary = CutSummary(
        n_total=n_total,
        n_after_z=n_after_z,
        n_after_logl=n_after_logl,
        n_after_d50=n_after_d50,
        n_after_max_objects=n_after_max,
    )

    metadata = {
        "catalogue_luminosity_source": catalogue_source,
        "band": band,
        "m_win_key": m_win_key,
        "ln_d_key": ln_d_key,
        "n_total": n_total,
        "k_total": k_total,
        "k_idx": k_idx.tolist(),
        "cuts": {
            "z_min": float(z_min),
            "z_max": float(z_max),
            "logl3000_min": float(logl3000_min),
            "logl3000_max": float(logl3000_max),
            "d50_min": float(d50_min),
            "d50_max": float(d50_max),
            "max_objects": int(max_objects) if max_objects is not None else None,
        },
        "cut_counts": {
            "n_total": cut_summary.n_total,
            "n_after_z": cut_summary.n_after_z,
            "n_after_logl3000": cut_summary.n_after_logl,
            "n_after_d50": cut_summary.n_after_d50,
            "n_after_max_objects": cut_summary.n_after_max_objects,
        },
        "arrays": {
            "N": int(z.shape[0]),
            "K": int(m_win.shape[1]),
        },
        "nan_handling": nan_handling,
    }
    if manifest_md is not None:
        metadata["source_id_manifest_filter"] = manifest_md

    return source_id, z, logl3000_cat, m_win, log10_d, metadata


def compute_single_band_terms(
    z: np.ndarray,
    passband: str,
    wave_grids: dict,
    s_unifieds: dict,
    jax_interpolator,
):
    # Match Stage-2 strategy: batch expensive SED-based computations to avoid GPU OOM.
    """Compute the SED/passband terms shared by all draws of each source.
    
    Return the magnitude--log10(L3000) intercept a_band, K-corrections in magnitudes
    and log10(lambda_rf_eff / 3000 Angstrom). Redshift-dependent arrays have shape
    (N,). Batch the SED calculations to limit device memory use."""
    BATCH_SIZE_KCORR = 5000
    z = np.asarray(z, dtype=np.float64)
    n_obj = int(z.shape[0])
    a_band = su.compute_M_rest_logL3000_linear_coeff(
        jax_interpolator, wave_grids, s_unifieds, rest_band=passband
    )

    n_batches = int(np.ceil(n_obj / BATCH_SIZE_KCORR))
    if n_batches > 1:
        print(
            f"Computing K-correction/lambda_rf in {n_batches} batches "
            f"(batch_size~{BATCH_SIZE_KCORR}, N={n_obj}) to avoid GPU OOM..."
        )

    k_corr_all = []
    lambda_rf_all = []
    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE_KCORR
        end = min((batch_idx + 1) * BATCH_SIZE_KCORR, n_obj)
        z_batch = jnp.asarray(z[start:end], dtype=jnp.float64)

        k_batch = su.compute_K_correction(
            z_batch, jax_interpolator, wave_grids, s_unifieds, obs_band=passband, rest_band=passband
        )
        lambda_rf_batch = su.compute_effective_lambda_rf_simplified(
            z_batch, wave_grids, s_unifieds, jax_interpolator, fiducial_logL3000=45.0, band=passband
        )

        k_corr_all.append(np.asarray(k_batch, dtype=np.float64))
        lambda_rf_all.append(np.asarray(lambda_rf_batch, dtype=np.float64))

    k_corr = np.concatenate(k_corr_all)
    lambda_rf_eff = np.concatenate(lambda_rf_all)

    if not np.all(np.isfinite(lambda_rf_eff)):
        raise ValueError("Non-finite values in lambda_rf_eff")
    log10_lambda_ratio = np.log10(lambda_rf_eff / LAMBDA_0)

    return float(a_band), jnp.asarray(k_corr), jnp.asarray(log10_lambda_ratio)


def model_mwin_chains(
    *,
    z: jnp.ndarray,  # [N]
    m_win: jnp.ndarray,  # [N, K]
    log10_d: jnp.ndarray,  # [N, K]
    k_corr: jnp.ndarray,  # [N]
    log10_lambda_ratio: jnp.ndarray,  # [N]
    a_band: float,  # scalar
    distmod_fn,
    sample_H0: bool,
    H0_fixed: float,
    sample_Om0: bool,
    Om0_fixed: float,
    sample_w0: bool,
    w0_fixed: float,
    use_wpiv: bool,
    wpiv_prior_mode: str,
    c_piv: float,
    sample_w_piv: bool,
    w_piv_fixed: float,
    sample_wa: bool,
    wa_fixed: float,
    sample_D_pivot: bool,
    D_pivot_fixed: float,
    D_pivot_prior_mu: float | None,
    D_pivot_prior_sigma: float | None,
    sample_d_D: bool,
    d_D_fixed: float,
    d_D_prior_mu: float | None,
    d_D_prior_sigma: float | None,
    sample_alpha_D: bool,
    alpha_D_fixed: float,
    alpha_D_prior_mu: float | None,
    alpha_D_prior_sigma: float | None,
    sample_log_sigma_int: bool,
    log_sigma_int_fixed: float,
    log_sigma_int_prior_mu: float | None,
    log_sigma_int_prior_sigma: float | None,
    H0_min: float,
    H0_max: float,
    Om0_min: float,
    Om0_max: float,
    w0_min: float,
    w0_max: float,
    w_piv_min: float,
    w_piv_max: float,
    wa_min: float,
    wa_max: float,
    sig_hyp: float,
    sig_alpha: float,
    sig_ell: float,
    logL3000_pivot: float,
):
    """Evaluate the population density at paired local posterior draws.
    
    For each (source, draw), map magnitude to log10(L3000) using the trial distance
    modulus, then predict eta = log10(D) from luminosity and effective wavelength.
    D_pivot is the intercept in eta, not a linear variability rate. sigma_int is the
    scatter in eta (dex), parametrised by its natural logarithm log_sigma_int.
    
    Average the Gaussian population density over the K paired draws for each
    source, then sum the per-source log averages. The reference-prior and support
    assumptions of this estimator are described in the module overview and paper."""
    def sample_with_optional_normal(name: str, default_dist, sample_flag: bool, fixed_value: float, prior_mu, prior_sigma):
        if not sample_flag:
            return fixed_value
        if prior_mu is not None and prior_sigma is not None:
            return numpyro.sample(name, dist.Normal(float(prior_mu), float(prior_sigma)))
        return numpyro.sample(name, default_dist)

    H0 = numpyro.sample("H0", dist.Uniform(H0_min, H0_max)) if sample_H0 else H0_fixed
    Om0 = numpyro.sample("Om0", dist.Uniform(Om0_min, Om0_max)) if sample_Om0 else Om0_fixed
    if use_wpiv:
        wa = numpyro.sample("wa", dist.Uniform(wa_min, wa_max)) if sample_wa else wa_fixed
        if sample_w_piv:
            if wpiv_prior_mode == "preserve-w0":
                w_piv = numpyro.sample("w_piv", dist.Uniform(w0_min + c_piv * wa, w0_max + c_piv * wa))
            else:
                w_piv = numpyro.sample("w_piv", dist.Uniform(w_piv_min, w_piv_max))
        else:
            w_piv = numpyro.deterministic("w_piv", jnp.asarray(w_piv_fixed))
        w0 = numpyro.deterministic("w0", w_piv - c_piv * wa)
    else:
        w0 = numpyro.sample("w0", dist.Uniform(w0_min, w0_max)) if sample_w0 else w0_fixed
        wa = numpyro.sample("wa", dist.Uniform(wa_min, wa_max)) if sample_wa else wa_fixed

    d_pivot = sample_with_optional_normal(
        "D_pivot",
        dist.Uniform(-20.0, 40.0),
        sample_D_pivot,
        D_pivot_fixed,
        D_pivot_prior_mu,
        D_pivot_prior_sigma,
    )
    d_d = sample_with_optional_normal("d_D", dist.Normal(0.0, sig_hyp), sample_d_D, d_D_fixed, d_D_prior_mu, d_D_prior_sigma)
    alpha_d = sample_with_optional_normal(
        "alpha_D", dist.Normal(0.0, sig_alpha), sample_alpha_D, alpha_D_fixed, alpha_D_prior_mu, alpha_D_prior_sigma
    )
    log_sigma_int = sample_with_optional_normal(
        "log_sigma_int",
        dist.Normal(0.0, sig_ell),
        sample_log_sigma_int,
        log_sigma_int_fixed,
        log_sigma_int_prior_mu,
        log_sigma_int_prior_sigma,
    )
    sigma_int = numpyro.deterministic("sigma_int", jnp.exp(log_sigma_int))

    mu_dm = distmod_fn(H0, Om0, w0, wa)  # [N]

    # At fixed SED/passband terms, every magnitude draw implies its own luminosity.
    log10_l3000 = (a_band + mu_dm[:, None] + k_corr[:, None] - m_win) / 2.5  # [N, K]
    mu_d = d_pivot + d_d * (log10_l3000 - logL3000_pivot) - alpha_d * log10_lambda_ratio[:, None]  # [N, K]

    resid = log10_d - mu_d
    logp = -0.5 * (resid / sigma_int) ** 2 - jnp.log(sigma_int) - 0.5 * jnp.log(2 * jnp.pi)  # [N, K]

    k = jnp.asarray(log10_d.shape[1], dtype=jnp.float64)
    # Average densities along the draw axis before multiplying source likelihoods.
    logp_per_obj = logsumexp(logp, axis=1) - jnp.log(k)  # [N]
    numpyro.factor("loglike", jnp.sum(logp_per_obj))


def run_mcmc(
    *,
    z: np.ndarray,
    m_win: np.ndarray,
    log10_d: np.ndarray,
    k_corr: jnp.ndarray,
    log10_lambda_ratio: jnp.ndarray,
    a_band: float,
    n_steps_z: int,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int,
    chain_method: str | None,
    target_accept_prob: float,
    max_tree_depth: int,
    sample_H0: bool,
    H0_fixed: float,
    sample_Om0: bool,
    Om0_fixed: float,
    sample_w0: bool,
    w0_fixed: float,
    use_wpiv: bool,
    wpiv_prior_mode: str,
    c_piv: float,
    sample_w_piv: bool,
    w_piv_fixed: float,
    sample_wa: bool,
    wa_fixed: float,
    sample_D_pivot: bool,
    D_pivot_fixed: float,
    D_pivot_prior_mu: float | None,
    D_pivot_prior_sigma: float | None,
    sample_d_D: bool,
    d_D_fixed: float,
    d_D_prior_mu: float | None,
    d_D_prior_sigma: float | None,
    sample_alpha_D: bool,
    alpha_D_fixed: float,
    alpha_D_prior_mu: float | None,
    alpha_D_prior_sigma: float | None,
    sample_log_sigma_int: bool,
    log_sigma_int_fixed: float,
    log_sigma_int_prior_mu: float | None,
    log_sigma_int_prior_sigma: float | None,
    H0_min: float,
    H0_max: float,
    Om0_min: float,
    Om0_max: float,
    w0_min: float,
    w0_max: float,
    w_piv_min: float,
    w_piv_max: float,
    wa_min: float,
    wa_max: float,
    sig_hyp: float,
    sig_alpha: float,
    sig_ell: float,
    logL3000_pivot: float,
):
    """Run NumPyro NUTS on the selected joint draws and return its MCMC object.
    
    The seed controls the JAX sampler. num_samples counts retained draws per chain;
    num_warmup is excluded from those draws. Source arrays are converted to float64
    and the distance-modulus function is prepared for the selected redshifts."""
    _, auto_chain_method = setup_jax_devices()
    if chain_method is None:
        chain_method = auto_chain_method
    if chain_method not in {"parallel", "vectorized", "sequential"}:
        print(f"Warning: Unknown chain_method '{chain_method}', using '{auto_chain_method}'")
        chain_method = auto_chain_method

    z_j = jnp.asarray(z, dtype=jnp.float64)
    m_win_j = jnp.asarray(m_win, dtype=jnp.float64)
    log10_d_j = jnp.asarray(log10_d, dtype=jnp.float64)

    distmod_fn = su.make_distmod_fn(z_j, n_steps=n_steps_z)

    kernel = NUTS(
        model_mwin_chains,
        dense_mass=True,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=True,
    )

    mcmc.run(
        jax.random.PRNGKey(seed),
        extra_fields=("num_steps", "accept_prob"),
        z=z_j,
        m_win=m_win_j,
        log10_d=log10_d_j,
        k_corr=k_corr,
        log10_lambda_ratio=log10_lambda_ratio,
        a_band=a_band,
        distmod_fn=distmod_fn,
        sample_H0=sample_H0,
        H0_fixed=H0_fixed,
        sample_Om0=sample_Om0,
        Om0_fixed=Om0_fixed,
        sample_w0=sample_w0,
        w0_fixed=w0_fixed,
        use_wpiv=use_wpiv,
        wpiv_prior_mode=wpiv_prior_mode,
        c_piv=c_piv,
        sample_w_piv=sample_w_piv,
        w_piv_fixed=w_piv_fixed,
        sample_wa=sample_wa,
        wa_fixed=wa_fixed,
        sample_D_pivot=sample_D_pivot,
        D_pivot_fixed=D_pivot_fixed,
        D_pivot_prior_mu=D_pivot_prior_mu,
        D_pivot_prior_sigma=D_pivot_prior_sigma,
        sample_d_D=sample_d_D,
        d_D_fixed=d_D_fixed,
        d_D_prior_mu=d_D_prior_mu,
        d_D_prior_sigma=d_D_prior_sigma,
        sample_alpha_D=sample_alpha_D,
        alpha_D_fixed=alpha_D_fixed,
        alpha_D_prior_mu=alpha_D_prior_mu,
        alpha_D_prior_sigma=alpha_D_prior_sigma,
        sample_log_sigma_int=sample_log_sigma_int,
        log_sigma_int_fixed=log_sigma_int_fixed,
        log_sigma_int_prior_mu=log_sigma_int_prior_mu,
        log_sigma_int_prior_sigma=log_sigma_int_prior_sigma,
        H0_min=H0_min,
        H0_max=H0_max,
        Om0_min=Om0_min,
        Om0_max=Om0_max,
        w0_min=w0_min,
        w0_max=w0_max,
        w_piv_min=w_piv_min,
        w_piv_max=w_piv_max,
        wa_min=wa_min,
        wa_max=wa_max,
        sig_hyp=sig_hyp,
        sig_alpha=sig_alpha,
        sig_ell=sig_ell,
        logL3000_pivot=logL3000_pivot,
    )

    return mcmc


def main():
    ap = argparse.ArgumentParser(
        description="Stage-2 (m_win chains): hierarchical inference using m_win and ln(D) posterior draws"
    )
    ap.add_argument("--chains-npz", required=True, help="NPZ file with per-object posterior draws (m_win and ln_D)")
    ap.add_argument("--pair-npz", type=Path, help="Matching closure pair/truth NPZ; supplies mock_logl3000 by source ID only when the Stage 1 catalogue-luminosity column is absent.")
    ap.add_argument(
        "--chains-npz-original",
        required=True,
        help=(
            "Original Stage-1 NPZ path, checked and recorded in the fit settings."
        ),
    )
    ap.add_argument(
        "--band",
        type=str,
        default=None,
        help="Stage-1 band label to disambiguate NPZ keys (e.g. gaia_g, gaia_bp, gaia_rp, ztf_g, ztf_r). "
        "If omitted, the script auto-detects only when the NPZ contains a unique m_win/ln_D pair.",
    )
    ap.add_argument("--sed-grid", required=True, help="Path to QSOSed grid file")
    ap.add_argument("--passband-file", required=True, help="Path to passband.dat file (Gaia)")
    ap.add_argument("--ztf-filter-dir", default="assets/ZTF_filters", help="Directory containing ZTF filter files")

    ap.add_argument("--max-objects", type=int, default=None, help="Maximum number of objects after cuts (keep first N)")
    ap.add_argument("--draw-stride", type=int, default=5, help="Stride to subsample posterior draws (default: 5)")
    ap.add_argument(
        "--draw-offset",
        type=int,
        default=0,
        help="First posterior draw index to consider before applying stride/K (default: 0).",
    )
    ap.add_argument("--K", type=int, default=None, help="Keep only the first K draws after applying offset/stride")
    ap.add_argument(
        "--nan-policy",
        choices=["strict", "finite-paired-draws"],
        default="strict",
        help=(
            "How to handle non-finite m_win/ln_D posterior draws. "
            "'strict' drops any source with any non-finite draw. "
            "'finite-paired-draws' keeps sources with enough finite paired draws and uses a common K."
        ),
    )
    ap.add_argument(
        "--min-finite-draws",
        type=int,
        default=None,
        help="Minimum finite paired m_win/ln_D draws required when --nan-policy=finite-paired-draws.",
    )

    ap.add_argument("--z-min", type=float, default=0.5)
    ap.add_argument("--z-max", type=float, default=3.0)
    ap.add_argument("--logl3000-min", type=float, default=44.0)
    ap.add_argument("--logl3000-max", type=float, default=47.0)
    ap.add_argument(
        "--d50-min",
        type=float,
        default=float("-inf"),
        help="Hard cut: median(log10D) >= d50-min (default: -inf)",
    )
    ap.add_argument(
        "--d50-max",
        type=float,
        default=-2.0,
        help="Hard cut: median(log10D) < d50-max (default: -2)",
    )

    ap.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
    ap.add_argument("--samples", type=int, default=500, help="Number of sampling steps")
    ap.add_argument("--chains", type=int, default=3, help="Number of chains")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--output",
        default="stage2_mwin_chains_posterior.npz",
        help=(
            "Posterior NPZ output. A .npz path selects that file; otherwise writes "
            "`<dir>/<dir_name>.npz` inside the selected directory."
        ),
    )

    ap.add_argument(
        "--chain-method",
        type=str,
        choices=["parallel", "vectorized", "sequential"],
        default="sequential",
        help="Chain execution method (default: sequential; set to vectorized for 1-device parallelism)",
    )
    ap.add_argument("--target-accept-prob", type=float, default=0.85, help="Target acceptance probability for NUTS")
    ap.add_argument("--max-tree-depth", type=int, default=10, help="Maximum tree depth for NUTS")
    ap.add_argument("--n-integration-wavelength", type=int, default=1000, help="Number of wavelength grid points")
    ap.add_argument("--n-steps-z", type=int, default=512, help="Number of integration steps for distance modulus")
    ap.add_argument(
        "--wpiv-parametrization",
        action="store_true",
        help="Sample w_piv and wa, deriving w0 = w_piv - c_piv * wa.",
    )
    ap.add_argument(
        "--c-piv",
        type=float,
        default=None,
        help="Pivot coefficient c_piv = z_piv / (1 + z_piv). Required with --wpiv-parametrization.",
    )
    ap.add_argument(
        "--wpiv-prior-mode",
        choices=["preserve-w0", "independent"],
        default="preserve-w0",
        help=(
            "Prior used for w_piv runs. 'preserve-w0' keeps the original rectangular prior in "
            "(w0, wa); 'independent' uses an independent top-hat in (w_piv, wa)."
        ),
    )

    ap.add_argument("--sample-H0", action="store_true", help="Sample H0 (default: fixed)")
    ap.add_argument("--H0-fixed", type=float, default=70.0)
    ap.add_argument("--sample-Om0", action="store_true", help="Sample Om0 (default: fixed)")
    ap.add_argument("--Om0-fixed", type=float, default=0.3)
    ap.add_argument("--sample-w0", action="store_true", help="Sample w0 (default: fixed)")
    ap.add_argument("--w0-fixed", type=float, default=-1.0)
    ap.add_argument("--sample-w-piv", action="store_true", help="Sample w_piv in --wpiv-parametrization mode")
    ap.add_argument("--w-piv-fixed", type=float, default=-1.0)
    ap.add_argument("--sample-wa", action="store_true", help="Sample wa (default: fixed)")
    ap.add_argument("--wa-fixed", type=float, default=0.0)

    ap.add_argument(
        "--sample-params",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of parameters to sample. Overrides individual --sample-* flags. "
            "Recognized: H0,Om0,w0,wa,D_pivot,d_D,alpha_D,log_sigma_int. "
            "Use w_piv instead of w0 with --wpiv-parametrization."
        ),
    )

    # Stage-2 hyperparameters (sample by default; allow fixing any of them).
    ap.add_argument(
        "--sample-D-pivot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample D_pivot (default: true; use --no-sample-D-pivot to fix it)",
    )
    ap.add_argument("--D-pivot-fixed", type=float, default=0.0, help="Fixed D_pivot used with --no-sample-D-pivot")
    ap.add_argument("--D-pivot-prior-mu", type=float, default=None, help="Optional Gaussian prior mean for sampled D_pivot")
    ap.add_argument(
        "--D-pivot-prior-sigma", type=float, default=None, help="Optional Gaussian prior sigma for sampled D_pivot"
    )

    ap.add_argument(
        "--sample-d-D",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample d_D (default: true; use --no-sample-d-D to fix it)",
    )
    ap.add_argument("--d-D-fixed", type=float, default=0.0, help="Fixed d_D used with --no-sample-d-D")
    ap.add_argument("--d-D-prior-mu", type=float, default=None, help="Optional Gaussian prior mean for sampled d_D")
    ap.add_argument("--d-D-prior-sigma", type=float, default=None, help="Optional Gaussian prior sigma for sampled d_D")

    ap.add_argument(
        "--sample-alpha-D",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample alpha_D (default: true; use --no-sample-alpha-D to fix it)",
    )
    ap.add_argument("--alpha-D-fixed", type=float, default=0.0, help="Fixed alpha_D used with --no-sample-alpha-D")
    ap.add_argument(
        "--alpha-D-prior-mu", type=float, default=None, help="Optional Gaussian prior mean for sampled alpha_D"
    )
    ap.add_argument(
        "--alpha-D-prior-sigma", type=float, default=None, help="Optional Gaussian prior sigma for sampled alpha_D"
    )

    ap.add_argument(
        "--sample-log-sigma-int",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample log_sigma_int (default: true; sigma_int is always deterministic exp(log_sigma_int))",
    )
    ap.add_argument(
        "--log-sigma-int-fixed", type=float, default=0.0, help="Fixed log_sigma_int used with --no-sample-log-sigma-int"
    )
    ap.add_argument(
        "--log-sigma-int-prior-mu",
        type=float,
        default=None,
        help="Optional Gaussian prior mean for sampled log_sigma_int",
    )
    ap.add_argument(
        "--log-sigma-int-prior-sigma",
        type=float,
        default=None,
        help="Optional Gaussian prior sigma for sampled log_sigma_int",
    )

    ap.add_argument("--H0-min", type=float, default=60.0)
    ap.add_argument("--H0-max", type=float, default=80.0)
    ap.add_argument("--Om0-min", type=float, default=0.0)
    ap.add_argument("--Om0-max", type=float, default=1.0)
    ap.add_argument("--w0-min", type=float, default=-5.0)
    ap.add_argument("--w0-max", type=float, default=1.0)
    ap.add_argument("--w-piv-min", type=float, default=-2.0)
    ap.add_argument("--w-piv-max", type=float, default=0.0)
    ap.add_argument("--wa-min", type=float, default=-27.0)
    ap.add_argument("--wa-max", type=float, default=7.0)

    ap.add_argument("--sig-hyp", type=float, default=2.0)
    ap.add_argument("--sig-alpha", type=float, default=1.0)
    ap.add_argument("--sig-ell", type=float, default=1.0)
    ap.add_argument("--logL3000-pivot", type=float, default=45.5)

    ap.add_argument("--source-id-key", type=str, default="source_id")
    ap.add_argument(
        "--source-id-manifest",
        type=str,
        default=None,
        help="Optional NPZ/TXT source-ID manifest to use as an additional deterministic row filter.",
    )
    ap.add_argument("--z-key", type=str, default="z")
    ap.add_argument("--logl3000-key", type=str, default="logl3000")
    ap.add_argument("--m-win-key", type=str, default=None, help="Override NPZ key for m_win array")
    ap.add_argument("--lnD-key", type=str, default=None, help="Override NPZ key for ln_D array")

    args = ap.parse_args()

    c_piv = None
    z_piv = None
    if args.wpiv_parametrization:
        if args.c_piv is None:
            raise SystemExit("--c-piv is required with --wpiv-parametrization.")
        c_piv = float(args.c_piv)
        if not (0.0 < c_piv < 1.0):
            raise SystemExit(
                f"--c-piv must be in (0, 1), got {c_piv}. "
                "It equals z_piv / (1 + z_piv), so z_piv > 0 implies c_piv in (0,1)."
            )
        z_piv = c_piv / (1.0 - c_piv)
    else:
        c_piv = 0.0
    if not args.wpiv_parametrization and args.wpiv_prior_mode != "preserve-w0":
        raise SystemExit("--wpiv-prior-mode is only meaningful with --wpiv-parametrization.")

    informative_pairs = {
        "D_pivot": (args.D_pivot_prior_mu, args.D_pivot_prior_sigma),
        "d_D": (args.d_D_prior_mu, args.d_D_prior_sigma),
        "alpha_D": (args.alpha_D_prior_mu, args.alpha_D_prior_sigma),
        "log_sigma_int": (args.log_sigma_int_prior_mu, args.log_sigma_int_prior_sigma),
    }
    for pname, (mu, sig) in informative_pairs.items():
        if (mu is None) ^ (sig is None):
            raise SystemExit(f"{pname}: informative prior requires both mu and sigma.")
        if sig is not None and float(sig) <= 0:
            raise SystemExit(f"{pname}: informative prior sigma must be > 0.")

    output_path = _resolve_output_path(args.output)
    args.output = str(output_path)

    if args.sample_params is not None:
        sample_set = {s.strip() for s in str(args.sample_params).split(",") if s.strip()}
        known = (
            {"H0", "Om0", "w_piv", "wa", "D_pivot", "d_D", "alpha_D", "log_sigma_int"}
            if args.wpiv_parametrization
            else {"H0", "Om0", "w0", "wa", "D_pivot", "d_D", "alpha_D", "log_sigma_int"}
        )
        unknown = sorted(sample_set - known)
        if unknown:
            raise SystemExit(f"Unknown --sample-params entries: {unknown}. Known: {sorted(known)}")
        args.sample_H0 = "H0" in sample_set
        args.sample_Om0 = "Om0" in sample_set
        args.sample_w0 = (not args.wpiv_parametrization) and ("w0" in sample_set)
        args.sample_w_piv = args.wpiv_parametrization and ("w_piv" in sample_set)
        args.sample_wa = "wa" in sample_set
        args.sample_D_pivot = "D_pivot" in sample_set
        args.sample_d_D = "d_D" in sample_set
        args.sample_alpha_D = "alpha_D" in sample_set
        args.sample_log_sigma_int = "log_sigma_int" in sample_set
    elif args.wpiv_parametrization and args.sample_w0:
        raise SystemExit("Use --sample-w-piv, not --sample-w0, with --wpiv-parametrization.")

    w_key = "w_piv" if args.wpiv_parametrization else "w0"
    sample_flags = {
        "H0": bool(args.sample_H0),
        "Om0": bool(args.sample_Om0),
        w_key: bool(args.sample_w_piv if args.wpiv_parametrization else args.sample_w0),
        "wa": bool(args.sample_wa),
        "D_pivot": bool(args.sample_D_pivot),
        "d_D": bool(args.sample_d_D),
        "alpha_D": bool(args.sample_alpha_D),
        "log_sigma_int": bool(args.sample_log_sigma_int),
    }
    fixed_values = {
        "H0": float(args.H0_fixed),
        "Om0": float(args.Om0_fixed),
        w_key: float(args.w_piv_fixed if args.wpiv_parametrization else args.w0_fixed),
        "wa": float(args.wa_fixed),
        "D_pivot": float(args.D_pivot_fixed),
        "d_D": float(args.d_D_fixed),
        "alpha_D": float(args.alpha_D_fixed),
        "log_sigma_int": float(args.log_sigma_int_fixed),
    }
    sampled_now = [k for k, v in sample_flags.items() if v]
    fixed_now = [f"{k}={fixed_values[k]}" for k, v in sample_flags.items() if not v]
    informative_priors = {
        "D_pivot": {"mu": args.D_pivot_prior_mu, "sigma": args.D_pivot_prior_sigma},
        "d_D": {"mu": args.d_D_prior_mu, "sigma": args.d_D_prior_sigma},
        "alpha_D": {"mu": args.alpha_D_prior_mu, "sigma": args.alpha_D_prior_sigma},
        "log_sigma_int": {"mu": args.log_sigma_int_prior_mu, "sigma": args.log_sigma_int_prior_sigma},
    }
    active_informative = {
        k: v for k, v in informative_priors.items() if v["mu"] is not None or v["sigma"] is not None
    }

    print("\n" + "=" * 70)
    print("PARAMETER CONFIGURATION (EARLY CHECK)")
    print("=" * 70)
    print(f"Parametrization: {'w_piv/wa' if args.wpiv_parametrization else 'w0/wa'}")
    if args.wpiv_parametrization:
        print(f"c_piv = {c_piv:.12g}   z_piv = {z_piv:.12g}")
    print(f"--sample-params requested: {args.sample_params}")
    print(f"Will sample: {sampled_now if sampled_now else '[]'}")
    print(f"Will fix:    {fixed_now if fixed_now else '[]'}")
    print(f"Informative priors: {active_informative if active_informative else '[]'}")
    print("=" * 70 + "\n")

    chains_path = Path(args.chains_npz)
    if not chains_path.exists():
        raise SystemExit(f"Chains NPZ not found: {chains_path}")
    chains_path_for_config = Path(args.chains_npz_original)
    if not chains_path_for_config.exists():
        raise SystemExit(f"Original chains NPZ not found: {chains_path_for_config}")

    # Auto-detect Stage-1 band and array keys from the Stage-1 chains NPZ before starting.
    dat = np.load(chains_path, allow_pickle=False, mmap_mode="r")
    stage1_band, m_win_key, ln_d_key = _auto_detect_stage1_band_and_keys(
        dat,
        band_arg=(str(args.band).strip() if args.band is not None else None),
        m_win_key_arg=(str(args.m_win_key).strip() if args.m_win_key else None),
        ln_d_key_arg=(str(args.lnD_key).strip() if args.lnD_key else None),
    )
    passband = _map_stage1_band_to_passband(stage1_band)

    print("\n" + "=" * 70)
    print("STAGE-2 (M_WIN CHAINS) HIERARCHICAL INFERENCE")
    print("=" * 70)
    print(f"Chains NPZ: {chains_path}")
    print(f"Band arg: {args.band}")
    print(f"Detected Stage-1 band: {stage1_band}")
    print(f"Using passband for SED/K-correction: {passband}")
    print(f"Detected m_win key: {m_win_key}")
    print(f"Detected ln_D key:  {ln_d_key}")
    print(f"Max objects: {args.max_objects if args.max_objects is not None else 'all after cuts'}")
    print(
        "Posterior draw subsampling: "
        f"offset={args.draw_offset}, stride={args.draw_stride}, "
        f"K={args.K if args.K is not None else 'all after offset/stride'}"
    )
    print(
        f"Cuts: z∈[{args.z_min}, {args.z_max}], logl3000∈[{args.logl3000_min}, {args.logl3000_max}], "
        f"d50∈[{args.d50_min}, {args.d50_max})"
    )
    print("=" * 70 + "\n")

    # Filters + SED
    su.passband_data = su.load_passband_data(args.passband_file)
    ztf_g_path = os.path.join(args.ztf_filter_dir, "Palomar_ZTF.g.dat")
    ztf_r_path = os.path.join(args.ztf_filter_dir, "Palomar_ZTF.r.dat")
    if os.path.exists(ztf_g_path):
        su.ztf_filter_data["ZTF_g"] = su.load_ztf_filter(ztf_g_path)
        print(f"✓ Loaded ZTF g filter: {ztf_g_path}")
    if os.path.exists(ztf_r_path):
        su.ztf_filter_data["ZTF_r"] = su.load_ztf_filter(ztf_r_path)
        print(f"✓ Loaded ZTF r filter: {ztf_r_path}")

    jax_interpolator = su.create_jax_interpolator(args.sed_grid)

    bands = [passband]
    if passband.startswith("ZTF_") and passband not in su.ztf_filter_data:
        raise SystemExit(
            f"{passband} filter not loaded; check --ztf-filter-dir for the corresponding Palomar_ZTF.*.dat file"
        )
    wave_grids, s_unifieds = su.setup_wavelength_grid(bands, args.n_integration_wavelength)

    # Load + cut chains
    source_id, z, logl3000_cat, m_win, log10_d, data_md = load_mwin_chains(
        chains_path,
        band=stage1_band,
        draw_stride=args.draw_stride,
        draw_offset=args.draw_offset,
        k_keep=args.K,
        z_min=args.z_min,
        z_max=args.z_max,
        logl3000_min=args.logl3000_min,
        logl3000_max=args.logl3000_max,
        d50_min=args.d50_min,
        d50_max=args.d50_max,
        max_objects=args.max_objects,
        source_id_key=args.source_id_key,
        z_key=args.z_key,
        logl3000_key=args.logl3000_key,
        m_win_key=m_win_key,
        ln_d_key=ln_d_key,
        nan_policy=args.nan_policy,
        min_finite_draws=args.min_finite_draws,
        source_id_manifest=(Path(args.source_id_manifest) if args.source_id_manifest else None),
        pair_npz=args.pair_npz,
    )

    n, k = m_win.shape
    print(f"✓ Data after cuts: N={n}, K={k}")
    print(f"  z range: [{float(np.min(z)):.3f}, {float(np.max(z)):.3f}]")
    print(f"  logl3000(cat) range: [{float(np.min(logl3000_cat)):.3f}, {float(np.max(logl3000_cat)):.3f}]")

    # Single-band terms (Stage-2 style)
    a_band, k_corr, log10_lambda_ratio = compute_single_band_terms(
        z, passband=passband, wave_grids=wave_grids, s_unifieds=s_unifieds, jax_interpolator=jax_interpolator
    )
    print(f"✓ a_band (M_rest→logL3000): {a_band:.6f}")
    print(
        f"✓ log10(lambda_rf/3000) range: [{float(log10_lambda_ratio.min()):.3f}, {float(log10_lambda_ratio.max()):.3f}]"
    )

    # Run MCMC
    import time as time_module

    start = time_module.time()
    mcmc = run_mcmc(
        z=z,
        m_win=m_win,
        log10_d=log10_d,
        k_corr=k_corr,
        log10_lambda_ratio=log10_lambda_ratio,
        a_band=a_band,
        n_steps_z=args.n_steps_z,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        seed=args.seed,
        chain_method=args.chain_method,
        target_accept_prob=args.target_accept_prob,
        max_tree_depth=args.max_tree_depth,
        sample_H0=args.sample_H0,
        H0_fixed=args.H0_fixed,
        sample_Om0=args.sample_Om0,
        Om0_fixed=args.Om0_fixed,
        sample_w0=args.sample_w0,
        w0_fixed=args.w0_fixed,
        use_wpiv=args.wpiv_parametrization,
        wpiv_prior_mode=args.wpiv_prior_mode,
        c_piv=float(c_piv),
        sample_w_piv=args.sample_w_piv,
        w_piv_fixed=args.w_piv_fixed,
        sample_wa=args.sample_wa,
        wa_fixed=args.wa_fixed,
        sample_D_pivot=args.sample_D_pivot,
        D_pivot_fixed=args.D_pivot_fixed,
        D_pivot_prior_mu=args.D_pivot_prior_mu,
        D_pivot_prior_sigma=args.D_pivot_prior_sigma,
        sample_d_D=args.sample_d_D,
        d_D_fixed=args.d_D_fixed,
        d_D_prior_mu=args.d_D_prior_mu,
        d_D_prior_sigma=args.d_D_prior_sigma,
        sample_alpha_D=args.sample_alpha_D,
        alpha_D_fixed=args.alpha_D_fixed,
        alpha_D_prior_mu=args.alpha_D_prior_mu,
        alpha_D_prior_sigma=args.alpha_D_prior_sigma,
        sample_log_sigma_int=args.sample_log_sigma_int,
        log_sigma_int_fixed=args.log_sigma_int_fixed,
        log_sigma_int_prior_mu=args.log_sigma_int_prior_mu,
        log_sigma_int_prior_sigma=args.log_sigma_int_prior_sigma,
        H0_min=args.H0_min,
        H0_max=args.H0_max,
        Om0_min=args.Om0_min,
        Om0_max=args.Om0_max,
        w0_min=args.w0_min,
        w0_max=args.w0_max,
        w_piv_min=args.w_piv_min,
        w_piv_max=args.w_piv_max,
        wa_min=args.wa_min,
        wa_max=args.wa_max,
        sig_hyp=args.sig_hyp,
        sig_alpha=args.sig_alpha,
        sig_ell=args.sig_ell,
        logL3000_pivot=args.logL3000_pivot,
    )
    elapsed = time_module.time() - start

    samples_flat = mcmc.get_samples(group_by_chain=False)
    samples_grouped = mcmc.get_samples(group_by_chain=True)
    extra_fields = mcmc.get_extra_fields()
    param_names = list(samples_flat.keys())

    n_divergences = None
    divergence_rate = None
    mean_accept_prob = None
    max_steps_allowed = int(2 ** int(args.max_tree_depth) - 1)
    num_steps_max = None
    num_steps_p50 = None
    num_steps_p90 = None
    num_steps_p99 = None
    n_at_max_tree_depth = None
    frac_at_max_tree_depth = None
    if "diverging" in extra_fields:
        n_div = int(jnp.sum(extra_fields["diverging"]))
        total = int(args.chains * args.samples)
        n_divergences = n_div
        divergence_rate = float(n_div / total) if total > 0 else None
    if "accept_prob" in extra_fields:
        mean_accept_prob = float(jnp.mean(extra_fields["accept_prob"]))
    if "num_steps" in extra_fields:
        num_steps = np.asarray(jax.device_get(extra_fields["num_steps"]), dtype=np.int64).reshape(-1)
        if num_steps.size:
            num_steps_max = int(np.max(num_steps))
            num_steps_p50, num_steps_p90, num_steps_p99 = [
                float(x) for x in np.percentile(num_steps, [50, 90, 99])
            ]
            n_at_max_tree_depth = int(np.sum(num_steps >= max_steps_allowed))
            frac_at_max_tree_depth = float(n_at_max_tree_depth / num_steps.size)

    # Save settings with the draws so retries cannot leave a mismatched sidecar.
    run_config_json = json.dumps({
        "argv": [sys.executable, *sys.argv],
        "working_directory": os.getcwd(),
        "arguments": {
            key: str(value) if isinstance(value, Path) or (
                isinstance(value, float) and not np.isfinite(value)
            ) else value
            for key, value in vars(args).items()
        },
        "resolved": {
            "stage1_band": stage1_band,
            "passband": passband,
            "m_win_key": m_win_key,
            "ln_D_key": ln_d_key,
            "chain_method": mcmc.chain_method,
            "num_sources": n,
            "draws_per_source": k,
        },
    }, allow_nan=False)
    _atomic_savez(
        output_path,
        run_config_json=run_config_json,
        samples=samples_flat,
        samples_grouped=samples_grouped,
        columns=np.array(param_names, dtype=object),
        source_id=np.asarray(source_id, dtype=np.int64),
        num_chains=args.chains,
        num_samples=args.samples,
        num_warmup=args.warmup,
        chain_method=args.chain_method,
        band=passband,
        lambda_0=LAMBDA_0,
        a_band=a_band,
        data_metadata=data_md,
        target_accept_prob=float(args.target_accept_prob),
        max_tree_depth=int(args.max_tree_depth),
        max_steps_allowed=max_steps_allowed,
        num_steps_max=(-1 if num_steps_max is None else int(num_steps_max)),
        num_steps_p50=(np.nan if num_steps_p50 is None else float(num_steps_p50)),
        num_steps_p90=(np.nan if num_steps_p90 is None else float(num_steps_p90)),
        num_steps_p99=(np.nan if num_steps_p99 is None else float(num_steps_p99)),
        n_at_max_tree_depth=(-1 if n_at_max_tree_depth is None else int(n_at_max_tree_depth)),
        frac_at_max_tree_depth=(
            np.nan if frac_at_max_tree_depth is None else float(frac_at_max_tree_depth)
        ),
        n_divergences=(-1 if n_divergences is None else int(n_divergences)),
        divergence_rate=(np.nan if divergence_rate is None else float(divergence_rate)),
        mean_accept_prob=(np.nan if mean_accept_prob is None else float(mean_accept_prob)),
        runtime_seconds=float(elapsed),
        wpiv_parametrization=bool(args.wpiv_parametrization),
        wpiv_prior_mode=str(args.wpiv_prior_mode if args.wpiv_parametrization else ""),
        pivot_coeff=(np.nan if not args.wpiv_parametrization else float(c_piv)),
        pivot_redshift=(np.nan if not args.wpiv_parametrization else float(z_piv)),
    )

    print("\n" + "=" * 70)
    print(f"✓ Saved posterior samples to: {args.output}")
    print(f"Runtime: {elapsed/60.0:.2f} minutes")
    print(f"Parameters: {param_names}")
    if args.wpiv_parametrization:
        print(f"Pivot: c_piv={c_piv:.12g}, z_piv={z_piv:.12g}; w0 is deterministic from w_piv and wa")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
