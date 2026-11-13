"""Posterior summaries, recovery statistics and shared figure helpers."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import textwrap
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, to_rgba
from matplotlib.legend_handler import HandlerBase, HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.ticker import FixedLocator, FormatStrFormatter, FuncFormatter, MaxNLocator, ScalarFormatter
import numpy as np

from framework.photometry import real_logl3000_from_mwin
from framework import PROJECT_ROOT, DATA_ROOT

try:
    import corner as corner_lib
except Exception:
    corner_lib = None




class HandlerPriorGrid(HandlerBase):
    """Lay a tuple of color-patch handles out as a two-row grid.

    matplotlib's built-in ``HandlerTuple`` only packs the swatches into a single
    horizontal row, which makes a 5-prior legend very wide.  This handler instead
    arranges the swatches as ``n_top`` on the upper row and the remainder below, so
    the legend is roughly half as long.  The legend draws the accompanying label
    vertically centred against the (now two-row-tall) handle box, which puts the
    "Different priors" text at the mid-height between the two rows.
    """

    def __init__(self, n_top: int = 2, x_pad: float = 0.18, y_pad: float = 0.22, **kwargs):
        self._n_top = int(n_top)
        self._x_pad = float(x_pad)
        self._y_pad = float(y_pad)
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        patches = list(orig_handle)
        n = len(patches)
        n_top = max(1, min(self._n_top, n))
        rows = [patches[:n_top], patches[n_top:]]
        ncols = max(1, max(len(r) for r in rows))
        cw = width / ncols
        ch = height / 2.0
        sw = cw * (1.0 - self._x_pad)
        sh = ch * (1.0 - self._y_pad)
        artists = []
        for ri, row in enumerate(rows):
            y = height - (ri + 1) * ch - ydescent + (ch - sh) / 2.0
            for ci, patch in enumerate(row):
                x = ci * cw - xdescent + (cw - sw) / 2.0
                artists.append(
                    Rectangle(
                        (x, y),
                        sw,
                        sh,
                        facecolor=patch.get_facecolor(),
                        edgecolor=patch.get_edgecolor(),
                        linewidth=patch.get_linewidth(),
                        transform=trans,
                    )
                )
        return artists


LN10 = np.log(10.0)


CORNER_PARAMS = ["log10_D", "m_win_cont", "mean_mag", "log10_sigma2", "log10_tau", "log10_jitter"]


DERIVED_CORNER_PARAMS = {"log10_D", "m_win_cont"}


CORNER_DIAG_TITLE_X = {"log10_jitter": 0.58}


CORNER_PARAM_SOURCE = {}


CORNER_RANGE_QUANTILES = (1.0, 99.0)


CORNER_RANGE_QUANTILES_BY_PARAM = {"mean_mag": (5.0, 95.0)}


CORNER_RANGE_PAD_FRAC = 0.06


CORNER_HIST_BINS = 28


CORNER_CONTOUR_LEVELS = (0.68, 0.95)


CORNER_CONTOUR_SMOOTH = 0.9


CORNER_AXIS_LABEL_FONTSIZE = 28


CORNER_TICK_LABEL_FONTSIZE = 20


CORNER_LEGEND_FONTSIZE = 28


CORNER_DEFAULT_STYLE = {
    "spine_lw": 1.8,
    "tick_width": 1.8,
    "tick_length": 5.5,
    "tick_labelsize": float(CORNER_TICK_LABEL_FONTSIZE),
    "axis_labelsize": float(CORNER_AXIS_LABEL_FONTSIZE),
    "diag_lw": 2.0,
    "diag_annotated_lw": 2.3,
    "contour_lw": 2.1,
    "truth_lw_1d": 1.8,
    "truth_lw_2d": 1.6,
    "truth_color": "red",
    "fill_contours": False,
    "contour_fill_alpha": 0.20,
    "contour_fill_inner_alpha": 0.34,
    "legend_patch_handles": False,
    "quantile_lw": 1.4,
    "median_lw": 1.6,
    "legend_lw": 2.5,
    "truth_legend_lw": 2.0,
    "legend_fontsize": float(CORNER_LEGEND_FONTSIZE),
    "legend_bbox_x": 0.97,
    "legend_bbox_y": 0.90,
    "diag_annotation_fontsize": 30.0,
}


CORNER_PRESENTATION_STYLE = {
    **CORNER_DEFAULT_STYLE,
    "spine_lw": 4.4,
    "tick_width": 3.8,
    "tick_length": 9.0,
    "tick_labelsize": 30.0,
    "axis_labelsize": 36.0,
    "diag_lw": 3.5,
    "diag_annotated_lw": 3.9,
    "contour_lw": 3.5,
    "truth_lw_1d": 3.4,
    "truth_lw_2d": 3.2,
    "truth_color": "red",
    "fill_contours": False,
    "contour_fill_alpha": 0.20,
    "contour_fill_inner_alpha": 0.34,
    "legend_patch_handles": False,
    "quantile_lw": 2.7,
    "median_lw": 3.0,
    "legend_lw": 3.8,
    "truth_legend_lw": 3.4,
    "legend_fontsize": 38.0,
    "legend_bbox_x": 0.92,
    "legend_bbox_y": 0.95,
    "diag_annotation_fontsize": 36.0,
}


PARAM_LABELS = {
    "log10_D":      r"$\log_{10}(D)$",
    "log10_sigma2": r"$\log_{10}(\sigma^2)$",
    "log10_tau":    r"$\log_{10}(\tau)$",
    "log10_jitter": r"$\log_{10}(\sigma_{\mathrm{add}})$",
    "mean_mag":     r"$\bar{m}_0$",
    "m_win_cont":   r"$m_{\mathrm{win}}^{\mathrm{cont}}$",
    "m_win":        r"$m_{\mathrm{win}}^{\mathrm{trap}}$",
}


CORNER_PARAM_LABELS = {
    **PARAM_LABELS,
    "mean_mag":     r"$m_0$",
    "m_win_cont":   r"$m_{\mathrm{win}}$",
    "m_win":        r"$m_{\mathrm{win}}$",
    "log10_sigma2": r"$\log_{10}(\sigma^2)$",
}


PARAM_ANNOTATION_MATH = {
    "log10_D":      r"\log_{10}(D)",
    "log10_sigma2": r"\log_{10}(\sigma^2)",
    "log10_tau":    r"\log_{10}(\tau)",
    "log10_jitter": r"\log_{10}(\sigma_{\mathrm{add}})",
    "mean_mag":     r"\bar{m}_0",
    "m_win_cont":   r"m_{\mathrm{win}}^{\mathrm{cont}}",
    "m_win":        r"m_{\mathrm{win}}^{\mathrm{trap}}",
}


CORNER_PARAM_ANNOTATION_MATH = {
    **PARAM_ANNOTATION_MATH,
    "mean_mag":     r"m_0",
    "m_win_cont":   r"m_{\mathrm{win}}",
    "m_win":        r"m_{\mathrm{win}}",
    "log10_sigma2": r"\log_{10}(\sigma^2)",
}


PRESENTATION_CORNER_PARAMS = ["log10_D", "m_win_cont", "log10_sigma2", "log10_tau"]


PRESENTATION_CORNER_PARAM_LABELS = {
    **CORNER_PARAM_LABELS,
    "log10_D": r"$\log_{10}D$",
    "m_win_cont": r"$m$",
    "log10_sigma2": r"$\log_{10}\sigma^2$",
    "log10_tau": r"$\log_{10}\tau$",
}


PRESENTATION_CORNER_PARAM_ANNOTATION_MATH = {
    **CORNER_PARAM_ANNOTATION_MATH,
    "log10_D": r"\log_{10}D",
    "m_win_cont": r"m",
    "log10_sigma2": r"\log_{10}\sigma^2",
    "log10_tau": r"\log_{10}\tau",
}


_POSTERIOR_Q_CACHE: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


_POINT_ESTIMATE_CACHE: dict[tuple[int, str], np.ndarray] = {}


_PRIOR_TOKEN = {
    "lnD":      "ln D",
    "lnsigma2": r"ln $\sigma^2$",
    "lnsigma":  r"ln $\sigma$",
    "lntau":    r"ln $\tau$",
    "lnjitter": r"ln $\sigma_{\mathrm{add}}$",
}


def _tex_escape_text(text: Any) -> str:
    """Escape plain-text chunks for usetex while leaving $...$ math intact."""
    s = str(text)
    if not plt.rcParams.get("text.usetex", False):
        return s

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "<": r"$<$",
        ">": r"$>$",
    }

    def escape_chunk(chunk: str) -> str:
        return "".join(replacements.get(ch, ch) for ch in chunk)

    parts = s.split("$")
    for idx in range(0, len(parts), 2):
        parts[idx] = escape_chunk(parts[idx])
    return "$".join(parts)


def _format_sampling_label(raw: str) -> str:
    """Convert e.g. 'lnsigma2_lntau_flat' → 'Flat ln σ², ln τ'."""
    base = raw.removesuffix("_flat")
    parts = [_PRIOR_TOKEN.get(p, p) for p in base.split("_")]
    label = ", ".join(parts)
    if raw.endswith("_flat"):
        label = "Flat " + label
    return label


def _format_log10_bounds(bounds: list[float] | tuple[float, float] | None) -> str:
    """Format natural-log prior bounds as base-10 logarithmic bounds."""
    if bounds is None or len(bounds) != 2:
        return "[?, ?]"
    lo, hi = float(bounds[0]) / LN10, float(bounds[1]) / LN10
    return f"[{lo:.2f}, {hi:.2f}]"


def _format_sampling_label_log10(raw: str) -> str:
    """Human-readable sampling family for corner legends, in log10 units."""
    mapping = {
        "lnD_lntau_flat": r"Flat $\log_{10}D$, $\log_{10}\tau$",
        "lnsigma2_lntau_flat": r"Flat $\log_{10}\sigma^2$, $\log_{10}\tau$",
        "lnsigma_lntau_flat": r"Flat $\log_{10}\sigma$, $\log_{10}\tau$",
    }
    return mapping.get(raw, _format_sampling_label(raw))


def _corner_legend_label(run: dict[str, Any]) -> str:
    meta = run.get("meta", {})
    sampling = meta.get("sampling_model", {}) if isinstance(meta, dict) else {}
    if not isinstance(sampling, dict):
        return _tex_escape_text(run["label"])

    raw_label = str(sampling.get("label", run.get("sampling_label", "unknown")))
    prior_bounds = sampling.get("prior_bounds_used", {})
    if not isinstance(prior_bounds, dict):
        prior_bounds = {}

    if raw_label == "lnD_lntau_flat":
        bound_txt = rf"$\log_{{10}}D$ {_format_log10_bounds(prior_bounds.get('ln_D'))}, $\log_{{10}}\tau$ {_format_log10_bounds(prior_bounds.get('ln_tau'))}"
    elif raw_label == "lnsigma2_lntau_flat":
        bound_txt = rf"$\log_{{10}}\sigma^2$ {_format_log10_bounds(prior_bounds.get('ln_sigma2'))}, $\log_{{10}}\tau$ {_format_log10_bounds(prior_bounds.get('ln_tau'))}"
    elif raw_label == "lnsigma_lntau_flat":
        bounds = prior_bounds.get("ln_sigma")
        if bounds is None and "ln_sigma2" in prior_bounds:
            lo, hi = prior_bounds["ln_sigma2"]
            bounds = [0.5 * float(lo), 0.5 * float(hi)]
        bound_txt = rf"$\log_{{10}}\sigma$ {_format_log10_bounds(bounds)}, $\log_{{10}}\tau$ {_format_log10_bounds(prior_bounds.get('ln_tau'))}"
    else:
        bound_txt = "bounds unknown"

    return _tex_escape_text(f"{run['label']}\n{_format_sampling_label_log10(raw_label)}\n{bound_txt}")


def _comparison_palette(n: int) -> list[Any]:
    base_colors = list(plt.get_cmap("tab10").colors)
    preferred_order = [0, 1, 2, 4, 5, 6, 7, 8, 9, 3]
    ordered = [base_colors[i] for i in preferred_order]
    if n <= len(ordered):
        return ordered[:n]
    return [ordered[i % len(ordered)] for i in range(n)]


def _infer_metadata_path(npz_path: Path) -> Path | None:
    candidate = npz_path.with_name(npz_path.stem + "_metadata.json")
    if candidate.exists():
        return candidate
    return None


def _detect_bands(data: np.lib.npyio.NpzFile) -> list[str]:
    bands: list[str] = []
    for key in data.files:
        if key.startswith("posterior_ln_D_"):
            bands.append(key.replace("posterior_ln_D_", ""))
    if bands:
        return bands
    if "posterior_ln_D" in data.files:
        return [""]
    raise ValueError("Could not find posterior_ln_D[_<band>] arrays in NPZ.")


def _suffix_for_band(band: str) -> str:
    return f"_{band}" if band else ""


def _truth_log10_array(data: np.lib.npyio.NpzFile, key_base: str, key_suffix: str) -> np.ndarray:
    ln_key = f"true_ln_{key_base}{key_suffix}"
    log10_key = f"true_log10_{key_base}{key_suffix}"
    if ln_key in data.files:
        return np.asarray(data[ln_key], dtype=float) / LN10
    if log10_key in data.files:
        return np.asarray(data[log10_key], dtype=float)
    n_obj = int(np.asarray(data[f"source_id"]).shape[0])
    return np.full(n_obj, np.nan, dtype=float)


def _optional_array(data: np.lib.npyio.NpzFile, key: str, fill_value: float, n_obj: int) -> np.ndarray:
    if key in data.files:
        return np.asarray(data[key], dtype=float)
    return np.full(n_obj, fill_value, dtype=float)


def _optional_posterior_array(data: np.lib.npyio.NpzFile, key: str, template: np.ndarray) -> np.ndarray:
    if key in data.files:
        return np.asarray(data[key], dtype=float)
    template_arr = np.asarray(template)
    n_obj = int(template_arr.shape[0]) if template_arr.ndim >= 1 else 0
    return np.empty((n_obj, 0), dtype=float)


def _optional_truth_array(
    data: np.lib.npyio.NpzFile,
    key_suffix: str,
    per_band_key: str,
    shared_key: str,
    n_obj: int,
) -> np.ndarray:
    if per_band_key in data.files:
        return np.asarray(data[per_band_key], dtype=float)
    if key_suffix and shared_key in data.files:
        return np.asarray(data[shared_key], dtype=float)
    return np.full(n_obj, np.nan, dtype=float)


def _load_run(npz_path: Path, label: str | None, band: str | None) -> dict[str, Any]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing NPZ: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    bands = _detect_bands(data)
    if band is None:
        if len(bands) != 1:
            raise ValueError(
                f"{npz_path} contains multiple bands {bands}; please pass --band explicitly."
            )
        band = bands[0]
    if band not in bands:
        raise ValueError(f"Band '{band}' not found in {npz_path}; available bands: {bands}")
    key_suffix = _suffix_for_band(band)

    meta_path = _infer_metadata_path(npz_path)
    meta = None
    if meta_path is not None:
        meta = json.loads(meta_path.read_text())

    source_id = np.asarray(data["source_id"], dtype=int)
    n_obj = int(source_id.size)
    posterior_log10_D = np.asarray(data[f"posterior_ln_D{key_suffix}"], dtype=float) / LN10
    posterior_log10_tau = np.asarray(data[f"posterior_ln_tau{key_suffix}"], dtype=float) / LN10
    posterior_log10_sigma2 = posterior_log10_D + posterior_log10_tau
    posterior_log10_jitter = np.asarray(data[f"posterior_ln_jitter{key_suffix}"], dtype=float) / LN10
    posterior_mean_mag = np.asarray(data[f"posterior_m{key_suffix}"], dtype=float)
    posterior_m_win_cont = _optional_posterior_array(
        data,
        f"posterior_m_win_cont{key_suffix}",
        posterior_log10_D,
    )
    posterior_m_win = _optional_posterior_array(
        data,
        f"posterior_m_win{key_suffix}",
        posterior_m_win_cont,
    )

    true_log10_tau = _truth_log10_array(data, "tau", key_suffix)
    true_log10_sigma = _optional_array(data, f"true_log10_sigma{key_suffix}", np.nan, n_obj)
    true_log10_D = _truth_log10_array(data, "D", key_suffix)
    true_log10_sigma2 = _truth_log10_array(data, "sigma2", key_suffix)
    primitive_truth = np.isfinite(true_log10_sigma) & np.isfinite(true_log10_tau)
    if np.any(primitive_truth):
        true_log10_D[primitive_truth] = 2.0 * true_log10_sigma[primitive_truth] - true_log10_tau[primitive_truth]
        true_log10_sigma2[primitive_truth] = 2.0 * true_log10_sigma[primitive_truth]
    missing_sigma2 = ~np.isfinite(true_log10_sigma2)
    if np.any(missing_sigma2) and np.any(np.isfinite(true_log10_sigma)):
        true_log10_sigma2[missing_sigma2] = 2.0 * true_log10_sigma[missing_sigma2]
    missing_sigma2 = ~np.isfinite(true_log10_sigma2)
    if np.any(missing_sigma2):
        derived_sigma2 = true_log10_D + true_log10_tau
        true_log10_sigma2[missing_sigma2] = derived_sigma2[missing_sigma2]
    true_mean_mag = _optional_truth_array(data, key_suffix, f"true_mean_mag{key_suffix}", "true_mean_mag", n_obj)
    true_m_win = _optional_truth_array(data, key_suffix, f"true_m_win{key_suffix}", "true_m_win", n_obj)
    true_m_win_cont = _optional_truth_array(data, key_suffix, f"true_m_win_cont{key_suffix}", "true_m_win_cont", n_obj)

    diag = {
        "baseline_rest": np.asarray(data.get(f"diag_baseline_rest{key_suffix}", np.full(source_id.size, np.nan)), dtype=float),
        "npts": np.asarray(data.get(f"diag_npts{key_suffix}", np.full(source_id.size, np.nan)), dtype=float),
        "nsteps_used": np.asarray(data.get(f"diag_nsteps_used{key_suffix}", np.full(source_id.size, np.nan)), dtype=float),
        "accept_mean": np.asarray(data.get(f"diag_accept_mean{key_suffix}", np.full(source_id.size, np.nan)), dtype=float),
        "iat_ln_D": np.asarray(data.get(f"diag_iat_ln_D{key_suffix}", np.full(source_id.size, np.nan)), dtype=float),
        "iat_ln_tau": np.asarray(data.get(f"diag_iat_ln_tau{key_suffix}", np.full(source_id.size, np.nan)), dtype=float),
        "neff_ln_D": np.asarray(data.get(f"diag_neff_ln_D{key_suffix}", np.full(source_id.size, np.nan)), dtype=float),
        "neff_ln_tau": np.asarray(data.get(f"diag_neff_ln_tau{key_suffix}", np.full(source_id.size, np.nan)), dtype=float),
    }

    sampling_label = None
    if isinstance(meta, dict):
        sampling = meta.get("sampling_model", {})
        if isinstance(sampling, dict):
            sampling_label = sampling.get("label", None)

    if label:
        run_label = label
    elif sampling_label:
        run_label = _format_sampling_label(sampling_label)
    else:
        run_label = npz_path.parent.name or npz_path.stem

    return {
        "label": run_label,
        "band": band,
        "npz_path": npz_path,
        "meta_path": meta_path,
        "meta": meta,
        "sampling_label": sampling_label or "unknown",
        "source_id": source_id,
        "grid_i": np.asarray(data.get("grid_i", np.full(source_id.size, -1)), dtype=int),
        "grid_j": np.asarray(data.get("grid_j", np.full(source_id.size, -1)), dtype=int),
        "posterior": {
            "log10_D": posterior_log10_D,
            "log10_sigma2": posterior_log10_sigma2,
            "log10_tau": posterior_log10_tau,
            "log10_jitter": posterior_log10_jitter,
            "mean_mag": posterior_mean_mag,
            "m_win_cont": posterior_m_win_cont,
            "m_win": posterior_m_win,
        },
        "truth": {
            "log10_D": true_log10_D,
            "log10_sigma2": true_log10_sigma2,
            "log10_tau": true_log10_tau,
            "log10_sigma": true_log10_sigma,
            "mean_mag": true_mean_mag,
            "m_win_cont": true_m_win_cont,
            "m_win": true_m_win,
        },
        "diag": diag,
    }


def _validate_compatible_runs(runs: list[dict[str, Any]]) -> None:
    ref = runs[0]
    for run in runs[1:]:
        if run["band"] != ref["band"]:
            raise ValueError(f"Band mismatch: {ref['label']} uses {ref['band']}, {run['label']} uses {run['band']}")
        for key in ["source_id", "grid_i", "grid_j"]:
            if not np.array_equal(run[key], ref[key]):
                raise ValueError(f"Run {run['label']} is not aligned with {ref['label']} for {key}.")
        for key in ["log10_D", "log10_sigma2", "log10_tau", "mean_mag", "m_win_cont", "m_win"]:
            a = np.asarray(run["truth"][key], dtype=float)
            b = np.asarray(ref["truth"][key], dtype=float)
            both = np.isfinite(a) & np.isfinite(b)
            if np.any(both) and not np.allclose(a[both], b[both], atol=1e-10, rtol=0.0):
                raise ValueError(f"Truth mismatch between {ref['label']} and {run['label']} for {key}.")


def _empty_bias_stats() -> dict[str, Any]:
    return {
        "median_bias": np.nan,
        "mean_bias": np.nan,
        "rmse": np.nan,
        "median_width": np.nan,
        "frac_within_0p3dex": np.nan,
        "frac_within_0p5dex": np.nan,
        "frac_uninformative": np.nan,
        "n_valid": 0,
    }


def _empty_metric(n_obj: int) -> dict[str, Any]:
    return {
        "bias": _empty_bias_stats(),
        "coverage": {"frac": np.nan, "n_within": 0, "n_total": 0},
        "posterior_median": np.full(int(n_obj), np.nan, dtype=float),
        "posterior_width": np.full(int(n_obj), np.nan, dtype=float),
    }


def _posterior_q16_q50_q84(post_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    post = np.asarray(post_samples, dtype=float)
    n_obj = int(post.shape[0]) if post.ndim >= 1 else 0
    cache_key = id(post_samples)
    cached = _POSTERIOR_Q_CACHE.get(cache_key)
    if cached is not None and cached[0].shape[0] == n_obj:
        return cached

    q16 = np.full(n_obj, np.nan, dtype=float)
    q50 = np.full(n_obj, np.nan, dtype=float)
    q84 = np.full(n_obj, np.nan, dtype=float)
    if post.ndim != 2 or post.shape[1] == 0:
        out = (q16, q50, q84)
        _POSTERIOR_Q_CACHE[cache_key] = out
        return out

    good_rows = np.any(np.isfinite(post), axis=1)
    if np.any(good_rows):
        q = np.nanpercentile(post[good_rows], [16, 50, 84], axis=1)
        q16[good_rows] = q[0]
        q50[good_rows] = q[1]
        q84[good_rows] = q[2]

    out = (q16, q50, q84)
    _POSTERIOR_Q_CACHE[cache_key] = out
    _POINT_ESTIMATE_CACHE[(cache_key, "median")] = q50
    return out


def _metric_from_quantiles(post_samples: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    post = np.asarray(post_samples, dtype=float)
    truth = np.asarray(truth, dtype=float)
    n_obj = int(truth.size)
    if post.ndim != 2 or post.shape[0] != n_obj or post.shape[1] == 0:
        return _empty_metric(n_obj)

    q16, q50, q84 = _posterior_q16_q50_q84(post_samples)
    width = q84 - q16
    valid = np.isfinite(truth) & np.isfinite(q50)
    if not np.any(valid):
        return {
            "bias": _empty_bias_stats(),
            "coverage": {"frac": np.nan, "n_within": 0, "n_total": 0},
            "posterior_median": q50,
            "posterior_width": width,
        }

    bias_values = q50[valid] - truth[valid]
    width_values = width[valid]
    finite_width = width_values[np.isfinite(width_values)]
    within = (truth[valid] >= q16[valid]) & (truth[valid] <= q84[valid])
    bias = {
        "median_bias": float(np.nanmedian(bias_values)),
        "mean_bias": float(np.nanmean(bias_values)),
        "rmse": float(np.sqrt(np.nanmean(bias_values * bias_values))),
        "median_width": float(np.nanmedian(finite_width)) if finite_width.size else np.nan,
        "frac_within_0p3dex": float(np.nanmean(np.abs(bias_values) < 0.3)),
        "frac_within_0p5dex": float(np.nanmean(np.abs(bias_values) < 0.5)),
        "frac_uninformative": float(np.nanmean(finite_width > 2.0)) if finite_width.size else np.nan,
        "n_valid": int(np.count_nonzero(valid)),
    }
    coverage = {
        "frac": float(np.mean(within)),
        "n_within": int(np.sum(within)),
        "n_total": int(np.count_nonzero(valid)),
    }
    return {
        "bias": bias,
        "coverage": coverage,
        "posterior_median": q50,
        "posterior_width": width,
    }


def _prepare_metrics(run: dict[str, Any]) -> None:
    metrics: dict[str, Any] = {}
    for param in ["log10_D", "log10_sigma2", "log10_tau", "mean_mag", "m_win_cont", "m_win"]:
        post = run["posterior"][param]
        truth = run["truth"][param]
        metrics[param] = _metric_from_quantiles(post, truth)
    run["metrics"] = metrics


def _point_estimate(post_samples: np.ndarray, estimator: str) -> np.ndarray:
    post = np.asarray(post_samples, dtype=float)
    n_obj = int(post.shape[0]) if post.ndim >= 1 else 0
    cache_key = (id(post_samples), estimator)
    cached = _POINT_ESTIMATE_CACHE.get(cache_key)
    if cached is not None and cached.shape[0] == n_obj:
        return cached

    out = np.full(n_obj, np.nan, dtype=float)
    if post.ndim != 2 or post.shape[1] == 0:
        _POINT_ESTIMATE_CACHE[cache_key] = out
        return out
    good_rows = np.any(np.isfinite(post), axis=1)
    if not np.any(good_rows):
        _POINT_ESTIMATE_CACHE[cache_key] = out
        return out
    if estimator == "median":
        _, out, _ = _posterior_q16_q50_q84(post_samples)
    elif estimator == "mean":
        out[good_rows] = np.nanmean(post[good_rows], axis=1)
    else:
        raise ValueError(f"Unknown estimator '{estimator}'")
    _POINT_ESTIMATE_CACHE[cache_key] = out
    return out


def _baseline_tau_ratio(run: dict[str, Any]) -> np.ndarray:
    baseline = np.asarray(run["diag"].get("baseline_rest", np.full(run["source_id"].size, np.nan)), dtype=float)
    tau_days = np.power(10.0, np.asarray(run["truth"]["log10_tau"], dtype=float))
    ratio = np.full_like(tau_days, np.nan, dtype=float)
    good = np.isfinite(baseline) & np.isfinite(tau_days) & (baseline > 0.0) & (tau_days > 0.0)
    ratio[good] = baseline[good] / tau_days[good]
    return ratio


def _common_log_ratio_edges(runs: list[dict[str, Any]], *, n_bins: int = 10) -> np.ndarray:
    vals: list[np.ndarray] = []
    for run in runs:
        x = np.log10(_baseline_tau_ratio(run))
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(x)
    if not vals:
        return np.linspace(-2.0, 4.0, n_bins + 1)
    cat = np.concatenate(vals)
    lo = float(np.floor(np.nanpercentile(cat, 1) * 2.0) / 2.0)
    hi = float(np.ceil(np.nanpercentile(cat, 99) * 2.0) / 2.0)
    lo = min(lo, float(np.floor(np.nanmin(cat) * 2.0) / 2.0))
    hi = max(hi, float(np.ceil(np.nanmax(cat) * 2.0) / 2.0))
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, n_bins + 1)


def _binned_summary(
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    *,
    min_count: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    centers: list[float] = []
    med: list[float] = []
    q16: list[float] = []
    q84: list[float] = []
    counts: list[int] = []
    for i in range(len(edges) - 1):
        if i == len(edges) - 2:
            mbin = m & (x >= edges[i]) & (x <= edges[i + 1])
        else:
            mbin = m & (x >= edges[i]) & (x < edges[i + 1])
        n = int(np.count_nonzero(mbin))
        if n < min_count:
            continue
        yb = y[mbin]
        centers.append(0.5 * (float(edges[i]) + float(edges[i + 1])))
        med.append(float(np.nanmedian(yb)))
        q16.append(float(np.nanpercentile(yb, 16)))
        q84.append(float(np.nanpercentile(yb, 84)))
        counts.append(n)
    return (
        np.asarray(centers, dtype=float),
        np.asarray(med, dtype=float),
        np.asarray(q16, dtype=float),
        np.asarray(q84, dtype=float),
        np.asarray(counts, dtype=int),
    )


def _kde1d(x: np.ndarray, xx: np.ndarray, bw_factor: float = 0.55) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.full_like(xx, np.nan, dtype=float)
    std = float(np.std(x, ddof=1))
    if not np.isfinite(std) or std <= 0:
        std = max(1e-3, 0.1 * (float(np.max(x)) - float(np.min(x)) + 1e-6))
    bw = float(bw_factor) * 1.06 * std * (x.size ** (-1.0 / 5.0))
    bw = max(float(bw), 1e-4)
    z = (xx[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * z * z).sum(axis=1) / (x.size * bw * np.sqrt(2.0 * np.pi))
    return dens


def _hist2d_contour(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: Any,
    xr: tuple[float, float],
    yr: tuple[float, float],
    linewidth: float | list[float] | None = None,
    fill_contours: bool = False,
    fill_alpha: float = 0.20,
    inner_fill_alpha: float = 0.34,
    bins: int | None = None,
    smooth: float | None = None,
    levels: tuple[float, ...] | list[float] | None = None,
) -> None:
    m = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[m], dtype=float)
    y = np.asarray(y[m], dtype=float)
    if x.size < 10:
        return
    if linewidth is None:
        contour_lw: float | list[float] = float(CORNER_DEFAULT_STYLE["contour_lw"])
    elif isinstance(linewidth, (list, tuple, np.ndarray)):
        contour_lw = [float(v) for v in linewidth]
    else:
        contour_lw = float(linewidth)
    bins = int(CORNER_HIST_BINS if bins is None else bins)
    smooth = float(CORNER_CONTOUR_SMOOTH if smooth is None else smooth)
    levels = list(CORNER_CONTOUR_LEVELS if levels is None else levels)
    if corner_lib is not None:
        try:
            corner_lib.hist2d(
                x,
                y,
                ax=ax,
                color=color,
                bins=int(bins),
                range=[xr, yr],
                smooth=float(smooth),
                levels=list(levels),
                plot_datapoints=False,
                plot_density=False,
                plot_contours=True,
                fill_contours=bool(fill_contours),
                no_fill_contours=True,
                contourf_kwargs={
                    "colors": [
                        to_rgba(color, 0.0),
                        to_rgba(color, float(fill_alpha)),
                        to_rgba(color, float(inner_fill_alpha)),
                    ]
                } if fill_contours else None,
                contour_kwargs={"linewidths": contour_lw, "alpha": 1.0},
            )
            return
        except Exception:
            pass
    H, xe, ye = np.histogram2d(x, y, bins=40, range=[xr, yr], density=True)
    H = H.T
    if not np.any(np.isfinite(H)) or np.nanmax(H) <= 0:
        return
    Hf = H[np.isfinite(H)].ravel()
    order = np.argsort(Hf)[::-1]
    csum = np.cumsum(Hf[order])
    csum /= csum[-1]
    levels = []
    for frac in (0.393, 0.865):
        idx = np.searchsorted(csum, frac, side="left")
        idx = min(idx, len(order) - 1)
        levels.append(Hf[order[idx]])
    levels = sorted(set(float(lv) for lv in levels if np.isfinite(lv)))
    if not levels:
        return
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    if fill_contours:
        fill_levels = [0.0, *levels, float(np.nanmax(H)) * (1.0 + 1.0e-4)]
        fill_colors = [
            to_rgba(color, 0.0),
            to_rgba(color, float(fill_alpha)),
            to_rgba(color, float(inner_fill_alpha)),
        ]
        ax.contourf(xc, yc, H, levels=fill_levels, colors=fill_colors, antialiased=False)
    ax.contour(xc, yc, H, levels=levels, colors=[color], linewidths=contour_lw)


def _kde2d_contour(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: Any,
    xr: tuple[float, float],
    yr: tuple[float, float],
    *,
    linewidths: float | list[float],
    fill_contours: bool = False,
    fill_alpha: float = 0.20,
    inner_fill_alpha: float = 0.34,
    levels: tuple[float, ...] = (0.68, 0.95),
    bw_factor: float = 1.3,
    gridsize: int = 200,
) -> None:
    """Smooth 2D credible contours via a Gaussian KDE (GetDist-style).

    Replaces the binned-histogram contours, which look jagged for small/odd-shaped
    posteriors.  ``levels`` are interpreted as enclosed posterior-mass fractions.
    """
    from scipy.stats import gaussian_kde

    m = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[m], dtype=float)
    y = np.asarray(y[m], dtype=float)
    if x.size < 10 or np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return
    try:
        kde = gaussian_kde(np.vstack([x, y]))
        kde.set_bandwidth(kde.factor * float(bw_factor))
    except Exception:
        return
    xs = np.linspace(float(xr[0]), float(xr[1]), int(gridsize))
    ys = np.linspace(float(yr[0]), float(yr[1]), int(gridsize))
    grid_x, grid_y = np.meshgrid(xs, ys)
    dens = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
    if not np.any(np.isfinite(dens)) or np.nanmax(dens) <= 0.0:
        return
    flat = dens[np.isfinite(dens)].ravel()
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order])
    csum /= csum[-1]
    lvls: list[float] = []
    for frac in levels:
        idx = min(int(np.searchsorted(csum, float(frac), side="left")), order.size - 1)
        lvls.append(float(flat[order[idx]]))
    lvls = sorted(set(lv for lv in lvls if np.isfinite(lv) and lv > 0.0))
    if not lvls:
        return
    if fill_contours:
        fill_levels = [0.0, *lvls, float(np.nanmax(dens)) * (1.0 + 1.0e-4)]
        n_bands = len(fill_levels) - 1
        if n_bands <= 1:
            fill_colors = [to_rgba(color, float(inner_fill_alpha))]
        elif n_bands == 2:
            fill_colors = [to_rgba(color, 0.0), to_rgba(color, float(inner_fill_alpha))]
        else:
            inner_alphas = np.linspace(float(fill_alpha), float(inner_fill_alpha), n_bands - 1)
            fill_colors = [to_rgba(color, 0.0)] + [to_rgba(color, float(a)) for a in inner_alphas]
        ax.contourf(grid_x, grid_y, dens, levels=fill_levels, colors=fill_colors, antialiased=True)
    ax.contour(grid_x, grid_y, dens, levels=lvls, colors=[color], linewidths=linewidths, antialiased=True)


def _clean_corner_tick_text(value: float, _pos: int | None = None) -> str:
    """Compact tick labels without implying unnecessary decimal precision."""
    x = float(value)
    if not np.isfinite(x):
        return ""

    def _fixed(decimals: int) -> str:
        scale = 10.0 ** int(decimals)
        rounded = np.sign(x) * np.floor(abs(x) * scale + 0.5 + 1.0e-12) / scale
        return f"{rounded:.{int(decimals)}f}".replace("-", "\N{MINUS SIGN}")

    if abs(x - round(x)) < 1.0e-8:
        return f"{int(round(x))}".replace("-", "\N{MINUS SIGN}")
    if abs(x) >= 10.0:
        return _fixed(2)
    if abs(10.0 * x - round(10.0 * x)) < 1.0e-8:
        return _fixed(1)
    return _fixed(2)


def _apply_corner_plain_tick_format(ax: plt.Axes, style: dict[str, Any] | None = None) -> None:
    """Avoid Matplotlib offset labels like '+1.906e1' on tightly zoomed panels."""
    if style is not None and bool(style.get("clean_tick_formatter", False)):
        formatter = FuncFormatter(_clean_corner_tick_text)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(formatter)
            axis.get_offset_text().set_visible(False)
        return
    for axis in (ax.xaxis, ax.yaxis):
        formatter = ScalarFormatter(useOffset=False, useMathText=False)
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
        axis.get_offset_text().set_visible(False)


def _apply_corner_panel_style(ax: plt.Axes, style: dict[str, float] | None = None) -> None:
    style = CORNER_DEFAULT_STYLE if style is None else style
    for spine in ax.spines.values():
        spine.set_linewidth(float(style["spine_lw"]))
    ax.tick_params(
        axis="both",
        which="both",
        width=float(style["tick_width"]),
        length=float(style["tick_length"]),
        labelsize=float(style["tick_labelsize"]),
        pad=8,
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.xaxis.labelpad = 9
    ax.yaxis.labelpad = 20


def _corner_annotation_decimals(err_lo: float, err_hi: float, default: int = 2, max_decimals: int = 8) -> int:
    """Use enough decimals that small nonzero uncertainties do not print as 0.00."""
    errs = [abs(float(e)) for e in (err_lo, err_hi) if np.isfinite(e) and abs(float(e)) > 0.0]
    if not errs:
        return default
    for ndp in range(default, max_decimals + 1):
        if all(float(f"{err:.{ndp}f}") > 0.0 for err in errs):
            return ndp
    return max_decimals


def _corner_source_param(param: str) -> str:
    return CORNER_PARAM_SOURCE.get(param, param)


def _corner_posterior_values(run: dict[str, Any], param: str, obj_idx: int) -> np.ndarray:
    source_param = _corner_source_param(param)
    return np.asarray(run["posterior"][source_param][obj_idx], dtype=float)


def _corner_truth_value(run: dict[str, Any], param: str, obj_idx: int) -> float:
    source_param = _corner_source_param(param)
    truth = run["truth"].get(source_param, None)
    if truth is None:
        return float("nan")
    return float(truth[obj_idx])


def _add_corner_page(
    pdf: PdfPages,
    runs: list[dict[str, Any]],
    colors: list[Any],
    obj_idx: int,
    source_id: int,
    title_note: str,
    annotate_run_idx: int = 0,
    show_title: bool = True,
    params: list[str] | None = None,
    param_labels: dict[str, str] | None = None,
    annotation_math: dict[str, str] | None = None,
    derived_params: set[str] | None = None,
    corner_style: dict[str, float] | None = None,
    draw_order: list[int] | None = None,
) -> None:
    params = CORNER_PARAMS if params is None else params
    param_labels = CORNER_PARAM_LABELS if param_labels is None else param_labels
    annotation_math = CORNER_PARAM_ANNOTATION_MATH if annotation_math is None else annotation_math
    derived_params = DERIVED_CORNER_PARAMS if derived_params is None else derived_params
    style = dict(CORNER_DEFAULT_STYLE)
    if corner_style is not None:
        style.update(corner_style)
    n = len(params)
    pad_frac = float(style.get("range_pad_frac", CORNER_RANGE_PAD_FRAC))
    # Order in which runs are *drawn* (later = on top). Defaults to natural order.
    # Legend order and the median-annotation target are kept in natural order, so
    # this only controls overplotting/z-order, not labels.
    if draw_order is None:
        order = list(range(len(runs)))
    else:
        order = [int(k) for k in draw_order if 0 <= int(k) < len(runs)]
        for k in range(len(runs)):
            if k not in order:
                order.append(k)
    ranges: dict[str, tuple[float, float]] = {}
    for p in params:
        post_vals = []
        truth_vals = []
        for run in runs:
            x = _corner_posterior_values(run, p, obj_idx)
            x = x[np.isfinite(x)]
            if x.size:
                post_vals.append(x)
            tv = _corner_truth_value(run, p, obj_idx)
            if np.isfinite(tv):
                truth_vals.append(float(tv))
        if post_vals:
            cat = np.concatenate(post_vals)
            qlo, qhi = CORNER_RANGE_QUANTILES_BY_PARAM.get(p, CORNER_RANGE_QUANTILES)
            lo, hi = np.percentile(cat, [qlo, qhi])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = float(np.min(cat)), float(np.max(cat))
        elif truth_vals:
            cat = np.asarray(truth_vals, dtype=float)
            lo, hi = float(np.min(cat)), float(np.max(cat))
        else:
            continue
        lo, hi = float(lo), float(hi)
        if lo == hi:
            lo -= 1e-3
            hi += 1e-3
        span = hi - lo
        # Per-parameter, per-edge padding overrides (fractions of the data span);
        # falls back to the scalar pad_frac on both edges when not specified.
        edge = style.get("range_pad_frac_by_param", {}).get(p)
        if edge is not None:
            lo_pad = float(edge[0]) * span
            hi_pad = float(edge[1]) * span
        else:
            lo_pad = hi_pad = pad_frac * span
        lo, hi = lo - lo_pad, hi + hi_pad
        for tv in truth_vals:
            lo = min(lo, tv - lo_pad)
            hi = max(hi, tv + hi_pad)
        ranges[p] = (lo, hi)

    fig, axes = plt.subplots(n, n, figsize=(20.0, 18.5))
    axes = np.atleast_2d(axes)
    ann_idx = min(annotate_run_idx, len(runs) - 1)

    for i, py in enumerate(params):
        for j, px in enumerate(params):
            ax = axes[i, j]
            if i < j:
                ax.axis("off")
                continue
            if i == j:
                for idx in order:
                    run, color = runs[idx], colors[idx]
                    x = _corner_posterior_values(run, px, obj_idx)
                    x = x[np.isfinite(x)]
                    if x.size:
                        xx = np.linspace(*ranges[px], 300)
                        line_lw = style["diag_annotated_lw"] if idx == ann_idx else style["diag_lw"]
                        diag_kwargs: dict[str, Any] = {}
                        cap = style.get("diag_capstyle")
                        if cap:
                            diag_kwargs["solid_capstyle"] = str(cap)
                            diag_kwargs["solid_joinstyle"] = str(cap)
                        bw_factor = float(style.get("diag_kde_bw_factor", 0.55))
                        ax.plot(
                            xx,
                            _kde1d(x, xx, bw_factor=bw_factor),
                            color=color,
                            lw=float(line_lw),
                            alpha=1.0,
                            **diag_kwargs,
                        )
                tv = _corner_truth_value(runs[0], px, obj_idx)
                if np.isfinite(tv):
                    ax.axvline(tv, color=style["truth_color"], lw=float(style["truth_lw_1d"]), ls="-", zorder=5)
                # Annotate with median^{+hi}_{-lo} above the diagonal panel (matching reference style)
                ann_run = runs[ann_idx]
                ann_x = _corner_posterior_values(ann_run, px, obj_idx)
                ann_x = ann_x[np.isfinite(ann_x)]
                if ann_x.size >= 2:
                    med = float(np.median(ann_x))
                    q16, q84 = float(np.percentile(ann_x, 16)), float(np.percentile(ann_x, 84))
                    ann_color = colors[ann_idx]
                    for qv, lw in (
                        (q16, style["quantile_lw"]),
                        (med, style["median_lw"]),
                        (q84, style["quantile_lw"]),
                    ):
                        ax.axvline(float(qv), color=ann_color, lw=lw, ls="--", alpha=0.85, zorder=4)
                    err_lo, err_hi = med - q16, q84 - med
                    math_lbl = annotation_math.get(px, param_labels[px])
                    ndp = _corner_annotation_decimals(err_lo, err_hi)
                    title_weight = "bold" if px in derived_params else "normal"
                    ann_xpos = CORNER_DIAG_TITLE_X.get(px, 0.5)
                    ann_y_by_param = style.get("diag_title_y_by_param", {})
                    ann_ypos = (
                        float(ann_y_by_param.get(px, style.get("diag_title_y", 1.025)))
                        if isinstance(ann_y_by_param, dict)
                        else float(style.get("diag_title_y", 1.025))
                    )
                    box_colors = style.get("diag_label_box_colors", {})
                    box_color = box_colors.get(px) if isinstance(box_colors, dict) else None
                    if box_color is not None:
                        fs = float(style["diag_annotation_fontsize"])
                        external_patches = style.get("diag_label_box_external_axes_patch_by_param", {})
                        external_patch = (
                            external_patches.get(px)
                            if isinstance(external_patches, dict)
                            else None
                        )
                        if external_patch is not None:
                            x0, y0, w, h = [float(v) for v in external_patch]
                            ax.add_patch(
                                FancyBboxPatch(
                                    (x0, y0),
                                    w,
                                    h,
                                    transform=ax.transAxes,
                                    boxstyle=str(
                                        style.get(
                                            "diag_label_box_external_boxstyle",
                                            "round,pad=0.01",
                                        )
                                    ),
                                    facecolor=str(box_color),
                                    edgecolor="none",
                                    alpha=float(style.get("diag_label_box_alpha", 0.92)),
                                    clip_on=False,
                                    zorder=2,
                                )
                            )
                        math_pad_by_param = style.get("diag_label_box_math_pad_by_param", {})
                        math_pad = (
                            str(math_pad_by_param.get(px, ""))
                            if isinstance(math_pad_by_param, dict)
                            else ""
                        )
                        math_left_pad_by_param = style.get("diag_label_box_math_left_pad_by_param", {})
                        math_right_pad_by_param = style.get("diag_label_box_math_right_pad_by_param", {})
                        if isinstance(math_left_pad_by_param, dict) or isinstance(math_right_pad_by_param, dict):
                            left_pad = (
                                str(math_left_pad_by_param.get(px, math_pad))
                                if isinstance(math_left_pad_by_param, dict)
                                else math_pad
                            )
                            right_pad = (
                                str(math_right_pad_by_param.get(px, math_pad))
                                if isinstance(math_right_pad_by_param, dict)
                                else math_pad
                            )
                            boxed_math_lbl = (
                                f"{left_pad} {math_lbl} {right_pad}"
                                if (left_pad or right_pad)
                                else math_lbl
                            )
                        else:
                            boxed_math_lbl = f"{math_pad} {math_lbl} {math_pad}" if math_pad else math_lbl
                        makebox_width_by_param = style.get("diag_label_box_makebox_width_by_param", {})
                        makebox_align_by_param = style.get("diag_label_box_makebox_align_by_param", {})
                        makebox_width = (
                            str(makebox_width_by_param.get(px, ""))
                            if isinstance(makebox_width_by_param, dict)
                            else ""
                        )
                        if makebox_width:
                            makebox_align = (
                                str(makebox_align_by_param.get(px, "c"))
                                if isinstance(makebox_align_by_param, dict)
                                else "c"
                            )
                            label_text = rf"\makebox[{makebox_width}][{makebox_align}]{{${math_lbl}$}}"
                        else:
                            label_text = f"${boxed_math_lbl}$"
                        if external_patch is not None:
                            label_text = f"${math_lbl}$"
                        label_textprops = {
                                "fontsize": fs,
                                "color": "black",
                                "fontweight": title_weight,
                        }
                        if external_patch is None:
                            label_textprops["bbox"] = {
                                    "boxstyle": str(
                                        style.get("diag_label_boxstyle_by_param", {}).get(
                                            px,
                                            style.get("diag_label_boxstyle", "round,pad=0.08"),
                                        )
                                    ),
                                    "facecolor": str(box_color),
                                    "edgecolor": "none",
                                    "alpha": float(style.get("diag_label_box_alpha", 0.92)),
                            }
                        label_area = TextArea(
                            label_text,
                            textprops=label_textprops,
                        )
                        value_area = TextArea(
                            f"$ = {med:.{ndp}f}^{{+{err_hi:.{ndp}f}}}_{{-{err_lo:.{ndp}f}}}$",
                            textprops={
                                "fontsize": fs,
                                "color": "black",
                                "fontweight": title_weight,
                            },
                        )
                        sep_by_param = style.get("diag_label_box_sep_by_param", {})
                        label_value_sep = (
                            sep_by_param.get(px, style.get("diag_label_box_sep", 1.5))
                            if isinstance(sep_by_param, dict)
                            else style.get("diag_label_box_sep", 1.5)
                        )
                        packed = HPacker(
                            children=[label_area, value_area],
                            align="baseline",
                            pad=0.0,
                            sep=float(label_value_sep),
                        )
                        ax.add_artist(
                            AnnotationBbox(
                                packed,
                                (ann_xpos, ann_ypos),
                                xycoords=ax.transAxes,
                                box_alignment=(0.5, 0.0),
                                frameon=False,
                                pad=0.0,
                                annotation_clip=False,
                            )
                        )
                    else:
                        ann_txt = f"${math_lbl} = {med:.{ndp}f}^{{+{err_hi:.{ndp}f}}}_{{-{err_lo:.{ndp}f}}}$"
                        ax.text(
                            ann_xpos,
                            ann_ypos,
                            ann_txt,
                            transform=ax.transAxes,
                            ha="center",
                            va="bottom",
                            fontsize=float(style["diag_annotation_fontsize"]),
                            color="black",
                            fontweight=title_weight,
                            clip_on=False,
                        )
                ax.set_xlim(*ranges[px])
                ax.set_yticks([])
            else:
                for idx in order:
                    run, color = runs[idx], colors[idx]
                    x = _corner_posterior_values(run, px, obj_idx)
                    y = _corner_posterior_values(run, py, obj_idx)
                    if bool(style.get("corner_kde_2d", False)):
                        _kde2d_contour(
                            ax,
                            x,
                            y,
                            color,
                            ranges[px],
                            ranges[py],
                            linewidths=style["contour_lw"],
                            fill_contours=bool(style.get("fill_contours", False)),
                            fill_alpha=float(style.get("contour_fill_alpha", 0.20)),
                            inner_fill_alpha=float(style.get("contour_fill_inner_alpha", 0.34)),
                            levels=tuple(style.get("corner_levels", CORNER_CONTOUR_LEVELS)),
                            bw_factor=float(style.get("kde2d_bw_factor", 1.3)),
                            gridsize=int(style.get("kde2d_gridsize", 200)),
                        )
                    else:
                        _hist2d_contour(
                            ax,
                            x,
                            y,
                            color,
                            ranges[px],
                            ranges[py],
                            linewidth=style["contour_lw"],
                            fill_contours=bool(style.get("fill_contours", False)),
                            fill_alpha=float(style.get("contour_fill_alpha", 0.20)),
                            inner_fill_alpha=float(style.get("contour_fill_inner_alpha", 0.34)),
                            bins=style.get("corner_bins"),
                            smooth=style.get("corner_smooth"),
                            levels=style.get("corner_levels"),
                        )
                tx = _corner_truth_value(runs[0], px, obj_idx)
                ty = _corner_truth_value(runs[0], py, obj_idx)
                if np.isfinite(tx):
                    ax.axvline(tx, color=style["truth_color"], lw=float(style["truth_lw_2d"]), ls="-", zorder=5)
                if np.isfinite(ty):
                    ax.axhline(ty, color=style["truth_color"], lw=float(style["truth_lw_2d"]), ls="-", zorder=5)
                ax.set_xlim(*ranges[px])
                ax.set_ylim(*ranges[py])

            _apply_corner_plain_tick_format(ax, style)
            _apply_corner_panel_style(ax, style)
            if i == j:
                ax.set_yticks([])
                ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
            if i == n - 1:
                ax.set_xlabel(
                    param_labels[px],
                    fontsize=float(style["axis_labelsize"]),
                    color="black",
                    fontweight="bold" if px in derived_params else "normal",
                )
            else:
                ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
            if j == 0 and i != j:
                ax.set_ylabel(
                    param_labels[py],
                    fontsize=float(style["axis_labelsize"]),
                    color="black",
                    fontweight="bold" if py in derived_params else "normal",
                )
                if "ylabel_x" in style:
                    ax.yaxis.set_label_coords(float(style["ylabel_x"]), 0.5)
            elif i != j:
                ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    handles = []
    legend_labels = []
    handler_map = {}
    if bool(style.get("legend_group_priors", False)):
        if bool(style.get("legend_patch_handles", False)):
            prior_handle = tuple(
                Patch(
                    facecolor=to_rgba(color, 0.75),
                    edgecolor=color,
                    linewidth=1.2,
                )
                for color in colors
            )
        else:
            prior_handle = tuple(
                Line2D([0], [0], color=color, lw=float(style["legend_lw"]))
                for color in colors
            )
        handles.append(prior_handle)
        legend_labels.append(str(style.get("legend_group_label", "Different priors")))
        if bool(style.get("legend_prior_grid", False)):
            handler_map[tuple] = HandlerPriorGrid(
                n_top=int(style.get("legend_prior_grid_top", 2)),
            )
        else:
            handler_map[tuple] = HandlerTuple(
                ndivide=None,
                pad=float(style.get("legend_group_pad", 0.12)),
            )
    else:
        for idx, (run, color) in enumerate(zip(runs, colors)):
            label = _corner_legend_label(run)
            if idx == ann_idx:
                label = rf"{label} (median $\pm$ 68\%)"
            if bool(style.get("legend_patch_handles", False)):
                handles.append(
                    Patch(
                        facecolor=to_rgba(color, 0.75),
                        edgecolor=color,
                        linewidth=1.6,
                    )
                )
            else:
                handles.append(Line2D([0], [0], color=color, lw=float(style["legend_lw"])))
            legend_labels.append(label)
    handles.append(Line2D([0], [0], color=style["truth_color"], lw=float(style["truth_legend_lw"])))
    legend_labels.append("Truth")
    title = f"Overlaid Corner Comparison for source_id={source_id}"
    if title_note:
        title = f"{title}: {title_note}"
    if show_title:
        fig.suptitle(_tex_escape_text(title), fontsize=28, fontweight="bold", y=0.985)
    if bool(style.get("show_legend", True)):
        fig.legend(
        handles=handles,
        labels=legend_labels,
        loc="upper right",
        bbox_to_anchor=(float(style["legend_bbox_x"]), float(style["legend_bbox_y"])),
        frameon=False,
        fontsize=float(style["legend_fontsize"]),
        handlelength=float(style.get("legend_handlelength", 2.0)),
        handleheight=float(style.get("legend_handleheight", 0.7)),
        handler_map=handler_map,
    )
    fig.subplots_adjust(left=0.085, right=0.94, bottom=0.09, top=0.90 if show_title else 0.94, wspace=0.06, hspace=0.06)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


CORNER_GROUP_COLORS = {
    "cosmology": "#009E73",
    "variability": "#E69F00",
    "scatter": "#CC79A7",
}


def load_stage2_output(posterior_npz: Path) -> Dict[str, Any]:
    """Load a saved Stage-2 parameter dictionary and validate its chain/draw shapes.
    
    Grouped arrays have shape (chains, retained draws); pooled arrays supply
    posterior summaries. These NPZ files contain pickled dictionaries and should
    only be loaded from trusted sources. Return the samples and stored metadata."""
    npz = np.load(posterior_npz, allow_pickle=True)

    required = {"samples_grouped", "samples", "columns", "num_chains", "num_samples", "num_warmup"}
    missing = [k for k in required if k not in npz.files]
    if missing:
        raise ValueError(f"Missing required fields in {posterior_npz}: {missing}. Available keys: {list(npz.files)}")

    samples_grouped = npz["samples_grouped"].item()
    samples_flat = npz["samples"].item()
    columns = [str(x) for x in npz["columns"].tolist()]
    num_chains = int(npz["num_chains"])
    num_samples = int(npz["num_samples"])
    num_warmup = int(npz["num_warmup"])

    chain_method = None
    if "chain_method" in npz.files:
        try:
            chain_method = str(npz["chain_method"])
        except Exception:
            chain_method = None

    data_metadata = None
    if "data_metadata" in npz.files:
        try:
            data_metadata = npz["data_metadata"].item()
        except Exception:
            data_metadata = None

    a_band = None
    if "a_band" in npz.files:
        try:
            a_band = float(npz["a_band"])
        except Exception:
            a_band = None

    band = None
    if "band" in npz.files:
        try:
            band = str(npz["band"])
        except Exception:
            band = None
    source_id = None
    if "source_id" in npz.files:
        try:
            source_id = np.asarray(npz["source_id"], dtype=np.int64)
        except Exception:
            source_id = None

    # Validate grouped shapes
    for p in columns:
        if p not in samples_grouped:
            raise ValueError(f"Parameter '{p}' missing from samples_grouped keys: {list(samples_grouped.keys())}")
        arr = np.asarray(samples_grouped[p])
        if arr.ndim != 2:
            raise ValueError(f"Expected samples_grouped['{p}'] to be 2D (chains, draws), got shape {arr.shape}")
        if arr.shape[0] != num_chains or arr.shape[1] != num_samples:
            raise ValueError(
                f"Shape mismatch for '{p}': expected ({num_chains}, {num_samples}), got {arr.shape}"
            )

    return {
        "samples_grouped": samples_grouped,
        "samples_flat": samples_flat,
        "columns": columns,
        "num_chains": num_chains,
        "num_samples": num_samples,
        "num_warmup": num_warmup,
        "chain_method": chain_method,
        "data_metadata": data_metadata,
        "a_band": a_band,
        "band": band,
        "source_id": source_id,
        "npz_keys": list(npz.files),
    }


def _validate_chain_source_ids_with_manifest(
    loaded_by_file: list[tuple[Path, Dict[str, Any]]],
    run_config_obj: Dict[str, Any],
) -> np.ndarray:
    sel = run_config_obj.get("selection_manifest", {}) if isinstance(run_config_obj.get("selection_manifest", {}), dict) else {}
    used_path_raw = str(sel.get("path", "")).strip()
    if not used_path_raw:
        raise ValueError('run_config missing selection_manifest.path (selected_source_ids.npz)')
    used_path = Path(used_path_raw)
    if not used_path.exists():
        raise FileNotFoundError(f"selection manifest does not exist: {used_path}")

    used = np.load(used_path, allow_pickle=False)
    if "source_id_selected" not in used.files:
        raise KeyError(f"{used_path} missing key 'source_id_selected'")
    canonical = np.asarray(used["source_id_selected"], dtype=np.int64)
    if canonical.ndim != 1:
        raise ValueError("source_id_selected must be a 1D array")
    if np.unique(canonical).size != canonical.size:
        raise ValueError("source_id_selected contains duplicates")
    canon_sorted = np.sort(canonical)

    for p, loaded in loaded_by_file:
        sid = loaded.get("source_id", None)
        if sid is None:
            raise ValueError(
                f"{p} missing source_id array. Re-run Stage 2 with code/framework/stage2_fitting.py to include source_id."
            )
        sid = np.asarray(sid, dtype=np.int64)
        if sid.ndim != 1:
            raise ValueError(f"{p}: source_id must be 1D")
        if np.unique(sid).size != sid.size:
            raise ValueError(f"{p}: source_id contains duplicates")

        sid_sorted = np.sort(sid)
        if sid_sorted.size != canon_sorted.size or not np.array_equal(sid_sorted, canon_sorted):
            missing = np.setdiff1d(canon_sorted, sid_sorted, assume_unique=False)
            extra = np.setdiff1d(sid_sorted, canon_sorted, assume_unique=False)
            raise ValueError(
                f"{p}: source_id set mismatch vs used_source_ids.npz (missing={missing.size}, extra={extra.size})"
            )

        # Crossmatch to canonical order (no order assumption).
        order = np.argsort(sid)
        sid_s = sid[order]
        idx = np.searchsorted(sid_s, canonical)
        ok = (idx < sid_s.size) & (sid_s[idx] == canonical)
        if not np.all(ok):
            raise ValueError(f"{p}: failed to crossmatch all canonical source IDs")
        loaded["source_id"] = sid[order[idx]]

    return canonical


def _merge_stage2_outputs(loaded_by_file: list[tuple[Path, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Merge multiple checker-compatible stage2 outputs into one in-memory object.
    Keeps chains separate for diagnostics by stacking on chain axis, and creates
    pooled flat arrays for posterior summaries/PPC.
    """
    if not loaded_by_file:
        raise ValueError("No stage2 outputs provided for merge.")

    base_path, base = loaded_by_file[0]
    columns = list(base["columns"])
    num_samples = int(base["num_samples"])
    num_warmup = int(base["num_warmup"])

    grouped_acc: Dict[str, list[np.ndarray]] = {p: [] for p in columns}
    for p, arr in base["samples_grouped"].items():
        grouped_acc[p].append(np.asarray(arr))

    total_chains = int(base["num_chains"])
    files_used = [str(base_path)]

    for path, loaded in loaded_by_file[1:]:
        files_used.append(str(path))
        if list(loaded["columns"]) != columns:
            raise ValueError(
                f"Columns mismatch while merging.\nBase ({base_path}): {columns}\nThis ({path}): {loaded['columns']}"
            )
        if int(loaded["num_samples"]) != num_samples:
            raise ValueError(
                f"num_samples mismatch while merging.\nBase ({base_path}): {num_samples}\nThis ({path}): {loaded['num_samples']}"
            )
        if int(loaded["num_warmup"]) != num_warmup:
            raise ValueError(
                f"num_warmup mismatch while merging.\nBase ({base_path}): {num_warmup}\nThis ({path}): {loaded['num_warmup']}"
            )
        for p in columns:
            grouped_acc[p].append(np.asarray(loaded["samples_grouped"][p]))
        total_chains += int(loaded["num_chains"])

    samples_grouped = {p: np.concatenate(grouped_acc[p], axis=0) for p in columns}
    samples_flat = {p: samples_grouped[p].reshape(-1) for p in columns}

    out = dict(base)
    out["samples_grouped"] = samples_grouped
    out["samples_flat"] = samples_flat
    out["num_chains"] = int(total_chains)
    out["files_merged"] = files_used
    return out


def load_config_json(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def _ordered_display_columns(columns: list[str]) -> list[str]:
    """
    Order parameters for display across pages (tables/traces/prior-vs-post/corner).

    Preferred order:
      Om0, w0, (any other cosmology), then d_D, D_pivot, alpha_D, sigma_int, then any remaining.
    Also drops log_sigma_int (keep sigma_int only).
    """
    drop = {"log_sigma_int"}
    cols = [c for c in columns if c not in drop]

    preferred = ["Om0", "w0", "H0", "wa", "d_D", "D_pivot", "alpha_D", "sigma_int"]
    ordered: list[str] = []
    seen = set()
    for p in preferred:
        if p in cols and p not in seen:
            ordered.append(p)
            seen.add(p)
    for p in cols:
        if p not in seen:
            ordered.append(p)
            seen.add(p)
    return ordered


def _mock_truth_by_param(
    run_config_obj: Dict[str, Any] | None,
    columns: list[str],
) -> Dict[str, float]:
    """
    Extract truth values for sampled/displayed parameters from a mock-aware
    run_config. Converts log_sigma_int truth into sigma_int for display pages.

    Read injected values from run_config["truth"] or
    run_config["mock_inputs"]["truth"], including available pivot conversions.
    """
    if not isinstance(run_config_obj, dict):
        return {}

    truth_sources: list[Dict[str, Any]] = []

    top_truth = run_config_obj.get("truth", {})
    if isinstance(top_truth, dict):
        truth_sources.append(top_truth)

    mock_inputs = run_config_obj.get("mock_inputs", {})
    if isinstance(mock_inputs, dict):
        truth = mock_inputs.get("truth", {})
        if isinstance(truth, dict):
            truth_sources.append(truth)
        truth_wpiv = mock_inputs.get("truth_wpiv", {})
        if isinstance(truth_wpiv, dict):
            truth_sources.append(truth_wpiv)

    wpiv_cfg = run_config_obj.get("wpiv_parametrization", {})
    if not isinstance(wpiv_cfg, dict):
        wpiv_cfg = {}

    combined_truth: Dict[str, float] = {}
    for source in truth_sources:
        for key, value in source.items():
            try:
                combined_truth[str(key)] = float(value)
            except Exception:
                continue

    if "w0" not in combined_truth and "w_piv" in combined_truth:
        try:
            c_piv = float(wpiv_cfg.get("pivot_coeff", np.nan))
            if np.isfinite(c_piv) and "wa" in combined_truth:
                combined_truth["w0"] = float(combined_truth["w_piv"] - c_piv * combined_truth["wa"])
        except Exception:
            pass

    out: Dict[str, float] = {}
    for p in columns:
        if p in combined_truth:
            try:
                out[p] = float(combined_truth[p])
            except Exception:
                continue
        elif p == "sigma_int" and "log_sigma_int" in combined_truth:
            try:
                out[p] = float(np.exp(float(combined_truth["log_sigma_int"])))
            except Exception:
                continue
    return out


def _display_param_label(param: str) -> str:
    """Pretty mathtext labels for common Stage-2 parameters."""
    return {
        "H0": r"$H_0$",
        "Om0": r"$\Omega_{\mathrm{m}}$",
        "w0": r"$w_0$",
        "w_piv": r"$w_{\rm piv}$",
        "wa": r"$w_a$",
        "D_pivot": r"$\eta_{D,0}$",
        "d_D": r"$d_D$",
        "alpha_D": r"$\alpha_D$",
        "sigma_int": r"$\sigma_{\rm int}$",
        "log_sigma_int": r"$\log \sigma_{\rm int}$",
        "a_D": r"$a_D$",
    }.get(param, param)


def _poster_corner_param_group(param: str) -> str:
    if param in {"Om0", "w0", "w_piv", "wa", "H0"}:
        return "cosmology"
    if param in {"d_D", "D_pivot", "alpha_D", "a_D"}:
        return "variability"
    if param in {"sigma_int", "log_sigma_int"}:
        return "scatter"
    return "cosmology"


POSTERIOR_LN10 = float(np.log(10.0))


STAGE1_SUMMARY_CACHE_VERSION = 1


DEFAULT_CLOSURE_DIR = Path(
    str(DATA_ROOT / 'results/cosmology_closure/closure_tests/cadence_matched_sample')
)


DEFAULT_SAMPLE_LABEL = 'cadence_matched_sample'


DEFAULT_RUN_NAME = "run1_prior1_lnsigma_lntau_K1500"


def _default_pair_npz(closure_dir: Path, sample_label: str) -> Path:
    return closure_dir / f"{sample_label}_pairs_and_truth.npz"


def _default_stage1_npz(closure_dir: Path, run_name: str) -> Path:
    return closure_dir / "stage1" / run_name / f"posteriors_{run_name}.npz"


def _align_to_reference_ids(
    *,
    values: np.ndarray,
    value_source_ids: np.ndarray,
    reference_source_ids: np.ndarray,
    name: str,
) -> np.ndarray:
    values = np.asarray(values)
    value_source_ids = np.asarray(value_source_ids, dtype=np.int64).reshape(-1)
    reference_source_ids = np.asarray(reference_source_ids, dtype=np.int64).reshape(-1)

    if values.shape[0] != value_source_ids.size:
        raise ValueError(f"{name}: value rows and source_id rows differ")
    if np.array_equal(value_source_ids, reference_source_ids):
        return values

    order = np.argsort(value_source_ids)
    sorted_ids = value_source_ids[order]
    loc = np.searchsorted(sorted_ids, reference_source_ids)
    found = loc < sorted_ids.size
    found[found] &= sorted_ids[loc[found]] == reference_source_ids[found]
    if not np.all(found):
        missing = reference_source_ids[~found][:10]
        raise ValueError(f"{name}: missing {np.count_nonzero(~found)} source IDs; first missing={missing}")
    return values[order[loc]]


def _nanmedian_rows(arr: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D posterior array; got shape={arr.shape}")
    return np.nanmedian(arr, axis=1, overwrite_input=True)


def _cache_n_draws_value(n_draws_limit: int | None) -> int:
    return -1 if n_draws_limit is None else int(n_draws_limit)


def _load_stage1_summary_cache(
    *,
    cache_npz: Path | None,
    stage1_npz: Path,
    reference_source_ids: np.ndarray,
    reference_z: np.ndarray,
    band: str,
    n_draws_limit: int | None,
) -> dict[str, Any] | None:
    if cache_npz is None or not cache_npz.exists():
        return None
    try:
        stat = stage1_npz.stat()
        reference_source_ids = np.asarray(reference_source_ids, dtype=np.int64)
        reference_z = np.asarray(reference_z, dtype=np.float64)
        with np.load(cache_npz, allow_pickle=False) as cache:
            required = {
                "cache_version",
                "stage1_npz",
                "stage1_size",
                "stage1_mtime_ns",
                "band",
                "n_draws_limit",
                "reference_source_id",
                "reference_z",
                "z",
                "logl3000",
                "log10d",
                "log10d_halfwidth",
                "m_win",
                "has_log10tau",
                "log10tau",
            }
            missing = sorted(required - set(cache.files))
            if missing:
                raise ValueError(f"missing keys {missing}")
            if int(cache["cache_version"]) != STAGE1_SUMMARY_CACHE_VERSION:
                raise ValueError("cache version mismatch")
            if str(np.asarray(cache["stage1_npz"]).item()) != str(stage1_npz.resolve()):
                raise ValueError("stage1 path mismatch")
            if int(cache["stage1_size"]) != int(stat.st_size):
                raise ValueError("stage1 file size changed")
            if int(cache["stage1_mtime_ns"]) != int(stat.st_mtime_ns):
                raise ValueError("stage1 file mtime changed")
            if str(np.asarray(cache["band"]).item()) != str(band):
                raise ValueError("band mismatch")
            if int(cache["n_draws_limit"]) != _cache_n_draws_value(n_draws_limit):
                raise ValueError("draw-limit mismatch")
            if not np.array_equal(np.asarray(cache["reference_source_id"], dtype=np.int64), reference_source_ids):
                raise ValueError("reference source IDs changed")
            if not np.allclose(
                np.asarray(cache["reference_z"], dtype=np.float64),
                reference_z,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            ):
                raise ValueError("reference redshifts changed")

            print(f"Using cached Stage-1 posterior summaries: {cache_npz}", flush=True)
            row = {
                "z": np.asarray(cache["z"], dtype=np.float64),
                "logl3000": np.asarray(cache["logl3000"], dtype=np.float64),
                "log10d": np.asarray(cache["log10d"], dtype=np.float64),
                "log10d_halfwidth": np.asarray(cache["log10d_halfwidth"], dtype=np.float64),
                "m_win": np.asarray(cache["m_win"], dtype=np.float64),
                "label": r"Mock Stage-1 posterior medians",
                "value_kind": "med",
                "logl_axis_label": r"$\log_{10}(L_{3000}\ [\mathrm{post.}\ m])$",
                "mwin_axis_label": r"$m_{50}$",
            }
            if bool(np.asarray(cache["has_log10tau"]).item()):
                row["log10tau"] = np.asarray(cache["log10tau"], dtype=np.float64)
            return row
    except Exception as exc:
        print(f"[cache] Ignoring stale/unusable Stage-1 summary cache {cache_npz}: {exc}", flush=True)
        return None


def _write_stage1_summary_cache(
    *,
    cache_npz: Path | None,
    stage1_npz: Path,
    reference_source_ids: np.ndarray,
    reference_z: np.ndarray,
    band: str,
    n_draws_limit: int | None,
    row: dict[str, Any],
) -> None:
    if cache_npz is None:
        return
    try:
        stat = stage1_npz.stat()
        cache_npz.parent.mkdir(parents=True, exist_ok=True)
        has_log10tau = "log10tau" in row
        log10tau = (
            np.asarray(row["log10tau"], dtype=np.float64)
            if has_log10tau
            else np.asarray([], dtype=np.float64)
        )
        np.savez_compressed(
            cache_npz,
            cache_version=np.asarray(STAGE1_SUMMARY_CACHE_VERSION, dtype=np.int16),
            stage1_npz=np.asarray(str(stage1_npz.resolve())),
            stage1_size=np.asarray(int(stat.st_size), dtype=np.int64),
            stage1_mtime_ns=np.asarray(int(stat.st_mtime_ns), dtype=np.int64),
            band=np.asarray(str(band)),
            n_draws_limit=np.asarray(_cache_n_draws_value(n_draws_limit), dtype=np.int64),
            reference_source_id=np.asarray(reference_source_ids, dtype=np.int64),
            reference_z=np.asarray(reference_z, dtype=np.float64),
            z=np.asarray(row["z"], dtype=np.float64),
            logl3000=np.asarray(row["logl3000"], dtype=np.float64),
            log10d=np.asarray(row["log10d"], dtype=np.float64),
            log10d_halfwidth=np.asarray(row["log10d_halfwidth"], dtype=np.float64),
            m_win=np.asarray(row["m_win"], dtype=np.float64),
            has_log10tau=np.asarray(bool(has_log10tau)),
            log10tau=log10tau,
        )
        print(f"Cached Stage-1 posterior summaries: {cache_npz}", flush=True)
    except Exception as exc:
        print(f"[cache] Could not write Stage-1 summary cache {cache_npz}: {exc}", flush=True)


def _load_pair_rows(pair_npz: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    with np.load(pair_npz, allow_pickle=False) as pair:
        mock_source_id = np.asarray(pair["mock_source_id"], dtype=np.int64)
        real_temple_row = {
            "z": np.asarray(pair["real_z"], dtype=np.float64),
            "logl3000": np.asarray(pair["real_logl3000_temple"], dtype=np.float64),
            "log10d": np.asarray(pair["real_log10_d_med"], dtype=np.float64),
            "m_win": np.asarray(pair["real_m_win_med"], dtype=np.float64),
            "label": "Real templates",
            "value_kind": "med",
            "logl_axis_label": r"$\log_{10}(L_{3000}\ [\mathrm{Temple{+}2021}])$",
        }
        real_mwin_row = {
            "z": np.asarray(pair["real_z"], dtype=np.float64),
            "logl3000": np.asarray(pair["real_logl3000_from_mwin"], dtype=np.float64),
            "log10d": np.asarray(pair["real_log10_d_med"], dtype=np.float64),
            "m_win": np.asarray(pair["real_m_win_med"], dtype=np.float64),
            "label": r"Real templates from $m$",
            "value_kind": "med",
            "logl_axis_label": r"$\log_{10}(L_{3000}\ [\mathrm{from}\ m])$",
        }
        mock_truth_row = {
            "z": np.asarray(pair["mock_z"], dtype=np.float64),
            "logl3000": np.asarray(pair["mock_logl3000"], dtype=np.float64),
            "log10d": np.asarray(pair["mock_log10_d_true"], dtype=np.float64),
            "m_win": np.asarray(pair["mock_m_win_cont_true"], dtype=np.float64),
            "label": r"Mock LC $m_{\rm true}$",
            "value_kind": "true",
            "mwin_axis_label": r"$m_{\rm true}$",
        }
    return [real_temple_row, real_mwin_row, mock_truth_row], mock_source_id


def _load_stage1_posterior_row(
    *,
    stage1_npz: Path,
    reference_source_ids: np.ndarray,
    reference_z: np.ndarray,
    band: str,
    n_draws_limit: int | None = None,
    summary_cache_npz: Path | None = None,
) -> dict[str, Any]:
    """Build source-aligned Stage-1 medians and widths for observed/mock comparisons.
    
    Align by source ID, convert natural-log DRW draws to base-10 values, and map
    window magnitudes to luminosity using the fixed plotting cosmology. A matching
    summary cache can supply the same reductions. n_draws_limit selects the first
    requested columns, preserving their original ordering."""
    key_ln_d = f"posterior_ln_D_{band}"
    key_ln_tau = f"posterior_ln_tau_{band}"
    key_mwin_cont = f"posterior_m_win_cont_{band}"

    cached = _load_stage1_summary_cache(
        cache_npz=summary_cache_npz,
        stage1_npz=stage1_npz,
        reference_source_ids=reference_source_ids,
        reference_z=reference_z,
        band=band,
        n_draws_limit=n_draws_limit,
    )
    if cached is not None:
        return cached

    with np.load(stage1_npz, allow_pickle=False) as post:
        keys = set(post.files)
        missing = sorted({key_ln_d, key_mwin_cont, "source_id"} - keys)
        if missing:
            raise KeyError(f"Stage-1 NPZ missing required keys: {missing}")

        post_source_id = np.asarray(post["source_id"], dtype=np.int64)

        def _load_draws(name: str) -> np.ndarray:
            raw = post[name]
            if n_draws_limit is None:
                return np.asarray(raw)
            n_avail = int(raw.shape[1])
            if n_draws_limit <= 0 or n_draws_limit > n_avail:
                raise ValueError(
                    f"{name}: requested first {n_draws_limit} draws but only "
                    f"{n_avail} are available"
                )
            print(
                f"  Restricting {name} to first {n_draws_limit:,} of {n_avail:,} draws.",
                flush=True,
            )
            return np.asarray(raw[:, :n_draws_limit])

        print(f"Reading {key_mwin_cont} and taking row medians ...", flush=True)
        mwin_cont_arr = _load_draws(key_mwin_cont)
        mwin_cont_med = _nanmedian_rows(mwin_cont_arr, name=key_mwin_cont)
        mwin_cont_med = _align_to_reference_ids(
            values=mwin_cont_med,
            value_source_ids=post_source_id,
            reference_source_ids=reference_source_ids,
            name=key_mwin_cont,
        )

        print(f"Reading {key_ln_d} and taking row quantiles ...", flush=True)
        log10d_arr = _load_draws(key_ln_d) / POSTERIOR_LN10
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
            log10d_q16, log10d_med, log10d_q84 = np.nanpercentile(
                log10d_arr,
                [16.0, 50.0, 84.0],
                axis=1,
                overwrite_input=True,
            )
        log10d_halfwidth = 0.5 * (log10d_q84 - log10d_q16)
        log10d_med = _align_to_reference_ids(
            values=log10d_med,
            value_source_ids=post_source_id,
            reference_source_ids=reference_source_ids,
            name=key_ln_d,
        )
        log10d_halfwidth = _align_to_reference_ids(
            values=log10d_halfwidth,
            value_source_ids=post_source_id,
            reference_source_ids=reference_source_ids,
            name=f"{key_ln_d}_halfwidth",
        )

        log10tau_med = None
        if key_ln_tau in keys:
            print(f"Reading {key_ln_tau} and taking row medians ...", flush=True)
            log10tau_arr = _load_draws(key_ln_tau) / POSTERIOR_LN10
            log10tau_med = _nanmedian_rows(log10tau_arr, name=key_ln_tau)
            log10tau_med = _align_to_reference_ids(
                values=log10tau_med,
                value_source_ids=post_source_id,
                reference_source_ids=reference_source_ids,
                name=key_ln_tau,
            )

    print("Converting posterior m_win_cont medians to log10(L3000) ...", flush=True)
    logl3000_med = real_logl3000_from_mwin(
        z=np.asarray(reference_z, dtype=np.float64),
        mwin_med=np.asarray(mwin_cont_med, dtype=np.float64),
        band="gaia_g",
    )

    row = {
        "z": np.asarray(reference_z, dtype=np.float64),
        "logl3000": logl3000_med,
        "log10d": log10d_med,
        "log10d_halfwidth": log10d_halfwidth,
        "m_win": mwin_cont_med,
        "label": r"Mock Stage-1 posterior medians",
        "value_kind": "med",
        "logl_axis_label": r"$\log_{10}(L_{3000}\ [\mathrm{post.}\ m])$",
        "mwin_axis_label": r"$m_{50}$",
    }
    if log10tau_med is not None:
        row["log10tau"] = log10tau_med
    _write_stage1_summary_cache(
        cache_npz=summary_cache_npz,
        stage1_npz=stage1_npz,
        reference_source_ids=reference_source_ids,
        reference_z=reference_z,
        band=band,
        n_draws_limit=n_draws_limit,
        row=row,
    )
    return row


PLOT_MPLCONFIGDIR = f"/tmp/agn-variability-cosmology-mpl-{os.getuid()}"


PLOT_MATPLOTLIBRC = str(PROJECT_ROOT / "matplotlibrc")


def _padded_range(*arrays: np.ndarray, pad: float = 0.04) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(a, dtype=np.float64)[np.isfinite(a)] for a in arrays])
    lo, hi = float(np.min(vals)), float(np.max(vals))
    span = hi - lo
    return (lo - pad * span, hi + pad * span)


def _write_selected_page(
    *,
    out_pdf: Path,
    rows_for_limits: list[dict[str, Any]],
    rows_to_plot: list[dict[str, Any]],
    scenario_name: str,
    title_suffix: str,
    selected_columns: tuple[int, ...] = (0, 1, 2, 3),
    presentation: bool = False,
    show_hist_row: bool = False,
    n_label_columns: tuple[int, ...] | None = None,
    show_n_out: bool = True,
    n_label_font_scale: float = 1.0,
    axis_label_font_scale: float = 1.0,
    tick_label_font_scale: float = 1.0,
    colorbar_tick_pad: float | None = None,
    colorbar_label_pad: float | None = None,
    compact_logl_label: bool = False,
    compact_z_label: bool = False,
    mwin_med_axis_label: bool = False,
    column_cmaps: dict[int, str] | None = None,
    d_panel_z_range: tuple[float, float] | None = None,
    d_panel_z_bin_width: float | None = None,
    d_binned_panel_xlim: tuple[float, float] | None = None,
    d_binned_panel_ylim: tuple[float, float] | None = None,
    d_binned_n_label_xlim: tuple[float, float] | None = None,
    d_binned_n_label_ylim: tuple[float, float] | None = None,
    d_binned_uncert_style: str = "none",
    mwin_panel_ylim: tuple[float, float] | None = None,
    logl_axis_min: float | None = None,
    row_label_x_override: float | None = None,
    row_label_y_override: float | None = None,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", PLOT_MPLCONFIGDIR)

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rc_file(PLOT_MATPLOTLIBRC)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator

    fs_tick = 44 if presentation else 32
    fs_label = 56 if presentation else 38
    fs_cbar_label = 50 if presentation else 34
    fs_cbar_tick = 42 if presentation else 30
    fs_row_label = 60 if presentation else fs_label
    tick_width = 3.4 if presentation else 2.5
    tick_length = 12 if presentation else 10
    spine_lw = 3.4 if presentation else 2.0
    row_label_x = -0.32 if presentation else -0.28
    row_label_y = 0.5
    if row_label_x_override is not None:
        row_label_x = float(row_label_x_override)
    if row_label_y_override is not None:
        row_label_y = float(row_label_y_override)
    fs_label *= float(axis_label_font_scale)
    fs_cbar_label *= float(axis_label_font_scale)
    fs_tick *= float(tick_label_font_scale)
    fs_cbar_tick *= float(tick_label_font_scale)
    fs_row_label = fs_label
    cbar_tick_pad = 9.0 if colorbar_tick_pad is None else float(colorbar_tick_pad)
    cbar_label_pad = 8.0 if colorbar_label_pad is None else float(colorbar_label_pad)
    column_cmaps = dict(column_cmaps or {})

    def panel_cmap(panel_col: int, default: str) -> str:
        return str(column_cmaps.get(int(panel_col), default))

    def style_ax(ax: Any) -> None:
        ax.tick_params(axis="both", labelsize=fs_tick, width=tick_width, length=tick_length, pad=10)
        ax.xaxis.label.set_size(fs_label)
        ax.yaxis.label.set_size(fs_label)
        for spine in ax.spines.values():
            spine.set_linewidth(spine_lw)

    def add_cbar(fig: Any, hb_or_sm: Any, ax: Any, label: str) -> Any:
        cbar = fig.colorbar(hb_or_sm, ax=ax)
        cbar.set_label(label, fontsize=fs_cbar_label, labelpad=cbar_label_pad)
        cbar.ax.tick_params(
            labelsize=fs_cbar_tick,
            width=tick_width,
            length=max(6, tick_length - 2),
            pad=cbar_tick_pad,
        )
        cbar.outline.set_linewidth(spine_lw)
        return cbar

    def add_n_label(
        ax: Any,
        n: int,
        n_outside: int | None = None,
        info_label: str | None = None,
    ) -> None:
        text = f"$N={n:,}$"
        has_n_outside = bool(show_n_out) and n_outside is not None and n_outside > 0
        if has_n_outside:
            text += "\n" + rf"$N_{{\rm out}}={n_outside:,}$"
        has_info = bool(info_label)
        if has_info:
            text += "\n" + str(info_label)
        if has_n_outside and has_info:
            scale = 0.6
        elif has_n_outside or has_info:
            scale = 0.72
        else:
            scale = 1.0
        n_lines = text.count("\n") + 1
        if n_lines >= 6:
            scale = min(scale, 0.46)
        elif n_lines >= 5:
            scale = min(scale, 0.52)
        elif n_lines >= 4:
            scale = min(scale, 0.60)
        ax.text(
            0.97,
            0.97,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=fs_tick * scale * float(n_label_font_scale),
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.75,
                edgecolor="none",
                linewidth=0.0,
            ),
        )

    def axis_visible_mask(
        x: np.ndarray,
        y: np.ndarray,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        *extra: np.ndarray,
    ) -> tuple[np.ndarray, int, int]:
        m_finite = np.isfinite(x) & np.isfinite(y)
        for arr in extra:
            m_finite &= np.isfinite(arr)
        m_visible = m_finite.copy()
        m_visible &= (x >= xlim[0]) & (x <= xlim[1])
        m_visible &= (y >= ylim[0]) & (y <= ylim[1])
        n_visible = int(np.count_nonzero(m_visible))
        n_outside = int(np.count_nonzero(m_finite) - n_visible)
        return m_visible, n_visible, n_outside

    def hex_panel(
        ax: Any,
        x: np.ndarray,
        y: np.ndarray,
        *,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        C: np.ndarray | None = None,
        reduce_C_function: Any = np.mean,
        gridsize: int = 60,
        cmap: str = "viridis",
        mincnt: int = 1,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> tuple[Any | None, int, int]:
        extra = () if C is None else (C,)
        m, n_visible, n_outside = axis_visible_mask(x, y, xlim, ylim, *extra)
        if n_visible == 0:
            return None, n_visible, n_outside
        extent = (float(xlim[0]), float(xlim[1]), float(ylim[0]), float(ylim[1]))
        if C is not None:
            hb = ax.hexbin(
                x[m],
                y[m],
                C=C[m],
                reduce_C_function=reduce_C_function,
                gridsize=gridsize,
                mincnt=mincnt,
                cmap=cmap,
                extent=extent,
                vmin=vmin,
                vmax=vmax,
            )
            return hb, n_visible, n_outside
        hb = ax.hexbin(x[m], y[m], gridsize=gridsize, mincnt=mincnt, cmap=cmap, extent=extent)
        return hb, n_visible, n_outside

    def plot_binned_log10d_vs_logl(
        ax: Any,
        *,
        z: np.ndarray,
        logl: np.ndarray,
        log10d: np.ndarray,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        d_binned_axis_label: str,
        logl_axis_label: str,
        n_z_bins: int = 6,
        n_l_bins: int = 16,
        cmap: str = "viridis",
        z_range: tuple[float, float] | None = None,
        z_bin_width: float | None = None,
        line_width: float = 2.6,
        uncert_style: str = "none",
        postprop: dict[str, Any] | None = None,
    ) -> tuple[Any | None, int, int]:
        m, n_visible, n_outside = axis_visible_mask(logl, log10d, xlim, ylim, z)
        z = np.asarray(z[m], dtype=np.float64)
        logl = np.asarray(logl[m], dtype=np.float64)
        log10d = np.asarray(log10d[m], dtype=np.float64)
        if z.size == 0:
            return None, n_visible, n_outside

        if z_range is None:
            z_lo_edge = float(np.min(z))
            z_hi_edge = float(np.max(z))
        else:
            z_lo_edge, z_hi_edge = map(float, z_range)
        if z_hi_edge <= z_lo_edge:
            raise ValueError(f"invalid redshift bin range: {z_range}")

        if z_bin_width is None:
            z_edges = np.linspace(z_lo_edge, z_hi_edge, int(n_z_bins) + 1)
        else:
            dz = float(z_bin_width)
            if dz <= 0.0:
                raise ValueError(f"invalid redshift bin width: {z_bin_width}")
            n_bins = int(np.ceil((z_hi_edge - z_lo_edge) / dz))
            z_edges = z_lo_edge + dz * np.arange(n_bins + 1, dtype=np.float64)
            if z_edges[-1] < z_hi_edge:
                z_edges = np.append(z_edges, z_hi_edge)
            else:
                z_edges[-1] = z_hi_edge
        logl_edges = np.linspace(float(xlim[0]), float(xlim[1]), int(n_l_bins) + 1)
        logl_centers = 0.5 * (logl_edges[:-1] + logl_edges[1:])
        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
        cmap_b = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=float(z_lo_edge), vmax=float(z_hi_edge))
        pp_q16 = pp_q50 = pp_q84 = None
        if uncert_style == "postprop" and postprop is not None:
            pp_q16 = np.asarray(postprop.get("q16", []), dtype=np.float64)
            pp_q50 = np.asarray(postprop.get("q50", []), dtype=np.float64)
            pp_q84 = np.asarray(postprop.get("q84", []), dtype=np.float64)

        n_z_intervals = int(z_edges.size - 1)
        for i in range(n_z_intervals):
            z_lo, z_hi = z_edges[i], z_edges[i + 1]
            mids: list[float] = []
            meds: list[float] = []
            q16s: list[float] = []
            q84s: list[float] = []
            for j in range(int(n_l_bins)):
                l_lo, l_hi = logl_edges[j], logl_edges[j + 1]
                mask = (z >= z_lo) & (z < z_hi if i < n_z_intervals - 1 else z <= z_hi)
                mask &= (logl >= l_lo) & (logl < l_hi if j < int(n_l_bins) - 1 else logl <= l_hi)
                if np.count_nonzero(mask) == 0:
                    continue
                mids.append(float(logl_centers[j]))
                q16, med, q84 = np.nanpercentile(log10d[mask], [16.0, 50.0, 84.0])
                if pp_q16 is not None and pp_q50 is not None and pp_q84 is not None:
                    if (
                        i < pp_q16.shape[0]
                        and j < pp_q16.shape[1]
                        and np.isfinite(pp_q16[i, j])
                        and np.isfinite(pp_q50[i, j])
                        and np.isfinite(pp_q84[i, j])
                    ):
                        q16 = float(pp_q16[i, j])
                        med = float(pp_q50[i, j])
                        q84 = float(pp_q84[i, j])
                    else:
                        q16 = float(med)
                        q84 = float(med)
                meds.append(float(med))
                q16s.append(float(q16))
                q84s.append(float(q84))
            if mids:
                mids_arr = np.asarray(mids, dtype=np.float64)
                meds_arr = np.asarray(meds, dtype=np.float64)
                q16_arr = np.asarray(q16s, dtype=np.float64)
                q84_arr = np.asarray(q84s, dtype=np.float64)
                color = cmap_b(norm(float(z_centers[i])))
                ax.plot(
                    mids_arr,
                    meds_arr,
                    marker="o",
                    ms=7.5,
                    lw=float(line_width),
                    color=color,
                    alpha=0.95,
                )
                if uncert_style in {"errorbar", "postprop", "postprop_paired"}:
                    yerr = np.vstack([meds_arr - q16_arr, q84_arr - meds_arr])
                    ax.errorbar(
                        mids_arr,
                        meds_arr,
                        yerr=yerr,
                        fmt="none",
                        ecolor=color,
                        elinewidth=0.9,
                        capsize=0.0,
                        alpha=0.35,
                    )
                elif uncert_style == "shade":
                    ax.fill_between(mids_arr, q16_arr, q84_arr, color=color, alpha=0.10, linewidth=0.0)

        ax.set_xlabel(logl_axis_label)
        ax.set_ylabel(d_binned_axis_label)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        sm = plt.cm.ScalarMappable(cmap=cmap_b, norm=norm)
        sm.set_array([])
        return sm, n_visible, n_outside

    z_lim = _padded_range(*(r["z"] for r in rows_for_limits))
    logl_lim = _padded_range(*(r["logl3000"] for r in rows_for_limits))
    logl_lower = 43.0 if logl_axis_min is None else float(logl_axis_min)
    d_panel_xlim = (logl_lower, logl_lim[1])
    d_panel_ylim = (-7.0, -2.0)
    d_binned_panel_xlim = d_panel_xlim if d_binned_panel_xlim is None else tuple(map(float, d_binned_panel_xlim))
    d_binned_panel_ylim = d_panel_ylim if d_binned_panel_ylim is None else tuple(map(float, d_binned_panel_ylim))
    d_binned_n_label_xlim = (
        d_binned_panel_xlim
        if d_binned_n_label_xlim is None
        else tuple(map(float, d_binned_n_label_xlim))
    )
    d_binned_n_label_ylim = (
        d_binned_panel_ylim
        if d_binned_n_label_ylim is None
        else tuple(map(float, d_binned_n_label_ylim))
    )
    zlogl_panel_ylim = (logl_lower, logl_lim[1])
    mwin_panel_ylim = (14.0, 22.0) if mwin_panel_ylim is None else tuple(map(float, mwin_panel_ylim))
    if d_panel_z_range is not None:
        d_panel_z_range = (float(d_panel_z_range[0]), float(d_panel_z_range[1]))

    n_plot_rows = len(rows_to_plot)
    n_plot_cols = len(selected_columns)
    n_label_column_set = set(selected_columns) if n_label_columns is None else set(n_label_columns)
    n_total_grid_rows = n_plot_rows + (1 if show_hist_row else 0)
    fig_width = 15.0 * n_plot_cols if presentation else 11.0 * n_plot_cols
    fig_height = (11.5 if presentation else 10.0) * n_total_grid_rows
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=150, constrained_layout=True)
    gs = GridSpec(n_total_grid_rows, n_plot_cols, figure=fig, wspace=0.08, hspace=0.06)

    for row_idx, row in enumerate(rows_to_plot):
        z = np.asarray(row["z"], dtype=np.float64)
        logl = np.asarray(row["logl3000"], dtype=np.float64)
        log10d = np.asarray(row["log10d"], dtype=np.float64)
        mwin = np.asarray(row["m_win"], dtype=np.float64)
        logl_axis_label = (
            r"$\log_{10}(L_{3000})$"
            if presentation
            else str(row.get("logl_axis_label", r"$\log_{10}(L_{3000})$"))
        )
        if compact_logl_label:
            logl_axis_label = r"$\ell$"
        z_axis_label = r"$z$" if compact_z_label else r"redshift $z$"
        is_true = row.get("value_kind") == "true"
        if is_true:
            d_label = r"$\eta_{\rm true}$"
            d_binned_label = r"$\mathrm{median}(\eta_{\rm true})$"
            default_m_label = r"$m_{\rm true}$"
        else:
            d_label = r"$\eta_{50}$"
            d_binned_label = r"$\mathrm{median}(\eta_{50})$"
            default_m_label = r"$m_{50}$"
        mwin_label = str(row.get("mwin_axis_label", default_m_label))
        if mwin_med_axis_label:
            mwin_label = r"$m_{50}$"

        axes = {
            panel_col: fig.add_subplot(gs[row_idx, out_col])
            for out_col, panel_col in enumerate(selected_columns)
        }
        for ax in axes.values():
            style_ax(ax)

        info_parts = [
            str(value)
            for value in (row.get("prior_label"), row.get("cut_label"))
            if value is not None and str(value)
        ]
        info_label = "\n".join(info_parts) if info_parts else None

        def maybe_add_n_label(panel_col: int, ax: Any, n_visible: int, n_outside: int) -> None:
            if panel_col in n_label_column_set:
                add_n_label(ax, n_visible, n_outside, info_label=info_label)

        if 0 in axes:
            sm, n_visible, n_outside = plot_binned_log10d_vs_logl(
                axes[0],
                z=z,
                logl=logl,
                log10d=log10d,
                xlim=d_binned_panel_xlim,
                ylim=d_binned_panel_ylim,
                d_binned_axis_label=d_binned_label,
                logl_axis_label=logl_axis_label,
                cmap=panel_cmap(0, "viridis"),
                z_range=d_panel_z_range,
                z_bin_width=d_panel_z_bin_width,
                line_width=float(row.get("d_binned_line_width", 2.6)),
                uncert_style=str(row.get("d_binned_uncert_style", d_binned_uncert_style)),
                postprop=row.get("d_binned_postprop"),
            )
            axes[0].set_xlim(d_binned_panel_xlim)
            axes[0].set_ylim(d_binned_panel_ylim)
            if d_binned_n_label_xlim != d_binned_panel_xlim or d_binned_n_label_ylim != d_binned_panel_ylim:
                _, n_visible, n_outside = axis_visible_mask(
                    logl,
                    log10d,
                    d_binned_n_label_xlim,
                    d_binned_n_label_ylim,
                    z,
                )
            maybe_add_n_label(0, axes[0], n_visible, n_outside)
            if sm is not None:
                add_cbar(fig, sm, axes[0], r"$z$ bin midpoint")

        if 1 in axes:
            hb, n_visible, n_outside = hex_panel(
                axes[1],
                logl,
                log10d,
                xlim=d_panel_xlim,
                ylim=d_panel_ylim,
                C=z,
                reduce_C_function=np.mean,
                cmap=panel_cmap(1, "viridis"),
                vmin=None if d_panel_z_range is None else d_panel_z_range[0],
                vmax=None if d_panel_z_range is None else d_panel_z_range[1],
            )
            axes[1].set_xlabel(logl_axis_label)
            axes[1].set_ylabel(d_label)
            axes[1].set_xlim(d_panel_xlim)
            axes[1].set_ylim(d_panel_ylim)
            maybe_add_n_label(1, axes[1], n_visible, n_outside)
            if hb is not None:
                add_cbar(fig, hb, axes[1], r"$\langle z \rangle$")

        if 2 in axes:
            hb, n_visible, n_outside = hex_panel(
                axes[2],
                z,
                logl,
                xlim=z_lim,
                ylim=zlogl_panel_ylim,
                cmap=panel_cmap(2, "cividis"),
            )
            axes[2].set_xlabel(z_axis_label)
            axes[2].set_ylabel(logl_axis_label)
            axes[2].set_xlim(z_lim)
            axes[2].set_ylim(zlogl_panel_ylim)
            maybe_add_n_label(2, axes[2], n_visible, n_outside)
            if hb is not None:
                add_cbar(fig, hb, axes[2], "count")

        if 3 in axes:
            hb, n_visible, n_outside = hex_panel(
                axes[3],
                z,
                mwin,
                xlim=z_lim,
                ylim=mwin_panel_ylim,
                cmap=panel_cmap(3, "plasma"),
            )
            axes[3].set_xlabel(z_axis_label)
            axes[3].set_ylabel(mwin_label)
            axes[3].set_xlim(z_lim)
            axes[3].set_ylim(mwin_panel_ylim)
            maybe_add_n_label(3, axes[3], n_visible, n_outside)
            if hb is not None:
                add_cbar(fig, hb, axes[3], "count")

        row_label = str(row["label"])
        if presentation:
            if row_label == "Real, with quality cuts":
                row_label = r"\textbf{Real data}"
            elif row_label == "Mock, run-matched":
                row_label = r"\textbf{Mock data}"
        axes[selected_columns[0]].annotate(
            row_label,
            xy=(0, row_label_y),
            xycoords="axes fraction",
            xytext=(row_label_x, row_label_y),
            textcoords="axes fraction",
            fontsize=fs_row_label,
            rotation=90,
            rotation_mode="anchor",
            va="center",
            ha="center",
            fontweight="bold",
            annotation_clip=False,
        )

    if show_hist_row:
        _HIST_QUANTITY: dict[int, str] = {0: "logl3000", 1: "log10d", 2: "z", 3: "m_win"}
        _HIST_XLIM: dict[int, tuple[float, float]] = {
            0: d_panel_xlim,
            1: d_panel_ylim,
            2: z_lim,
            3: mwin_panel_ylim,
        }
        _HIST_XLABEL: dict[int, str] = {
            0: r"$\log_{10}(L_{3000})$",
            1: r"$\eta$",
            2: r"redshift $z$",
            3: r"$m$",
        }
        _HIST_PALETTE = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        hist_colors = _HIST_PALETTE[: len(rows_to_plot)]
        hist_axes: dict[int, Any] = {}

        for out_col, panel_col in enumerate(selected_columns):
            ax_h = fig.add_subplot(gs[n_plot_rows, out_col])
            style_ax(ax_h)
            hist_axes[out_col] = ax_h

            qty = _HIST_QUANTITY.get(panel_col)
            xlim = _HIST_XLIM.get(panel_col)
            xlabel = _HIST_XLABEL.get(panel_col, "")

            if qty is None:
                ax_h.set_visible(False)
                continue

            for ridx, (row, color) in enumerate(zip(rows_to_plot, hist_colors)):
                vals = np.asarray(row[qty], dtype=np.float64)
                m = np.isfinite(vals)
                if xlim is not None:
                    m &= (vals >= xlim[0]) & (vals <= xlim[1])
                vals = vals[m]
                if vals.size == 0:
                    continue

                hist_label = str(row.get("label", f"Row {ridx}"))
                logl_ax = str(row.get("logl_axis_label", ""))
                if "Temple" in logl_ax:
                    hist_label = r"Real (Temple$+$2021)"
                elif logl_ax and ("from" in logl_ax or "m_win" in logl_ax):
                    hist_label = r"Real (from $m$)"

                ax_h.hist(
                    vals,
                    bins=50,
                    density=True,
                    histtype="step",
                    color=color,
                    lw=2.5,
                    label=hist_label,
                )

            ax_h.set_xlabel(xlabel)
            ax_h.set_ylabel("density")
            if xlim is not None:
                ax_h.set_xlim(xlim)
            ax_h.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax_h.yaxis.set_major_locator(MaxNLocator(nbins=4))

            if out_col == 0:
                ax_h.legend(fontsize=fs_tick * 0.85, loc="upper left", framealpha=0.8)

        if hist_axes:
            first_out_col = min(hist_axes.keys())
            hist_axes[first_out_col].annotate(
                "Distributions",
                xy=(0, row_label_y),
                xycoords="axes fraction",
                xytext=(row_label_x, row_label_y),
                textcoords="axes fraction",
                fontsize=fs_row_label,
                rotation=90,
                rotation_mode="anchor",
                va="center",
                ha="center",
                fontweight="bold",
                annotation_clip=False,
            )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
