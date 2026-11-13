#!/usr/bin/env python
"""Paper-style overlay corner plot for full-closure Stage-2 prior comparison."""

from __future__ import annotations

from framework import PROJECT_ROOT, DATA_ROOT

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator

try:
    import corner
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("The 'corner' package is required for this plot.") from exc


from plotting import diagnostics as base


DEFAULT_PRIOR5_RUN_DIR = DATA_ROOT / "results/cosmology_closure/prior_5/stage2"
DEFAULT_PRIOR1_RUN_DIR = DATA_ROOT / "results/cosmology_closure/prior_1/stage2"
DEFAULT_OUT_DIR = PROJECT_ROOT / 'figures'
DEFAULT_OUT_PDF = DEFAULT_OUT_DIR / 'figure_06_cosmology_closure.pdf'

PRIOR_COLOR_PALETTES = {
    "current": {
        1: "#009E73",
        2: "#0072B2",
        3: "#D55E00",
        4: "#E69F00",
        5: "#FF00FF",
    },
    "colorblindpalette": {
        1: "#009E73",
        2: "#0072B2",
        3: "#D55E00",
        4: "#E69F00",
        5: "#FF00FF",
    },
    "newpalette": {
        1: "#2C7A7B",
        2: "#6FA66E",
        3: "#F2A000",
        4: "#B51D3A",
        5: "#8A005D",
    },
    "extrapalette": {
        1: "#006D2C",
        2: "#0072B2",
        3: "#00A6D6",
        4: "#E69F00",
        5: "#FF00FF",
    },
}
DEFAULT_PALETTE_NAME = "current"
PRIOR_COLORS_BY_NUMBER = PRIOR_COLOR_PALETTES[DEFAULT_PALETTE_NAME]
PRIOR5_COLOR = PRIOR_COLORS_BY_NUMBER[5]
PRIOR1_COLOR = PRIOR_COLORS_BY_NUMBER[1]
DEFAULT_MULTI_COLORS = [PRIOR_COLORS_BY_NUMBER[idx] for idx in range(1, 6)] + ["#9467BD"]
CORNER_COLUMN_ORDER = ["Om0", "w0", "wa", "D_pivot", "d_D", "alpha_D", "sigma_int"]
RUN_HIST_LINEWIDTH = 2.8
RUN_CONTOUR_LINEWIDTH = 2.6
RUN_QUANTILE_LINEWIDTHS = (2.0, 2.5, 2.0)
TRUTH_COLOR = "#000000"
TRUTH_LINEWIDTH = 3.0
LEGEND_LINEWIDTH = 3.2
DIAGONAL_GROUP_BOXES = {
    "cosmology": {"label": "Cosmology", "color": "#4472C4", "alpha": 0.72},
    "variability_luminosity": {
        "label": "Variability-Luminosity relation", "color": "#4C956C", "alpha": 0.70,
    },
    "scatter": {"label": "Intrinsic scatter", "color": "#E6B84A", "alpha": 0.74},
}
RANGE_PAD_FRACTION = 0.26
RANGE_PAD_FRACTION_BY_PARAM = {
    "d_D": 0.55,
    "sigma_int": 0.55,
}
CONTOUR_FILL_ALPHA = 0.20
CONTOUR_FILL_INNER_ALPHA = 0.34
SMOOTH_STYLE_FILL_ALPHA = 0.22
SMOOTH_STYLE_FILL_INNER_ALPHA = 0.48
SMOOTH_STYLE_OUTER_CONTOUR_LW = 1.25
SMOOTH_STYLE_INNER_CONTOUR_LW = 2.55
SMOOTH_STYLE_BINS = 44
SMOOTH_STYLE_SMOOTH = 1.45
SMOOTH_STYLE_SMOOTH1D = 1.80
SMOOTH_STYLE_KDE_POINTS = 650
SMOOTH_STYLE_KDE_BW_FACTOR = 1.25


def _contour_fill_colors(
    color: str,
    *,
    outer_alpha: float = CONTOUR_FILL_ALPHA,
    inner_alpha: float = CONTOUR_FILL_INNER_ALPHA,
) -> list[tuple[float, float, float, float]]:
    return [
        to_rgba(color, 0.0),
        to_rgba(color, outer_alpha),
        to_rgba(color, inner_alpha),
    ]


def _chain_sort_key(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"chain_(\d+)\.npz", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def _chain_id_from_path(path: Path) -> int | None:
    match = re.fullmatch(r"chain_(\d+)\.npz", path.name)
    return int(match.group(1)) if match else None


def _chain_paths_from_run(
    run_dir: Path,
    run_config: Dict[str, Any],
    *,
    max_chain_id: int | None = None,
) -> list[Path]:
    chains = run_config.get("chains", []) if isinstance(run_config.get("chains", []), list) else []
    paths: list[Path] = []
    for chain in chains:
        raw = str(chain.get("npz_path", "")).strip() if isinstance(chain, dict) else ""
        if raw:
            # Resolve configured chain filenames within the selected run directory.
            path = run_dir / Path(raw).name
            raw_chain_id = chain.get("chain_id") if isinstance(chain, dict) else None
            try:
                chain_id = int(raw_chain_id)
            except (TypeError, ValueError):
                chain_id = _chain_id_from_path(path)
            if max_chain_id is None or chain_id is None or chain_id <= max_chain_id:
                paths.append(path)
    if not paths:
        paths = [p for p in run_dir.glob("chain_*.npz") if re.fullmatch(r"chain_\d+\.npz", p.name)]
        if max_chain_id is not None:
            paths = [p for p in paths if (_chain_id_from_path(p) or 0) <= max_chain_id]
    paths = sorted(paths, key=_chain_sort_key)
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing chain files: " + ", ".join(str(p) for p in missing))
    if not paths:
        suffix = f" with chain id <= {max_chain_id}" if max_chain_id is not None else ""
        raise FileNotFoundError(f"No chain_N.npz files found in {run_dir}{suffix}")
    return paths


def _load_run(
    run_dir: Path,
    label: str,
    color: str,
    *,
    max_chain_id: int | None = None,
) -> Dict[str, Any]:
    cfg_path = run_dir / 'inference_settings.json'
    if not cfg_path.exists():
        raise FileNotFoundError(f"Run config not found: {cfg_path}")
    cfg = base.load_config_json(cfg_path)
    chain_paths = _chain_paths_from_run(run_dir, cfg, max_chain_id=max_chain_id)
    loaded_by_file = [(p, base.load_stage2_output(p)) for p in chain_paths]
    validation_cfg = dict(cfg)
    selection_manifest = dict(validation_cfg.get("selection_manifest", {}))
    selection_manifest["path"] = str(run_dir / 'selected_source_ids.npz')
    validation_cfg["selection_manifest"] = selection_manifest
    base._validate_chain_source_ids_with_manifest(loaded_by_file, validation_cfg)
    loaded = base._merge_stage2_outputs(loaded_by_file)
    columns = base._ordered_display_columns(list(loaded["columns"]))
    loaded["columns"] = columns
    truth = base._mock_truth_by_param(cfg, columns)
    return {
        "label": label,
        "color": color,
        "run_dir": run_dir,
        "config": cfg,
        "chain_paths": chain_paths,
        "loaded": loaded,
        "truth": truth,
    }


def _prior_number_from_text(value: Any) -> int | None:
    text = str(value or "")
    for pattern in (r"prior\s*[-_ ]*(\d+)", r"\bp\s*[-_ ]*(\d+)\b"):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _run_prior_number(run: Dict[str, Any]) -> int | None:
    for value in (run.get("label"), run.get("short_label"), run.get("run_dir")):
        prior_number = _prior_number_from_text(value)
        if prior_number is not None:
            return prior_number
    return None


def _legend_run_order(runs: list[Dict[str, Any]]) -> list[tuple[int, Dict[str, Any]]]:
    def sort_key(item: tuple[int, Dict[str, Any]]) -> tuple[int, int | str, int]:
        idx, run = item
        prior_number = _run_prior_number(run)
        if prior_number is not None:
            return (0, prior_number, idx)
        return (1, str(run.get("label", "")).lower(), idx)

    return sorted(enumerate(runs), key=sort_key)


def _default_annotate_run_idx(runs: list[Dict[str, Any]]) -> int:
    for idx, run in enumerate(runs):
        if _run_prior_number(run) == 5:
            return idx
    return 0


def _finite_matrix(samples_flat: Dict[str, Any], columns: Iterable[str]) -> np.ndarray:
    mat = np.column_stack([np.asarray(samples_flat[p], dtype=float).reshape(-1) for p in columns])
    mask = np.all(np.isfinite(mat), axis=1)
    return mat[mask]


def _kde1d_smooth_curve(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.full_like(grid, np.nan, dtype=float)
    std = float(np.std(x, ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        span = float(np.nanmax(x) - np.nanmin(x)) if x.size else 0.0
        std = max(span, 1.0e-4)
    bw = SMOOTH_STYLE_KDE_BW_FACTOR * 1.06 * std * (x.size ** (-1.0 / 5.0))
    bw = max(float(bw), 1.0e-6)
    z = (np.asarray(grid, dtype=float)[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(axis=1) / (x.size * bw * np.sqrt(2.0 * np.pi))


def _draw_smooth_diagonal_kdes(
    fig: plt.Figure,
    runs: list[Dict[str, Any]],
    columns: list[str],
    ranges: list[tuple[float, float]],
) -> None:
    axes = np.array(fig.get_axes()).reshape((len(columns), len(columns)))
    for i, param in enumerate(columns):
        ax = axes[i, i]
        for line in list(ax.lines):
            line.remove()
        lo, hi = ranges[i]
        grid = np.linspace(float(lo), float(hi), int(SMOOTH_STYLE_KDE_POINTS))
        y_max = 0.0
        for run in runs:
            y = _kde1d_smooth_curve(run["loaded"]["samples_flat"][param], grid)
            finite = np.isfinite(y)
            if not np.any(finite):
                continue
            y_max = max(y_max, float(np.nanmax(y[finite])))
            ax.plot(
                grid,
                y,
                color=run["color"],
                lw=RUN_HIST_LINEWIDTH,
                alpha=1.0,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=5,
            )
        ax.set_xlim(float(lo), float(hi))
        if y_max > 0.0:
            ax.set_ylim(0.0, y_max * 1.22)


def _quantile_text(samples_flat: Dict[str, Any], param: str, digits: int) -> str:
    x = np.asarray(samples_flat[param], dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return ""
    q16, q50, q84 = np.percentile(x, [16.0, 50.0, 84.0])
    return f"{q50:.{digits}f} +{q84 - q50:.{digits}f}/-{q50 - q16:.{digits}f}"


def _quantiles(samples_flat: Dict[str, Any], param: str) -> tuple[float, float, float]:
    x = np.asarray(samples_flat[param], dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return (np.nan, np.nan, np.nan)
    q16, q50, q84 = np.percentile(x, [16.0, 50.0, 84.0])
    return float(q16), float(q50), float(q84)


def _diag_value_text(samples_flat: Dict[str, Any], param: str, digits: int) -> str:
    q16, q50, q84 = _quantiles(samples_flat, param)
    if not np.all(np.isfinite([q16, q50, q84])):
        return ""
    err_hi = q84 - q50
    err_lo = q50 - q16
    return rf"$= {q50:.{digits}f}^{{+{err_hi:.{digits}f}}}_{{-{err_lo:.{digits}f}}}$"


def _common_corner_columns(runs: list[Dict[str, Any]]) -> tuple[list[str], list[str]]:
    common = set(runs[0]["loaded"]["samples_flat"])
    for run in runs[1:]:
        common &= set(run["loaded"]["samples_flat"])
    ordered = [p for p in CORNER_COLUMN_ORDER if p in common]
    ordered.extend(p for p in sorted(common) if p not in ordered and p != "log_sigma_int")

    kept: list[str] = []
    omitted: list[str] = []
    for param in ordered:
        is_varying = True
        for run in runs:
            arr = np.asarray(run["loaded"]["samples_flat"].get(param, []), dtype=float).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size < 2 or float(np.nanmax(arr) - np.nanmin(arr)) <= 1.0e-12:
                is_varying = False
                break
        if is_varying:
            kept.append(param)
        else:
            omitted.append(param)
    if len(kept) < 2:
        raise RuntimeError(f"Fewer than two common varying parameters; omitted={omitted}")
    return kept, omitted


def _shared_truth_values(runs: list[Dict[str, Any]], columns: list[str]) -> list[float]:
    vals: list[float] = []
    for param in columns:
        raw = [run["truth"].get(param, np.nan) for run in runs]
        finite = [float(x) for x in raw if x is not None and np.isfinite(float(x))]
        if finite and np.allclose(finite, finite[0], rtol=0.0, atol=1.0e-10):
            vals.append(float(finite[0]))
        else:
            vals.append(np.nan)
    return vals


def _shared_ranges(
    runs: list[Dict[str, Any]],
    columns: list[str],
    truth_vals: list[float],
) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for idx, param in enumerate(columns):
        pieces = []
        for run in runs:
            x = np.asarray(run["loaded"]["samples_flat"][param], dtype=float).reshape(-1)
            x = x[np.isfinite(x)]
            if x.size:
                pieces.append(x)
        cat = np.concatenate(pieces)
        lo = float(np.nanpercentile(cat, 1.0))
        hi = float(np.nanpercentile(cat, 99.0))
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise RuntimeError(f"Non-finite range for {param}")
        tv = truth_vals[idx]
        if np.isfinite(float(tv)):
            lo = min(lo, float(tv))
            hi = max(hi, float(tv))
        if lo == hi:
            lo -= 1.0e-3
            hi += 1.0e-3
        pad_fraction = RANGE_PAD_FRACTION_BY_PARAM.get(param, RANGE_PAD_FRACTION)
        pad = max(pad_fraction * (hi - lo), 1.0e-10)
        ranges.append((lo - pad, hi + pad))
    return ranges


def _style_corner_axes(
    fig: plt.Figure,
    columns: list[str],
    *,
    bottom_xlabel_y: float = -0.36,
    left_ylabel_x: float = -0.33,
    left_ylabel_y: float = 0.5,
    tick_labelsize: float = 20.0,
    show_outer_tick_labels: bool = True,
) -> None:
    axes = np.array(fig.get_axes()).reshape((len(columns), len(columns)))
    tick_nbins = {
        "Om0": 4,
        "w0": 4,
        "wa": 4,
        "d_D": 3,
        "D_pivot": 3,
        "sigma_int": 3,
    }
    for i, pi in enumerate(columns):
        for j, pj in enumerate(columns):
            ax = axes[i, j]
            show_bottom = i == len(columns) - 1
            show_left = j == 0
            ax.xaxis.set_major_locator(MaxNLocator(nbins=tick_nbins.get(pj, 4)))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=tick_nbins.get(pi, 4)))
            ax.tick_params(
                axis="x",
                labelsize=tick_labelsize,
                bottom=show_bottom and show_outer_tick_labels,
                top=False,
                labelbottom=show_bottom and show_outer_tick_labels,
                labeltop=False,
                pad=5,
            )
            ax.tick_params(
                axis="y",
                labelsize=tick_labelsize,
                left=show_left and show_outer_tick_labels,
                right=False,
                labelleft=show_left and show_outer_tick_labels,
                labelright=False,
                pad=5,
            )
            ax.xaxis.labelpad = 10
            ax.yaxis.labelpad = 10
            if show_bottom:
                ax.xaxis.set_label_coords(0.5, bottom_xlabel_y)
            if show_left:
                ax.yaxis.set_label_coords(left_ylabel_x, left_ylabel_y)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)


def _expand_diagonal_y_limits(
    fig: plt.Figure,
    columns: list[str],
    *,
    headroom_fraction: float = 0.35,
) -> None:
    """Give overlaid 1D marginals enough headroom after all runs are drawn."""
    axes = np.array(fig.get_axes()).reshape((len(columns), len(columns)))
    for i in range(len(columns)):
        ax = axes[i, i]
        y_max = 0.0
        for line in ax.lines:
            xd = np.asarray(line.get_xdata(), dtype=float)
            xf = xd[np.isfinite(xd)]
            # Skip full-height vertical guide lines (truth / quantile axvlines):
            # their ydata is [0, 1] in axes-fraction, not a density, and would
            # otherwise force the y-scale up and crush low, wide marginals (e.g. wa).
            if xf.size and np.ptp(xf) == 0.0:
                continue
            y = np.asarray(line.get_ydata(), dtype=float)
            if y.size:
                finite = y[np.isfinite(y)]
                if finite.size:
                    y_max = max(y_max, float(np.nanmax(finite)))
        for patch in ax.patches:
            path = patch.get_path()
            if path is None or path.vertices.size == 0:
                continue
            y = np.asarray(path.vertices[:, 1], dtype=float)
            finite = y[np.isfinite(y)]
            if finite.size:
                y_max = max(y_max, float(np.nanmax(finite)))
        if y_max <= 0.0:
            continue
        lower, upper = ax.get_ylim()
        wanted_upper = y_max * (1.0 + headroom_fraction)
        if not np.isfinite(upper) or upper < wanted_upper:
            ax.set_ylim(lower, wanted_upper)


def parameter_group_for_index(index: int) -> str:
    if index < 0 or index > 5:
        raise ValueError(f"unsupported Stage-2 parameter index: {index}")
    if index <= 2:
        return "cosmology"
    if index <= 4:
        return "variability_luminosity"
    return "scatter"


def _axis_report_entry(
    ax: plt.Axes,
    *,
    panel: str,
    x_param: str,
    x_expected: tuple[float, float],
    y_param: str | None = None,
    y_expected: tuple[float, float] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "panel": panel,
        "x_param": x_param,
        "x_expected": [float(x_expected[0]), float(x_expected[1])],
        "x_actual": [float(v) for v in ax.get_xlim()],
    }
    if y_param is not None and y_expected is not None:
        entry.update(
            {
                "y_param": y_param,
                "y_expected": [float(y_expected[0]), float(y_expected[1])],
                "y_actual": [float(v) for v in ax.get_ylim()],
            }
        )
    return entry


def _assert_close_range(
    actual: tuple[float, float],
    expected: tuple[float, float],
    *,
    panel: str,
    axis: str,
) -> None:
    if not np.allclose(actual, expected, rtol=1.0e-9, atol=1.0e-11):
        raise RuntimeError(
            f"{panel} {axis}-axis range changed after plotting: "
            f"actual={actual}, expected={expected}"
        )


def _lock_corner_axis_ranges_and_check(
    fig: plt.Figure,
    columns: list[str],
    ranges: list[tuple[float, float]],
    *,
    axis_check_json: Path | None = None,
) -> None:
    """Keep smoothing/style changes from altering the parameter coordinates."""
    axes = np.array(fig.get_axes()).reshape((len(columns), len(columns)))
    report: list[dict[str, Any]] = []
    for i, y_param in enumerate(columns):
        for j, x_param in enumerate(columns):
            if j > i:
                continue
            ax = axes[i, j]
            x_expected = ranges[j]
            ax.set_xlim(float(x_expected[0]), float(x_expected[1]))
            if i == j:
                lower, upper = ax.get_ylim()
                if not np.isfinite(lower) or lower > 0.0:
                    lower = 0.0
                ax.set_ylim(float(lower), float(upper))
                entry = _axis_report_entry(
                    ax,
                    panel=f"{y_param}|diag",
                    x_param=x_param,
                    x_expected=x_expected,
                )
                _assert_close_range(tuple(ax.get_xlim()), x_expected, panel=entry["panel"], axis="x")
            else:
                y_expected = ranges[i]
                ax.set_ylim(float(y_expected[0]), float(y_expected[1]))
                entry = _axis_report_entry(
                    ax,
                    panel=f"{y_param}|{x_param}",
                    x_param=x_param,
                    x_expected=x_expected,
                    y_param=y_param,
                    y_expected=y_expected,
                )
                _assert_close_range(tuple(ax.get_xlim()), x_expected, panel=entry["panel"], axis="x")
                _assert_close_range(tuple(ax.get_ylim()), y_expected, panel=entry["panel"], axis="y")
            report.append(entry)

    if axis_check_json is not None:
        axis_check_json.parent.mkdir(parents=True, exist_ok=True)
        axis_check_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _add_diagonal_titles(
    fig: plt.Figure,
    runs: list[Dict[str, Any]],
    columns: list[str],
    *,
    annotate_run_idx: int,
    show_group_colors: bool = True,
    diagonal_group_boxes: bool = False,
    quantile_lines_annotated_only: bool = False,
    symbol_fontsize: float = 26.0,
    value_fontsize: float = 24.0,
    horizontal_offset: float = 0.0,
    vertical_offset: float = 0.0,
    group_boxes: Dict[str, Dict[str, Any]] | None = None,
) -> None:
    group_boxes = DIAGONAL_GROUP_BOXES if group_boxes is None else group_boxes
    axes = np.array(fig.get_axes()).reshape((len(columns), len(columns)))
    title_digits = {
        "Om0": 2,
        "w0": 2,
        "wa": 2,
        "d_D": 3,
        "D_pivot": 3,
        "sigma_int": 3,
    }
    ann_run = runs[annotate_run_idx]
    for i, param in enumerate(columns):
        ax = axes[i, i]
        ax.set_title("")
        digits = title_digits.get(param, 2)
        for run_idx, run in enumerate(runs):
            if quantile_lines_annotated_only and run_idx != annotate_run_idx:
                continue
            q16, q50, q84 = _quantiles(run["loaded"]["samples_flat"], param)
            if np.all(np.isfinite([q16, q50, q84])):
                for qv, lw in zip((q16, q50, q84), RUN_QUANTILE_LINEWIDTHS):
                    ax.axvline(
                        qv,
                        color=run["color"],
                        lw=lw,
                        ls="--",
                        alpha=0.88,
                        zorder=6,
                    )
        value_text = _diag_value_text(ann_run["loaded"]["samples_flat"], param, digits=digits)
        group = base._poster_corner_param_group(param)
        sym_textprops: dict[str, Any] = {"fontsize": symbol_fontsize}
        if show_group_colors:
            if diagonal_group_boxes:
                group_box = group_boxes[parameter_group_for_index(i)]
                box_color = group_box["color"]
                box_alpha = group_box["alpha"]
            else:
                box_color = base.CORNER_GROUP_COLORS[group]
                box_alpha = 0.18
            sym_textprops["bbox"] = dict(
                boxstyle="round,pad=0.16",
                facecolor=box_color,
                edgecolor="none",
                alpha=box_alpha,
            )
        sym_area = TextArea(
            base._display_param_label(param),
            textprops=sym_textprops,
        )
        val_area = TextArea(value_text, textprops={"fontsize": value_fontsize})
        packed = HPacker(children=[sym_area, val_area], align="center", pad=0, sep=3)
        xloc = diagonal_annotation_x_location(param, horizontal_offset)
        anchored = AnchoredOffsetbox(
            loc="upper center",
            child=packed,
            bbox_to_anchor=(xloc, 1.28 + vertical_offset),
            bbox_transform=ax.transAxes,
            frameon=False,
            pad=0.0,
            borderpad=0.0,
        )
        ax.add_artist(anchored)


def diagonal_annotation_x_location(param: str, horizontal_offset: float = 0.0) -> float:
    """Return the normalized horizontal anchor for a diagonal summary."""
    xloc = 0.46
    if param == "d_D":
        xloc = 0.58
    elif param in {"D_pivot", "sigma_int"}:
        xloc = 0.60
    return xloc + float(horizontal_offset)


def _parameter_group_legend_handles(
    group_boxes: Dict[str, Dict[str, Any]] | None = None,
) -> list[Any]:
    group_boxes = DIAGONAL_GROUP_BOXES if group_boxes is None else group_boxes
    return [
        Patch(
            facecolor=to_rgba(
                group_boxes[parameter_group_for_index(index)]["color"],
                group_boxes[parameter_group_for_index(index)]["alpha"],
            ),
            edgecolor="none",
            linewidth=0.0,
            label=group_boxes[parameter_group_for_index(index)]["label"],
        )
        for index in (0, 3, 5)
    ]


class _HandlerPriorGrid(HandlerBase):
    """Lay five prior swatches out as two on top and three below."""

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        rows = (list(orig_handle[:2]), list(orig_handle[2:]))
        cell_width = width / 3.0
        cell_height = height / 2.0
        artists = []
        for row_index, row in enumerate(rows):
            for column_index, patch in enumerate(row):
                artists.append(
                    Rectangle(
                        (
                            column_index * cell_width - xdescent + 0.08 * cell_width,
                            height - (row_index + 1) * cell_height - ydescent + 0.11 * cell_height,
                        ),
                        0.84 * cell_width,
                        0.78 * cell_height,
                        facecolor=patch.get_facecolor(),
                        edgecolor=patch.get_edgecolor(),
                        linewidth=patch.get_linewidth(),
                        transform=trans,
                    )
                )
        return artists


def _combined_group_prior_legend(
    runs: list[Dict[str, Any]],
    truth_vals: list[float],
    *,
    group_boxes: Dict[str, Dict[str, Any]] | None = None,
    show_parameter_group_legend: bool = True,
    prior_key_first: bool = False,
) -> tuple[list[Any], list[str], dict[Any, Any]]:
    group_handles = _parameter_group_legend_handles(group_boxes) \
        if show_parameter_group_legend else []
    handles: list[Any] = []
    labels: list[str] = []
    if any(np.isfinite(float(value)) for value in truth_vals):
        handles.append(Line2D([0], [0], color=TRUTH_COLOR, lw=LEGEND_LINEWIDTH))
        labels.append("Truth")
    handles.extend(group_handles)
    labels.extend(handle.get_label() for handle in group_handles)
    if len(runs) == 1:
        run = runs[0]
        run_handle = Line2D([0], [0], color=run["color"], lw=LEGEND_LINEWIDTH)
        if prior_key_first:
            handles.insert(0, run_handle)
            labels.insert(0, run["label"])
        else:
            handles.append(run_handle)
            labels.append(run["label"])
        return handles, labels, {}
    prior_handle = tuple(
        Patch(
            facecolor=to_rgba(run["color"], 0.75),
            edgecolor=run["color"],
            linewidth=1.2,
        )
        for _, run in _legend_run_order(runs)
    )
    if prior_key_first:
        handles.insert(0, prior_handle)
        labels.insert(0, "Dif{}ferent priors")
    else:
        handles.append(prior_handle)
        labels.append("Dif{}ferent priors")
    return handles, labels, {tuple: _HandlerPriorGrid()}


def _legend_handles(
    runs: list[Dict[str, Any]],
    *,
    annotate_run_idx: int,
    truth_vals: list[float],
    show_group_colors: bool = True,
    filled_contours: bool = False,
    parameter_group_legend_only: bool = False,
    truth_first_parameter_group_legend: bool = False,
    group_boxes: Dict[str, Dict[str, Any]] | None = None,
) -> list[Any]:
    if show_group_colors and parameter_group_legend_only:
        handles = _parameter_group_legend_handles(group_boxes)
        if truth_first_parameter_group_legend \
                and any(np.isfinite(float(value)) for value in truth_vals):
            handles.insert(0, Line2D(
                [0], [0], color=TRUTH_COLOR, lw=LEGEND_LINEWIDTH, label="Truth",
            ))
        return handles

    present_groups = {
        base._poster_corner_param_group(p)
        for p in runs[0].get("corner_columns", [])
    }
    group_labels = {
        "cosmology": "Cosmology",
        "variability": "Variability-Luminosity",
        "scatter": "Scatter",
    }
    handles: list[Any] = []
    for idx, run in _legend_run_order(runs):
        label = run["label"]
        if idx == annotate_run_idx:
            label = rf"{label} (median $\pm$ 68\%)"
        if filled_contours:
            handles.append(
                Patch(
                    facecolor=to_rgba(run["color"], 0.75),
                    edgecolor=run["color"],
                    linewidth=1.8,
                    label=label,
                )
            )
        else:
            handles.append(Line2D([0], [0], color=run["color"], lw=LEGEND_LINEWIDTH, label=label))
    if any(np.isfinite(float(v)) for v in truth_vals):
        handles.append(Line2D([0], [0], color=TRUTH_COLOR, lw=LEGEND_LINEWIDTH, label="Truth"))
    if show_group_colors:
        if group_boxes is None:
            handles.extend(
                Patch(facecolor=base.CORNER_GROUP_COLORS[group], edgecolor="none", alpha=0.35, label=group_labels[group])
                for group in ["cosmology", "variability", "scatter"]
                if group in present_groups
            )
        else:
            handles.extend(
                handle
                for index, handle in zip((0, 3, 5), _parameter_group_legend_handles(group_boxes))
                if base._poster_corner_param_group(runs[0]["corner_columns"][index])
                in present_groups
            )
    return handles


def make_multi_overlay_corner_pdf(
    out_pdf: Path,
    run_specs: list[tuple[Path, str, str, str | None]],
    *,
    annotate_run_idx: int | None = None,
    max_chain_ids: list[int | None] | None = None,
    show_group_colors: bool = True,
    diagonal_group_boxes: bool = False,
    parameter_group_legend_only: bool = False,
    truth_first_parameter_group_legend: bool = False,
    combined_group_prior_legend: bool = False,
    show_parameter_group_legend: bool = True,
    prior_key_first: bool = False,
    show_legend: bool = True,
    quantile_lines_annotated_only: bool = False,
    filled_contours: bool = False,
    smooth_reference_style: bool = False,
    bottom_xlabel_y: float = -0.36,
    left_ylabel_x: float = -0.33,
    left_ylabel_y: float = 0.5,
    axis_labelsize: float = 30.0,
    tick_labelsize: float = 20.0,
    diagonal_symbol_fontsize: float = 26.0,
    diagonal_value_fontsize: float = 24.0,
    legend_fontsize: float | None = None,
    legend_ncol: int | None = None,
    show_outer_tick_labels: bool = True,
    diagonal_annotation_x_offset: float = 0.0,
    diagonal_annotation_y_offset: float = 0.0,
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    legend_loc: str = "upper right",
    figure_bottom: float = 0.07,
    figure_right: float = 0.97,
    save_tight: bool = True,
    axis_check_json: Path | None = None,
    group_boxes: Dict[str, Dict[str, Any]] | None = None,
) -> None:
    if not run_specs:
        raise ValueError("At least one run directory is required")
    if max_chain_ids is not None and len(max_chain_ids) != len(run_specs):
        raise ValueError(f"max_chain_ids must have {len(run_specs)} entries, got {len(max_chain_ids)}")
    runs = []
    for idx, (run_dir, label, color, short_label) in enumerate(run_specs):
        max_chain_id = max_chain_ids[idx] if max_chain_ids is not None else None
        run = _load_run(run_dir, label, color, max_chain_id=max_chain_id)
        run["short_label"] = short_label or label
        run["max_chain_id"] = max_chain_id
        runs.append(run)
    if annotate_run_idx is None:
        annotate_run_idx = _default_annotate_run_idx(runs)
    if annotate_run_idx < 0:
        annotate_run_idx += len(runs)
    if annotate_run_idx < 0 or annotate_run_idx >= len(runs):
        raise ValueError(f"annotate_run_idx={annotate_run_idx} is outside 0..{len(runs) - 1}")

    columns, omitted = _common_corner_columns(runs)
    for run in runs:
        run["corner_columns"] = list(columns)
    truth_vals = _shared_truth_values(runs, columns)
    ranges = _shared_ranges(runs, columns, truth_vals)

    rc_path = PROJECT_ROOT / "matplotlibrc"
    rc_ctx = plt.rc_context(fname=str(rc_path)) if rc_path.exists() else plt.rc_context()
    with rc_ctx:
        fig = plt.figure(figsize=(15.2, 13.0))
        draw_filled_contours = filled_contours or smooth_reference_style
        corner_bins = SMOOTH_STYLE_BINS if smooth_reference_style else 28
        corner_smooth = SMOOTH_STYLE_SMOOTH if smooth_reference_style else 0.8
        corner_smooth1d = SMOOTH_STYLE_SMOOTH1D if smooth_reference_style else 0.8
        contour_linewidths: float | list[float] = RUN_CONTOUR_LINEWIDTH
        fill_outer_alpha = CONTOUR_FILL_ALPHA
        fill_inner_alpha = CONTOUR_FILL_INNER_ALPHA
        if smooth_reference_style:
            contour_linewidths = [SMOOTH_STYLE_OUTER_CONTOUR_LW, SMOOTH_STYLE_INNER_CONTOUR_LW]
            fill_outer_alpha = SMOOTH_STYLE_FILL_ALPHA
            fill_inner_alpha = SMOOTH_STYLE_FILL_INNER_ALPHA
        for idx, run in enumerate(runs):
            mat = _finite_matrix(run["loaded"]["samples_flat"], columns)
            fig = corner.corner(
                mat,
                fig=fig,
                labels=[base._display_param_label(p) for p in columns],
                range=ranges,
                color=run["color"],
                bins=corner_bins,
                smooth=corner_smooth,
                smooth1d=corner_smooth1d,
                plot_datapoints=False,
                plot_density=False,
                fill_contours=draw_filled_contours,
                no_fill_contours=True,
                levels=(0.393, 0.865),
                truths=None,
                truth_color=TRUTH_COLOR,
                hist_kwargs={"lw": RUN_HIST_LINEWIDTH, "color": run["color"]},
                contour_kwargs={"linewidths": contour_linewidths, "colors": [run["color"]]},
                contourf_kwargs=(
                    {
                        "colors": _contour_fill_colors(
                            run["color"],
                            outer_alpha=fill_outer_alpha,
                            inner_alpha=fill_inner_alpha,
                        )
                    }
                    if draw_filled_contours
                    else None
                ),
                label_kwargs={"fontsize": axis_labelsize},
                max_n_ticks=4,
            )

        if smooth_reference_style:
            _draw_smooth_diagonal_kdes(fig, runs, columns, ranges)

        if any(np.isfinite(float(v)) for v in truth_vals):
            truth_plot_vals = [float(v) if np.isfinite(float(v)) else np.nan for v in truth_vals]
            corner.overplot_lines(fig, truth_plot_vals, color=TRUTH_COLOR, lw=TRUTH_LINEWIDTH)
            corner.overplot_points(
                fig,
                [truth_plot_vals],
                marker="s",
                color=TRUTH_COLOR,
                markersize=5.5,
            )

        _style_corner_axes(
            fig,
            columns,
            bottom_xlabel_y=bottom_xlabel_y,
            left_ylabel_x=left_ylabel_x,
            left_ylabel_y=left_ylabel_y,
            tick_labelsize=tick_labelsize,
            show_outer_tick_labels=show_outer_tick_labels,
        )
        _expand_diagonal_y_limits(fig, columns)
        _add_diagonal_titles(
            fig,
            runs,
            columns,
            annotate_run_idx=annotate_run_idx,
            show_group_colors=show_group_colors,
            diagonal_group_boxes=diagonal_group_boxes,
            quantile_lines_annotated_only=quantile_lines_annotated_only,
            symbol_fontsize=diagonal_symbol_fontsize,
            value_fontsize=diagonal_value_fontsize,
            horizontal_offset=diagonal_annotation_x_offset,
            vertical_offset=diagonal_annotation_y_offset,
            group_boxes=group_boxes,
        )
        _lock_corner_axis_ranges_and_check(
            fig,
            columns,
            ranges,
            axis_check_json=axis_check_json,
        )

        legend_labels = None
        legend_handler_map = None
        if show_group_colors and combined_group_prior_legend:
            handles, legend_labels, legend_handler_map = _combined_group_prior_legend(
                runs,
                truth_vals,
                group_boxes=group_boxes,
                show_parameter_group_legend=show_parameter_group_legend,
                prior_key_first=prior_key_first,
            )
        else:
            handles = _legend_handles(
                runs,
                annotate_run_idx=annotate_run_idx,
                truth_vals=truth_vals,
                show_group_colors=show_group_colors,
                filled_contours=draw_filled_contours,
                parameter_group_legend_only=parameter_group_legend_only,
                truth_first_parameter_group_legend=truth_first_parameter_group_legend,
                group_boxes=group_boxes,
            )
        if show_legend and handles:
            if parameter_group_legend_only and show_group_colors:
                legend_fontsize_value = 25
                legend_labelspacing = 0.42
            else:
                legend_fontsize_value = 23 if len(runs) > 2 else 26
                legend_labelspacing = 0.36 if len(runs) > 2 else 0.45
            if legend_fontsize is not None:
                legend_fontsize_value = float(legend_fontsize)
            default_legend_anchor = (0.96, 0.97 if combined_group_prior_legend else 0.88)
            fig.legend(
                handles=handles,
                labels=legend_labels,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox_to_anchor or default_legend_anchor,
                ncol=(legend_ncol if legend_ncol is not None
                      else (2 if combined_group_prior_legend else 1)),
                frameon=False,
                fontsize=legend_fontsize_value,
                handlelength=1.2,
                handleheight=2.0 if combined_group_prior_legend else 0.7,
                labelspacing=legend_labelspacing,
                handler_map=legend_handler_map,
            )
        if omitted:
            print(
                "Omitted fixed/degenerate parameters: "
                + ", ".join(base._display_param_label(p) for p in omitted)
            )
        fig.subplots_adjust(left=0.085, right=figure_right, bottom=figure_bottom,
                            top=0.84, wspace=0.08, hspace=0.08)

        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(out_pdf) as pdf:
            pdf.savefig(fig, bbox_inches="tight" if save_tight else None,
                        facecolor="white", edgecolor="white")
        plt.close(fig)

    print(f"Wrote {out_pdf}")


def make_overlay_corner_pdf(
    out_pdf: Path,
    prior5_dir: Path,
    comparison_dir: Path,
    *,
    comparison_label: str = "Prior 1",
    comparison_short_label: str | None = None,
    comparison_color: str = PRIOR1_COLOR,
    show_group_colors: bool = True,
    diagonal_group_boxes: bool = False,
    parameter_group_legend_only: bool = False,
    combined_group_prior_legend: bool = False,
    show_legend: bool = True,
    quantile_lines_annotated_only: bool = False,
    filled_contours: bool = False,
    smooth_reference_style: bool = False,
    bottom_xlabel_y: float = -0.36,
    left_ylabel_x: float = -0.33,
    axis_labelsize: float = 30.0,
    tick_labelsize: float = 20.0,
    diagonal_symbol_fontsize: float = 26.0,
    diagonal_value_fontsize: float = 24.0,
    legend_fontsize: float | None = None,
    legend_ncol: int | None = None,
    axis_check_json: Path | None = None,
) -> None:
    comparison_short_label = comparison_short_label or comparison_label.replace("Prior ", "P")
    make_multi_overlay_corner_pdf(
        out_pdf,
        [
            (prior5_dir, "Prior 5", PRIOR5_COLOR, "P5"),
            (comparison_dir, comparison_label, comparison_color, comparison_short_label),
        ],
        annotate_run_idx=None,
        max_chain_ids=None,
        show_group_colors=show_group_colors,
        diagonal_group_boxes=diagonal_group_boxes,
        parameter_group_legend_only=parameter_group_legend_only,
        combined_group_prior_legend=combined_group_prior_legend,
        show_legend=show_legend,
        quantile_lines_annotated_only=quantile_lines_annotated_only,
        filled_contours=filled_contours,
        smooth_reference_style=smooth_reference_style,
        bottom_xlabel_y=bottom_xlabel_y,
        left_ylabel_x=left_ylabel_x,
        axis_labelsize=axis_labelsize,
        tick_labelsize=tick_labelsize,
        diagonal_symbol_fontsize=diagonal_symbol_fontsize,
        diagonal_value_fontsize=diagonal_value_fontsize,
        legend_fontsize=legend_fontsize,
        legend_ncol=legend_ncol,
        axis_check_json=axis_check_json,
    )


def _default_labels(run_dirs: list[Path]) -> list[str]:
    labels: list[str] = []
    for path in run_dirs:
        prior_number = _prior_number_from_text(path.name)
        labels.append(f"Prior {prior_number}" if prior_number is not None else path.name)
    return labels


def _default_short_labels(labels: list[str]) -> list[str]:
    short_labels: list[str] = []
    for label in labels:
        prior_number = _prior_number_from_text(label)
        if prior_number is not None and re.fullmatch(r"\s*prior\s*[-_ ]*\d+\s*", label, flags=re.IGNORECASE):
            short_labels.append(f"P{prior_number}")
        else:
            short_labels.append(label)
    return short_labels


def _palette_colors(name: str) -> dict[int, str]:
    try:
        return PRIOR_COLOR_PALETTES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(PRIOR_COLOR_PALETTES))
        raise SystemExit(f"Unknown --palette {name!r}; choose one of: {valid}") from exc


def _default_colors(
    labels: list[str],
    run_dirs: list[Path],
    palette_name: str = DEFAULT_PALETTE_NAME,
) -> list[str]:
    colors_by_number = _palette_colors(palette_name)
    fallback_colors = [colors_by_number[idx] for idx in sorted(colors_by_number)] + ["#9467BD"]
    colors: list[str] = []
    for idx, (label, run_dir) in enumerate(zip(labels, run_dirs)):
        prior_number = _prior_number_from_text(label)
        if prior_number is None:
            prior_number = _prior_number_from_text(run_dir.name)
        if prior_number in colors_by_number:
            colors.append(colors_by_number[prior_number])
        else:
            colors.append(fallback_colors[idx % len(fallback_colors)])
    return colors


def _validate_parallel_arg(name: str, values: list[str] | None, n_runs: int) -> list[str] | None:
    if values is not None and len(values) != n_runs:
        raise SystemExit(f"--{name} must have {n_runs} values, got {len(values)}")
    return values


def _validate_max_chain_ids(values: list[int] | None, n_runs: int) -> list[int | None] | None:
    if values is None:
        return None
    if len(values) != n_runs:
        raise SystemExit(f"--run-max-chain-ids must have {n_runs} values, got {len(values)}")
    return [None if int(v) <= 0 else int(v) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", default=None)
    parser.add_argument("--run-labels", nargs="+", default=None)
    parser.add_argument("--run-short-labels", nargs="+", default=None)
    parser.add_argument("--run-colors", nargs="+", default=None)
    parser.add_argument(
        "--run-max-chain-ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional per-run maximum chain id; use 0 for no cap.",
    )
    parser.add_argument("--palette", choices=sorted(PRIOR_COLOR_PALETTES), default=DEFAULT_PALETTE_NAME)
    parser.add_argument("--annotate-run-idx", type=int, default=None)
    parser.add_argument("--prior5-run-dir", type=Path, default=DEFAULT_PRIOR5_RUN_DIR)
    parser.add_argument("--prior1-run-dir", type=Path, default=DEFAULT_PRIOR1_RUN_DIR)
    parser.add_argument("--comparison-run-dir", type=Path, default=None)
    parser.add_argument("--comparison-label", default="Prior 1")
    parser.add_argument("--comparison-short-label", default=None)
    parser.add_argument("--comparison-color", default=PRIOR1_COLOR)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT_PDF)
    parser.add_argument(
        "--no-group-colors",
        action="store_true",
        help="Remove shaded parameter-group labels and group-color legend patches.",
    )
    parser.add_argument(
        "--no-diagonal-group-boxes",
        action="store_true",
        help="Disable the explicit diagonal group boxes if enabled by a wrapper script.",
    )
    parser.add_argument(
        "--diagonal-group-boxes",
        action="store_true",
        help="Use Kavli-style shaded boxes behind diagonal parameter symbols by group.",
    )
    parser.add_argument(
        "--parameter-group-legend",
        action="store_true",
        help="Replace the run/truth legend with the three-entry parameter-group legend.",
    )
    parser.add_argument(
        "--combined-group-prior-legend",
        action="store_true",
        help="Show parameter groups, a compact five-prior key, and the truth marker.",
    )
    parser.add_argument(
        "--run-legend",
        action="store_true",
        help="Show the run/truth legend even if a wrapper script requested the compact group legend.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Do not draw any figure legend.",
    )
    parser.add_argument(
        "--prior5-quantile-lines-only",
        action="store_true",
        help="Draw dashed 16/50/84 percentile lines only for the annotated run, normally Prior 5.",
    )
    parser.add_argument(
        "--filled-contours",
        action="store_true",
        help="Fill 2D corner contours with transparent prior colors and use filled legend handles.",
    )
    parser.add_argument(
        "--smooth-reference-style",
        action="store_true",
        help="Use smoother diagonal curves plus filled 95/68 contours with thicker inner outlines.",
    )
    parser.add_argument(
        "--bottom-xlabel-y",
        type=float,
        default=-0.36,
        help="Axes-coordinate y position for bottom-row x-axis labels.",
    )
    parser.add_argument(
        "--left-ylabel-x",
        type=float,
        default=-0.33,
        help="Axes-coordinate x position for left-column y-axis labels.",
    )
    parser.add_argument("--axis-labelsize", type=float, default=30.0)
    parser.add_argument("--tick-labelsize", type=float, default=20.0)
    parser.add_argument("--diagonal-symbol-fontsize", type=float, default=26.0)
    parser.add_argument("--diagonal-value-fontsize", type=float, default=24.0)
    parser.add_argument("--legend-fontsize", type=float, default=None)
    parser.add_argument(
        "--axis-check-json",
        type=Path,
        default=None,
        help="Optional JSON report of the final corner-panel axis ranges after style overlays.",
    )
    args = parser.parse_args()
    if args.run_dirs:
        run_dirs = list(args.run_dirs)
        n_runs = len(run_dirs)
        labels = _validate_parallel_arg("run-labels", args.run_labels, n_runs) or _default_labels(run_dirs)
        short_labels = (
            _validate_parallel_arg("run-short-labels", args.run_short_labels, n_runs)
            or _default_short_labels(labels)
        )
        colors = _validate_parallel_arg("run-colors", args.run_colors, n_runs) or _default_colors(
            labels,
            run_dirs,
            args.palette,
        )
        max_chain_ids = _validate_max_chain_ids(args.run_max_chain_ids, n_runs)
        make_multi_overlay_corner_pdf(
            args.out_pdf,
            list(zip(run_dirs, labels, colors, short_labels)),
            annotate_run_idx=args.annotate_run_idx,
            max_chain_ids=max_chain_ids,
            show_group_colors=not args.no_group_colors,
            diagonal_group_boxes=(
                not args.no_group_colors
                and args.diagonal_group_boxes
                and not args.no_diagonal_group_boxes
            ),
            parameter_group_legend_only=(
                not args.no_group_colors
                and args.parameter_group_legend
                and not args.run_legend
            ),
            combined_group_prior_legend=(
                not args.no_group_colors and args.combined_group_prior_legend
            ),
            show_legend=not args.no_legend,
            quantile_lines_annotated_only=args.prior5_quantile_lines_only,
            filled_contours=args.filled_contours,
            smooth_reference_style=args.smooth_reference_style,
            bottom_xlabel_y=args.bottom_xlabel_y,
            left_ylabel_x=args.left_ylabel_x,
            axis_labelsize=args.axis_labelsize,
            tick_labelsize=args.tick_labelsize,
            diagonal_symbol_fontsize=args.diagonal_symbol_fontsize,
            diagonal_value_fontsize=args.diagonal_value_fontsize,
            legend_fontsize=args.legend_fontsize,
            axis_check_json=args.axis_check_json,
        )
        return

    comparison_run_dir = args.comparison_run_dir or args.prior1_run_dir
    make_overlay_corner_pdf(
        args.out_pdf,
        args.prior5_run_dir,
        comparison_run_dir,
        comparison_label=args.comparison_label,
        comparison_short_label=args.comparison_short_label,
        comparison_color=args.comparison_color,
        show_group_colors=not args.no_group_colors,
        diagonal_group_boxes=(
            not args.no_group_colors
            and args.diagonal_group_boxes
            and not args.no_diagonal_group_boxes
        ),
        parameter_group_legend_only=(
            not args.no_group_colors
            and args.parameter_group_legend
            and not args.run_legend
        ),
        combined_group_prior_legend=(
            not args.no_group_colors and args.combined_group_prior_legend
        ),
        show_legend=not args.no_legend,
        quantile_lines_annotated_only=args.prior5_quantile_lines_only,
        filled_contours=args.filled_contours,
        smooth_reference_style=args.smooth_reference_style,
        bottom_xlabel_y=args.bottom_xlabel_y,
        left_ylabel_x=args.left_ylabel_x,
        axis_labelsize=args.axis_labelsize,
        tick_labelsize=args.tick_labelsize,
        diagonal_symbol_fontsize=args.diagonal_symbol_fontsize,
        diagonal_value_fontsize=args.diagonal_value_fontsize,
        legend_fontsize=args.legend_fontsize,
        axis_check_json=args.axis_check_json,
    )


if __name__ == "__main__":
    main()
