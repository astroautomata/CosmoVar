#!/usr/bin/env python3
"""
Build the two-panel Stage-1 prior-sensitivity figure:

  a) Ensemble recovery
  b) Single-source posterior

The left panel is the vertical recovery-bias-versus-baseline diagnostic.  The
right panel is the overlaid source-22692 corner comparison.  Both are generated
as vector PDFs first and then placed side-by-side into a final vector PDF.
"""

from __future__ import annotations

from framework import PROJECT_ROOT, DATA_ROOT

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/agn-variability-cosmology-mpl-{os.getuid()}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


import fitz
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from plotting import plot_recovery_posteriors as gallery
from plotting import plot_recovery_bias as baseline
from plotting import diagnostics as cmp

try:
    from scipy.interpolate import PchipInterpolator
except Exception:  # pragma: no cover - plotting nicety
    PchipInterpolator = None

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:  # pragma: no cover - plotting nicety
    gaussian_filter1d = None


PRIOR_SENSITIVITY_DIR = DATA_ROOT / 'results/prior_sensitivity'
CACHE_VERSION = 8
LEFT_SUMMARY_CACHE_VERSION = 2
LEFT_AXIS_LABELSIZE = 19.0
LEFT_YAXIS_LABELSIZE = 19.0
LEFT_TICK_LABELSIZE = 16.5
LEFT_REFERENCE_LABELSIZE = 13.5
LEFT_YLABEL_X = -0.17
LEFT_XLIM_MIN = -0.5
LEFT_PANEL_HSPACE = 0.15
DEFAULT_BODY_HEIGHT_PT = 510.0
RIGHT_PANEL_SCALE = 1.0
LEFT_PARAMS = ["log10_D", "m_win", "mean_mag", "log10_sigma2", "log10_tau"]
LEFT_AXIS_LABELS = {
    "log10_D": r"$\Delta \eta\ [\mathrm{dex}]$",
    "m_win": r"$\Delta m\ [\mathrm{mag}]$",
    "mean_mag": r"$\Delta m_0\ [\mathrm{mag}]$",
    "log10_sigma2": r"$\Delta \log_{10}(\sigma^{2})\ [\mathrm{dex}]$",
    "log10_tau": r"$\Delta \log_{10}(\tau)\ [\mathrm{dex}]$",
}
RIGHT_PARAM_LABELS = {
    "log10_D": r"$\eta$",
    "m_win": r"$m$",
    "m_win_cont": r"$m$",
}
RIGHT_ANNOTATION_MATH = {
    "log10_D": r"\eta",
    "m_win": r"m",
    "m_win_cont": r"m",
}
RIGHT_CORNER_STYLE = {
    "axis_labelsize": 34.0,
    "tick_labelsize": 29.0,
    "legend_fontsize": 31.0,
    "diag_annotation_fontsize": 31.0,
    "clean_tick_formatter": True,
    "ylabel_x": -0.34,
}
RIGHT_DIAG_TITLE_X = {"log10_sigma2": 0.56, "log10_tau": 0.50, "log10_jitter": 0.72}
RIGHT_DIAG_LABEL_BOX_COLORS = {
    "log10_D": "#F2A1EA",
    "m_win": "#F2A1EA",
    "m_win_cont": "#F2A1EA",
    "mean_mag": "#A7D8FF",
    "log10_sigma2": "#A7D8FF",
    "log10_tau": "#A7D8FF",
}

PRIOR_COLOR_PALETTES = {
    "original": {
        1: "#59A14F",
        2: "#4C78A8",
        3: "#F58518",
        4: "#00A6D6",
        5: "#FF00FF",
    },
    "colorblindpalette": {
        1: "#009E73",
        2: "#0072B2",
        3: "#D55E00",
        4: "#E69F00",
        5: "#FF00FF",
    },
    # Revision of "colorblindpalette": Prior 3 moved from vermilion (#D55E00, hue-close
    # to the amber Prior 4) to the IBM colorblind-safe violet, and Prior 5 softened from
    # pure magenta to the IBM magenta/pink so the violet/magenta high end stays separable.
    # Green/blue/violet/amber/magenta remain mutually distinct under deutan/protan/tritan.
    "colorblindpalette_v2": {
        1: "#009E73",
        2: "#0072B2",
        3: "#6A3D9A",
        4: "#E69F00",
        5: "#DC267F",
    },
    # Palette adopted from an external cosmology paper (DESI/CamSpec/Pantheon/Union3/
    # DESY5/Dovekie line colors).  Black (the paper's DESI color) is reserved for the
    # truth marker; the five priors take the remaining colors, with Prior 5 set to the
    # most vivid (Dovekie red #D7301F) so it pops out most.
    "paperpalette": {
        1: "#762A83",  # camspec (purple)
        2: "#2166AC",  # pantheon (blue)
        3: "#FF7F00",  # union3 (orange)
        4: "#7FBC41",  # desy5 (green)
        5: "#D7301F",  # dovekie (red) -- pops out the most
    },
}
DEFAULT_PALETTE_NAME = "paperpalette"


def _palette_colors(name: str) -> dict[int, str]:
    try:
        return PRIOR_COLOR_PALETTES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(PRIOR_COLOR_PALETTES))
        raise SystemExit(f"Unknown --palette {name!r}; choose one of: {valid}") from exc


def _base_prior_specs(palette_name: str = DEFAULT_PALETTE_NAME) -> list[tuple[str, Path, str]]:
    colors = _palette_colors(palette_name)
    return [
        ("Prior 1", PRIOR_SENSITIVITY_DIR / 'prior_1/posterior_samples.npz', colors[1]),
        ("Prior 2", PRIOR_SENSITIVITY_DIR / 'prior_2/posterior_samples.npz', colors[2]),
        ("Prior 3", PRIOR_SENSITIVITY_DIR / 'prior_3/posterior_samples.npz', colors[3]),
        ("Prior 4", PRIOR_SENSITIVITY_DIR / 'prior_4/posterior_samples.npz', colors[4]),
        ("Prior 5", PRIOR_SENSITIVITY_DIR / 'prior_5/posterior_samples.npz', colors[5]),
    ]


PRIOR_SPECS = _base_prior_specs()
BASE_PRIOR_SPECS = tuple(PRIOR_SPECS)


def _prior_specs_for_suffix(
    run_suffix: str,
    palette_name: str = DEFAULT_PALETTE_NAME,
) -> list[tuple[str, Path, str]]:
    suffix = str(run_suffix or "")
    base_specs = _base_prior_specs(palette_name)
    if not suffix:
        return list(base_specs)
    specs: list[tuple[str, Path, str]] = []
    for label, old_path, color in base_specs:
        run_name = old_path.parent.name
        new_run_name = f"{run_name}{suffix}"
        new_path = PRIOR_SENSITIVITY_DIR / new_run_name / f"{old_path.stem}{suffix}.npz"
        specs.append((label, new_path, color))
    return specs


def _metadata_path(npz_path: Path) -> Path:
    return npz_path.with_name(npz_path.stem + "_metadata.json")


def _file_signature(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    st = path.stat()
    return {
        "path": str(path),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _input_signatures() -> list[dict[str, Any]]:
    signatures: list[dict[str, Any]] = []
    for _, npz_path, _ in PRIOR_SPECS:
        signatures.append(_file_signature(npz_path))
        meta_path = _metadata_path(npz_path)
        if meta_path.exists():
            signatures.append(_file_signature(meta_path))
    return signatures


def _script_signatures(*paths: Path) -> list[dict[str, Any]]:
    scripts = list(paths)
    return [_file_signature(path) for path in scripts]


def _pdf_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with fitz.open(str(path)) as doc:
            return len(doc) >= 1
    except Exception:
        return False


def _cache_path(panel_pdf: Path) -> Path:
    return panel_pdf.with_suffix(".cache.json")


def _summary_cache_path(panel_pdf: Path) -> Path:
    return panel_pdf.with_suffix(".summary.npz")


def _draw_limit(args: argparse.Namespace) -> int | None:
    n_draws = int(getattr(args, "n_draws_per_source", 0) or 0)
    return n_draws if n_draws > 0 else None


def _exclude_rows_arg(args: argparse.Namespace) -> list[int]:
    spec = str(getattr(args, "exclude_rows", "") or "").strip()
    if not spec:
        return []
    rows: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        rows.append(int(part))
    return sorted(set(rows))


def _drop_rows_from_runs(runs: list[dict[str, Any]], rows: list[int]) -> None:
    if not rows:
        return
    for run in runs:
        n_obj = int(np.asarray(run["source_id"]).shape[0])
        keep = np.ones(n_obj, dtype=bool)
        valid_rows = [row for row in rows if 0 <= int(row) < n_obj]
        if valid_rows:
            keep[np.asarray(valid_rows, dtype=int)] = False
        for key in ["source_id", "grid_i", "grid_j"]:
            run[key] = np.asarray(run[key])[keep]
        for group in ["posterior", "truth", "diag"]:
            for key, arr in list(run[group].items()):
                run[group][key] = np.asarray(arr)[keep]
        run["excluded_rows"] = list(rows)


def _limit_posterior_draws(runs: list[dict[str, Any]], n_draws: int | None) -> None:
    if n_draws is None:
        return
    for run in runs:
        posterior = run.get("posterior", {})
        for key, values in list(posterior.items()):
            arr = np.asarray(values)
            if arr.ndim == 2 and arr.shape[1] > n_draws:
                posterior[key] = arr[:, :n_draws]


def _right_corner_style(args: argparse.Namespace) -> dict[str, Any]:
    style = dict(RIGHT_CORNER_STYLE)
    if getattr(args, "right_smooth_style", False):
        # Match the Stage-2 "smooth reference" corner look: oversmoothed
        # normalized KDE diagonals, more 2D smoothing, and filled 1/2-sigma contours
        # with a thin outer + thick inner outline.  Set as defaults here so explicit
        # --right-* flags below still take precedence if the user passes them.
        style.update(
            {
                "fill_contours": True,
                "legend_patch_handles": True,
                "diag_kde_bw_factor": 1.25,
                "diag_capstyle": "round",
                "diag_lw": 3.4,
                "diag_annotated_lw": 3.4,
                # Smooth 2D credible regions via a Gaussian KDE (GetDist-style) instead
                # of binned histograms, so the off-diagonal contours are not jagged.
                "corner_kde_2d": True,
                "kde2d_bw_factor": 1.30,
                # Contours enclose 68% / 95% of the posterior mass (cosmology standard).
                "corner_levels": (0.68, 0.95),
                "contour_lw": [1.8, 3.6],
                "contour_fill_alpha": 0.22,
                "contour_fill_inner_alpha": 0.48,
            }
        )
    if args.right_contour_lw is not None:
        style["contour_lw"] = float(args.right_contour_lw)
    if args.right_truth_lw is not None:
        style["truth_lw_1d"] = float(args.right_truth_lw)
        style["truth_lw_2d"] = float(args.right_truth_lw)
        style["truth_legend_lw"] = float(args.right_truth_lw)
    if args.right_truth_color is not None:
        style["truth_color"] = str(args.right_truth_color)
    if args.right_filled_contours:
        style["fill_contours"] = True
        style["legend_patch_handles"] = True
    if args.right_legend_lw is not None:
        style["legend_lw"] = float(args.right_legend_lw)
    if args.right_legend_fontsize is not None:
        style["legend_fontsize"] = float(args.right_legend_fontsize)
    if not bool(args.right_show_legend):
        style["show_legend"] = False
        style["legend_patch_handles"] = False
    if bool(args.right_group_prior_legend):
        style["legend_group_priors"] = True
        style["legend_group_label"] = "Dif{}ferent priors"
        style["legend_patch_handles"] = True
        style["legend_group_pad"] = 0.08
        # Render the 5 prior swatches as a 2-row grid (2 on top, 3 below) so the
        # legend is roughly half as wide; "Different priors" auto-centres at the
        # mid-height between the two rows.
        style["legend_prior_grid"] = True
        style["legend_prior_grid_top"] = 2
        style["legend_handlelength"] = 3.0
        style["legend_handleheight"] = 2.0
    if args.right_axis_labelsize is not None:
        style["axis_labelsize"] = float(args.right_axis_labelsize)
    if args.right_tick_labelsize is not None:
        style["tick_labelsize"] = float(args.right_tick_labelsize)
    if args.right_diag_annotation_fontsize is not None:
        style["diag_annotation_fontsize"] = float(args.right_diag_annotation_fontsize)
    if bool(args.right_diag_label_boxes):
        requested_box_params = {
            param.strip()
            for param in str(args.right_diag_label_box_params or "").split(",")
            if param.strip()
        }
        box_colors = {
            param: color
            for param, color in RIGHT_DIAG_LABEL_BOX_COLORS.items()
            if not requested_box_params or param in requested_box_params
        }
        if args.right_diag_label_box_color:
            box_colors = {
                param: str(args.right_diag_label_box_color)
                for param in box_colors
            }
        style["diag_label_box_colors"] = box_colors
        style["diag_label_box_math_pad_by_param"] = {
            "m_win": r"\hphantom{00}",
            "m_win_cont": r"\hphantom{00}",
            "log10_tau": r"\hphantom{00}",
        }
        style["diag_label_box_math_left_pad_by_param"] = {
            "log10_D": r"\hphantom{000000000000000000000000000000}",
            "log10_tau": r"\hphantom{000000}",
        }
        style["diag_label_box_math_right_pad_by_param"] = {
            "log10_D": r"\hphantom{00}",
            "log10_tau": r"\hphantom{0}",
        }
        style["diag_label_box_makebox_width_by_param"] = {
            "log10_D": "3.0em",
        }
        style["diag_label_box_makebox_align_by_param"] = {
            "log10_D": "r",
        }
        # Extra gap between the eta symbol and its "=" (eta only). Widening this gap
        # re-centres the annotation and nudges eta left, so the external box x0 below
        # is shifted to keep the box centred on the eta glyph.
        style["diag_label_box_sep_by_param"] = {"log10_D": 10.0}
        # eta box is drawn as an external rounded rectangle (mathtext ignores the
        # \hphantom padding, so the box can only be widened this way). Width 0.162 and
        # x0 size/center it on the eta glyph so it matches the m box's width.
        style["diag_label_box_external_axes_patch_by_param"] = {
            "log10_D": [0.023, 1.03, 0.162, 0.135],
        }
        style["diag_label_box_external_boxstyle"] = "round,pad=0.01"
        style["diag_label_boxstyle_by_param"] = {
            "log10_D": "round,pad=0.16",
            "m_win": "round,pad=0.16",
            "m_win_cont": "round,pad=0.16",
            "log10_tau": "round,pad=0.22",
        }
        style["diag_title_y_by_param"] = {
            "log10_tau": 1.065,
        }
        style["diag_label_box_alpha"] = 0.92
        style["diag_label_boxstyle"] = "round,pad=0.08"
        if bool(args.right_log10_d_labels):
            for key in (
                "diag_label_box_external_axes_patch_by_param",
                "diag_label_box_makebox_width_by_param",
                "diag_label_box_makebox_align_by_param",
                "diag_label_box_math_left_pad_by_param",
                "diag_label_box_math_right_pad_by_param",
                "diag_label_box_sep_by_param",
            ):
                style.get(key, {}).pop("log10_D", None)
    if args.right_diag_lw is not None:
        style["diag_lw"] = float(args.right_diag_lw)
        style["diag_annotated_lw"] = float(args.right_diag_lw)
    if getattr(args, "right_range_pad_frac", None) is not None:
        style["range_pad_frac"] = float(args.right_range_pad_frac)
    # Per-edge range widening: m0 on both edges, log10(sigma^2) and log10(tau) on the
    # upper edge, so those posteriors are not clipped at the panel boundary.
    style["range_pad_frac_by_param"] = {
        "mean_mag": [0.15, 0.15],
        "log10_sigma2": [0.12, 0.32],
        "log10_tau": [0.12, 0.32],
    }
    return style


def _left_ylim_overrides(args: argparse.Namespace) -> dict[str, float]:
    """Parse plot-time symmetric y-limit overrides, e.g. 'log10_D:0.5,m_win:0.02'.

    Applied when rendering the ensemble panel from the cached summary, so it does NOT
    change the summary payload and therefore does not trigger a summary rebuild.
    """
    spec = str(getattr(args, "left_ylim_override", "") or "").strip()
    out: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, val = part.partition(":")
        out[key.strip()] = abs(float(val))
    return out


def _right_draw_order(args: argparse.Namespace) -> list[int] | None:
    spec = str(getattr(args, "right_draw_order", "") or "").strip()
    if not spec:
        return None
    return [int(p.strip()) for p in spec.split(",") if p.strip()]


def _drop_params_arg(args: argparse.Namespace) -> list[str]:
    spec = str(getattr(args, "right_drop_params", "") or "").strip()
    return [p.strip() for p in spec.split(",") if p.strip()]


def _right_corner_params(args: argparse.Namespace) -> list[str]:
    drop = set(_drop_params_arg(args))
    return [p for p in cmp.CORNER_PARAMS if p not in drop]


def _left_axis_label(param: str) -> str:
    return LEFT_AXIS_LABELS.get(param, baseline._axis_label(param))


def _right_param_labels(
    video_coordinate_labels: bool = False,
    log10_d_labels: bool = False,
) -> dict[str, str]:
    labels = dict(cmp.CORNER_PARAM_LABELS)
    labels.update(RIGHT_PARAM_LABELS)
    if video_coordinate_labels:
        labels["log10_D"] = r"$\log(D)$"
        labels["m_win"] = r"$m_{\rm win}$"
        labels["m_win_cont"] = r"$m_{\rm win}$"
    if log10_d_labels:
        labels["log10_D"] = r"$\log_{10}(D)$"
    return labels


def _right_annotation_math(
    video_coordinate_labels: bool = False,
    log10_d_labels: bool = False,
) -> dict[str, str]:
    labels = dict(cmp.CORNER_PARAM_ANNOTATION_MATH)
    labels.update(RIGHT_ANNOTATION_MATH)
    if video_coordinate_labels:
        labels["log10_D"] = r"\log(D)"
        labels["m_win"] = r"m_{\rm win}"
        labels["m_win_cont"] = r"m_{\rm win}"
    if log10_d_labels:
        labels["log10_D"] = r"\log_{10}(D)"
    return labels


def _right_diag_title_x(args: argparse.Namespace) -> dict[str, float]:
    diag_title_x = dict(RIGHT_DIAG_TITLE_X)
    if args.right_log10_d_diag_title_x is not None:
        diag_title_x["log10_D"] = float(args.right_log10_d_diag_title_x)
    if args.right_mwin_diag_title_x is not None:
        diag_title_x["m_win"] = float(args.right_mwin_diag_title_x)
        diag_title_x["m_win_cont"] = float(args.right_mwin_diag_title_x)
    if args.right_logsigma2_diag_title_x is not None:
        diag_title_x["log10_sigma2"] = float(args.right_logsigma2_diag_title_x)
    if args.right_logtau_diag_title_x is not None:
        diag_title_x["log10_tau"] = float(args.right_logtau_diag_title_x)
    return diag_title_x


def _read_cache(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _cache_valid(panel_pdf: Path, payload: dict[str, Any], *, force: bool) -> bool:
    if force or not _pdf_ok(panel_pdf):
        return False
    cached = _read_cache(_cache_path(panel_pdf))
    return cached == json.loads(json.dumps(payload))


def _without_prior_colors(payload: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(payload))
    priors = out.get("priors")
    if isinstance(priors, list):
        for prior in priors:
            if isinstance(prior, dict):
                prior.pop("color", None)
    return out


def _common_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "priors": [
            {"label": label, "path": str(path.expanduser().resolve()), "color": color}
            for label, path, color in PRIOR_SPECS
        ],
        "inputs": _input_signatures(),
        "band": str(args.band),
        "n_draws_per_source": _draw_limit(args),
    }


def _left_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = _common_payload(args)
    payload.pop("cache_version", None)
    payload.update(
        {
            "panel": "ensemble_recovery",
            "min_count": int(args.min_count),
            "mwin_source": str(args.mwin_source),
            "summary_cache_version": LEFT_SUMMARY_CACHE_VERSION,
            "params": LEFT_PARAMS,
            "left_default_dex_ylim": float(args.left_default_dex_ylim),
            "left_log10_d_ylim": (
                None if args.left_log10_d_ylim is None else float(args.left_log10_d_ylim)
            ),
            "n_bins": 11,
            "layout": "vertical",
            "include_m0": True,
            "exclude_rows": _exclude_rows_arg(args),
        }
    )
    return payload


def _left_style_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "panel": "ensemble_recovery_pdf",
        "summary_payload": _left_payload(args),
        "xlim": [float(LEFT_XLIM_MIN), "data_max"],
        "realistic_band": [-0.5, 1.2],
        "show_gaia_metric": False,
        "axis_labelsize": LEFT_AXIS_LABELSIZE,
        "yaxis_labelsize": LEFT_YAXIS_LABELSIZE,
        "tick_labelsize": LEFT_TICK_LABELSIZE,
        "ylabel_x": LEFT_YLABEL_X,
        "hspace": LEFT_PANEL_HSPACE,
        "reference_labelsize": LEFT_REFERENCE_LABELSIZE,
        "show_prior_legend": False,
        "method_in_panel_title": "MCMC posterior median + 16--84\\% mock scatter",
        "subtitle_fontsize": float(args.subtitle_fontsize),
        "colors": [color for _, _, color in PRIOR_SPECS],
        "labels": [label for label, _, _ in PRIOR_SPECS],
        "axis_label_variant": "eta_m_with_units",
        "mwin_fill_mode": str(args.mwin_fill_mode),
        "mwin_fill_alpha": float(args.mwin_fill_alpha),
        "left_interpolation": str(args.left_interpolation),
        "left_interp_points": int(args.left_interp_points),
        "left_smooth_sigma": float(args.left_smooth_sigma),
        "left_ylim_override": _left_ylim_overrides(args),
        "left_draw_order": _right_draw_order(args),
        "line_widths": {
            "prior": float(args.left_prior_lw),
            "reference": float(args.left_reference_lw),
            "vertical": float(args.left_reference_lw),
        },
    }


def _right_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = _common_payload(args)
    payload.update(
        {
            "panel": "single_source_corner",
            "source_id": int(args.source_id),
            "annotate_prior_label": "Prior 5",
            "annotate_run_idx": 4,
            "show_title": False,
            "corner_style": _right_corner_style(args),
            "corner_params": _right_corner_params(args),
            "corner_label_variant": (
                "log10_D_m" if bool(args.right_log10_d_labels) else "eta_m"
            ),
            "video_coordinate_labels": bool(args.right_video_coordinate_labels),
            "log10_d_labels": bool(args.right_log10_d_labels),
            "draw_order": _right_draw_order(args),
            "diag_title_x": _right_diag_title_x(args),
            "scripts": _script_signatures(
                Path(__file__).with_name("plot_recovery_posteriors.py"),
                Path(__file__).with_name("diagnostics.py"),
            ),
        }
    )
    return payload


def _left_summary_valid(summary_npz: Path, payload: dict[str, Any], *, force: bool) -> bool:
    if force or not summary_npz.exists() or summary_npz.stat().st_size <= 0:
        return False
    try:
        with np.load(summary_npz, allow_pickle=False) as data:
            cached_payload = json.loads(str(data["payload_json"]))
        cached_payload.pop("scripts", None)
        cached_payload.pop("cache_version", None)
        expected_payload = dict(payload)
        expected_payload.pop("scripts", None)
        expected_payload.pop("cache_version", None)
        return _without_prior_colors(cached_payload) == _without_prior_colors(expected_payload)
    except Exception:
        return False


def _build_left_summary(args: argparse.Namespace, summary_npz: Path, payload: dict[str, Any]) -> None:
    labels = [label for label, _, _ in PRIOR_SPECS]
    inputs = [str(path) for _, path, _ in PRIOR_SPECS]
    print(f"Building left summary cache: {summary_npz}", flush=True)
    runs = baseline._load_runs(inputs, labels, args.band, args.mwin_source)
    _limit_posterior_draws(runs, _draw_limit(args))
    _drop_rows_from_runs(runs, _exclude_rows_arg(args))
    edges = cmp._common_log_ratio_edges(runs, n_bins=11)

    n_params = len(LEFT_PARAMS)
    n_runs = len(runs)
    n_bins = edges.size - 1
    x = np.full((n_params, n_runs, n_bins), np.nan, dtype=float)
    med = np.full_like(x, np.nan)
    q16 = np.full_like(x, np.nan)
    q84 = np.full_like(x, np.nan)

    for ip, param in enumerate(LEFT_PARAMS):
        for ir, run in enumerate(runs):
            xc, ymed, y16, y84 = baseline._binned_bias(
                run,
                param,
                edges,
                min_count=int(args.min_count),
            )
            n = min(n_bins, int(xc.size))
            x[ip, ir, :n] = xc[:n]
            med[ip, ir, :n] = ymed[:n]
            q16[ip, ir, :n] = y16[:n]
            q84[ip, ir, :n] = y84[:n]

    default_dex_ylim = abs(float(args.left_default_dex_ylim))
    ylims = np.tile(np.asarray([-default_dex_ylim, default_dex_ylim], dtype=float), (n_params, 1))
    if args.left_log10_d_ylim is not None:
        log10_d_ylim = abs(float(args.left_log10_d_ylim))
        for ip, param in enumerate(LEFT_PARAMS):
            if param == "log10_D":
                ylims[ip] = [-log10_d_ylim, log10_d_ylim]
                break
    for ip, param in enumerate(LEFT_PARAMS):
        if param not in {"m_win", "mean_mag"}:
            continue
        mag_values = np.concatenate(
            [
                arr[np.isfinite(arr)]
                for arr in (med[ip], q16[ip], q84[ip])
                if np.any(np.isfinite(arr))
            ]
        )
        if mag_values.size:
            lim_mag = float(np.nanpercentile(np.abs(mag_values), 98))
            lim_mag = max(0.02, 1.10 * lim_mag)
        else:
            lim_mag = 0.05
        ylims[ip] = [-lim_mag, lim_mag]

    summary_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        summary_npz,
        payload_json=json.dumps(payload, sort_keys=True),
        params=np.asarray(LEFT_PARAMS, dtype=str),
        labels=np.asarray(labels, dtype=str),
        colors=np.asarray([color for _, _, color in PRIOR_SPECS], dtype=str),
        edges=edges,
        x=x,
        med=med,
        q16=q16,
        q84=q84,
        ylims=ylims,
    )


def _load_or_build_left_summary(args: argparse.Namespace, summary_npz: Path) -> dict[str, Any]:
    payload = _left_payload(args)
    if not _left_summary_valid(summary_npz, payload, force=bool(args.force_summary)):
        _build_left_summary(args, summary_npz, payload)
    else:
        print(f"Using cached left summary: {summary_npz}", flush=True)

    with np.load(summary_npz, allow_pickle=True) as data:
        return {
            "params": [str(p) for p in data["params"].tolist()],
            "labels": [str(x) for x in data["labels"].tolist()],
            "colors": [color for _, _, color in PRIOR_SPECS],
            "x": np.asarray(data["x"], dtype=float),
            "med": np.asarray(data["med"], dtype=float),
            "q16": np.asarray(data["q16"], dtype=float),
            "q84": np.asarray(data["q84"], dtype=float),
            "ylims": np.asarray(data["ylims"], dtype=float),
        }


def _plot_left_from_summary(summary: dict[str, Any], out_pdf: Path, args: argparse.Namespace) -> Path:
    params = summary["params"]
    labels = summary["labels"]
    colors = summary["colors"]
    x = summary["x"]
    med = summary["med"]
    q16 = summary["q16"]
    q84 = summary["q84"]
    ylims = np.array(summary["ylims"], dtype=float, copy=True)
    overrides = _left_ylim_overrides(args)
    if overrides:
        for ip, param in enumerate(params):
            if param in overrides:
                v = float(overrides[param])
                ylims[ip] = [-v, v]
    # Draw/overplot order for the priors (later = on top), matching the corner panel so
    # the same prior ends up on top in both panels. Defaults to the right-panel order.
    raw_order = _right_draw_order(args)
    if raw_order is None:
        left_order = list(range(len(colors)))
    else:
        left_order = [k for k in raw_order if 0 <= k < len(colors)]
        for k in range(len(colors)):
            if k not in left_order:
                left_order.append(k)
    finite_x = x[np.isfinite(x) & (x >= float(LEFT_XLIM_MIN))]
    xlim_right = float(np.nanmax(finite_x)) if finite_x.size else 3.0

    def _display_curve(xx: np.ndarray, yy: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mode = str(getattr(args, "left_interpolation", "linear"))
        finite = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(lo) & np.isfinite(hi)
        xx = np.asarray(xx[finite], dtype=float)
        yy = np.asarray(yy[finite], dtype=float)
        lo = np.asarray(lo[finite], dtype=float)
        hi = np.asarray(hi[finite], dtype=float)
        if mode != "pchip" or PchipInterpolator is None or xx.size < 4:
            return xx, yy, lo, hi
        n_points = max(int(getattr(args, "left_interp_points", 240) or 240), int(xx.size))
        x_dense = np.linspace(float(xx[0]), float(xx[-1]), n_points)
        yy_dense = PchipInterpolator(xx, yy)(x_dense)
        lo_dense = PchipInterpolator(xx, lo)(x_dense)
        hi_dense = PchipInterpolator(xx, hi)(x_dense)
        # PCHIP passes exactly through every grid node, so the curve still has visible
        # "corners" (slope discontinuities) at each finite-grid knot.  An optional
        # Gaussian low-pass over the densified curve rounds those edges off without
        # shifting the overall shape; sigma is measured in display points.
        smooth_sigma = float(getattr(args, "left_smooth_sigma", 0.0) or 0.0)
        if smooth_sigma > 0.0 and gaussian_filter1d is not None and x_dense.size >= 3:
            yy_dense = gaussian_filter1d(yy_dense, smooth_sigma, mode="nearest")
            lo_dense = gaussian_filter1d(lo_dense, smooth_sigma, mode="nearest")
            hi_dense = gaussian_filter1d(hi_dense, smooth_sigma, mode="nearest")
        crossed = lo_dense > hi_dense
        if np.any(crossed):
            mid = 0.5 * (lo_dense[crossed] + hi_dense[crossed])
            lo_dense[crossed] = mid
            hi_dense[crossed] = mid
        return x_dense, yy_dense, lo_dense, hi_dense

    fig_height = 8.10 + max(0, len(params) - 4) * 1.85
    fig, axes = plt.subplots(len(params), 1, figsize=(5.25, fig_height), sharex=True)
    axes = np.asarray(axes).ravel()

    for ip, (ax, param) in enumerate(zip(axes, params)):
        baseline._add_realistic_band(ax, (-0.5, 1.2), label=None, label_loc="bottom-left")
        mwin_fill_mode = str(getattr(args, "mwin_fill_mode", "all"))
        if param == "m_win" and mwin_fill_mode == "single_gray":
            xx = x[ip, 0]
            lo = q16[ip, 0]
            hi = q84[ip, 0]
            finite = np.isfinite(xx) & np.isfinite(lo) & np.isfinite(hi)
            if np.any(finite):
                ax.fill_between(
                    xx[finite],
                    lo[finite],
                    hi[finite],
                    color="0.45",
                    alpha=float(args.mwin_fill_alpha),
                    linewidth=0,
                    zorder=2,
                )
        for ir in left_order:
            color = colors[ir]
            xx, yy, lo, hi = _display_curve(x[ip, ir], med[ip, ir], q16[ip, ir], q84[ip, ir])
            if xx.size == 0:
                continue
            show_fill = not (param == "m_win" and mwin_fill_mode in {"none", "single_gray"})
            if show_fill:
                fill_alpha = float(args.mwin_fill_alpha) if param == "m_win" else 0.16
                ax.fill_between(xx, lo, hi, color=color, alpha=fill_alpha, linewidth=0, zorder=2)
            ax.plot(xx, yy, color=color, lw=float(args.left_prior_lw), zorder=3)
        baseline._decorate_axis(
            ax,
            show_ylabel=True,
            ylabel=_left_axis_label(param),
            show_xlabel=(ip == len(params) - 1),
            vertical_line_labels=(ip == 0),
            show_reference_text=(ip == 0),
            vertical_line_label_fontsize=LEFT_REFERENCE_LABELSIZE,
            reference_lw=float(args.left_reference_lw),
            vertical_lw=float(args.left_reference_lw),
        )
        ax.xaxis.label.set_size(LEFT_AXIS_LABELSIZE)
        ax.yaxis.label.set_size(LEFT_YAXIS_LABELSIZE)
        ax.yaxis.set_label_coords(LEFT_YLABEL_X, 0.5)
        ax.tick_params(axis="both", which="major", labelsize=LEFT_TICK_LABELSIZE)
        ax.set_xlim(float(LEFT_XLIM_MIN), xlim_right)
        ax.set_ylim(*ylims[ip])

    fig.subplots_adjust(left=0.285, right=0.985, bottom=0.09, top=0.965, hspace=LEFT_PANEL_HSPACE)
    out_pdf = out_pdf.expanduser().resolve()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


def _generate_recovery_panel(args: argparse.Namespace) -> Path:
    out_pdf = Path(args.left_panel_pdf).expanduser().resolve()
    summary_npz = Path(args.left_summary_npz).expanduser().resolve()
    summary = _load_or_build_left_summary(args, summary_npz)
    if args.export_plot_data is not None:
        from plotting.plot_data import save_snapshot
        save_snapshot(args.export_plot_data, 'figure_03_prior_sensitivity_summary', {"summary": summary, "args": _plot_style_args(args)})
        return out_pdf
    payload = _left_style_payload(args)
    if _cache_valid(out_pdf, payload, force=bool(args.force)):
        print(f"Using cached left panel: {out_pdf}", flush=True)
        return out_pdf

    print(f"Rendering left panel from summary cache: {out_pdf}", flush=True)
    _plot_left_from_summary(summary, out_pdf, args)
    _write_cache(_cache_path(out_pdf), payload)
    return out_pdf


def _generate_corner_panel(args: argparse.Namespace) -> Path:
    labels = [label for label, _, _ in PRIOR_SPECS]
    colors = [color for _, _, color in PRIOR_SPECS]
    paths = [path.expanduser().resolve() for _, path, _ in PRIOR_SPECS]
    out_pdf = Path(args.right_panel_pdf).expanduser().resolve()
    payload = _right_payload(args)
    if args.export_plot_data is None and _cache_valid(out_pdf, payload, force=bool(args.force)):
        print(f"Using cached right panel: {out_pdf}", flush=True)
        return out_pdf

    print(f"Building right panel cache: {out_pdf}", flush=True)
    selected_source_ids = gallery._selected_source_ids(
        paths[0],
        args.band,
        n_sources=1,
        seed=42,
        object_ids=[int(args.source_id)],
    )
    runs = [
        gallery._load_sparse_run(path, labels[i], args.band, selected_source_ids)
        for i, path in enumerate(paths)
    ]
    _limit_posterior_draws(runs, _draw_limit(args))
    cmp._validate_compatible_runs(runs)

    if args.export_plot_data is not None:
        from plotting.plot_data import save_snapshot
        compact_runs = []
        for run in runs:
            compact_run = {k: v for k, v in run.items() if k not in {"npz_path", "meta_path", "meta"}}
            compact_run["meta"] = {"sampling_model": (run.get("meta") or {}).get("sampling_model", {})}
            compact_runs.append(compact_run)
        save_snapshot(args.export_plot_data, 'figure_03_prior_sensitivity_example', {"runs": compact_runs, "colors": colors, "args": _plot_style_args(args)})
        return out_pdf
    _plot_corner_from_runs(runs, colors, out_pdf, args)
    _write_cache(_cache_path(out_pdf), payload)
    return out_pdf


def _plot_style_args(args: argparse.Namespace) -> dict[str, Any]:
    # Output locations and local-cache metadata are not portable plot settings.
    omit = {"left_panel_pdf", "left_summary_npz", "right_panel_pdf", "output_pdf", "export_plot_data"}
    return {key: value for key, value in vars(args).items() if key not in omit}


def _plot_corner_from_runs(runs, colors, out_pdf: Path, args: argparse.Namespace) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    original_legend = cmp._corner_legend_label
    original_diag_title_x = dict(cmp.CORNER_DIAG_TITLE_X)
    cmp._corner_legend_label = lambda run: cmp._tex_escape_text(run["label"])
    cmp.CORNER_DIAG_TITLE_X.update(_right_diag_title_x(args))
    try:
        with PdfPages(out_pdf) as pdf:
            cmp._add_corner_page(
                pdf,
                runs,
                colors,
                obj_idx=0,
                source_id=int(args.source_id),
                title_note="",
                annotate_run_idx=4,
                show_title=False,
                corner_style=_right_corner_style(args),
                params=_right_corner_params(args),
                param_labels=_right_param_labels(
                    bool(args.right_video_coordinate_labels),
                    bool(args.right_log10_d_labels),
                ),
                annotation_math=_right_annotation_math(
                    bool(args.right_video_coordinate_labels),
                    bool(args.right_log10_d_labels),
                ),
                draw_order=_right_draw_order(args),
            )
    finally:
        cmp._corner_legend_label = original_legend
        cmp.CORNER_DIAG_TITLE_X.clear()
        cmp.CORNER_DIAG_TITLE_X.update(original_diag_title_x)


def _page_rect(path: Path) -> fitz.Rect:
    with fitz.open(str(path)) as doc:
        return doc[0].rect


def _title_strip_pdf(
    page_w: float,
    title_h: float,
    left_x: float,
    left_w: float,
    right_x: float,
    right_w: float,
    title_fontsize: float,
    subtitle_fontsize: float,
) -> Path:
    path = Path("/tmp/agn_prior_sensitivity_titles.pdf")
    fig = plt.figure(figsize=(page_w / 72.0, title_h / 72.0))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    title_kwargs = dict(
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=float(title_fontsize),
        color="black",
    )
    ax.text(left_x / page_w, 0.70, r"a) Ensemble recovery", **title_kwargs)
    ax.text(
        left_x / page_w,
        0.23,
        r"MCMC posterior median + 16--84\% mock scatter",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=float(subtitle_fontsize),
        color="black",
    )
    ax.text(right_x / page_w, 0.70, r"b) Single-source posterior", **title_kwargs)
    fig.savefig(path, transparent=True)
    plt.close(fig)
    return path


def _compose(left_pdf: Path, right_pdf: Path, output_pdf: Path, args: argparse.Namespace) -> Path:
    left_rect = _page_rect(left_pdf)
    right_rect = _page_rect(right_pdf)
    right_layout_ref = str(getattr(args, "right_layout_reference_pdf", "") or "").strip()
    right_layout_rect = _page_rect(Path(right_layout_ref)) if right_layout_ref else right_rect

    margin = 24.0
    gap = 24.0
    title_h = 46.0
    bottom = 16.0
    body_h = float(args.body_height_pt)
    right_body_h = body_h * float(RIGHT_PANEL_SCALE)
    left_w = body_h * (left_rect.width / left_rect.height)
    right_w = right_body_h * (right_layout_rect.width / right_layout_rect.height)
    page_w = margin * 2.0 + gap + left_w + right_w
    page_h = margin + title_h + max(body_h, right_body_h) + bottom

    output_pdf = output_pdf.expanduser().resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    out = fitz.open()
    page = out.new_page(width=page_w, height=page_h)

    title_top = margin - 4.0
    body_top = margin + title_h
    left_x = margin
    right_x = margin + left_w + gap
    left_target = fitz.Rect(left_x, body_top, left_x + left_w, body_top + body_h)
    right_target = fitz.Rect(right_x, body_top, right_x + right_w, body_top + right_body_h)

    title_pdf = _title_strip_pdf(
        page_w,
        title_h,
        left_x,
        left_w,
        right_x,
        right_w,
        float(args.title_fontsize),
        float(args.subtitle_fontsize),
    )
    with fitz.open(str(title_pdf)) as src:
        page.show_pdf_page(fitz.Rect(0.0, title_top, page_w, title_top + title_h), src, 0)

    with fitz.open(str(left_pdf)) as src:
        page.show_pdf_page(left_target, src, 0)
    with fitz.open(str(right_pdf)) as src:
        page.show_pdf_page(right_target, src, 0)

    out.set_metadata(
        {
            "title": "",
            "author": "",
            "subject": "",
            "keywords": "",
            "creator": "plot_prior_sensitivity.py",
            "producer": "",
        }
    )
    out.save(str(output_pdf))
    out.close()
    return output_pdf


def build_parser() -> argparse.ArgumentParser:
    default_right = PROJECT_ROOT / "figures/figure_03_prior_sensitivity_right.pdf"
    default_left = PROJECT_ROOT / "figures/figure_03_prior_sensitivity_left.pdf"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-plot-data", type=Path)
    ap.add_argument("--band", default="Gaia_G")
    ap.add_argument("--source-id", type=int, default=22692)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--mwin-source", choices=["trap", "cont"], default="cont")
    ap.add_argument(
        "--n-draws-per-source",
        type=int,
        default=2000,
        help=(
            "If positive, use only the first N posterior draws per object when "
            "building the recovery and corner panels.  This leaves the input "
            "NPZ files untouched."
        ),
    )
    ap.add_argument("--run-suffix", default="", help="Optional suffix for custom prior directories and posterior sample files.")
    ap.add_argument("--palette", choices=sorted(PRIOR_COLOR_PALETTES), default=DEFAULT_PALETTE_NAME)
    ap.add_argument("--left-prior-lw", type=float, default=3.0)
    ap.add_argument("--left-reference-lw", type=float, default=1.25)
    ap.add_argument(
        "--left-default-dex-ylim",
        type=float,
        default=2.0,
        help="Symmetric y-limit used for dex-valued ensemble-recovery rows by default.",
    )
    ap.add_argument(
        "--left-log10-d-ylim",
        type=float,
        default=1.0,
        help="Optional symmetric y-limit override for only the Delta log10(D) ensemble row.",
    )
    ap.add_argument(
        "--subtitle-fontsize",
        type=float,
        default=11.4,
        help="Font size for the 'MCMC posterior median...' subtitle in the combined figure.",
    )
    ap.add_argument(
        "--title-fontsize",
        type=float,
        default=14.5,
        help="Font size for the a/b panel headings in the combined figure.",
    )
    ap.add_argument("--right-diag-lw", type=float, default=None)
    ap.add_argument("--right-contour-lw", type=float, default=None)
    ap.add_argument("--right-truth-lw", type=float, default=3.8)
    ap.add_argument("--right-truth-color", default="black")
    ap.add_argument("--right-filled-contours", action="store_true")
    ap.add_argument(
        "--right-smooth-style",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Render the right corner panel in the Stage-2 'smooth reference' style: "
            "oversmoothed KDE diagonals, smooth 2D KDE contours, filled contours with a "
            "thin outer / thick inner outline. On by default; use --no-right-smooth-style "
            "to disable."
        ),
    )
    ap.add_argument(
        "--right-drop-params",
        default="log10_jitter",
        help=(
            "Comma-separated corner parameters to omit from the right panel. "
            "Defaults to dropping log10_jitter (the log10(sigma_add) row/column); "
            "pass an empty string to keep all parameters."
        ),
    )
    ap.add_argument("--right-legend-lw", type=float, default=None)
    ap.add_argument("--right-legend-fontsize", type=float, default=None)
    ap.add_argument("--right-show-legend", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--right-group-prior-legend", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--right-video-coordinate-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Display log(D) and m_win labels in the right corner panel for video use.",
    )
    ap.add_argument(
        "--right-log10-d-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Display log10(D) instead of eta in the right-panel axis and annotation.",
    )
    ap.add_argument("--right-axis-labelsize", type=float, default=None)
    ap.add_argument("--right-tick-labelsize", type=float, default=None)
    ap.add_argument("--right-diag-annotation-fontsize", type=float, default=None)
    ap.add_argument("--right-diag-label-boxes", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--right-diag-label-box-params",
        default="",
        help=(
            "Comma-separated parameters whose diagonal labels receive shading. "
            "Empty keeps the existing behavior of shading every configured parameter."
        ),
    )
    ap.add_argument(
        "--right-diag-label-box-color",
        default=None,
        help="Optional color applied to every selected diagonal label box.",
    )
    ap.add_argument(
        "--right-draw-order",
        default="2,3,4,1,0",
        help=(
            "Comma-separated 0-based prior indices giving the draw/overplot order in the "
            "corner panel (later = on top). Legend order and the median annotation stay in "
            "natural Prior 1..5 order. Default 2,3,4,1,0 = draw Prior3,4,5,2,1. Empty = natural."
        ),
    )
    ap.add_argument(
        "--right-range-pad-frac",
        type=float,
        default=0.12,
        help=(
            "Fractional padding added to each corner subplot axis range so posteriors are "
            "not clipped at the panel edges (compare_checker default is 0.06)."
        ),
    )
    ap.add_argument(
        "--right-log10-d-diag-title-x",
        type=float,
        default=None,
        help="Optional x-position override for the log10(D) diagonal corner annotation.",
    )
    ap.add_argument(
        "--right-mwin-diag-title-x",
        type=float,
        default=None,
        help="Optional x-position override for the m_win top-diagonal corner annotation.",
    )
    ap.add_argument(
        "--right-logsigma2-diag-title-x",
        type=float,
        default=None,
        help="Optional x-position override for the log10(sigma^2) diagonal corner annotation.",
    )
    ap.add_argument(
        "--right-logtau-diag-title-x",
        type=float,
        default=None,
        help="Optional x-position override for the log10(tau) diagonal corner annotation.",
    )
    ap.add_argument(
        "--mwin-fill-mode",
        choices=["all", "none", "single_gray"],
        default="all",
        help="How to render the m_win shaded 16--84 mock-scatter band in the ensemble panel.",
    )
    ap.add_argument(
        "--mwin-fill-alpha",
        type=float,
        default=0.16,
        help="Alpha used for the m_win shaded band; other ensemble bands keep the paper default.",
    )
    ap.add_argument(
        "--left-interpolation",
        choices=["linear", "pchip"],
        default="pchip",
        help="Display interpolation for binned median/quantile curves in the ensemble panel.",
    )
    ap.add_argument(
        "--left-interp-points",
        type=int,
        default=300,
        help="Number of display points used when --left-interpolation=pchip.",
    )
    ap.add_argument(
        "--left-smooth-sigma",
        type=float,
        default=5.0,
        help=(
            "If positive, apply a Gaussian low-pass (sigma in display points) to the "
            "PCHIP-densified ensemble curves to round off the hard corners left at each "
            "finite mock-grid node. Typical gentle values: 2-5 with --left-interp-points 300."
        ),
    )
    ap.add_argument(
        "--left-ylim-override",
        default="",
        help=(
            "Comma-separated param:halfrange pairs to override symmetric ensemble-panel "
            "y-limits at plot time, e.g. 'log10_D:0.5,m_win:0.02'. Applied when rendering "
            "from the cached summary, so it does NOT trigger a summary rebuild."
        ),
    )
    ap.add_argument("--body-height-pt", type=float, default=DEFAULT_BODY_HEIGHT_PT)
    ap.add_argument("--force", action="store_true", help="Rebuild cached panel PDFs even if cache metadata matches.")
    ap.add_argument("--force-summary", action="store_true", help="Recompute the expensive binned-summary cache.")
    ap.add_argument(
        "--exclude-rows",
        default="9624,14729,14818,20092",
        help=(
            "Comma-separated zero-based full-grid row indices to exclude from the "
            "ensemble recovery panel only. The single-source corner panel is left unchanged."
        ),
    )
    ap.add_argument("--left-panel-pdf", default=str(default_left))
    ap.add_argument("--left-summary-npz", default=str(default_left.with_suffix(".summary.npz")))
    ap.add_argument("--right-panel-pdf", default=str(default_right))
    ap.add_argument(
        "--right-layout-reference-pdf",
        default="",
        help=(
            "Optional PDF whose page aspect is used for laying out the right panel. "
            "Useful when a variant changes text size but should keep the original page box."
        ),
    )
    ap.add_argument(
        "--output-pdf",
        default=str(PRIOR_SENSITIVITY_DIR / 'figure_03_prior_sensitivity.pdf'),
    )
    ap.add_argument(
        "--right-only",
        action="store_true",
        help="Write only the single-source corner panel to --output-pdf.",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()
    global PRIOR_SPECS
    PRIOR_SPECS = _prior_specs_for_suffix(str(args.run_suffix), str(args.palette))
    for _, path, _ in PRIOR_SPECS:
        if not path.exists():
            raise FileNotFoundError(str(path))
    if bool(args.right_only):
        args.right_panel_pdf = str(Path(args.output_pdf))
        out_pdf = _generate_corner_panel(args)
        print(f"Wrote right-only PDF: {out_pdf}")
        print(f"Wrote cache manifest: {_cache_path(out_pdf)}")
        print("Prior mapping:")
        for label, path, color in PRIOR_SPECS:
            print(f"  {label}: {path.parent.name}  {color}")
        return
    left_pdf = _generate_recovery_panel(args)
    right_pdf = _generate_corner_panel(args)
    if args.export_plot_data is not None:
        return
    out_pdf = _compose(left_pdf, right_pdf, Path(args.output_pdf), args)
    print(f"Wrote left panel: {left_pdf}")
    print(f"Wrote right panel: {right_pdf}")
    print(f"Wrote combined PDF: {out_pdf}")
    print(f"Wrote cache manifests: {_cache_path(left_pdf)}, {_cache_path(right_pdf)}")
    print("Prior mapping:")
    for label, path, color in PRIOR_SPECS:
        print(f"  {label}: {path.parent.name}  {color}")


if __name__ == "__main__":
    main()
