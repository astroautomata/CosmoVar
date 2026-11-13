#!/usr/bin/env python3
"""
Paper-ready Stage-1 mock recovery bias versus rest-frame baseline.

This makes the compact version of the diagnostic page:
  - one row of panels for Delta log10(D), Delta log10(tau), Delta log10(sigma^2)
  - different colours for the fitting/prior setups
  - per-object point estimates are MCMC posterior medians
  - shaded regions are the 16th--84th percentile spread of recovery biases
    across mock light curves in each baseline bin, not posterior intervals.
"""

from __future__ import annotations

from framework import PROJECT_ROOT

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/agn-variability-cosmology-mpl-{os.getuid()}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
matplotlib.use("Agg")

_DEFAULT_MPLRC = PROJECT_ROOT / "matplotlibrc"
if _DEFAULT_MPLRC.exists():
    matplotlib.rc_file(str(_DEFAULT_MPLRC))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

from plotting import diagnostics as cmp

if _DEFAULT_MPLRC.exists():
    matplotlib.rc_file(str(_DEFAULT_MPLRC))


DEFAULT_INPUTS = [
    "results/stage1/stage1_mock_recoverytests/grid_3/grid3_run1/posteriors_grid3_run1.npz",
    "results/stage1/stage1_mock_recoverytests/grid_3/grid3_run2/posteriors_grid3_run2.npz",
    "results/stage1/stage1_mock_recoverytests/grid_3/grid3_run3/posteriors_grid3_run3.npz",
]
DEFAULT_LABELS = [
    r"Flat $\log\sigma^2,\log\tau$ (bounded)",
    r"Flat $\log D,\log\tau$ (bounded)",
    r"Flat $\log\sigma^2,\log\tau$ (wide)",
]
DEFAULT_OUT = "results/stage1/stage1_mock_recoverytests/grid_3/grid3_recovery_bias_vs_baseline_paper.pdf"


def _split_csv(spec: str | None) -> list[str] | None:
    if spec is None:
        return None
    parts = next(csv.reader([str(spec)], skipinitialspace=True))
    parts = [part.strip() for part in parts if part.strip()]
    return parts or None


def _use_requested_mwin_source(run: dict[str, Any], source: str) -> None:
    source = str(source).lower().strip()
    if source == "trap":
        return
    if source != "cont":
        raise ValueError("mwin source must be 'trap' or 'cont'")
    if "m_win_cont" not in run["posterior"] or "m_win_cont" not in run["truth"]:
        raise KeyError("Requested continuous m_win, but m_win_cont posterior/truth arrays are missing.")
    # Keep the paper label as m_win, but use the continuous-integral definition.
    run["posterior"]["m_win"] = run["posterior"]["m_win_cont"]
    run["truth"]["m_win"] = run["truth"]["m_win_cont"]


def _load_runs(inputs: list[str], labels: list[str], band: str, mwin_source: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path, label in zip(inputs, labels):
        label = label.replace("\\n", "\n")
        run = cmp._load_run(Path(path).expanduser().resolve(), label=label, band=band)
        _use_requested_mwin_source(run, mwin_source)
        cmp._prepare_metrics(run)
        runs.append(run)
    cmp._validate_compatible_runs(runs)
    return runs


def _baseline_log10(run: dict[str, Any]) -> np.ndarray:
    return np.log10(cmp._baseline_tau_ratio(run))


def _binned_bias(
    run: dict[str, Any],
    param: str,
    edges: np.ndarray,
    *,
    min_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = _baseline_log10(run)
    estimate = cmp._point_estimate(run["posterior"][param], "median")
    bias = estimate - np.asarray(run["truth"][param], dtype=float)
    xc, med, q16, q84, counts = cmp._binned_summary(x, bias, edges, min_count=min_count)
    return xc, med, q16, q84


def _axis_label(param: str) -> str:
    if param == "log10_D":
        return r"$\Delta \log_{10}(D)$"
    if param == "log10_tau":
        return r"$\Delta \log_{10}(\tau)$"
    if param == "log10_sigma2":
        return r"$\Delta \log_{10}(\sigma^{2})$"
    if param == "m_win":
        return r"$\Delta m_{\rm win}\ [\mathrm{mag}]$"
    if param == "mean_mag":
        return r"$\Delta m_0\ [\mathrm{mag}]$"
    return rf"$\Delta$ {param}"


def _metric_param_label(param: str) -> str:
    if param == "log10_D":
        return r"D"
    if param == "log10_tau":
        return r"\tau"
    if param == "log10_sigma2":
        return r"\sigma^{2}"
    if param == "m_win":
        return r"m_{\rm win}"
    if param == "mean_mag":
        return r"m_0"
    return param


def _all_binned_values(
    runs: list[dict[str, Any]],
    params: list[str],
    edges: np.ndarray,
    *,
    min_count: int,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for run in runs:
        for param in params:
            _, med, q16, q84 = _binned_bias(run, param, edges, min_count=min_count)
            for arr in (med, q16, q84):
                arr = np.asarray(arr, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    values.append(arr)
    if not values:
        return np.array([], dtype=float)
    return np.concatenate(values)


def _decorate_axis(
    ax: plt.Axes,
    *,
    show_ylabel: bool,
    ylabel: str,
    show_xlabel: bool = True,
    vertical_line_labels: bool = False,
    show_reference_text: bool = True,
    vertical_line_label_fontsize: float = 8.0,
    reference_text_fontsize: float = 7.5,
    reference_lw: float = 1.0,
    vertical_lw: float = 1.0,
) -> None:
    ax.axhline(0.0, color="black", ls="--", lw=float(reference_lw), alpha=0.80, zorder=1)
    ax.set_xlabel(r"$T_{\rm rf}/\tau_{\rm true}$" if show_xlabel else "")
    ax.set_ylabel(ylabel if show_ylabel else "", labelpad=3)
    ax.xaxis.set_major_locator(FixedLocator([-2, -1, 0, 1, 2, 3]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: rf"$10^{{{int(round(x))}}}$"))
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    for value in (3.0, 5.0, 10.0):
        xpos = np.log10(value)
        ax.axvline(xpos, color="0.25", ls=":", lw=float(vertical_lw), alpha=0.75, zorder=1)
    if vertical_line_labels:
        for value, label in [(3.0, r"$3\tau$"), (5.0, r"$5\tau$"), (10.0, r"$10\tau$")]:
            xpos = np.log10(value)
            ax.text(
                xpos + 0.035,
                0.940,
                label,
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=float(vertical_line_label_fontsize),
                color="0.25",
            )
    elif show_reference_text:
        ax.text(
            0.985,
            0.965,
            r"$T_{\rm rf}/\tau_{\rm true}=3,5,10$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=float(reference_text_fontsize),
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.0),
        )


def _median_abs_bias_in_band(
    run: dict[str, Any],
    param: str,
    band: tuple[float, float],
) -> float:
    x = _baseline_log10(run)
    estimate = cmp._point_estimate(run["posterior"][param], "median")
    bias = estimate - np.asarray(run["truth"][param], dtype=float)
    lo, hi = band
    m = np.isfinite(x) & np.isfinite(bias) & (x >= lo) & (x <= hi)
    if not np.any(m):
        return np.nan
    return float(np.nanmedian(np.abs(bias[m])))


def _bias_summary_in_band(
    run: dict[str, Any],
    param: str,
    band: tuple[float, float],
) -> tuple[float, float]:
    x = _baseline_log10(run)
    estimate = cmp._point_estimate(run["posterior"][param], "median")
    bias = estimate - np.asarray(run["truth"][param], dtype=float)
    lo, hi = band
    m = np.isfinite(x) & np.isfinite(bias) & (x >= lo) & (x <= hi)
    if not np.any(m):
        return np.nan, np.nan
    b = bias[m]
    return float(np.nanmedian(b)), float(np.sqrt(np.nanmean(b * b)))


def _add_gaia_metric_block(
    ax: plt.Axes,
    runs: list[dict[str, Any]],
    param: str,
    colors: list[str],
    band: tuple[float, float] | None,
    mode: str = "median_abs",
) -> None:
    if band is None:
        return
    x0 = 0.86
    label = _metric_param_label(param)
    if mode == "median_rmse":
        if param in {"m_win", "mean_mag"}:
            header = rf"Gaia range" "\n" rf"median $\Delta$ / RMSE" "\n" rf"$[{{\rm mmag}}]$"
            fmt = ".1f"
        else:
            header = rf"Gaia range" "\n" rf"median $\Delta$ / RMSE"
            fmt = ".2f"
    else:
        if param in {"m_win", "mean_mag"}:
            header = rf"Gaia range median" "\n" rf"$|\Delta {label}|$"
            fmt = ".3f"
        else:
            header = rf"Gaia range median" "\n" rf"$|\Delta\log_{{10}} {label}|$"
            fmt = ".2f"
    ax.text(
        x0,
        0.935,
        header,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.1,
        color="0.18",
        linespacing=1.05,
    )
    for idx, (run, color) in enumerate(zip(runs, colors)):
        if mode == "median_rmse":
            med, rmse = _bias_summary_in_band(run, param, band)
            if param in {"m_win", "mean_mag"}:
                med *= 1000.0
                rmse *= 1000.0
                text = "--" if not (np.isfinite(med) and np.isfinite(rmse)) else f"{med:.2f}/{rmse:.1f}"
            else:
                text = "--" if not (np.isfinite(med) and np.isfinite(rmse)) else f"{med:{fmt}}/{rmse:{fmt}}"
        else:
            value = _median_abs_bias_in_band(run, param, band)
            text = "--" if not np.isfinite(value) else format(value, fmt)
        ax.text(
            x0,
            0.765 - 0.075 * idx,
            text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.6,
            color=color,
            fontweight="bold",
        )


def _parse_optional_band(spec: str | None) -> tuple[float, float] | None:
    if spec is None or str(spec).strip().lower() in {"", "none", "off"}:
        return None
    parts = [float(x.strip()) for x in str(spec).split(",") if x.strip()]
    if len(parts) != 2 or not np.isfinite(parts[0]) or not np.isfinite(parts[1]) or parts[0] >= parts[1]:
        raise ValueError(f"Expected band as lo,hi; got {spec!r}")
    return float(parts[0]), float(parts[1])


def _add_realistic_band(
    ax: plt.Axes,
    band: tuple[float, float] | None,
    *,
    label: str | None = None,
    label_loc: str = "bottom-left",
) -> None:
    if band is None:
        return
    lo, hi = band
    ax.axvspan(lo, hi, color="0.70", alpha=0.16, lw=0.0, zorder=0)
    if label:
        label_color = "0.55"
        if label_loc == "top":
            x_text, y_text, ha, va = 0.5 * (lo + hi), 0.965, "center", "top"
            transform = ax.get_xaxis_transform()
        else:
            x_text, y_text, ha, va = lo + 0.04 * (hi - lo), 0.055, "left", "bottom"
            transform = ax.get_xaxis_transform()
        ax.text(
            x_text,
            y_text,
            label,
            transform=transform,
            ha=ha,
            va=va,
            fontsize=7.4,
            color=label_color,
        )


def make_figure(
    runs: list[dict[str, Any]],
    out_pdf: Path,
    *,
    min_count: int = 20,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    realistic_band: tuple[float, float] | None = None,
    realistic_label: str | None = None,
    layout: str = "horizontal",
    include_m0: bool = False,
    gaia_metric: str = "median_abs",
    colors: list[str] | None = None,
    save_png: bool = True,
    axis_labelsize: float | None = None,
    tick_labelsize: float | None = None,
    show_gaia_metric: bool = True,
    dpi: int = 250,
) -> Path:
    params = ["log10_D", "log10_tau", "log10_sigma2", "m_win"]
    if include_m0:
        params.append("mean_mag")
    dex_params = params[:3]
    if colors is None:
        base_colors = ["#FF00FF", "#59A14F", "#4C78A8", "#F58518", "#9467BD", "#E45756"]
        colors = [base_colors[i % len(base_colors)] for i in range(len(runs))]
    elif len(colors) != len(runs):
        raise ValueError("colors must have the same length as runs")
    linestyles = ["-" for _ in runs]

    edges = cmp._common_log_ratio_edges(runs, n_bins=11)
    if xlim is None:
        xlim = (-2.0, float(edges[-1]))

    # ylim for the three dex panels (shared)
    if ylim is None:
        yvals = _all_binned_values(runs, dex_params, edges, min_count=min_count)
        if yvals.size:
            lim = float(np.nanpercentile(np.abs(yvals), 98))
            lim = max(0.35, min(3.5, 1.10 * lim))
        else:
            lim = 1.0
        ylim = (-lim, lim)

    # independent y-limits for magnitude panels: m_win is much tighter than m0.
    mag_ylims: dict[str, tuple[float, float]] = {}
    for param in [p for p in params if p in {"m_win", "mean_mag"}]:
        yvals_mag = _all_binned_values(runs, [param], edges, min_count=min_count)
        if yvals_mag.size:
            lim_mag = float(np.nanpercentile(np.abs(yvals_mag), 98))
            lim_mag = max(0.02, 1.10 * lim_mag)
        else:
            lim_mag = 0.05
        mag_ylims[param] = (-lim_mag, lim_mag)

    if layout not in {"horizontal", "vertical"}:
        raise ValueError("layout must be 'horizontal' or 'vertical'")
    if layout == "vertical":
        fig_height = 8.10 + max(0, len(params) - 4) * 1.85
        fig, axes = plt.subplots(len(params), 1, figsize=(5.25, fig_height), sharex=True)
    else:
        fig_width = 10.70 + max(0, len(params) - 4) * 2.45
        fig, axes = plt.subplots(1, len(params), figsize=(fig_width, 2.65), sharex=True)
    axes = np.asarray(axes).ravel()

    for idx, (ax, param) in enumerate(zip(axes, params)):
        _add_realistic_band(
            ax,
            realistic_band,
            label=realistic_label if idx == 0 else None,
            label_loc="bottom-left",
        )
        for run, color, linestyle in zip(runs, colors, linestyles):
            x, med, q16, q84 = _binned_bias(run, param, edges, min_count=min_count)
            if x.size == 0:
                continue
            ax.fill_between(x, q16, q84, color=color, alpha=0.16, linewidth=0, zorder=2)
            ax.plot(x, med, color=color, ls=linestyle, lw=1.25, zorder=3)
        _decorate_axis(
            ax,
            show_ylabel=True,
            ylabel=_axis_label(param),
            show_xlabel=(layout == "horizontal" or idx == len(params) - 1),
            vertical_line_labels=(layout == "vertical" and idx == 0),
            show_reference_text=(layout == "horizontal" or idx == 0),
        )
        if axis_labelsize is not None:
            ax.xaxis.label.set_size(float(axis_labelsize))
            ax.yaxis.label.set_size(float(axis_labelsize))
        if tick_labelsize is not None:
            ax.tick_params(axis="both", which="major", labelsize=float(tick_labelsize))
        if layout == "vertical" and show_gaia_metric:
            _add_gaia_metric_block(ax, runs, param, colors, realistic_band, mode=gaia_metric)
        ax.set_xlim(*xlim)
        ax.set_ylim(*(mag_ylims[param] if param in mag_ylims else ylim))

    handles = [
        Line2D([0], [0], color=color, lw=1.35, ls=linestyle, label=label)
        for color, linestyle, label in zip(colors, linestyles, [run["label"] for run in runs])
    ]
    if layout == "vertical":
        axes[0].legend(
            handles=handles,
            loc="upper left",
            ncol=1,
            frameon=False,
            fontsize=7.6,
            handlelength=2.2,
            title="MCMC posterior median\n+ 16--84\\% mock scatter",
            title_fontsize=7.8,
        )
    else:
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.045),
            ncol=min(len(runs), 4),
            frameon=False,
            fontsize=8.0,
            handlelength=2.4,
            columnspacing=1.0,
            title="MCMC posterior median\n+ 16--84\\% mock scatter",
            title_fontsize=8.2,
        )
    if layout == "vertical":
        left_margin = 0.17 if axis_labelsize is not None and float(axis_labelsize) > 12.0 else 0.14
        fig.subplots_adjust(left=left_margin, right=0.985, bottom=0.09, top=0.965, hspace=0.08)
    else:
        fig.subplots_adjust(left=0.065, right=0.99, bottom=0.19, top=0.78, wspace=0.22)

    out_pdf = out_pdf.expanduser().resolve()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    if save_png:
        out_png = out_pdf.with_suffix(".png")
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", default=None, help="Comma-separated Stage-1 recovery NPZ paths.")
    ap.add_argument("--labels", default=None, help="Optional CSV curve labels. Quote labels that contain commas; use \\n for line breaks.")
    ap.add_argument("--band", default="Gaia_G", help="Band to plot.")
    ap.add_argument("--out-pdf", default=DEFAULT_OUT, help="Output PDF path.")
    ap.add_argument("--min-count", type=int, default=20, help="Minimum objects per baseline bin.")
    ap.add_argument("--ylim", default=None, help="Optional y limits as lo,hi.")
    ap.add_argument("--xlim", default=None, help="Optional x limits as lo,hi.")
    ap.add_argument("--realistic-band", default=None, help="Optional shaded x-band as lo,hi, e.g. -0.5,1.2.")
    ap.add_argument("--realistic-label", default=None, help="Optional label for the shaded x-band.")
    ap.add_argument("--layout", choices=["horizontal", "vertical"], default="horizontal", help="Panel layout.")
    ap.add_argument("--include-m0", action="store_true", help="Add an extra bias-vs-baseline row/panel for the stationary mean m0.")
    ap.add_argument(
        "--gaia-metric",
        choices=["median_abs", "median_rmse"],
        default="median_abs",
        help="Metric shown in the Gaia-range annotation for vertical plots.",
    )
    ap.add_argument(
        "--mwin-source",
        choices=["trap", "cont"],
        default="trap",
        help="Which posterior/truth definition to plot under the label m_win. Default: trap.",
    )
    ap.add_argument("--colors", default=None, help="Optional comma-separated curve colors.")
    ap.add_argument("--no-png", action="store_true", help="Only write the PDF, not a PNG preview.")
    ap.add_argument("--axis-labelsize", type=float, default=None, help="Optional x/y axis label font size.")
    ap.add_argument("--tick-labelsize", type=float, default=None, help="Optional tick-label font size.")
    ap.add_argument("--hide-gaia-metric", action="store_true", help="Do not print Gaia-range median annotations.")
    return ap


def _parse_limits(spec: str | None) -> tuple[float, float] | None:
    if spec is None:
        return None
    parts = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if len(parts) != 2 or not np.isfinite(parts[0]) or not np.isfinite(parts[1]) or parts[0] >= parts[1]:
        raise ValueError(f"Expected limits as lo,hi; got {spec!r}")
    return float(parts[0]), float(parts[1])


def main() -> None:
    args = build_parser().parse_args()
    inputs = _split_csv(args.inputs) or DEFAULT_INPUTS
    labels = _split_csv(args.labels) or DEFAULT_LABELS
    if len(inputs) != len(labels):
        raise SystemExit("--labels must have the same number of entries as --inputs")
    runs = _load_runs(inputs, labels, args.band, args.mwin_source)
    colors = _split_csv(args.colors)
    if colors is not None and len(colors) != len(runs):
        raise SystemExit("--colors must have the same number of entries as --inputs")
    out_pdf = make_figure(
        runs,
        Path(args.out_pdf),
        min_count=int(args.min_count),
        xlim=_parse_limits(args.xlim),
        ylim=_parse_limits(args.ylim),
        realistic_band=_parse_optional_band(args.realistic_band),
        realistic_label=args.realistic_label,
        layout=args.layout,
        include_m0=bool(args.include_m0),
        gaia_metric=args.gaia_metric,
        colors=colors,
        save_png=not bool(args.no_png),
        axis_labelsize=args.axis_labelsize,
        tick_labelsize=args.tick_labelsize,
        show_gaia_metric=not bool(args.hide_gaia_metric),
    )
    print(f"Wrote {out_pdf}")
    if not args.no_png:
        print(f"Wrote {out_pdf.with_suffix('.png')}")


if __name__ == "__main__":
    main()
