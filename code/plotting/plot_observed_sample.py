#!/usr/bin/env python
"""Build the Gaia-only SDSS sample figure for the manuscript.

The Gaia quality-summary NPZ supplies the Gaia-only template pool.
"""

from framework import PROJECT_ROOT, DATA_ROOT

import argparse
import json
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

SCRIPT_DIR = PROJECT_ROOT
SHARED_MPL_RC = SCRIPT_DIR / "matplotlibrc"
if SHARED_MPL_RC.exists():
    matplotlib.rc_file(str(SHARED_MPL_RC))

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams


DEFAULT_QUALITY_NPZ = DATA_ROOT / 'results/observed_sample/gaia_sdss_sample.npz'
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figures"
DEFAULT_OUTPUT_STEM = 'figure_02_observed_sample'
DEFAULT_SPLIT_SLUG = 'figure_02_observed_sample'

rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    }
)


def apply_presentation_style(ax, *, spine_lw=1.6):
    for spine in ax.spines.values():
        spine.set_linewidth(spine_lw)
    ax.tick_params(axis="both", which="both", width=spine_lw, length=6, labelsize=18)
    ax.xaxis.label.set_size(22)
    ax.yaxis.label.set_size(22)


def save_single_page_pdf(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig)


def _load_quality_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        required = [
            "candidate_z",
            "candidate_logl3000",
            "gaia_g_mag_med",
            "gaia_g_baseline_rest",
        ]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{path} is missing required arrays: {missing}")
        arrays = {key: np.asarray(data[key], dtype=np.float64) for key in required}
        if "gaia_g_n_epochs" in data.files:
            arrays["gaia_g_n_epochs"] = np.asarray(data["gaia_g_n_epochs"], dtype=np.float64)
        if "candidate_source_id" in data.files:
            arrays["candidate_source_id"] = np.asarray(data["candidate_source_id"])
    return arrays


def _finite_range(values: np.ndarray, *, pad_fraction: float = 0.02) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("cannot determine plot range from an all-nonfinite array")
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("plot range is not finite")
    if hi <= lo:
        pad = 0.5 if hi == lo else abs(lo) * pad_fraction
    else:
        pad = pad_fraction * (hi - lo)
    return lo - pad, hi + pad


def make_figure(
    *,
    z: np.ndarray,
    logl3000: np.ndarray,
    gmag: np.ndarray,
    baseline_rest_yr: np.ndarray,
    label: str,
    gridsize: int,
) -> plt.Figure:
    _sample_color = "black"
    _slw = 1.6

    zlo, zhi = _finite_range(z)
    llo, lhi = _finite_range(logl3000)
    zbins = np.linspace(zlo, zhi, 80)
    lbins = np.linspace(llo, lhi, 80)

    fig = plt.figure(figsize=(12.5, 7.0), dpi=200)
    gs = gridspec.GridSpec(
        2,
        4,
        figure=fig,
        height_ratios=[1, 4],
        width_ratios=[4, 1, 0.35, 2.2],
        left=0.09,
        right=0.965,
        top=0.93,
        bottom=0.11,
        hspace=0.05,
        wspace=0.15,
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_cb = fig.add_subplot(gs[0, 1])

    gs_rc = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=gs[:, 3],
        hspace=0.30,
        height_ratios=[1, 1],
    )
    ax_gmag = fig.add_subplot(gs_rc[0])
    ax_baseline = fig.add_subplot(gs_rc[1])

    hb = ax_main.hexbin(
        z,
        logl3000,
        gridsize=gridsize,
        bins="log",
        mincnt=1,
        cmap="viridis",
        linewidths=0.0,
        edgecolors="none",
    )
    hb.set_antialiased(False)
    hb.set_rasterized(True)

    ax_main.set_xlim(zlo, zhi)
    ax_main.set_ylim(llo, lhi)
    ax_main.set_xlabel(r"$z$")
    ax_main.set_ylabel(r"$\log_{10}(L_{3000})\,[\mathrm{erg\,s^{-1}}]$")
    ax_main.text(
        0.97,
        0.04,
        f"{label}\nN\\,=\\,{len(z):,}",
        transform=ax_main.transAxes,
        ha="right",
        va="bottom",
        fontsize=14,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.0},
    )
    apply_presentation_style(ax_main, spine_lw=_slw)
    ax_main.xaxis.label.set_size(17)
    ax_main.yaxis.label.set_size(17)

    cbar = fig.colorbar(hb, cax=ax_cb)
    cbar.set_label("Count (log scale)", fontsize=17)
    cbar.outline.set_linewidth(_slw)
    cbar.ax.tick_params(width=_slw, length=4, labelsize=12)

    ax_top.hist(z, bins=zbins, density=True, histtype="step", linewidth=1.8, color=_sample_color)
    apply_presentation_style(ax_top, spine_lw=_slw)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_top.tick_params(axis="y", which="both", labelsize=13)
    ax_top.set_ylabel("Density", fontsize=17)

    ax_right.hist(
        logl3000,
        bins=lbins,
        density=True,
        histtype="step",
        linewidth=1.8,
        color=_sample_color,
        orientation="horizontal",
    )
    apply_presentation_style(ax_right, spine_lw=_slw)
    ax_right.tick_params(
        axis="y",
        which="both",
        left=False,
        labelleft=False,
        right=False,
        labelright=False,
    )
    ax_right.tick_params(axis="x", which="both", labelsize=13)
    ax_right.set_xlabel("Density", fontsize=17)

    gmag_finite = gmag[np.isfinite(gmag)]
    gmag_bins = np.linspace(float(np.nanmin(gmag_finite)), float(np.nanmax(gmag_finite)), 60)
    med_gmag = float(np.nanmedian(gmag_finite))
    ax_gmag.hist(gmag_finite, bins=gmag_bins, density=True, histtype="step", linewidth=1.8, color=_sample_color)
    ax_gmag.axvline(med_gmag, color=_sample_color, lw=1.4, ls="--", alpha=0.8)
    ax_gmag.set_xlabel(r"$G$ [mag]", fontsize=12)
    ax_gmag.set_ylabel("Density", fontsize=14)
    ax_gmag.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                color=_sample_color,
                lw=1.8,
                ls="--",
                label=rf"$\mathrm{{med}}_{{\rm cat}}(G) = {med_gmag:.2f}$",
            ),
        ],
        frameon=False,
        fontsize=12,
        loc="upper left",
        handlelength=1.2,
    )
    apply_presentation_style(ax_gmag, spine_lw=_slw)
    ax_gmag.tick_params(axis="both", which="both", labelsize=13)
    ax_gmag.xaxis.label.set_size(17)
    ax_gmag.yaxis.label.set_size(17)

    baseline_finite = baseline_rest_yr[
        np.isfinite(baseline_rest_yr) & (baseline_rest_yr > 0)
    ]
    baseline_hi = float(np.nanmax(baseline_finite))
    baseline_bins = np.linspace(0.0, baseline_hi * 1.02, 65)
    med_baseline = float(np.nanmedian(baseline_finite))
    ax_baseline.hist(
        baseline_finite,
        bins=baseline_bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        color=_sample_color,
    )
    ax_baseline.axvline(med_baseline, color=_sample_color, lw=1.4, ls="--", alpha=0.8)
    ax_baseline.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                color=_sample_color,
                lw=1.8,
                ls="--",
                label=(
                    r"$\mathrm{med}_{\rm cat}(T_{\rm rf})="
                    + f"{med_baseline:.2f}"
                    + r"\,\mathrm{yr}$"
                ),
            ),
        ],
        frameon=False,
        fontsize=11,
        loc="upper right",
        handlelength=1.2,
        handletextpad=0.4,
        borderaxespad=0.2,
    )
    ax_baseline.set_xlabel(r"$T_{\rm rf}$ [yr]", fontsize=11)
    ax_baseline.set_ylabel("Density", fontsize=14)
    apply_presentation_style(ax_baseline, spine_lw=_slw)
    ax_baseline.tick_params(axis="both", which="both", labelsize=13)
    ax_baseline.xaxis.label.set_size(17)
    ax_baseline.yaxis.label.set_size(17)

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make the corrected Gaia-only SDSS sample manuscript figure."
    )
    parser.add_argument("--quality-npz", type=Path, default=DEFAULT_QUALITY_NPZ)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--split-slug", default=DEFAULT_SPLIT_SLUG)
    parser.add_argument("--label", default=r"Gaia $G$ LCs")
    parser.add_argument("--gridsize", type=int, default=220)
    parser.add_argument("--export-plot-data", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()

    quality_npz = args.quality_npz.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_stem = str(args.output_stem).strip()
    split_slug = str(args.split_slug).strip()
    if not output_stem:
        raise SystemExit("--output-stem must not be empty")
    if not split_slug:
        raise SystemExit("--split-slug must not be empty")

    out_pdf = output_dir / f"{output_stem}.pdf"
    split_dir = output_dir / output_stem
    split_pdf = split_dir / f"plot10_{split_slug}.pdf"
    meta_json = output_dir / f"{output_stem}.json"

    print(f"[progress] quality_npz={quality_npz}", flush=True)
    arrays = _load_quality_arrays(quality_npz)

    z_raw = arrays["candidate_z"]
    logl_raw = arrays["candidate_logl3000"]
    gmag_raw = arrays["gaia_g_mag_med"]
    baseline_rest_yr_raw = arrays["gaia_g_baseline_rest"] / 365.25

    main_mask = np.isfinite(z_raw) & np.isfinite(logl_raw)
    z = z_raw[main_mask]
    logl3000 = logl_raw[main_mask]
    gmag = gmag_raw[main_mask]
    baseline_rest_yr = baseline_rest_yr_raw[main_mask]

    if len(z) == 0:
        raise RuntimeError(f"No finite z/logL3000 rows found in {quality_npz}")

    print(f"[progress] finite plotting rows={len(z):,}", flush=True)
    if args.export_plot_data is not None:
        from plotting.plot_data import save_snapshot
        save_snapshot(args.export_plot_data, 'figure_02_observed_sample', {
            'z': z,
            'logl3000': logl3000,
            'gmag': gmag,
            'baseline_rest_yr': baseline_rest_yr,
            'label': str(args.label),
            'gridsize': int(args.gridsize),
        })
        return 0

    fig = make_figure(
        z=z,
        logl3000=logl3000,
        gmag=gmag,
        baseline_rest_yr=baseline_rest_yr,
        label=str(args.label),
        gridsize=int(args.gridsize),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    save_single_page_pdf(fig, split_pdf)
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

    n_epochs = arrays.get("gaia_g_n_epochs")
    meta = {
        "creator": Path(__file__).name,
        "quality_npz": str(quality_npz),
        "output_pdf": str(out_pdf),
        "split_pdf": str(split_pdf),
        "n_rows_input": int(len(z_raw)),
        "n_rows_plotted": int(len(z)),
        "n_finite_gmag": int(np.sum(np.isfinite(gmag))),
        "n_finite_baseline_rest": int(np.sum(np.isfinite(baseline_rest_yr) & (baseline_rest_yr > 0))),
        "median_gaia_g_mag": float(np.nanmedian(gmag[np.isfinite(gmag)])),
        "median_rest_baseline_yr": float(
            np.nanmedian(baseline_rest_yr[np.isfinite(baseline_rest_yr) & (baseline_rest_yr > 0)])
        ),
        "z_min": float(np.nanmin(z)),
        "z_max": float(np.nanmax(z)),
        "logl3000_min": float(np.nanmin(logl3000)),
        "logl3000_max": float(np.nanmax(logl3000)),
    }
    if n_epochs is not None:
        n_epochs_main = n_epochs[main_mask]
        meta["n_epochs_min"] = int(np.nanmin(n_epochs_main))
        meta["n_epochs_max"] = int(np.nanmax(n_epochs_main))
        meta["n_epochs_lt10"] = int(np.sum(n_epochs_main < 10))

    with open(meta_json, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"[progress] wrote {split_pdf}", flush=True)
    print(f"[progress] wrote {meta_json}", flush=True)
    print(f"[progress] finished in {time.time() - t0:.2f}s", flush=True)
    print(f"WROTE={out_pdf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
