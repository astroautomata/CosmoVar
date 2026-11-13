#!/usr/bin/env python3
"""Rest-frame cadence support and short-lag variability calibration.

Plot epoch-pair support and the calibration of D = sigma_DRW^2 / tau_DRW,
including rest-frame 1--31.6 day support across the redshift--luminosity plane.
"""

from __future__ import annotations

from framework import PROJECT_ROOT, DATA_ROOT

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib

DEFAULT_MATPLOTLIBRC = PROJECT_ROOT / "matplotlibrc"
DEFAULT_TINYTEX_BIN = Path.home() / ".TinyTeX/bin/x86_64-linux"
DEFAULT_MPLCONFIGDIR = Path(os.environ.get("TMPDIR") or "/tmp") / f"agn-variability-cosmology-mpl-{os.getuid()}"


def _preparse_style_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matplotlibrc", type=Path, default=DEFAULT_MATPLOTLIBRC)
    parser.add_argument("--tinytex-bin", type=Path, default=DEFAULT_TINYTEX_BIN)
    parser.add_argument("--mplconfigdir", type=Path, default=DEFAULT_MPLCONFIGDIR)
    parser.add_argument("--no-tex", action="store_true")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


_STYLE_ARGS = _preparse_style_args()
os.environ.setdefault("MPLCONFIGDIR", str(_STYLE_ARGS.mplconfigdir))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
if _STYLE_ARGS.tinytex_bin and _STYLE_ARGS.tinytex_bin.exists():
    tinytex = str(_STYLE_ARGS.tinytex_bin)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if tinytex not in path_parts:
        os.environ["PATH"] = tinytex + os.pathsep + os.environ.get("PATH", "")

matplotlib.use("Agg")
if _STYLE_ARGS.matplotlibrc and _STYLE_ARGS.matplotlibrc.exists():
    matplotlib.rc_file(str(_STYLE_ARGS.matplotlibrc))
if _STYLE_ARGS.no_tex:
    matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm


LN10 = float(np.log(10.0))

DEFAULT_STAGE1_NPZ = DATA_ROOT / "results/variability_calibration/posterior_samples.npz"
DEFAULT_PAIR_NPZ = DATA_ROOT / "results/variability_calibration/simulated_sources.npz"
DEFAULT_PAIR_SUPPORT_NPZ = DATA_ROOT / "results/variability_calibration/lag_pair_support.npz"
DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "figures/figure_05_short_lag_calibration"
SHORT_COLOR = "#4C78A8"
EXTENDED_COLOR = "#F58518"
CALIBRATION_COLOR = "#54A24B"
POSTERIOR_WIDTH_COLOR = "#B279A2"
REFERENCE_COLOR = "0.35"
PRIOR_OVERLAY_COLORS = {
    1: "#009E73",
    2: "#0072B2",
    3: "#D55E00",
    4: "#E69F00",
    5: "#FF00FF",
}


def _tex_int(value):
    return f"{int(value):,}".replace(",", r"{,}")


def _align_by_source_id(values, value_source_id, target_source_id, name):
    values = np.asarray(values)
    value_source_id = np.asarray(value_source_id, dtype=np.int64)
    target_source_id = np.asarray(target_source_id, dtype=np.int64)
    if values.shape[0] != value_source_id.size:
        raise ValueError(f"{name}: values/source IDs have incompatible lengths")
    if np.array_equal(value_source_id, target_source_id):
        return values
    order = np.argsort(value_source_id)
    sorted_id = value_source_id[order]
    loc = np.searchsorted(sorted_id, target_source_id)
    ok = loc < sorted_id.size
    ok[ok] &= sorted_id[loc[ok]] == target_source_id[ok]
    if not np.all(ok):
        missing = target_source_id[~ok][:10]
        raise ValueError(f"{name}: missing {np.count_nonzero(~ok)} IDs; first={missing}")
    return values[order[loc]]


def _load_inputs(
    stage1_npz,
    pair_npz,
    pair_support_npz,
    band,
    *,
    apply_retained_cuts=True,
    match_2x4_d_panel_visible=False,
):
    """Align fit, truth and cadence-support arrays by source ID for Figure 5.
    
    Convert logged DRW draws from ln to log10 and form per-source medians, central
    68-percent half-widths and residuals from injected truth. The retained mask
    optionally selects m_win <= 20, median log10(tau / day) >= 1 and
    0.5 <= z <= 3.5. A separate option matches the visible Figure-4 D-panel cuts."""
    suffix = f"_{band}"
    with np.load(stage1_npz, allow_pickle=False) as post:
        source_id = np.asarray(post["source_id"], dtype=np.int64)
        z = np.asarray(post["redsh"], dtype=np.float64)
        post_log10d = np.asarray(post[f"posterior_ln_D{suffix}"], dtype=np.float64) / LN10
        post_log10tau = np.asarray(post[f"posterior_ln_tau{suffix}"], dtype=np.float64) / LN10
        post_mwin = np.asarray(post[f"posterior_m_win_cont{suffix}"], dtype=np.float64)
        true_log10d = np.asarray(post[f"true_log10_D{suffix}"], dtype=np.float64)
        true_mwin_key = "true_m_win_cont" if "true_m_win_cont" in post.files else "true_m_win"
        true_mwin = np.asarray(post[true_mwin_key], dtype=np.float64)
        npts = np.asarray(post[f"diag_npts{suffix}"], dtype=np.float64)

    with np.errstate(all="ignore"):
        d_q16, d_q50, d_q84 = np.nanpercentile(post_log10d, [16.0, 50.0, 84.0], axis=1)
        mwin_q16, mwin_q50, mwin_q84 = np.nanpercentile(post_mwin, [16.0, 50.0, 84.0], axis=1)
        tau_med = np.nanmedian(post_log10tau, axis=1)
        mwin_med = np.nanmedian(post_mwin, axis=1)
    d_half = 0.5 * (d_q84 - d_q16)
    d_resid = d_q50 - true_log10d
    mwin_half = 0.5 * (mwin_q84 - mwin_q16)
    mwin_resid = mwin_q50 - true_mwin

    with np.load(pair_npz, allow_pickle=False) as pair:
        pair_sid = np.asarray(pair["mock_source_id"], dtype=np.int64)
        logl3000 = _align_by_source_id(
            np.asarray(pair["mock_logl3000"], dtype=np.float64),
            pair_sid,
            source_id,
            "mock_logl3000",
        )
        gaia_g_snr = _align_by_source_id(
            np.asarray(pair["gaia_g_snr"], dtype=np.float64),
            pair_sid,
            source_id,
            "gaia_g_snr",
        )

    with np.load(pair_support_npz, allow_pickle=False) as support:
        support_sid = np.asarray(support["source_id"], dtype=np.int64)
        n_pair_31 = _align_by_source_id(
            np.asarray(support["n_pair_1_31p6"], dtype=np.float64),
            support_sid,
            source_id,
            "n_pair_1_31p6",
        )
        n_pair_100 = _align_by_source_id(
            np.asarray(support["n_pair_1_100"], dtype=np.float64),
            support_sid,
            source_id,
            "n_pair_1_100",
        )
        n_adj_31 = _align_by_source_id(
            np.asarray(support["n_adj_0p5_31p6"], dtype=np.float64),
            support_sid,
            source_id,
            "n_adj_0p5_31p6",
        )
        n_epoch_time = _align_by_source_id(
            np.asarray(support["n_epoch_time"], dtype=np.float64),
            support_sid,
            source_id,
            "n_epoch_time",
        )
        g_mag = _align_by_source_id(
            np.asarray(support["g_mag"], dtype=np.float64),
            support_sid,
            source_id,
            "g_mag",
        )

    final = (
        np.isfinite(d_q50)
        & np.isfinite(d_half)
        & np.isfinite(true_log10d)
        & np.isfinite(mwin_med)
        & np.isfinite(tau_med)
        & np.isfinite(z)
    )
    if apply_retained_cuts:
        final &= (mwin_med <= 20.0) & (tau_med >= np.log10(10.0)) & (z >= 0.5) & (z <= 3.5)
    if match_2x4_d_panel_visible:
        from framework.photometry import real_logl3000_from_mwin

        logl3000_from_mwin50 = real_logl3000_from_mwin(
            z=z,
            mwin_med=mwin_med,
            band="gaia_g",
        )
        final &= (
            np.isfinite(logl3000_from_mwin50)
            & np.isfinite(d_q50)
            & (logl3000_from_mwin50 >= 43.0)
            & (d_q50 >= -7.0)
            & (d_q50 <= -2.0)
        )

    return {
        "source_id": source_id,
        "z": z,
        "logl3000": logl3000,
        "gaia_g_snr": gaia_g_snr,
        "g_mag": g_mag,
        "npts": np.where(np.isfinite(n_epoch_time) & (n_epoch_time > 0), n_epoch_time, npts),
        "n_pair_31": n_pair_31,
        "n_pair_100": n_pair_100,
        "n_adj_31": n_adj_31,
        "d_q16": d_q16,
        "d_q50": d_q50,
        "d_q84": d_q84,
        "d_half": d_half,
        "d_resid": d_resid,
        "true_log10d": true_log10d,
        "mwin_q16": mwin_q16,
        "mwin_q50": mwin_q50,
        "mwin_q84": mwin_q84,
        "mwin_half": mwin_half,
        "mwin_resid": mwin_resid,
        "true_mwin": true_mwin,
        "final": final,
        "match_2x4_d_panel_visible": bool(match_2x4_d_panel_visible),
    }


def _survival_curve(values):
    x = np.sort(np.asarray(values, dtype=np.float64))
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x, x
    y = 1.0 - np.arange(x.size, dtype=np.float64) / float(x.size)
    return x, y


def _quantile_edges(values, n_bins):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    edges = np.nanpercentile(x, np.linspace(0.0, 100.0, int(n_bins) + 1))
    edges = np.unique(edges[np.isfinite(edges)])
    if edges.size < 3:
        return None
    edges[0] -= 1.0e-9
    edges[-1] += 1.0e-9
    return edges


def _calibration_by_axis(
    data,
    x_values,
    n_bins=8,
    *,
    quantity="d",
    log_x=False,
    log_x_offset=0.0,
):
    """Compare recovery scatter with reported posterior widths in quantile bins.
    
    Use finite retained sources with positive posterior half-widths, and require
    at least 100 sources per bin. The empirical width is half the 84th-minus-16th
    percentile span of median-minus-truth residuals. Divide it by the median
    per-source posterior half-width to obtain the reported calibration ratio."""
    mask = data["final"].copy()
    x = np.asarray(x_values, dtype=np.float64)
    if quantity == "d":
        resid = np.asarray(data["d_resid"], dtype=np.float64)
        half = np.asarray(data["d_half"], dtype=np.float64)
        q16 = np.asarray(data["d_q16"], dtype=np.float64)
        q84 = np.asarray(data["d_q84"], dtype=np.float64)
        truth = np.asarray(data["true_log10d"], dtype=np.float64)
    elif quantity == "mwin":
        resid = np.asarray(data["mwin_resid"], dtype=np.float64)
        half = np.asarray(data["mwin_half"], dtype=np.float64)
        q16 = np.asarray(data["mwin_q16"], dtype=np.float64)
        q84 = np.asarray(data["mwin_q84"], dtype=np.float64)
        truth = np.asarray(data["true_mwin"], dtype=np.float64)
    else:
        raise ValueError(f"unknown calibration quantity: {quantity}")
    mask &= np.isfinite(x) & np.isfinite(resid) & np.isfinite(half) & (half > 0.0)
    mask &= np.isfinite(q16) & np.isfinite(q84) & np.isfinite(truth)
    if log_x:
        mask &= (x + float(log_x_offset)) > 0.0

    x_bin = np.log10(x + float(log_x_offset)) if log_x else x
    edges = _quantile_edges(x_bin[mask], n_bins=n_bins)
    rows = []
    if edges is None:
        return rows
    for i in range(edges.size - 1):
        in_bin = (x_bin >= edges[i]) & (x_bin < edges[i + 1])
        if i == edges.size - 2:
            in_bin = (x_bin >= edges[i]) & (x_bin <= edges[i + 1])
        m = mask & in_bin
        if np.count_nonzero(m) < 100:
            continue
        r = resid[m]
        h = half[m]
        r16, r50, r84 = np.nanpercentile(r, [16.0, 50.0, 84.0])
        empirical_half = 0.5 * (r84 - r16)
        post_half = np.nanmedian(h)
        rows.append(
            {
                "x": float(np.nanmedian(x[m])),
                "x16": float(np.nanpercentile(x[m], 16.0)),
                "x84": float(np.nanpercentile(x[m], 84.0)),
                "r16": float(r16),
                "r50": float(r50),
                "r84": float(r84),
                "post_half": float(post_half),
                "ratio": float(empirical_half / post_half),
                "n": int(np.count_nonzero(m)),
            }
        )
    return rows


def _calibration_by_pair_support(data, n_bins=8, *, quantity="d"):
    return _calibration_by_axis(
        data,
        np.asarray(data["n_pair_31"], dtype=np.float64),
        n_bins=n_bins,
        quantity=quantity,
        log_x=True,
        log_x_offset=1.0,
    )


def _calibration_summary(data, *, quantity):
    if quantity == "d":
        resid = np.asarray(data["d_resid"], dtype=np.float64)
        half = np.asarray(data["d_half"], dtype=np.float64)
        q16 = np.asarray(data["d_q16"], dtype=np.float64)
        q84 = np.asarray(data["d_q84"], dtype=np.float64)
        truth = np.asarray(data["true_log10d"], dtype=np.float64)
    elif quantity == "mwin":
        resid = np.asarray(data["mwin_resid"], dtype=np.float64)
        half = np.asarray(data["mwin_half"], dtype=np.float64)
        q16 = np.asarray(data["mwin_q16"], dtype=np.float64)
        q84 = np.asarray(data["mwin_q84"], dtype=np.float64)
        truth = np.asarray(data["true_mwin"], dtype=np.float64)
    else:
        raise ValueError(f"unknown calibration quantity: {quantity}")
    mask = (
        data["final"]
        & np.isfinite(resid)
        & np.isfinite(half)
        & (half > 0.0)
        & np.isfinite(q16)
        & np.isfinite(q84)
        & np.isfinite(truth)
    )
    empirical_half = 0.5 * (np.nanpercentile(resid[mask], 84.0) - np.nanpercentile(resid[mask], 16.0))
    post_half = float(np.nanmedian(half[mask]))
    coverage = float(np.nanmean((truth[mask] >= q16[mask]) & (truth[mask] <= q84[mask])))
    return {
        "empirical_half": float(empirical_half),
        "post_half": post_half,
        "ratio": float(empirical_half / post_half),
        "coverage": coverage,
    }


def _parse_float_range(text):
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be two comma-separated values, e.g. -0.03,0.03")
    lo, hi = float(parts[0]), float(parts[1])
    if hi <= lo:
        raise argparse.ArgumentTypeError("range upper bound must be larger than lower bound")
    return (lo, hi)


def _parse_overlay_run(text):
    if "=" not in str(text):
        raise argparse.ArgumentTypeError("overlay run must be LABEL=PATH")
    label, path = str(text).split("=", 1)
    label = label.strip()
    path = Path(path.strip())
    if not label:
        raise argparse.ArgumentTypeError("overlay run label cannot be empty")
    if not str(path):
        raise argparse.ArgumentTypeError("overlay run path cannot be empty")
    return label, path


def _prior_number_from_label(label):
    match = re.search(r"prior\s*[-_ ]*(\d+)", str(label), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _overlay_color_for_label(label, index):
    prior_number = _prior_number_from_label(label)
    if prior_number in PRIOR_OVERLAY_COLORS:
        return PRIOR_OVERLAY_COLORS[prior_number]
    fallback = [SHORT_COLOR, EXTENDED_COLOR, CALIBRATION_COLOR, POSTERIOR_WIDTH_COLOR]
    return fallback[int(index) % len(fallback)]


def _heatmap_median_pair_support(data, z_edges, logl_edges, min_count=100):
    z = data["z"]
    logl = data["logl3000"]
    n_pair = data["n_pair_31"]
    mask = data["final"] & np.isfinite(z) & np.isfinite(logl) & np.isfinite(n_pair)
    med = np.full((z_edges.size - 1, logl_edges.size - 1), np.nan)
    counts = np.zeros_like(med, dtype=int)
    frac_ge10 = np.full_like(med, np.nan, dtype=np.float64)
    for iz in range(z_edges.size - 1):
        mz = (z >= z_edges[iz]) & (z < z_edges[iz + 1])
        if iz == z_edges.size - 2:
            mz = (z >= z_edges[iz]) & (z <= z_edges[iz + 1])
        for il in range(logl_edges.size - 1):
            ml = (logl >= logl_edges[il]) & (logl < logl_edges[il + 1])
            if il == logl_edges.size - 2:
                ml = (logl >= logl_edges[il]) & (logl <= logl_edges[il + 1])
            m = mask & mz & ml
            n = int(np.count_nonzero(m))
            counts[iz, il] = n
            if n < min_count:
                continue
            med[iz, il] = float(np.nanmedian(n_pair[m]))
            frac_ge10[iz, il] = float(np.nanmean(n_pair[m] >= 10.0))
    return med, counts, frac_ge10


def _binned_count_summary(x, y, mask, n_bins=9, min_count=100):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(x) & np.isfinite(y)
    edges = _quantile_edges(x[mask], n_bins=n_bins)
    rows = []
    if edges is None:
        return rows
    for i in range(edges.size - 1):
        in_bin = (x >= edges[i]) & (x < edges[i + 1])
        if i == edges.size - 2:
            in_bin = (x >= edges[i]) & (x <= edges[i + 1])
        m = mask & in_bin
        if np.count_nonzero(m) < min_count:
            continue
        q16, q50, q84 = np.nanpercentile(y[m], [16.0, 50.0, 84.0])
        rows.append(
            {
                "x": float(np.nanmedian(x[m])),
                "q16": float(q16),
                "q50": float(q50),
                "q84": float(q84),
                "n": int(np.count_nonzero(m)),
            }
        )
    return rows


def make_main_figure(data, output_prefix, *, save_png=False):
    final = data["final"]
    n31 = data["n_pair_31"][final]
    n100 = data["n_pair_100"][final]
    resid = data["d_resid"][final]
    half = data["d_half"][final]
    truth = data["true_log10d"][final]
    q16 = data["d_q16"][final]
    q84 = data["d_q84"][final]

    med31 = float(np.nanmedian(n31))
    p16_31, p84_31 = np.nanpercentile(n31, [16.0, 84.0])
    frac_lt10 = float(np.nanmean(n31 < 10.0))
    coverage = float(np.nanmean((truth >= q16) & (truth <= q84)))
    empirical_half = 0.5 * (np.nanpercentile(resid, 84.0) - np.nanpercentile(resid, 16.0))
    post_half = float(np.nanmedian(half))
    ratio = float(empirical_half / post_half)

    fig = plt.figure(figsize=(7.25, 5.85))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.18],
        left=0.100,
        right=0.985,
        bottom=0.09,
        top=0.940,
        hspace=0.46,
        wspace=0.30,
    )

    ax = fig.add_subplot(gs[0, 0])
    for vals, color, label in [
        (n31, SHORT_COLOR, r"$1<\Delta t_{\rm RF}<31.6\,{\rm d}$"),
        (n100, EXTENDED_COLOR, r"$1<\Delta t_{\rm RF}<100\,{\rm d}$"),
    ]:
        x, y = _survival_curve(vals)
        ax.step(x, y, where="post", color=color, lw=1.5, label=label)
    ax.axvline(10.0, color=REFERENCE_COLOR, lw=0.9, ls="--")
    ax.axvline(med31, color=SHORT_COLOR, lw=1.0, ls=":")
    ax.text(
        0.04,
        0.08,
        rf"$N={_tex_int(np.count_nonzero(final))}$"
        "\n"
        rf"median $N_{{31.6}}={med31:.0f}$"
        rf" ($16$--$84\%$: ${p16_31:.0f}$--${p84_31:.0f}$)"
        "\n"
        rf"$P(N_{{31.6}}<10)={100.0 * frac_lt10:.2f}" + r"\%$",
        transform=ax.transAxes,
        fontsize=8.0,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.92, pad=2.5),
    )
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel(r"pair-count threshold")
    ax.set_ylabel("fraction of retained sources")
    ax.set_title(r"\textbf{(a)} pair support", loc="left", pad=8)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", handlelength=2.2)

    ax = fig.add_subplot(gs[1, :])
    z_edges = np.arange(0.5, 3.5 + 0.5, 0.5)
    logl_fin = data["logl3000"][final]
    l_lo = np.floor(np.nanpercentile(logl_fin, 1.0) / 0.25) * 0.25
    l_hi = np.ceil(np.nanpercentile(logl_fin, 99.0) / 0.25) * 0.25
    logl_edges = np.arange(l_lo, l_hi + 0.001, 0.25)
    med, counts, frac_ge10 = _heatmap_median_pair_support(data, z_edges, logl_edges, min_count=150)
    plot_med = np.ma.masked_invalid(med.T)
    finite_med = med[np.isfinite(med)]
    norm = LogNorm(vmin=50.0, vmax=400.0)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="0.90")
    mesh = ax.pcolormesh(z_edges, logl_edges, plot_med, cmap=cmap, norm=norm, shading="auto")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.010, fraction=0.028)
    cbar.set_label(r"median $N_{31.6}$")
    cbar_ticks = [50, 100, 200, 400]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([str(tick) for tick in cbar_ticks])
    bad = np.isfinite(frac_ge10) & (frac_ge10 < 0.99)
    if np.any(bad):
        zc = 0.5 * (z_edges[:-1] + z_edges[1:])
        lc = 0.5 * (logl_edges[:-1] + logl_edges[1:])
        for iz, il in zip(*np.where(bad)):
            ax.plot(zc[iz], lc[il], "x", color="white", ms=4.2, mew=1.0)
    ax.set_xlabel("redshift")
    ax.set_ylabel(r"$\log_{10}(L_{3000}/{\rm erg\,s^{-1}})$")
    ax.set_title(r"\textbf{(c)} support across the Stage--2 plane", loc="left", pad=8)
    good_frac = frac_ge10[np.isfinite(frac_ge10)]
    if good_frac.size:
        ax.text(
            0.02,
            0.04,
            (
                rf"worst populated bin: {100.0 * np.nanmin(good_frac):.2f}"
                + r"\%"
                "\n"
                rf"with $N_{{31.6}}\geq10$"
            ),
            transform=ax.transAxes,
            fontsize=8.0,
            color="white",
            ha="left",
            va="bottom",
            bbox=dict(facecolor="black", edgecolor="none", alpha=0.42, pad=2.5),
        )

    ax = fig.add_subplot(gs[0, 1])
    rows = _calibration_by_pair_support(data, n_bins=8)
    x = np.asarray([row["x"] for row in rows])
    r16 = np.asarray([row["r16"] for row in rows])
    r50 = np.asarray([row["r50"] for row in rows])
    r84 = np.asarray([row["r84"] for row in rows])
    ph = np.asarray([row["post_half"] for row in rows])
    if x.size:
        ax.errorbar(
            x,
            r50,
            yerr=np.vstack([r50 - r16, r84 - r50]),
            fmt="o-",
            color=SHORT_COLOR,
            ecolor=SHORT_COLOR,
            lw=1.3,
            ms=3.8,
            capsize=2.0,
            label=r"empirical residual 16--84\%",
        )
        ax.plot(x, ph, "--", color=EXTENDED_COLOR, lw=1.3, label=r"median posterior half-width")
        ax.plot(x, -ph, "--", color=EXTENDED_COLOR, lw=1.3)
    ax.axhline(0.0, color=REFERENCE_COLOR, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_{\rm pair}(1<\Delta t_{\rm RF}<31.6\,{\rm d})$")
    ax.set_ylabel(r"$\eta_{\rm med}-\eta_{\rm true}$ [dex]")
    ax.set_ylim(-0.27, 0.27)
    ax.set_title(r"\textbf{(b)} $\log_{10}D$ calibration", loc="left", pad=8)
    ax.legend(frameon=False, fontsize=7.3, loc="upper right", handlelength=2.0)
    ax.text(
        0.04,
        0.07,
        rf"$R_{{\rm cal}}={ratio:.3f}$"
        "\n"
        rf"coverage $={100.0 * coverage:.1f}" + r"\%$",
        transform=ax.transAxes,
        fontsize=8.0,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.92, pad=2.5),
    )

    pdf = output_prefix.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    png = None
    if save_png:
        png = output_prefix.with_suffix(".png")
        fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def _plot_pair_support_survival(ax, data, *, title, font_scale=1.0):
    final = data["final"]
    n31 = data["n_pair_31"][final]
    n100 = data["n_pair_100"][final]
    med31 = float(np.nanmedian(n31))
    p16_31, p84_31 = np.nanpercentile(n31, [16.0, 84.0])
    frac_lt10 = float(np.nanmean(n31 < 10.0))

    for vals, color, label in [
        (n31, SHORT_COLOR, r"$1<\Delta t_{\rm RF}<31.6\,{\rm d}$"),
        (n100, EXTENDED_COLOR, r"$1<\Delta t_{\rm RF}<100\,{\rm d}$"),
    ]:
        x, y = _survival_curve(vals)
        ax.step(x, y, where="post", color=color, lw=1.5, label=label)
    ax.axvline(10.0, color=REFERENCE_COLOR, lw=0.9, ls="--")
    ax.axvline(med31, color=SHORT_COLOR, lw=1.0, ls=":")
    ax.text(
        0.04,
        0.08,
        rf"$N={_tex_int(np.count_nonzero(final))}$"
        "\n"
        rf"median $N_{{31.6}}={med31:.0f}$"
        rf" ($16$--$84\%$: ${p16_31:.0f}$--${p84_31:.0f}$)"
        "\n"
        rf"$P(N_{{31.6}}<10)={100.0 * frac_lt10:.2f}" + r"\%$",
        transform=ax.transAxes,
        fontsize=8.0 * float(font_scale),
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.92, pad=2.5),
    )
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel(r"pair-count threshold")
    ax.set_ylabel("fraction of retained sources")
    ax.set_title(title, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=7.3 * float(font_scale), loc="upper right", handlelength=2.0)


def _plot_stage2_plane_support(
    ax,
    fig,
    data,
    *,
    title,
    cmap_name="viridis",
    text_side="left",
    sources_wording=False,
    font_scale=1.0,
):
    final = data["final"]
    z_edges = np.arange(0.5, 3.5 + 0.5, 0.5)
    logl_fin = data["logl3000"][final]
    l_lo = np.floor(np.nanpercentile(logl_fin, 1.0) / 0.25) * 0.25
    l_hi = np.ceil(np.nanpercentile(logl_fin, 99.0) / 0.25) * 0.25
    logl_edges = np.arange(l_lo, l_hi + 0.001, 0.25)
    med, counts, frac_ge10 = _heatmap_median_pair_support(data, z_edges, logl_edges, min_count=150)
    plot_med = np.ma.masked_invalid(med.T)
    norm = LogNorm(vmin=50.0, vmax=400.0)
    cmap = plt.get_cmap(str(cmap_name)).copy()
    cmap.set_bad(color="0.90")
    mesh = ax.pcolormesh(z_edges, logl_edges, plot_med, cmap=cmap, norm=norm, shading="auto")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.012, fraction=0.046)
    cbar.set_label(r"median $N_{31.6}$")
    cbar.ax.yaxis.label.set_size(9.6 * float(font_scale))
    cbar.ax.tick_params(labelsize=8.6 * float(font_scale))
    cbar_ticks = [50, 100, 200, 400]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([str(tick) for tick in cbar_ticks])
    bad = np.isfinite(frac_ge10) & (frac_ge10 < 0.99)
    if np.any(bad):
        zc = 0.5 * (z_edges[:-1] + z_edges[1:])
        lc = 0.5 * (logl_edges[:-1] + logl_edges[1:])
        for iz, il in zip(*np.where(bad)):
            ax.plot(zc[iz], lc[il], "x", color="white", ms=4.2, mew=1.0)
    ax.set_xlabel("redshift")
    ax.set_ylabel(r"$\log_{10}(L_{3000}/{\rm erg\,s^{-1}})$")
    ax.set_title(title, loc="left", pad=8)
    good_frac = frac_ge10[np.isfinite(frac_ge10)]
    if good_frac.size:
        text_right = str(text_side).lower() == "right"
        second_line = (
            rf"sources with $N_{{31.6}}\geq10$"
            if sources_wording
            else rf"with $N_{{31.6}}\geq10$"
        )
        ax.text(
            0.96 if text_right else 0.04,
            0.05,
            (
                rf"worst populated bin: {100.0 * np.nanmin(good_frac):.2f}"
                + r"\%"
                "\n"
                + second_line
            ),
            transform=ax.transAxes,
            fontsize=7.5 * float(font_scale),
            color="white",
            ha="right" if text_right else "left",
            va="bottom",
            bbox=dict(facecolor="black", edgecolor="none", alpha=0.42, pad=2.5),
        )


def _plot_calibration_panel(
    ax,
    data,
    *,
    quantity,
    title,
    ylabel,
    ylim,
    font_scale=1.0,
    x_values=None,
    xlabel=None,
    xscale="log",
    log_x_bins=True,
    log_x_offset=1.0,
):
    if x_values is None:
        rows = _calibration_by_pair_support(data, n_bins=8, quantity=quantity)
        xlabel = xlabel or r"$N_{\rm pair}(1<\Delta t_{\rm RF}<31.6\,{\rm d})$"
    else:
        rows = _calibration_by_axis(
            data,
            x_values,
            n_bins=8,
            quantity=quantity,
            log_x=log_x_bins,
            log_x_offset=log_x_offset,
        )
    summary = _calibration_summary(data, quantity=quantity)
    x = np.asarray([row["x"] for row in rows])
    r16 = np.asarray([row["r16"] for row in rows])
    r50 = np.asarray([row["r50"] for row in rows])
    r84 = np.asarray([row["r84"] for row in rows])
    ph = np.asarray([row["post_half"] for row in rows])
    if x.size:
        ax.errorbar(
            x,
            r50,
            yerr=np.vstack([r50 - r16, r84 - r50]),
            fmt="o-",
            color=CALIBRATION_COLOR,
            ecolor=CALIBRATION_COLOR,
            lw=1.3,
            ms=3.6,
            capsize=2.0,
            label=r"empirical residual 16--84\%",
        )
        ax.plot(x, ph, "--", color=POSTERIOR_WIDTH_COLOR, lw=1.3, label=r"median posterior half-width")
        ax.plot(x, -ph, "--", color=POSTERIOR_WIDTH_COLOR, lw=1.3)
        check_vals = np.concatenate([r16, r50, r84, ph, -ph])
        check_vals = check_vals[np.isfinite(check_vals)]
        if check_vals.size and (np.nanmin(check_vals) < ylim[0] or np.nanmax(check_vals) > ylim[1]):
            print(
                f"Warning: {quantity} calibration values extend outside y-limits "
                f"[{ylim[0]:.4g}, {ylim[1]:.4g}] "
                f"(data range [{np.nanmin(check_vals):.4g}, {np.nanmax(check_vals):.4g}])",
                flush=True,
            )
    ax.axhline(0.0, color=REFERENCE_COLOR, lw=0.8)
    if xscale:
        ax.set_xscale(xscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=6.9 * float(font_scale), loc="upper right", handlelength=1.8)
    ax.text(
        0.04,
        0.07,
        rf"$R_{{\rm cal}}={summary['ratio']:.3f}$"
        "\n"
        rf"coverage $={100.0 * summary['coverage']:.1f}" + r"\%$",
        transform=ax.transAxes,
        fontsize=8.0 * float(font_scale),
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.92, pad=2.5),
    )


def make_paper_figure(
    data,
    output_prefix,
    *,
    save_png=False,
    paper_output_pdf=None,
    stage2_plane_cmap="viridis",
    stage2_plane_text_side="left",
    stage2_plane_sources_wording=False,
    eta_calib_ylim=(-0.27, 0.27),
    mwin_calib_ylim=(-0.04, 0.04),
    font_scale=1.0,
    figure_label=None,
):
    fig = plt.figure(figsize=(7.25, 6.35))
    if figure_label:
        fig.text(
            0.5,
            0.992,
            str(figure_label),
            ha="center",
            va="top",
            fontsize=11.0 * float(font_scale),
            fontweight="bold",
        )
    gs = fig.add_gridspec(
        2,
        2,
        left=0.095,
        right=0.985,
        bottom=0.085,
        top=0.900 if figure_label else 0.940,
        hspace=0.43,
        wspace=0.34,
    )

    ax = fig.add_subplot(gs[0, 0])
    _plot_pair_support_survival(
        ax,
        data,
        title=r"\textbf{(a)} pair support",
        font_scale=font_scale,
    )

    ax = fig.add_subplot(gs[0, 1])
    _plot_stage2_plane_support(
        ax,
        fig,
        data,
        title=r"\textbf{(b)} support across the Stage--2 plane",
        cmap_name=stage2_plane_cmap,
        text_side=stage2_plane_text_side,
        sources_wording=stage2_plane_sources_wording,
        font_scale=font_scale,
    )

    ax = fig.add_subplot(gs[1, 0])
    _plot_calibration_panel(
        ax,
        data,
        quantity="d",
        title=r"\textbf{(c)} $\eta\equiv\log_{10}D$ calibration",
        ylabel=r"$\eta_{50}-\eta_{\rm true}$ [dex]",
        ylim=eta_calib_ylim,
        font_scale=font_scale,
    )

    ax = fig.add_subplot(gs[1, 1])
    _plot_calibration_panel(
        ax,
        data,
        quantity="mwin",
        title=r"\textbf{(d)} $m\equiv m_{\rm win}$ calibration",
        ylabel=r"$m_{50}-m_{\rm true}$ [mag]",
        ylim=mwin_calib_ylim,
        font_scale=font_scale,
    )

    for axis in fig.axes:
        axis.xaxis.label.set_size(10.8 * float(font_scale))
        axis.yaxis.label.set_size(10.8 * float(font_scale))
        axis.title.set_size(10.0 * float(font_scale))
        for title_obj in (axis.title, axis._left_title, axis._right_title):
            title_obj.set_size(10.0 * float(font_scale))
        axis.tick_params(axis="both", labelsize=8.6 * float(font_scale))

    pdf = (
        Path(paper_output_pdf).expanduser().resolve()
        if paper_output_pdf is not None
        else output_prefix.with_name(output_prefix.name + "_paper").with_suffix(".pdf")
    )
    fig.savefig(pdf, bbox_inches="tight")
    png = None
    if save_png:
        png = output_prefix.with_name(output_prefix.name + "_paper").with_suffix(".png")
        fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def make_calibration_figure(
    data,
    output_prefix,
    *,
    save_png=False,
    calibration_output_pdf=None,
    stage2_plane_cmap="viridis",
    stage2_plane_text_side="left",
    stage2_plane_sources_wording=False,
    eta_calib_ylim=(-0.27, 0.27),
    mwin_calib_ylim=(-0.04, 0.04),
    font_scale=1.0,
    figure_label=None,
):
    fig = plt.figure(figsize=(7.25, 14.75))
    if figure_label:
        fig.text(
            0.5,
            0.992,
            str(figure_label),
            ha="center",
            va="top",
            fontsize=11.0 * float(font_scale),
            fontweight="bold",
        )
    gs = fig.add_gridspec(
        6,
        2,
        height_ratios=[1.05, 1.0, 1.0, 1.0, 1.0, 1.0],
        left=0.095,
        right=0.985,
        bottom=0.045,
        top=0.948 if figure_label else 0.975,
        hspace=0.67,
        wspace=0.34,
    )

    ax = fig.add_subplot(gs[0, 0])
    _plot_pair_support_survival(
        ax,
        data,
        title=r"\textbf{(a)} pair support",
        font_scale=font_scale,
    )

    ax = fig.add_subplot(gs[0, 1])
    _plot_stage2_plane_support(
        ax,
        fig,
        data,
        title=r"\textbf{(b)} support across the Stage--2 plane",
        cmap_name=stage2_plane_cmap,
        text_side=stage2_plane_text_side,
        sources_wording=stage2_plane_sources_wording,
        font_scale=font_scale,
    )

    axis_specs = [
        (
            r"$N_{31.6}$",
            None,
            r"$N_{\rm pair}(1<\Delta t_{\rm RF}<31.6\,{\rm d})$",
            "log",
            True,
            1.0,
        ),
        ("redshift", data["z"], "redshift", None, False, 0.0),
        (r"Gaia $G$ S/N", data["gaia_g_snr"], r"Gaia $G$ signal-to-noise", "log", True, 0.0),
        (r"true $m$", data["true_mwin"], r"true $m\equiv m_{\rm win}$ [mag]", None, False, 0.0),
        (
            r"true $\eta$",
            data["true_log10d"],
            r"true $\eta\equiv\log_{10}D$",
            None,
            False,
            0.0,
        ),
    ]
    letters = list("cdefghijkl")
    for row, (axis_name, x_values, xlabel, xscale, log_x_bins, log_x_offset) in enumerate(axis_specs, start=1):
        left_letter = letters[2 * (row - 1)]
        right_letter = letters[2 * (row - 1) + 1]

        ax = fig.add_subplot(gs[row, 0])
        _plot_calibration_panel(
            ax,
            data,
            quantity="d",
            title=rf"\textbf{{({left_letter})}} $\eta$ calibration vs {axis_name}",
            ylabel=r"$\eta_{50}-\eta_{\rm true}$ [dex]",
            ylim=eta_calib_ylim,
            font_scale=font_scale,
            x_values=x_values,
            xlabel=xlabel,
            xscale=xscale,
            log_x_bins=log_x_bins,
            log_x_offset=log_x_offset,
        )

        ax = fig.add_subplot(gs[row, 1])
        _plot_calibration_panel(
            ax,
            data,
            quantity="mwin",
            title=rf"\textbf{{({right_letter})}} $m$ calibration vs {axis_name}",
            ylabel=r"$m_{50}-m_{\rm true}$ [mag]",
            ylim=mwin_calib_ylim,
            font_scale=font_scale,
            x_values=x_values,
            xlabel=xlabel,
            xscale=xscale,
            log_x_bins=log_x_bins,
            log_x_offset=log_x_offset,
        )

    for axis in fig.axes:
        axis.xaxis.label.set_size(10.4 * float(font_scale))
        axis.yaxis.label.set_size(10.4 * float(font_scale))
        axis.title.set_size(9.6 * float(font_scale))
        for title_obj in (axis.title, axis._left_title, axis._right_title):
            title_obj.set_size(9.6 * float(font_scale))
        axis.tick_params(axis="both", labelsize=8.1 * float(font_scale))
    fig.align_ylabels()

    pdf = (
        Path(calibration_output_pdf).expanduser().resolve()
        if calibration_output_pdf is not None
        else output_prefix.with_name(output_prefix.name + "_calibration").with_suffix(".pdf")
    )
    fig.savefig(pdf, bbox_inches="tight")
    png = None
    if save_png:
        png = pdf.with_suffix(".png")
        fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def _plot_calibration_overlay_panel(
    ax,
    runs,
    *,
    quantity,
    title,
    ylabel,
    ylim,
    x_values_func,
    xlabel,
    xscale,
    log_x_bins,
    log_x_offset,
    font_scale=1.0,
):
    n_runs = max(1, len(runs))
    offset_center = 0.5 * (n_runs - 1)
    for idx, run in enumerate(runs):
        label = run["label"]
        data = run["data"]
        color = run["color"]
        x_values = x_values_func(data)
        if x_values is None:
            rows = _calibration_by_pair_support(data, n_bins=8, quantity=quantity)
        else:
            rows = _calibration_by_axis(
                data,
                x_values,
                n_bins=8,
                quantity=quantity,
                log_x=log_x_bins,
                log_x_offset=log_x_offset,
            )
        if not rows:
            continue
        x = np.asarray([row["x"] for row in rows], dtype=np.float64)
        r16 = np.asarray([row["r16"] for row in rows], dtype=np.float64)
        r50 = np.asarray([row["r50"] for row in rows], dtype=np.float64)
        r84 = np.asarray([row["r84"] for row in rows], dtype=np.float64)
        ph = np.asarray([row["post_half"] for row in rows], dtype=np.float64)
        x_plot = x.copy()
        offset_index = float(idx) - offset_center
        if n_runs > 1 and np.any(np.isfinite(x_plot)):
            if xscale == "log":
                x_plot = x_plot * np.power(10.0, 0.018 * offset_index)
            else:
                span = float(np.nanmax(x_plot) - np.nanmin(x_plot))
                if np.isfinite(span) and span > 0.0:
                    x_plot = x_plot + 0.018 * offset_index * span
        ax.errorbar(
            x_plot,
            r50,
            yerr=np.vstack([r50 - r16, r84 - r50]),
            fmt="o-",
            color=color,
            ecolor=color,
            lw=1.25,
            ms=3.1,
            capsize=1.8,
            alpha=0.94,
            label=label,
        )
        ax.plot(x_plot, ph, "--", color=color, lw=1.15, alpha=0.86)
        ax.plot(x_plot, -ph, "--", color=color, lw=1.15, alpha=0.86)
        check_vals = np.concatenate([r16, r50, r84, ph, -ph])
        check_vals = check_vals[np.isfinite(check_vals)]
        if check_vals.size and (np.nanmin(check_vals) < ylim[0] or np.nanmax(check_vals) > ylim[1]):
            print(
                f"Warning: {label} {quantity} overlay values extend outside y-limits "
                f"[{ylim[0]:.4g}, {ylim[1]:.4g}] "
                f"(data range [{np.nanmin(check_vals):.4g}, {np.nanmax(check_vals):.4g}])",
                flush=True,
            )

    ax.axhline(0.0, color=REFERENCE_COLOR, lw=0.75, zorder=0)
    if xscale:
        ax.set_xscale(xscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", pad=8)
    ax.tick_params(axis="both", labelsize=8.1 * float(font_scale))


def make_prior_overlay_calibration_figure(
    runs,
    output_pdf,
    *,
    eta_calib_ylim=(-0.45, 0.45),
    mwin_calib_ylim=(-0.03, 0.03),
    font_scale=1.0,
    title="Stage 1 short-lag calibration prior overlay",
):
    fig = plt.figure(figsize=(7.45, 12.65))
    fig.suptitle(str(title), y=0.992, fontsize=11.0 * float(font_scale), fontweight="bold")

    prior_handles = [
        Line2D([0], [0], color=run["color"], lw=2.2, label=run["label"])
        for run in runs
    ]
    style_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            lw=1.35,
            marker="o",
            ms=3.6,
            linestyle="-",
            label=r"empirical residual 16--84\%",
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            lw=1.35,
            linestyle="--",
            label="median posterior half-width",
        ),
    ]
    fig.legend(
        handles=prior_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.969),
        ncol=max(1, len(prior_handles)),
        frameon=False,
        fontsize=8.6 * float(font_scale),
        handlelength=2.4,
        columnspacing=1.2,
    )
    fig.legend(
        handles=style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.946),
        ncol=2,
        frameon=False,
        fontsize=7.8 * float(font_scale),
        handlelength=2.4,
        columnspacing=1.3,
    )
    fig.text(
        0.5,
        0.923,
        "small horizontal offsets separate nearly identical prior curves",
        ha="center",
        va="center",
        fontsize=6.7 * float(font_scale),
        color="0.35",
    )

    gs = fig.add_gridspec(
        5,
        2,
        left=0.095,
        right=0.985,
        bottom=0.055,
        top=0.915,
        hspace=0.64,
        wspace=0.34,
    )
    axis_specs = [
        (
            r"$N_{31.6}$",
            lambda data: None,
            r"$N_{\rm pair}(1<\Delta t_{\rm RF}<31.6\,{\rm d})$",
            "log",
            True,
            1.0,
        ),
        ("redshift", lambda data: data["z"], "redshift", None, False, 0.0),
        (r"Gaia $G$ S/N", lambda data: data["gaia_g_snr"], r"Gaia $G$ signal-to-noise", "log", True, 0.0),
        (r"true $m$", lambda data: data["true_mwin"], r"true $m\equiv m_{\rm win}$ [mag]", None, False, 0.0),
        (
            r"true $\eta$",
            lambda data: data["true_log10d"],
            r"true $\eta\equiv\log_{10}D$",
            None,
            False,
            0.0,
        ),
    ]
    letters = list("abcdefghij")
    for row, (axis_name, x_values_func, xlabel, xscale, log_x_bins, log_x_offset) in enumerate(axis_specs):
        left_letter = letters[2 * row]
        right_letter = letters[2 * row + 1]

        ax = fig.add_subplot(gs[row, 0])
        _plot_calibration_overlay_panel(
            ax,
            runs,
            quantity="d",
            title=rf"\textbf{{({left_letter})}} $\eta$ calibration vs {axis_name}",
            ylabel=r"$\eta_{50}-\eta_{\rm true}$ [dex]",
            ylim=eta_calib_ylim,
            x_values_func=x_values_func,
            xlabel=xlabel,
            xscale=xscale,
            log_x_bins=log_x_bins,
            log_x_offset=log_x_offset,
            font_scale=font_scale,
        )

        ax = fig.add_subplot(gs[row, 1])
        _plot_calibration_overlay_panel(
            ax,
            runs,
            quantity="mwin",
            title=rf"\textbf{{({right_letter})}} $m$ calibration vs {axis_name}",
            ylabel=r"$m_{50}-m_{\rm true}$ [mag]",
            ylim=mwin_calib_ylim,
            x_values_func=x_values_func,
            xlabel=xlabel,
            xscale=xscale,
            log_x_bins=log_x_bins,
            log_x_offset=log_x_offset,
            font_scale=font_scale,
        )

    for axis in fig.axes:
        axis.xaxis.label.set_size(10.4 * float(font_scale))
        axis.yaxis.label.set_size(10.4 * float(font_scale))
        axis.title.set_size(9.6 * float(font_scale))
        for title_obj in (axis.title, axis._left_title, axis._right_title):
            title_obj.set_size(9.6 * float(font_scale))
        axis.tick_params(axis="both", labelsize=8.1 * float(font_scale))
    fig.align_ylabels()

    pdf = Path(output_pdf).expanduser().resolve()
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf


def make_binned_support_figure(data, output_prefix, *, save_png=False):
    final = data["final"]
    n31 = data["n_pair_31"]
    n100 = data["n_pair_100"]
    specs = [
        (r"\textbf{(a)} redshift", data["z"], "redshift"),
        (r"\textbf{(b)} luminosity", data["logl3000"], r"$\log_{10}(L_{3000}/{\rm erg\,s^{-1}})$"),
        (r"\textbf{(c)} Gaia $G$", data["g_mag"], r"Gaia $G$ magnitude"),
        (r"\textbf{(d)} epoch count", data["npts"], r"$N_{\rm epoch}$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.25), sharey=True)
    axes = axes.ravel()
    for ax, (title, x, xlabel) in zip(axes, specs):
        for counts, color, label in [
            (n31, SHORT_COLOR, r"$1<\Delta t_{\rm RF}<31.6\,{\rm d}$"),
            (n100, EXTENDED_COLOR, r"$1<\Delta t_{\rm RF}<100\,{\rm d}$"),
        ]:
            rows = _binned_count_summary(x, counts, final, n_bins=9, min_count=150)
            if not rows:
                continue
            cx = np.asarray([row["x"] for row in rows])
            q16 = np.asarray([row["q16"] for row in rows])
            q50 = np.asarray([row["q50"] for row in rows])
            q84 = np.asarray([row["q84"] for row in rows])
            ax.errorbar(
                cx,
                q50,
                yerr=np.vstack([q50 - q16, q84 - q50]),
                fmt="o-",
                color=color,
                lw=1.2,
                ms=3.5,
                capsize=2.0,
                label=label,
            )
        ax.axhline(10.0, color=REFERENCE_COLOR, ls=":", lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(1.0, 2000.0)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", pad=2)
    axes[0].set_ylabel("pair count per source")
    axes[2].set_ylabel("pair count per source")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=REFERENCE_COLOR, lw=0.8, ls=":"))
    labels.append(r"$N_{\rm pair}=10$")
    axes[0].legend(handles, labels, frameon=False, fontsize=6.7, loc="lower right")
    fig.tight_layout()

    pdf = output_prefix.with_name(output_prefix.name + "_binned").with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    png = None
    if save_png:
        png = output_prefix.with_name(output_prefix.name + "_binned").with_suffix(".png")
        fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-plot-data", type=Path)
    parser.add_argument("--stage1-npz", type=Path, default=DEFAULT_STAGE1_NPZ)
    parser.add_argument("--pair-npz", type=Path, default=DEFAULT_PAIR_NPZ)
    parser.add_argument("--pair-support-npz", type=Path, default=DEFAULT_PAIR_SUPPORT_NPZ)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--band", default="Gaia_G")
    parser.add_argument("--matplotlibrc", type=Path, default=DEFAULT_MATPLOTLIBRC)
    parser.add_argument("--tinytex-bin", type=Path, default=DEFAULT_TINYTEX_BIN)
    parser.add_argument("--mplconfigdir", type=Path, default=DEFAULT_MPLCONFIGDIR)
    parser.add_argument("--no-tex", action="store_true", help="Disable LaTeX text rendering for quick local previews.")
    parser.add_argument("--save-png", action="store_true", help="Also write PNG previews next to the paper PDFs.")
    parser.add_argument("--paper-only", action="store_true", help="Only write the new four-panel *_paper figure.")
    parser.add_argument(
        "--paper-output-pdf",
        type=Path,
        default=None,
        help="Optional exact output path for the four-panel paper PDF.",
    )
    parser.add_argument(
        "--calibration-output-pdf",
        type=Path,
        default=None,
        help="Optional exact output path for the extended calibration PDF.",
    )
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Only write the extended calibration PDF.",
    )
    parser.add_argument(
        "--prior-overlay-run",
        action="append",
        type=_parse_overlay_run,
        default=None,
        metavar="LABEL=NPZ",
        help="Add one labelled Stage-1 posterior NPZ to the prior-overlay calibration figure.",
    )
    parser.add_argument(
        "--prior-overlay-output-pdf",
        type=Path,
        default=None,
        help="Exact output PDF path for the prior-overlay calibration figure.",
    )
    parser.add_argument(
        "--prior-overlay-title",
        default="Stage 1 short-lag calibration prior overlay",
        help="Title printed above the prior-overlay calibration figure.",
    )
    parser.add_argument(
        "--prior-overlay-only",
        action="store_true",
        help="Only write the prior-overlay calibration figure.",
    )
    parser.add_argument(
        "--stage2-plane-cmap",
        default="viridis",
        help="Colormap for panel (b), the Stage-2-plane pair-support heatmap.",
    )
    parser.add_argument(
        "--stage2-plane-text-side",
        choices=("left", "right"),
        default="left",
        help="Horizontal placement for the panel (b) worst-bin annotation.",
    )
    parser.add_argument(
        "--stage2-plane-sources-wording",
        action="store_true",
        help="Use 'sources with' instead of 'with' in the panel (b) worst-bin annotation.",
    )
    parser.add_argument(
        "--match-2x4-mock-d-panel-visible-sample",
        action="store_true",
        help=(
            "Further restrict the final sample to the same mock sources visible "
            "in the real-vs-mock 2x4 D-L panel: m_win-derived ell>=43 and "
            "-7<=eta_50<=-2."
        ),
    )
    parser.add_argument(
        "--eta-calib-ylim",
        type=_parse_float_range,
        default=(-0.27, 0.27),
        help="Symmetric or explicit y-limits for eta calibration panels, e.g. -0.45,0.45.",
    )
    parser.add_argument(
        "--mwin-calib-ylim",
        type=_parse_float_range,
        default=(-0.04, 0.04),
        help="Symmetric or explicit y-limits for panel (d), e.g. -0.03,0.03.",
    )
    parser.add_argument(
        "--paper-font-scale",
        type=float,
        default=1.0,
        help="Multiplicative font-size scale for the four-panel paper figure.",
    )
    parser.add_argument(
        "--figure-label",
        default=None,
        help="Optional label printed above the figure, e.g. the prior/run name.",
    )
    parser.add_argument(
        "--without-retained-cuts",
        action="store_true",
        help="Do not apply the retained-sample cuts m_win<=20, tau_med>=10 d, and 0.5<=z<=3.5.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    overlay_runs = args.prior_overlay_run or []
    if args.calibration_only and args.paper_only:
        parser.error("--calibration-only and --paper-only are mutually exclusive")
    if args.prior_overlay_only and (args.calibration_only or args.paper_only):
        parser.error("--prior-overlay-only cannot be combined with --calibration-only or --paper-only")
    if args.prior_overlay_only and not overlay_runs:
        parser.error("--prior-overlay-only requires at least one --prior-overlay-run LABEL=NPZ")
    paths_to_check = [args.pair_npz, args.pair_support_npz]
    if args.prior_overlay_only:
        paths_to_check.extend(path for _, path in overlay_runs)
    else:
        paths_to_check.append(args.stage1_npz)
        if overlay_runs and args.prior_overlay_output_pdf is not None:
            paths_to_check.extend(path for _, path in overlay_runs)
    for path in paths_to_check:
        if not path.exists():
            raise FileNotFoundError(str(path))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paper_rc = {
            "font.size": 9.2 * float(args.paper_font_scale),
            "axes.labelsize": 9.6 * float(args.paper_font_scale),
            "axes.titlesize": 10.0 * float(args.paper_font_scale),
            "xtick.labelsize": 8.6 * float(args.paper_font_scale),
            "ytick.labelsize": 8.6 * float(args.paper_font_scale),
            "legend.fontsize": 8.0 * float(args.paper_font_scale),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.grid": False,
        }
    plt.rcParams.update(paper_rc)
    if args.no_tex:
        plt.rcParams["text.usetex"] = False

    if overlay_runs and (args.prior_overlay_only or args.prior_overlay_output_pdf is not None):
        loaded_runs = []
        for idx, (label, stage1_npz) in enumerate(overlay_runs):
            run_data = _load_inputs(
                stage1_npz,
                args.pair_npz,
                args.pair_support_npz,
                args.band,
                apply_retained_cuts=not bool(args.without_retained_cuts),
                match_2x4_d_panel_visible=bool(args.match_2x4_mock_d_panel_visible_sample),
            )
            final = run_data["final"]
            n31 = run_data["n_pair_31"][final]
            n100 = run_data["n_pair_100"][final]
            print(
                f"{label}: Final retained N = {int(np.count_nonzero(final)):,}; "
                f"N31 median={np.nanmedian(n31):.0f}; "
                f"N100 median={np.nanmedian(n100):.0f}"
            )
            loaded_runs.append(
                {
                    "label": label,
                    "npz": stage1_npz,
                    "data": run_data,
                    "color": _overlay_color_for_label(label, idx),
                }
            )
        overlay_pdf = (
            args.prior_overlay_output_pdf
            if args.prior_overlay_output_pdf is not None
            else args.output_prefix.with_name(args.output_prefix.name + "_prior_overlay_calibration").with_suffix(".pdf")
        )
        written_overlay_pdf = make_prior_overlay_calibration_figure(
            loaded_runs,
            overlay_pdf,
            eta_calib_ylim=args.eta_calib_ylim,
            mwin_calib_ylim=args.mwin_calib_ylim,
            font_scale=float(args.paper_font_scale),
            title=args.prior_overlay_title,
        )
        print(f"Wrote {written_overlay_pdf}")
        if args.prior_overlay_only:
            return

    data = _load_inputs(
        args.stage1_npz,
        args.pair_npz,
        args.pair_support_npz,
        args.band,
        apply_retained_cuts=not bool(args.without_retained_cuts),
        match_2x4_d_panel_visible=bool(args.match_2x4_mock_d_panel_visible_sample),
    )
    if args.export_plot_data is not None:
        if not args.paper_only or overlay_runs:
            parser.error("--export-plot-data supports the paper preset only")
        from plotting.plot_data import save_snapshot
        save_snapshot(args.export_plot_data, 'figure_05_short_lag_calibration', {
            "data": data, "rc": paper_rc, "kwargs": {
                "stage2_plane_cmap": str(args.stage2_plane_cmap),
                "stage2_plane_text_side": str(args.stage2_plane_text_side),
                "stage2_plane_sources_wording": bool(args.stage2_plane_sources_wording),
                "eta_calib_ylim": args.eta_calib_ylim,
                "mwin_calib_ylim": args.mwin_calib_ylim,
                "font_scale": float(args.paper_font_scale),
                "figure_label": args.figure_label,
            },
        })
        return
    main_pdf = main_png = binned_pdf = binned_png = None
    paper_pdf = paper_png = None
    calibration_pdf = calibration_png = None
    if not args.calibration_only and not args.paper_only:
        main_pdf, main_png = make_main_figure(data, args.output_prefix, save_png=args.save_png)
        binned_pdf, binned_png = make_binned_support_figure(data, args.output_prefix, save_png=args.save_png)
    if not args.calibration_only:
        paper_pdf, paper_png = make_paper_figure(
            data,
            args.output_prefix,
            save_png=args.save_png,
            paper_output_pdf=args.paper_output_pdf,
            stage2_plane_cmap=str(args.stage2_plane_cmap),
            stage2_plane_text_side=str(args.stage2_plane_text_side),
            stage2_plane_sources_wording=bool(args.stage2_plane_sources_wording),
            eta_calib_ylim=args.eta_calib_ylim,
            mwin_calib_ylim=args.mwin_calib_ylim,
            font_scale=float(args.paper_font_scale),
            figure_label=args.figure_label,
        )
    if args.calibration_only or args.calibration_output_pdf is not None:
        calibration_pdf, calibration_png = make_calibration_figure(
            data,
            args.output_prefix,
            save_png=args.save_png,
            calibration_output_pdf=args.calibration_output_pdf,
            stage2_plane_cmap=str(args.stage2_plane_cmap),
            stage2_plane_text_side=str(args.stage2_plane_text_side),
            stage2_plane_sources_wording=bool(args.stage2_plane_sources_wording),
            eta_calib_ylim=args.eta_calib_ylim,
            mwin_calib_ylim=args.mwin_calib_ylim,
            font_scale=float(args.paper_font_scale),
            figure_label=args.figure_label,
        )

    final = data["final"]
    n31 = data["n_pair_31"][final]
    n100 = data["n_pair_100"][final]
    print(f"Final retained N = {int(np.count_nonzero(final)):,}")
    print(
        "N_pair(1-31.6 d RF): "
        f"median={np.nanmedian(n31):.0f}, "
        f"16-84=[{np.nanpercentile(n31, 16):.0f}, {np.nanpercentile(n31, 84):.0f}], "
        f"frac<10={100.0 * np.nanmean(n31 < 10):.3f}%"
    )
    print(
        "N_pair(1-100 d RF): "
        f"median={np.nanmedian(n100):.0f}, "
        f"16-84=[{np.nanpercentile(n100, 16):.0f}, {np.nanpercentile(n100, 84):.0f}]"
    )
    if main_pdf is not None:
        print(f"Wrote {main_pdf}")
    if binned_pdf is not None:
        print(f"Wrote {binned_pdf}")
    if paper_pdf is not None:
        print(f"Wrote {paper_pdf}")
    if calibration_pdf is not None:
        print(f"Wrote {calibration_pdf}")
    if main_png is not None:
        print(f"Wrote {main_png}")
    if binned_png is not None:
        print(f"Wrote {binned_png}")
    if paper_png is not None:
        print(f"Wrote {paper_png}")
    if calibration_png is not None:
        print(f"Wrote {calibration_png}")


if __name__ == "__main__":
    main()
