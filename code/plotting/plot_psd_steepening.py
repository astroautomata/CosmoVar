"""Render the supplementary short-lag PSD steepening figure for Section 5.1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
TB_GRID_D = (1.0 / 12.0, 0.25, 0.5, 1.0, 2.0, 3.0)
POP_S_RANGE = (3.0, 3.4)

def parse_fixed(name: str):
    if name == "drw":
        return 2.0, None
    s_str, tb_str = name[1:].split("_tb")
    return float(s_str), float(tb_str)


def parse_population(variant):
    """Split a population name into (base, fixed value): popLup_s3 -> ("popLup", 3.0),
    popSLdown_tb0.5 -> ("popSLdown", 0.5).  Without a suffix the original defaults hold:
    s = 3.4 for popLup/popLdown and t_b = 1 d for popSLup/popSLdown."""
    base, _, suffix = variant.partition("_")
    if base in ("popLup", "popLdown"):
        return base, float(suffix[1:]) if suffix.startswith("s") else max(POP_S_RANGE)
    if base in ("popSLup", "popSLdown"):
        return base, float(suffix[2:]) if suffix.startswith("tb") else 1.0
    return base, None


class DeltaKTable:
    """
    Exact high-frequency correction to the DRW kernel for the bent PSD
    P(f) = P_DRW(f) [1 + (2 pi f t_b)^2]^(-p), p = (s-2)/2, in the scaled
    variables x = dt/tau and r = t_b/tau:

        k(dt) = sigma^2 [ exp(-x) - dk(x; r, s) ],
        dk(x) = int_0^inf 4/(1+4 pi^2 u^2) [1 - (1 + 4 pi^2 u^2 r^2)^(-p)] cos(2 pi u x) du.

    The bend factor is a Matern spectral density, so its time-domain response
    is known in closed form:  b(y) = c_nu y^nu K_nu(y), y = |t|/t_b,
    nu = p - 1/2, normalised to unit integral (b = exp(-y)/2 for s = 4,
    K_0(y)/pi for s = 3).  The correction is then the smooth convolution
        dk(x) = int b(y) [exp(-x) - exp(-|x - r y|)] dy,
    evaluated with ordinary adaptive quadrature (no oscillatory weights).
    Tabulated once per object (its own r) on a log grid of x and interpolated.
    For s = 4 the closed form is dk(x) = exp(-x) - [exp(-x) - r exp(-x/r)]/(1-r^2),
    used as the start-up self-check.  For r -> 0, dk -> r h_s(x/r) with
    h_4 = exp(-x/r) (the t_b << tau limit).
    """

    def __init__(self, s: float, r: float, x_max: float, n: int = 2000):
        from scipy.integrate import quad
        from scipy.interpolate import CubicSpline
        from scipy.special import gamma, kv

        self.s, self.r = float(s), float(r)
        nu = (self.s - 2.0) / 2.0 - 0.5
        if nu < 0.0:
            raise ValueError("slope s must be >= 3")
        if abs(nu - 0.5) < 1e-12:
            def b(y):
                return 0.5 * np.exp(-abs(y))
        else:
            c_nu = (1.0 / (2.0 * np.pi)) * (2.0 * np.sqrt(np.pi) / gamma(nu + 0.5)) * 0.5**nu

            def b(y):
                y = abs(y)
                if y < 1e-300:
                    return 0.0  # single point; the nu = 0 log singularity is integrable
                return c_nu * y**nu * kv(nu, y)

        r = self.r
        tol = dict(epsabs=1e-12, epsrel=1e-10, limit=500)

        def dk_of_x(x):
            f = lambda y: b(y) * (np.exp(-x) - np.exp(-abs(x - r * y)))
            yk = x / r  # kink of exp(-|x - r y|)
            if yk < 60.0:
                return (quad(f, -np.inf, 0.0, **tol)[0] + quad(f, 0.0, yk, **tol)[0]
                        + quad(f, yk, np.inf, **tol)[0])
            return quad(f, -np.inf, 0.0, **tol)[0] + quad(f, 0.0, np.inf, **tol)[0]

        xs = np.concatenate([[0.0], np.logspace(-7.0, np.log10(max(x_max, 1e-5)) + 0.05, n)])
        self.x = xs
        self.dk = np.array([dk_of_x(float(x)) for x in xs])
        self._spl = CubicSpline(self.x, self.dk)

    def __call__(self, x):
        x = np.abs(np.asarray(x, dtype=float))
        return np.where(x <= self.x[-1], self._spl(np.minimum(x, self.x[-1])), 0.0)

    @staticmethod
    def closed_form_s4(x, r):
        x = np.asarray(x, dtype=float)
        return np.exp(-x) - (np.exp(-x) - r * np.exp(-x / r)) / (1.0 - r * r)


INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
S_COLORS = {3.0: "#2a78d6", 3.4: "#eb6834", 4.0: "#1baf7a"}  # categorical slots 1-3
# population families: colour + short name; members differ in the fixed value (marker)
POP_FAMILY = {
    "popR": ("#6250d6", "random"),
    "popLup": ("#e34948", r"$t_b\uparrow L$"),
    "popLdown": ("#52514e", r"$t_b\downarrow L$"),
    "popSLdown": ("#e87ba4", r"$s\downarrow L$"),
    "popSLup": ("#008300", r"$s\uparrow L$"),
}
RANDOM_LABEL = "random: " + r"$s\in[3,3.4]$," + "\n" + r"$t_b\in[2\,{\rm h},3\,{\rm d}]$"
# marker = fixed value: s for the t_b-vs-L families, t_b [d] for the s-vs-L families
FIXED_MARKER = {("s", 3.0): "o", ("s", 3.4): "p", ("s", 4.0): "s",
                ("tb", 1.0 / 12.0): "^", ("tb", 0.5): "D", ("tb", 1.0): "P", ("tb", 3.0): "v"}
POPULATIONS = ("popR", "popLup", "popLup_s4", "popLdown",
               "popLdown_s4", "popSLdown", "popSLup")


def pop_key(v):
    """(family, kind, fixed value) of a population variant."""
    fam, val = parse_population(v)
    kind = "s" if fam in ("popLup", "popLdown") else "tb" if fam in ("popSLup", "popSLdown") else None
    return fam, kind, val


def pop_marker(v):
    fam, kind, val = pop_key(v)
    if kind is None:
        return "h"
    return min(FIXED_MARKER.items(), key=lambda kv: (kv[0][0] != kind, abs(kv[0][1] - val)))[1]


def fixed_text(kind, val):
    return f"$s={val:g}$" if kind == "s" else f"$t_b$ = {tb_text(val)}"


def pop_label(v):
    fam, kind, val = pop_key(v)
    return RANDOM_LABEL if kind is None else f"{POP_FAMILY[fam][1]} ({fixed_text(kind, val)})"
BENCH_STYLE = ("#0b0b0b", "*", r"Zw 229-15-like, $s=4.5$")
BENCH_COLORS = ("#8c510a", "#d4a000")  # Zw 229-15-like t_b = 22 h (brown), 1.5 d (gold): same in every panel


def bench_label(v):
    s, tb = parse_fixed(v)
    fb = 1.0 / (2.0 * np.pi * tb)
    return rf"Zw 229-15-like, $s={s:g}$, $u_b={fb:.2f}$ d$^{{-1}}$"
PSD_SLOPE_LS = {3.0: "-", 4.0: (0, (5, 2.5))}  # panel (a): line style = slope s
TB_SHOW_D = (1.0 / 12.0, 0.5, 3.0)  # bend lags drawn in (a), (b); colours keep their slot in TB_BLUES
TB_BLUES = ["#cfe0f5", "#a6c6ec", "#78a8e0", "#4a88d3", "#2a6cb8", "#1d4f8a"]


def sf2_ratio(lags, tau, s, tb):
    x, r = lags / tau, tb / tau
    if abs(s - 4.0) < 1e-12:
        dk, dk0 = DeltaKTable.closed_form_s4(x, r), DeltaKTable.closed_form_s4(0.0, r)
    else:
        tab = DeltaKTable(s, r, x_max=float(x.max()), n=600)
        dk, dk0 = tab(x), tab.dk[0]
    return (2.0 * ((1.0 - dk0) - (np.exp(-x) - dk))) / (2.0 * (1.0 - np.exp(-x)))


def psd(f, sigma, tau, s, tb):
    """Injected one-sided PSD: 4 sigma^2 tau / [1+(2 pi f tau)^2] x [1+(2 pi f t_b)^2]^(-(s-2)/2)."""
    p = 4.0 * sigma**2 * tau / (1.0 + (2 * np.pi * f * tau) ** 2)
    if tb is not None:
        p = p * (1.0 + (2 * np.pi * f * tb) ** 2) ** (-(s - 2.0) / 2.0)
    return p


def tb_text(tb):
    return f"{tb * 24:.0f} h" if tb < 1 else f"{tb:g} d"


def style(ax, top_ticks=True):
    """Paper style (matplotlibrc): full box, ticks inward on all four sides, no grid."""
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(INK)
        sp.set_linewidth(1.0)
    ax.tick_params(which="both", direction="in", top=top_ticks, right=True, bottom=True, left=True, colors=INK)


def panel_label(ax, letter):
    """Place a panel reference inside its reserved upper-left space."""
    ax.text(0.025, 0.97, f"({letter})", transform=ax.transAxes,
            ha="left", va="top", fontsize=16, color=INK, zorder=20)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "zenodo/results/figure_analysis/psd_steepening")
    args = ap.parse_args()
    matplotlib.rc_file(ROOT / "matplotlibrc", use_default_template=True)

    data = args.data_dir
    summary = json.loads((data / "summary.json").read_text())
    d = np.load(data / "representative.npz")
    rows = {row["variant"]: row for row in summary["rows"]}
    run_of = {variant: data / filename for variant, filename in summary["inputs"].items()}
    z_edges = np.asarray(summary["z_edges"])
    zc = 0.5 * (z_edges[1:] + z_edges[:-1])
    fixed = [v for v in rows if v.startswith("s") and parse_fixed(v)[0] in S_COLORS]
    bench = sorted([v for v in rows if v.startswith("s") and parse_fixed(v)[0] not in S_COLORS],
                   key=lambda v: parse_fixed(v)[1])
    pops = POPULATIONS
    families = [f for f in POP_FAMILY if any(pop_key(v)[0] == f for v in pops)]
    ctrl = d["variants"].tolist().index("drw")
    sel = d["ok"].astype(bool) & (d["variant_idx"] == ctrl)
    half_width = float(np.median(0.5 * (d["eta84"][sel] - d["eta16"][sel])))

    plt.rcParams.update({"font.size": 14, "axes.labelsize": 18, "axes.titlesize": 16, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 13, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.top": True, "ytick.right": True, "xtick.direction": "in", "ytick.direction": "in"})
    plt.rcParams["text.latex.preamble"] = plt.rcParams["text.latex.preamble"] + r"\usepackage{amsmath}"
    W, HEAD, BOT = 12.5, 0.65, 0.9
    rows_in = [3.8, 0.85, 3.8]
    H = HEAD + sum(rows_in) + BOT
    fig = plt.figure(figsize=(W, H))
    outer = GridSpec(len(rows_in), 2, figure=fig, height_ratios=rows_in, hspace=0.0, wspace=0.24, left=0.085, right=0.985,
                     top=1 - HEAD / H, bottom=BOT / H)

    # (a) injected power spectra: colour = bend lag t_b, solid s = 3, dashed s = 4 (the ends of the slope grid)
    f = np.logspace(-4.5, 2.0, 600)
    ctl = d["variant_idx"] == d["variants"].tolist().index("drw")
    tau_c = d["tpl_tau"][d["tid"][ctl].astype(int)]
    sig_med = float(np.median(np.sqrt(10.0 ** d["log10D_true"][ctl] * tau_c)))
    tau_med = float(np.median(tau_c))
    p0 = 4.0 * sig_med**2 * tau_med
    tau_sf = tau_med
    f_bend = (1 / (2 * np.pi * max(TB_GRID_D)), 1 / (2 * np.pi * min(TB_GRID_D)))
    ax = fig.add_subplot(outer[0, 0])
    ax.axvspan(*f_bend, color="#f2f1ee", zorder=0)
    ax.plot(f, psd(f, sig_med, tau_med, 2.0, None), color=INK, lw=2.4, zorder=5)
    for s_, ls in PSD_SLOPE_LS.items():
        for j, tb in enumerate(TB_GRID_D):
            if tb in TB_SHOW_D:
                ax.plot(f, psd(f, sig_med, tau_med, s_, tb), color=TB_BLUES[j], lw=1.8, ls=ls)
    for j, v in enumerate(bench):
        b_s, b_tb = parse_fixed(v)
        ax.plot(f, psd(f, sig_med, tau_med, b_s, b_tb), color=BENCH_COLORS[j], lw=2.2, zorder=4)
    ax.axvline(1 / (2 * np.pi * tau_med), color=INK2, lw=0.7, ls=":", zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(f[0], f[-1]); ax.set_ylim(1e-13 * p0, 40 * p0)
    ax.set_xlabel(r"rest-frame frequency $u$ [d$^{-1}$]")
    ax.set_ylabel(r"$P(u)$ [mag$^2$ d]")
    panel_label(ax, "a")
    sec = ax.secondary_xaxis("top", functions=(lambda x: 1 / np.clip(x, 1e-12, None),
                                               lambda x: 1 / np.clip(x, 1e-12, None)))
    sec.set_xlabel(r"timescale $1/u$ [d]", fontsize=14)
    sec.tick_params(which="both", direction="in", labelsize=13, colors=INK)
    style(ax, top_ticks=False)  # the timescale axis carries the top ticks
    ax.text(1 / (2 * np.pi * tau_med) * 1.35, 3e-11 * p0, r"DRW break $1/2\pi\tau$", rotation=90,
            fontsize=12.5, color=INK2, va="bottom")
    h_tb = [Line2D([], [], color=INK, lw=2.4,
                   label="pure DRW\n" + rf"($\sigma={sig_med:.2f}$ mag, $\tau={tau_med:.0f}$ d)")]
    h_tb += [Line2D([], [], color=TB_BLUES[j], lw=2.2, label=f"$t_b$ = {tb_text(tb)}")
             for j, tb in enumerate(TB_GRID_D) if tb in TB_SHOW_D]
    h_s = [Line2D([], [], color=INK2, lw=1.8, ls=ls, label=f"$s={s_:g}$") for s_, ls in PSD_SLOPE_LS.items()]
    h_s += [Line2D([], [], color=BENCH_COLORS[j], lw=2.2,
                   label=rf"Zw 229-15-like, $u_b={1.0 / (2.0 * np.pi * parse_fixed(v)[1]):.2f}$ d$^{{-1}}$")
            for j, v in enumerate(bench)]
    leg = ax.legend(handles=h_tb, loc="upper right", frameon=False, fontsize=13, handlelength=1.6)
    ax.add_artist(leg)
    x_break = (np.log10(1 / (2 * np.pi * tau_med)) - np.log10(f[0])) / (np.log10(f[-1]) - np.log10(f[0]))
    ax.legend(handles=h_s, loc="lower left", bbox_to_anchor=(x_break + 0.045, 0.0), frameon=False, fontsize=13,
              handlelength=1.8)

    ga = outer[0, 1].subgridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.06)

    # (b) injected structure functions and Gaia pair lags
    ax = fig.add_subplot(ga[0])
    lags = np.logspace(-2.3, 3, 400)
    for s_, ls in PSD_SLOPE_LS.items():
        for k, tb in enumerate(TB_GRID_D):
            if tb in TB_SHOW_D:
                ax.plot(lags, sf2_ratio(lags, tau_sf, s_, tb), color=TB_BLUES[k], lw=2, ls=ls)
    ax.axvspan(1 / 12, 3, color="#f2f1ee", zorder=0)
    ax.text(0.5, 0.97, "bends\nsimulated", ha="center", va="top", fontsize=12.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(lags[0], lags[-1])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(r"${\rm SF}^2/{\rm SF}^2_{\rm DRW}$")
    panel_label(ax, "b")
    ax.tick_params(labelbottom=False)
    ax.legend(handles=[Line2D([], [], color=TB_BLUES[k], lw=2.2, label=f"$t_b$ = {tb_text(tb)}")
                       for k, tb in enumerate(TB_GRID_D) if tb in TB_SHOW_D]
              + [Line2D([], [], color=INK2, lw=1.8, ls=ls, label=f"$s={s_:g}$") for s_, ls in PSD_SLOPE_LS.items()],
              fontsize=13, loc="lower right", frameon=False, ncol=2, columnspacing=0.8, handlelength=1.8)
    style(ax)
    axh = fig.add_subplot(ga[1], sharex=ax)
    bins = d["pair_lag_edges_days"]
    h = d["pair_lag_counts"]
    axh.stairs(h / h.sum(), bins, fill=True, color="#b9b8b2")
    axh.axvspan(1 / 12, 3, color="#f2f1ee", zorder=0)
    axh.set_xlabel(r"rest-frame lag $\Delta t$ [d]")
    axh.set_ylabel("Gaia\npairs", fontsize=13)
    axh.set_yticks([])
    style(axh)

    # shared helpers for (b), (c)
    def by_slope():
        out = {}
        for v in fixed:
            s, tb = parse_fixed(v)
            out.setdefault(s, []).append((tb, rows[v]))
        return {s: sorted(vals) for s, vals in out.items()}

    def fam_x(wide=False):  # x position of each population family, right of the t_b grid
        step = 1.7 if wide else 1.45
        return {f: 6.0 * step ** i for i, f in enumerate(families)}

    def pop_x(wide=False):  # members of a family are spread slightly around its position
        fx, out = fam_x(wide), {}
        for f in families:
            mem = [v for v in pops if pop_key(v)[0] == f]
            for k, v in enumerate(mem):
                out[v] = fx[f] * (1.7 if wide else 1.45) ** (0.22 * (k - (len(mem) - 1) / 2))
        return out
    one_value = len(pops) == len(families)  # original set: one member per family, fixed value in the label
    pop_tick = {f: POP_FAMILY[f][1] for f in families}
    pop_tick_wide = {f: (POP_FAMILY[f][1] + ("\n" + fixed_text(*pop_key(v)[1:]) if pop_key(v)[1] else "\n" + r"$s,\ t_b$")
                         if one_value else POP_FAMILY[f][1])
                     for f in families for v in pops if pop_key(v)[0] == f}

    def tb_axis(ax, wide=False):
        xpop = fam_x(wide)
        ax.set_xscale("log")
        ax.set_xticks(list(TB_GRID_D) + list(xpop.values()))
        names = pop_tick_wide if wide else pop_tick
        ax.set_xticklabels([tb_text(t) for t in TB_GRID_D] + [names[f] for f in families], fontsize=13)
        if not wide:
            for lab in ax.get_xticklabels()[len(TB_GRID_D):]:
                lab.set_rotation(90); lab.set_ha("center")
        ax.minorticks_off()
        ax.set_xlabel(r"rest-frame bend lag $t_b$", labelpad=2)
        if pops:
            x_sep = np.sqrt(3.0 * min(xpop.values()))
            ax.axvline(x_sep, color=INK, lw=1.5, ls=(0, (6, 3)), zorder=6)
            x0, x1 = ax.get_xlim()
            frac = lambda x: (np.log(x) - np.log(x0)) / (np.log(x1) - np.log(x0))
            ax.xaxis.label.set_x(frac(np.sqrt(min(TB_GRID_D) * max(TB_GRID_D))))  # label under the t_b grid only
            ax.text(np.sqrt(min(xpop.values()) * max(xpop.values())), 0.03, "population\nscenarios",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=13, color=INK)

    # (c) median shift vs t_b, full width
    ax = fig.add_subplot(outer[2, 0])
    for s, vals in by_slope().items():
        tb = np.array([t for t, _ in vals])
        med = np.array([r["median"] for _, r in vals])
        lo = np.array([r["p16"] for _, r in vals])
        hi = np.array([r["p84"] for _, r in vals])
        c = S_COLORS.get(s, INK2)
        ax.fill_between(tb, lo, hi, color=c, alpha=0.12, lw=0)
        ax.plot(tb, med, "-o", color=c, lw=2.2, ms=6, label=f"$s={s:g}$")
    for k, v in enumerate(bench):
        r, tb = rows[v], parse_fixed(v)[1]
        ax.errorbar(tb, r["median"], yerr=[[r["median"] - r["p16"]], [r["p84"] - r["median"]]], fmt=BENCH_STYLE[1],
                    color=BENCH_COLORS[k], ms=12, lw=1.4, capsize=0, label=bench_label(v), zorder=5)
    for p in pops:
        r = rows[p]
        ax.errorbar(pop_x()[p], r["median"], yerr=[[r["median"] - r["p16"]], [r["p84"] - r["median"]]],
                    fmt=pop_marker(p), color=POP_FAMILY[pop_key(p)[0]][0], ms=8, lw=1.6, capsize=0)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axhspan(-half_width, half_width, color="#f2f1ee", zorder=0)
    ax.text(1 / 12 * 1.05, half_width * 0.82, rf"per-object posterior half-width $\pm{half_width:.2f}$ dex",
            fontsize=12.5, color=INK2, va="top")
    tb_axis(ax)
    ax.xaxis.set_label_coords(ax.xaxis.label.get_position()[0], -0.11)
    ax.set_ylabel(r"$\Delta\eta=\eta_{\rm fit}-\eta_{\rm fit,DRW}$ [dex]")
    panel_label(ax, "c")
    leg = ax.legend(loc="lower left", frameon=False, ncol=1, columnspacing=0.8, handlelength=1.6)
    if not one_value:  # key to the fixed value of each population member, inside the population region
        ax.add_artist(leg)
        kinds = sorted({pop_key(v)[1:] for v in pops if pop_key(v)[1]}, key=lambda kv: (kv[0] != "s", kv[1]))
        ax.legend(handles=[Line2D([], [], ls="", marker=pop_marker(next(v for v in pops if pop_key(v)[1:] == kv)),
                                  color=INK2, ms=8, label=fixed_text(*kv)) for kv in kinds],
                  loc="upper right", frameon=False, fontsize=12, title="fixed value", title_fontsize=12,
                  handletextpad=0.3, borderaxespad=0.3)
    style(ax)

    # (d) paired shift by redshift after adjusting for catalogue luminosity
    def fixed_lum_curve(variant, rng=np.random.default_rng(3)):
        dd = np.load(run_of[variant])
        vl = [str(x) for x in dd["variants"]]
        ok, tid, vi = dd["ok"].astype(bool), dd["tid"].astype(int), dd["variant_idx"]
        ctrl = {int(t): e for t, e, o, v in zip(tid, dd["eta50"], ok, vi) if o and v == vl.index("drw")}
        sel = ok & (vi == vl.index(variant))
        t = tid[sel]
        y = np.array([e - ctrl.get(int(k), np.nan) for k, e in zip(t, dd["eta50"][sel])])
        good = np.isfinite(y)
        t, y = t[good], y[good]
        z = dd["tpl_z"][t]
        M = dd["tpl_ell"][t]
        r = y - rows[variant]["b_ell"] * (M - M.mean())
        zb = np.digitize(z, z_edges[1:-1])
        med, err = [], []
        for k in range(len(zc)):
            rr = r[zb == k]
            med.append(np.median(rr))
            err.append(np.std([np.median(rng.choice(rr, rr.size)) for _ in range(300)]))
        return np.array(med), np.array(err)

    ax = fig.add_subplot(outer[2, 1])
    curves = [(p, POP_FAMILY[pop_key(p)[0]][0], pop_marker(p), pop_label(p)) for p in pops]
    worst = [v for v in fixed if parse_fixed(v) == (4.0, 3.0)]
    if worst:
        curves.append((worst[0], S_COLORS[4.0], "s", r"$s=4$, $t_b=3$ d"))
    if bench:
        s_bench, tb_bench = parse_fixed(bench[-1])
        fb_bench = 1.0 / (2.0 * np.pi * tb_bench)
        curves.append((bench[-1], BENCH_COLORS[len(bench) - 1], BENCH_STYLE[1],
                       "Zw 229-15-like\n" + rf"($s={s_bench:g}$, $u_b={fb_bench:.2f}$ d$^{{-1}}$)"))
    for k, (v, c, m, lab) in enumerate(curves):
        med, err = fixed_lum_curve(v)
        ax.errorbar(zc + 0.015 * (k - len(curves) / 2), med, yerr=err, fmt="-" + m, color=c, lw=2, ms=7,
                    capsize=0, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("redshift $z$")
    ax.set_ylabel(r"median $\Delta\eta$ at fixed $L$ [dex]")
    panel_label(ax, "d")
    lo = ax.get_ylim()[0]
    ax.set_ylim(lo - 0.55 * (ax.get_ylim()[1] - lo), ax.get_ylim()[1] + 0.018)
    if one_value:
        ax.legend(fontsize=13, loc="lower left", frameon=False, ncol=2, columnspacing=0.8, handlelength=1.6)
    else:  # colour = family, marker = fixed value (key in (c)); one entry per family
        hs = [Line2D([], [], color=POP_FAMILY[f][0], lw=2, marker=pop_marker([v for v in pops if pop_key(v)[0] == f][0]),
                     ms=7, label=RANDOM_LABEL if f == "popR" else POP_FAMILY[f][1]) for f in families]
        hs += [Line2D([], [], color=c, lw=2, marker=m, ms=7, label=lab) for v, c, m, lab in curves if v not in pops]
        ax.legend(handles=hs, fontsize=13, loc="lower left", frameon=False, ncol=2, columnspacing=0.8,
                  handlelength=1.6)
    style(ax)

    save(fig, args.output)


def save(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print("wrote", output)


if __name__ == "__main__":
    main()
