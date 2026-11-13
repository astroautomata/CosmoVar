"""Generate short-lag PSD mocks and fit their matched DRW controls (Section 5.1)."""
from __future__ import annotations
import argparse
import json
import os
import tempfile
import time
import zlib
from pathlib import Path
import numpy as np
from framework.drw_fitting import run_drw_mcmc_jitter, trapezoid_time_weights
from framework import DATA_ROOT
PRIOR5_LN_D_BOUNDS = (-35.0, 26.0)
PRIOR5_LN_TAU_BOUNDS = (-6.0, 15.0)
TB_GRID_D = (1.0 / 12.0, 0.25, 0.5, 1.0, 2.0, 3.0)
S_GRID = (3.0, 3.4, 4.0)
TB_MIN_D, TB_MAX_D = 1.0 / 12.0, 3.0
POP_S_RANGE = (3.0, 3.4)
Z_EDGES = (0.5, 1.1, 1.7, 2.3, 2.9, 3.5)

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

def bent_covariance(t_rf, sigma2, tau, s, tb, force_table=False):
    """Covariance matrix of the bent-PSD process at rest-frame epochs t_rf.
    s = 4 uses the exact closed form; other slopes use the tabulated dk."""
    dt = np.abs(t_rf[:, None] - t_rf[None, :])
    x = dt / tau
    k = np.exp(-x)
    if tb is not None:
        r = tb / tau
        if abs(1.0 - r) < 1e-3:  # closed form is singular at r = 1; nudge
            r = 1.0 - 1e-3
        if abs(s - 4.0) < 1e-12 and not force_table:
            k = k - DeltaKTable.closed_form_s4(x, r)
        else:
            tab = DeltaKTable(s, r, x_max=float(x.max()))
            k = k - tab(x)
    return sigma2 * k

def self_check_kernel():
    """Numeric dk tables vs closed forms: s = 4 at all x (two-Lorentzian kernel,
    forcing the table route) and s = 3 at x = 0 (dk(0) = 1 - (2/pi) arccos(r)
    / sqrt(1-r^2), Laplace transform of K_0), for t_b/tau from 0.01 to 0.9.
    The mid-point tolerance (1e-10) is set by the smallest eigenvalues of the
    exact kernel for Gaia's 1.8 h epoch pairs (~1e-9 sigma^2)."""
    worst = 0.0
    for r in (0.01, 0.3, 0.9):
        tab = DeltaKTable(4.0, r, x_max=30.0)
        err_nodes = float(np.max(np.abs(tab.dk - tab.closed_form_s4(tab.x, r))))
        xm = np.sqrt(tab.x[1:-1] * tab.x[2:])
        err_interp = float(np.max(np.abs(tab(xm) - tab.closed_form_s4(xm, r))))
        tab3 = DeltaKTable(3.0, r, x_max=1.0, n=20)
        err3 = abs(tab3.dk[0] - (1.0 - (2.0 / np.pi) * np.arccos(r) / np.sqrt(1.0 - r * r)))
        worst = max(worst, err_nodes, err_interp, err3)
        if err_nodes > 1e-12 or err_interp > 1e-10 or not (err3 < 1e-10):
            raise RuntimeError(f"kernel self-check failed at r={r}: s4 nodes {err_nodes:.2e}, interp {err_interp:.2e}, s3 dk(0) {err3:.2e}")
    return worst

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lc-fits", required=True)
    ap.add_argument("--template-manifest", required=True)
    ap.add_argument("--cohort", choices=("representative", "common_support"), required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--variants", default=",".join(default_variants()))
    ap.add_argument("--min-epochs", type=int, default=10)
    ap.add_argument("--max-templates", type=int)
    ap.add_argument("--nwalkers", type=int, default=32)
    ap.add_argument("--nburn", type=int, default=200)
    ap.add_argument("--nsteps", type=int, default=500)
    ap.add_argument("--step-chunk", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--nsamples", type=int, default=500)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--mp-start-method", choices=("spawn", "forkserver", "fork"), default="spawn")
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--chain-seed", type=int, default=0)
    ap.add_argument("--save-joint", action="store_true")
    ap.add_argument("--d-D", type=float, default=-0.85)
    ap.add_argument("--analyze-only", action="store_true")
    return ap


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def tb_label(tb: float) -> str:
    return f"{tb * 24:.0f}h" if tb < 1.0 else f"{tb:g}d"

def default_variants():
    v = ["drw"]
    for s in S_GRID:
        for tb in TB_GRID_D:
            v.append(f"s{s:g}_tb{tb:.4g}")
    return v + ["popR", "popLup", "popLdown"]

def parse_fixed(name: str):
    if name == "drw":
        return 2.0, None
    s_str, tb_str = name[1:].split("_tb")
    return float(s_str), float(tb_str)

def load_manifest_templates(lc_fits, manifest_npz, *, cohort, injected_d_source, min_epochs, max_rows=None):
    """Load recorded simulation parameters and Gaia cadence/noise by source ID."""
    from astropy.io import fits

    manifest = np.load(manifest_npz, allow_pickle=False)
    use = np.asarray(manifest["cohort"]).astype(str) == str(cohort)
    if not np.any(use):
        available = sorted(set(np.asarray(manifest["cohort"]).astype(str).tolist()))
        raise ValueError(f"cohort {cohort!r} is absent; available={available}")
    d_key = {
        "population": "log10D_population",
        "population-mean": "log10D_population_mean",
        "empirical-prior5": "log10D_empirical_prior5",
    }[injected_d_source]
    fields = [
        "source_id", "z", "ell", "selection_probability", "selection_weight",
        "tau_true", d_key,
    ]
    rows = {key: np.asarray(manifest[key])[use] for key in fields}
    source_ids = np.asarray(rows["source_id"], dtype=np.int64)
    if np.unique(source_ids).size != source_ids.size:
        raise RuntimeError(f"manifest cohort {cohort!r} contains duplicate source IDs")
    by_sid = {int(sid): i for i, sid in enumerate(source_ids)}
    loaded = {}
    with fits.open(lc_fits, memmap=True) as hdul:
        data = hdul[1].data
        n_rows = len(data) if max_rows is None else min(len(data), int(max_rows))
        for i in range(n_rows):
            sid = int(data["source_id"][i])
            k = by_sid.get(sid)
            if k is None:
                continue
            z = float(data["z"][i])
            if abs(z - float(rows["z"][k])) > 1e-8:
                raise RuntimeError(f"redshift mismatch for source {sid}")
            t = np.asarray(data["g_transit_time"][i], dtype=float)
            fl = np.asarray(data["g_transit_flux"][i], dtype=float)
            fe = np.asarray(data["g_transit_flux_error"][i], dtype=float)
            mag = np.asarray(data["g_transit_mag"][i], dtype=float)
            good = np.isfinite(t) & np.isfinite(fl) & np.isfinite(fe) & np.isfinite(mag)
            good &= (fl > 0.0) & (fe > 0.0) & (mag > 0.0)
            if "variability_flag_g_reject" in data.dtype.names:
                good &= np.asarray(data["variability_flag_g_reject"][i]) == 0
            if int(good.sum()) < int(min_epochs):
                continue
            t, fl, fe, mag = t[good], fl[good], fe[good], mag[good]
            order = np.argsort(t)
            t, fl, fe, mag = t[order], fl[order], fe[order], mag[order]
            loaded[sid] = dict(
                source_id=sid, z=z, t_obs=t,
                merr=np.clip((2.5 / np.log(10.0)) * fe / fl, 1e-4, 1.0),
                g_med=float(np.median(mag)), npts=int(t.size),
                ell_real=float(rows["ell"][k]), log10D_real=float(rows[d_key][k]),
                tau_true=float(rows["tau_true"][k]),
                selection_probability=float(rows["selection_probability"][k]),
                selection_weight=float(rows["selection_weight"][k]), cohort=str(cohort),
            )
    missing = [int(sid) for sid in source_ids if int(sid) not in loaded]
    if missing:
        raise RuntimeError(f"{len(missing)} manifest sources missing from final light-curve FITS; first={missing[:5]}")
    return [loaded[int(sid)] for sid in source_ids]

def check_general_slope(s: float = 3.4) -> float:
    """Table dk(0) for a non-integer slope against the direct PSD integral of k(0).
    k(0)/sigma^2 = int 2 tau / (1 + (2 pi f tau)^2) (1 + (2 pi f t_b)^2)^(-p) df."""
    from scipy.integrate import quad

    p = (s - 2.0) / 2.0
    worst = 0.0
    for r in (1e-4, 3e-3, 0.1):
        tab = DeltaKTable(s, r, x_max=1.0, n=20)
        # u = 2 pi f tau; k(0)/sigma^2 = (2/pi) int_0^inf du / (1+u^2) (1 + (r u)^2)^(-p)
        f = lambda u: 1.0 / (1.0 + u * u) * (1.0 + (r * u) ** 2) ** (-p)
        val = (2.0 / np.pi) * (quad(f, 0, 1.0, epsabs=1e-14, epsrel=1e-12, limit=500)[0]
                               + quad(f, 1.0, np.inf, epsabs=1e-14, epsrel=1e-12, limit=500)[0])
        err = abs((1.0 - tab.dk[0]) - val)
        worst = max(worst, err)
        if err > 1e-9:
            raise RuntimeError(f"s={s} k(0) check failed at r={r}: {err:.2e}")
    return worst

def luminosity_proxy(g_med, z):
    from astropy.cosmology import Planck18

    return np.asarray(g_med) - Planck18.distmod(np.asarray(z)).value

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

def assign_population(variant, tpls, rng):
    """Per-template (s, t_b) for the population variants."""
    n = len(tpls)
    variant, fixed = parse_population(variant)
    if variant == "popR":
        s = rng.uniform(*POP_S_RANGE, size=n)
        tb = 10.0 ** rng.uniform(np.log10(TB_MIN_D), np.log10(TB_MAX_D), size=n)
        return s, tb
    ell = np.array([t["ell_real"] for t in tpls])
    rank = np.argsort(np.argsort(ell)) / max(n - 1, 1)  # 0 = least luminous, 1 = most luminous
    if variant in ("popSLdown", "popSLup"):
        # slope varies linearly with luminosity rank between 3 and 4 at a fixed Zw 229-15-like bend t_b = 1 d;
        # popSLdown: brighter objects shallower (direction of Smith et al. 2018); popSLup: the reverse
        s = 4.0 - rank if variant == "popSLdown" else 3.0 + rank
        return s, np.full(n, fixed)
    if variant == "popLdown":
        rank = 1.0 - rank
    tb = 10.0 ** (np.log10(TB_MIN_D) + rank * (np.log10(TB_MAX_D) - np.log10(TB_MIN_D)))
    return np.full(n, fixed), tb

def run_task(task):
    (tid, tpl, variant, s, tb, tau, log10D, u, v, fit_cfg) = task
    z, t_obs, merr = tpl["z"], tpl["t_obs"], tpl["merr"]
    t_rf = (t_obs - t_obs[0]) / (1.0 + z)
    D = 10.0 ** log10D
    sigma2 = D * tau
    K = bent_covariance(t_rf, sigma2, tau, s, tb)
    try:
        L = np.linalg.cholesky(K + 1e-8 * sigma2 * np.eye(K.shape[0]))
    except np.linalg.LinAlgError:
        return dict(tid=tid, variant=variant, ok=False, reason="cholesky")
    latent = tpl["g_med"] + L @ u
    m_win_true = float(np.sum(trapezoid_time_weights(t_rf) * latent))
    mag = latent + merr * v
    seed = int((fit_cfg["chain_seed"] + 1000003 * tid + zlib.crc32(variant.encode())) % (2**31 - 1))
    res = run_drw_mcmc_jitter(
        t_obs, mag, merr, z,
        parameterization="lnD", lnD_prior="flat",
        lnD_bounds=PRIOR5_LN_D_BOUNDS, ln_tau_bounds=PRIOR5_LN_TAU_BOUNDS,
        nwalkers=fit_cfg["nwalkers"], nburn=fit_cfg["nburn"], nsteps=fit_cfg["nsteps"],
        nsamples_output=fit_cfg["nsamples"], random_seed=seed,
        mwin_mode="trap", adaptive=True, target_thinned=fit_cfg["nsamples"],
        step_chunk=fit_cfg["step_chunk"], max_steps=fit_cfg["max_steps"],
    )
    if res is None:
        return dict(tid=tid, variant=variant, ok=False, reason="fit returned None")
    smp, names = res["samples"], res["param_names"]
    lnD_all = np.asarray(smp[:, names.index("ln_D")], dtype=float)
    lnD = lnD_all[np.isfinite(lnD_all)]
    mwin = smp[:, names.index("m_win")] if "m_win" in names else np.full(1, np.nan)
    q = lambda a, p: float(np.percentile(a, p)) / np.log(10.0) if a.size else np.nan
    out = dict(tid=tid, variant=variant, ok=bool(lnD.size), reason="",
               s=float(s), tb=np.nan if tb is None else float(tb),
               eta16=q(lnD, 16), eta50=q(lnD, 50), eta84=q(lnD, 84),
               mwin50=float(np.nanmedian(mwin)), m_win_true=m_win_true)
    diag = res.get("diag", {})
    out.update(
        accept_mean=float(diag.get("accept_mean", np.nan)),
        logprob_nonfinite_frac=float(diag.get("logprob_nonfinite_frac", np.nan)),
        iat_ln_D=float(diag.get("iat", {}).get("ln_D", np.nan)),
        neff_ln_D=float(diag.get("neff", {}).get("ln_D", np.nan)),
        thin_stride=float(diag.get("thin_stride", np.nan)),
        n_thinned_available=float(diag.get("n_thinned_available", np.nan)),
        n_thinned_used=float(diag.get("n_thinned_used", np.nan)),
    )
    if fit_cfg["save_joint"]:
        joint = np.isfinite(lnD_all) & np.isfinite(mwin) if mwin.size == lnD_all.size else np.zeros(lnD_all.size, bool)
        out["eta_samples"] = np.asarray(lnD_all[joint] / np.log(10.0), dtype=float)
        out["mwin_samples"] = np.asarray(mwin[joint], dtype=float)
    return out

def _median_regression(X, y):
    """Least-absolute-deviation (median) regression via linear programming."""
    from scipy.optimize import linprog

    n, k = X.shape
    c = np.concatenate([np.zeros(k), np.ones(n), np.ones(n)])
    A_eq = np.hstack([X, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * k + [(0, None)] * (2 * n)
    res = linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    return res.x[:k] if res.success else np.full(k, np.nan)

def _fit_trend(z, M, y, rng, n_boot=300):
    """Robust (median-regression) trend of y on luminosity proxy M and redshift z.
    Returns (b_z, err, b_M) with bootstrap errors; the OLS raw z-slope is also returned
    for reference.  Median regression is used because ~1% of pairs contain a runaway
    fit (|ddEta| of several dex, also present in the DRW control) that dominates OLS."""
    X = np.column_stack([np.ones_like(z), M - M.mean(), z - 1.5])
    beta = _median_regression(X, y)
    boots = []
    for _ in range(n_boot):
        k = rng.integers(0, y.size, y.size)
        boots.append(_median_regression(X[k], y[k])[2])
    boots = np.array(boots)
    raw = np.polyfit(z - 1.5, y, 1)[0]
    return float(beta[2]), float(np.nanstd(boots)), float(beta[1]), float(raw), \
        (float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5)))

def _fit_trend_quality_adjusted(z, ell, g_med, n_epoch, y, rng, n_boot=300):
    """Median-regression sensitivity including cadence/noise proxies."""
    log_n = np.log(np.asarray(n_epoch, dtype=float))
    X = np.column_stack([
        np.ones_like(z), ell - np.mean(ell), z - 1.5,
        g_med - np.mean(g_med), log_n - np.mean(log_n),
    ])
    beta = _median_regression(X, y)
    boots = []
    for _ in range(n_boot):
        k = rng.integers(0, y.size, y.size)
        boots.append(_median_regression(X[k], y[k])[2])
    boots = np.asarray(boots, dtype=float)
    return (
        float(beta[2]), float(np.nanstd(boots)),
        (float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))),
    )

def analyze(npz_path, out_dir, d_D=-0.85):
    d = np.load(npz_path, allow_pickle=False)
    variants = [str(v) for v in d["variants"]]
    ok = d["ok"].astype(bool)
    tid, vidx = d["tid"].astype(int), d["variant_idx"].astype(int)
    z_t, M_t = d["tpl_z"], d["tpl_M"]
    ell_t = np.asarray(d["tpl_ell"], dtype=float)
    ctrl = {int(t): e for t, e, o, vi in zip(tid, d["eta50"], ok, vidx) if o and vi == variants.index("drw")}
    rng = np.random.default_rng(1)
    rows, lines = [], []
    lines.append("ddEta = eta50(variant) - eta50(DRW control), same template and noise realisation (Gaia G, full cadence)")
    lines.append("b_z = redshift slope at fixed luminosity [dex per unit z], MEDIAN regression on (ell, z), bootstrap 1-sigma and 95% interval;")
    lines.append("ell = log10 L3000 from the true m_win via the paper's SED/passband mapping (Gaia G); b_ell = slope per dex of L3000.")
    lines.append("dmu/dz = 2.5 b_z / d_D (signed; d_D = %.2f): fitted-minus-true distance-modulus tilt in linear response" % d_D)
    lines.append(f"{'variant':>14} {'N':>5} {'median':>8} {'16..84':>17} {'b_z':>8} {'+-':>6} {'95% interval':>17} {'b_ell':>8} {'dmu/dz':>7} {'n>0.3':>6}")
    zc_idx = None
    for vi, v in enumerate(variants):
        sel = ok & (vidx == vi)
        t = tid[sel]
        y = np.array([e - ctrl.get(int(k), np.nan) for k, e in zip(t, d["eta50"][sel])])
        good = np.isfinite(y)
        t, y = t[good], y[good]
        dm = (d["mwin50"][sel][good] - d["m_win_true"][sel][good]) * 1e3
        row = dict(variant=v, n=int(y.size), n_fail=int(np.sum((vidx == vi) & ~ok)))
        if v != "drw" and y.size > 10:
            z, ell = z_t[t], ell_t[t]
            bz, bz_err, bM, raw, ci = _fit_trend(z, ell, y, rng)
            bz_q, bz_q_err, ci_q = _fit_trend_quality_adjusted(
                z, ell, d["tpl_gmed"][t], d["tpl_npts"][t], y, rng
            )
            bz0, bz0_err = np.nan, np.nan
            zb = np.digitize(z, Z_EDGES[1:-1])
            zbins = []
            for k in range(len(Z_EDGES) - 1):
                yy = y[zb == k]
                if yy.size < 5:
                    zbins.append([np.nan, np.nan]); continue
                boots = [np.median(rng.choice(yy, yy.size)) for _ in range(500)]
                zbins.append([float(np.median(yy)), float(np.std(boots))])
            n_big = int(np.sum(np.abs(y) > 0.3))
            row.update(median=float(np.median(y)), p16=float(np.percentile(y, 16)), p84=float(np.percentile(y, 84)),
                       b_z=bz, b_z_err=bz_err, b_z_ci95=ci, b_ell=bM, b_M=bM, ols_raw_dz=raw,
                       b_z_quality_adjusted=bz_q, b_z_quality_adjusted_err=bz_q_err,
                       b_z_quality_adjusted_ci95=ci_q,
                       dmu_dz=2.5 * bz / d_D, dmu_dz_err=2.5 * bz_err / abs(d_D),
                       dmu_dz_ci95=sorted([2.5 * ci[0] / d_D, 2.5 * ci[1] / d_D]),
                       n_abs_gt_0p3=n_big, median_dmwin_mmag=float(np.median(dm)), zbins=zbins)
            lines.append(f"{v:>14} {y.size:5d} {row['median']:+8.4f} {row['p16']:+8.3f}..{row['p84']:+7.3f} "
                         f"{bz:+8.4f} {bz_err:6.4f} [{ci[0]:+7.4f},{ci[1]:+7.4f}] {bM:+8.4f} {row['dmu_dz']:+7.3f} {n_big:6d}")
        else:
            e = d["eta50"][sel][good] - d["log10D_true"][sel][good]
            c68 = np.mean((d["eta16"][sel][good] <= d["log10D_true"][sel][good]) & (d["log10D_true"][sel][good] <= d["eta84"][sel][good]))
            row.update(median_eta_minus_true=float(np.median(e)), c68=float(c68))
            lines.append(f"{v:>14} {y.size:5d}  control: median(eta - eta_true) = {np.median(e):+.4f}, C68 = {c68:.3f}")
        rows.append(row)
    txt = "\n".join(lines)
    print(txt)
    (Path(out_dir) / "psd_summary.txt").write_text(txt + "\n")
    (Path(out_dir) / "psd_summary.json").write_text(json.dumps(dict(z_edges=Z_EDGES, d_D=d_D, rows=rows), indent=1))

def main():
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "psd_results.npz"
    if npz_path.exists() and not args.analyze_only:
        raise FileExistsError(npz_path)
    if args.analyze_only:
        analyze(npz_path, out_dir, d_D=args.d_D)
        return

    w4 = self_check_kernel()
    w34 = check_general_slope(3.4)
    _log(f"kernel checks passed: s=3/4 tables {w4:.1e}; s=3.4 k(0) vs PSD integral {w34:.1e}")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    rng = np.random.default_rng(args.seed)
    _log("Loading templates ...")
    tpls = load_manifest_templates(
        args.lc_fits, args.template_manifest, cohort=args.cohort,
        injected_d_source="population", min_epochs=args.min_epochs,
    )
    if args.max_templates is not None:
        tpls = tpls[:args.max_templates]
    _log(f"Loaded {len(tpls)} templates from cohort={args.cohort}.")
    Mp = luminosity_proxy([t["g_med"] for t in tpls], [t["z"] for t in tpls])
    for t, m in zip(tpls, Mp):
        t["M_proxy"] = float(m)
    taus = np.asarray([t["tau_true"] for t in tpls], dtype=float)
    noise = [(rng.normal(size=t["npts"]), rng.normal(size=t["npts"])) for t in tpls]
    pop = {v: assign_population(v, tpls, np.random.default_rng(args.seed + 17)) for v in variants if v.startswith("pop")}

    fit_cfg = dict(nwalkers=args.nwalkers, nburn=args.nburn, nsteps=args.nsteps,
                   nsamples=args.nsamples, step_chunk=args.step_chunk, max_steps=args.max_steps,
                   chain_seed=args.chain_seed, save_joint=args.save_joint)
    tasks = []
    for tid, tpl in enumerate(tpls):
        u, v = noise[tid]
        for variant in variants:
            if variant.startswith("pop"):
                s, tb = float(pop[variant][0][tid]), float(pop[variant][1][tid])
            else:
                s, tb = parse_fixed(variant)
            tasks.append((tid, tpl, variant, s, tb, float(taus[tid]), tpl["log10D_real"], u, v, fit_cfg))
    _log(f"{len(tasks)} fits ({len(tpls)} templates x {len(variants)} variants)")

    n_jobs = max(1, (os.cpu_count() or 1) - 2) if args.n_jobs < 0 else args.n_jobs
    if n_jobs < 1:
        raise ValueError("--n-jobs must be >= 1 or negative for automatic selection")
    t0 = time.time()
    results = []
    _log(f"running fits with n_jobs={n_jobs}, mp_start_method={args.mp_start_method}")

    if n_jobs == 1:
        result_iter = map(run_task, tasks)
        for k, r in enumerate(result_iter, start=1):
            results.append(r)
            if k % 500 == 0 or k == len(tasks):
                _log(f"  {k}/{len(tasks)} done, {time.time() - t0:.0f}s elapsed")
    else:
        import multiprocessing as mp

        with mp.get_context(args.mp_start_method).Pool(n_jobs) as pool_:
            for k, r in enumerate(pool_.imap_unordered(run_task, tasks, chunksize=2), start=1):
                results.append(r)
                if k % 500 == 0 or k == len(tasks):
                    _log(f"  {k}/{len(tasks)} done, {time.time() - t0:.0f}s elapsed")
    _log(f"fits finished in {time.time() - t0:.0f}s; failures: {sum(1 for r in results if not r['ok'])}")

    keys = [
        "s", "tb", "eta16", "eta50", "eta84", "mwin50", "m_win_true",
        "accept_mean", "logprob_nonfinite_frac", "iat_ln_D", "neff_ln_D",
        "thin_stride", "n_thinned_available", "n_thinned_used",
    ]
    out = {k: np.array([r.get(k, np.nan) for r in results], dtype=float) for k in keys}
    out["tid"] = np.array([r["tid"] for r in results])
    out["ok"] = np.array([r["ok"] for r in results])
    out["reason"] = np.array([r["reason"] for r in results])
    out["variant_idx"] = np.array([variants.index(r["variant"]) for r in results])
    out["log10D_true"] = np.array([tpls[r["tid"]]["log10D_real"] for r in results])
    out["variants"] = np.array(variants)
    out["tpl_z"] = np.array([t["z"] for t in tpls])
    out["tpl_source_id"] = np.array([t["source_id"] for t in tpls], dtype=np.int64)
    out["tpl_gmed"] = np.array([t["g_med"] for t in tpls])
    out["tpl_npts"] = np.array([t["npts"] for t in tpls])
    out["tpl_M"] = np.array([t["M_proxy"] for t in tpls])
    out["tpl_ell"] = np.array([t.get("ell_real", np.nan) for t in tpls])
    out["tpl_selection_probability"] = np.array([t.get("selection_probability", np.nan) for t in tpls])
    out["tpl_selection_weight"] = np.array([t.get("selection_weight", np.nan) for t in tpls])
    out["tpl_tau"] = taus
    out["tpl_tobs"] = np.array([t["t_obs"] for t in tpls], dtype=object)
    if args.save_joint:
        max_samples = max((len(r.get("eta_samples", ())) for r in results), default=0)
        eta_samples = np.full((len(results), max_samples), np.nan)
        mwin_samples = np.full((len(results), max_samples), np.nan)
        for i, r in enumerate(results):
            n = min(len(r.get("eta_samples", ())), len(r.get("mwin_samples", ())), max_samples)
            if n:
                eta_samples[i, :n] = r["eta_samples"][:n]
                mwin_samples[i, :n] = r["mwin_samples"][:n]
        out["eta_samples"] = eta_samples
        out["mwin_samples"] = mwin_samples
    out["config"] = np.array(json.dumps(vars(args)))
    # Do not leave a corrupt final archive if a Slurm job is interrupted while
    # writing.  The temporary file lives on the same filesystem, so replace is
    # atomic once the archive has been flushed and fsynced.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=".psd_results.", suffix=".npz",
            dir=out_dir, delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            np.savez(tmp, **out)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, npz_path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    _log(f"wrote {npz_path}")
    analyze(npz_path, out_dir, d_D=args.d_D)

if __name__ == "__main__":
    main()
