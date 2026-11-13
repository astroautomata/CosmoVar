"""DRW Gaussian-process fitting and finite-window summaries.

Stage 1 uses run_drw_mcmc_jitter to sample the light-curve posterior.
Input times are converted to the rest frame using the source redshift.
"""

import numpy as np
from celerite2 import terms, GaussianProcess
from scipy import optimize
from scipy.linalg import cho_factor, cho_solve


def fit_drw_unconstrained(times_rest, mags_centered, mag_errs):
    """
    Fit a Damped Random Walk (DRW) model using maximum likelihood optimization.

    The DRW/OU process has covariance: C(dt) = sigma^2 * exp(-|dt|/tau)

    This is implemented as a celerite2 RealTerm kernel with:
        a = sigma^2 (variance)
        c = 1/tau (inverse timescale)

    Parameters
    ----------
    times_rest : array-like
        Rest-frame times in days. Will be sorted internally.
    mags_centered : array-like
        Median-subtracted magnitudes. Must have same length as times_rest.
    mag_errs : array-like
        Photometric uncertainties on magnitudes. Must have same length as times_rest.

    Returns
    -------
    dict or None
        If successful, returns dictionary with:
            - 'log_sigma' : float, natural log of sigma
            - 'log_tau' : float, natural log of tau
            - 'sigma' : float, DRW amplitude in magnitudes
            - 'tau' : float, DRW timescale in days (rest-frame)
            - 'success' : bool, True if optimization converged
        Returns None if fitting fails (bad data, non-convergence, numerical issues).

    Notes
    -----
    - The optimizer uses bounds: sigma in [0.001, 100] mag, tau in [0.1, 10000] days
    - Initial guess: sigma ~ sqrt(variance), tau ~ 10% of baseline
    - Data is automatically sorted by time for celerite2 compatibility
    """
    try:
        # Ensure arrays are numpy arrays
        times_rest = np.asarray(times_rest)
        mags_centered = np.asarray(mags_centered)
        mag_errs = np.asarray(mag_errs)

        # Sort by time (celerite2 requires sorted times for correct GP computation)
        sort_idx = np.argsort(times_rest)
        times_rest = times_rest[sort_idx]
        mags_centered = mags_centered[sort_idx]
        mag_errs = mag_errs[sort_idx]

        def neg_ll(params):
            log_sigma, log_tau = params
            sigma = np.exp(log_sigma)
            tau = np.exp(log_tau)

            sigma2 = sigma**2
            kernel = terms.RealTerm(a=sigma2, c=1.0 / tau)
            gp = GaussianProcess(kernel)
            try:
                gp.compute(times_rest, yerr=mag_errs)
                ll = gp.log_likelihood(mags_centered)
            except Exception:
                # Any numerical issue in the GP gets penalised
                return np.inf

            # Extra safety: if ll is NaN or +/-inf, penalise it
            if not np.isfinite(ll):
                return np.inf

            return -ll

        # Initialise from the measured variance and observing baseline.
        y_var = np.nanvar(mags_centered)
        log_sigma_init = 0.5 * np.log(max(y_var, 1e-6))
        baseline = np.max(times_rest) - np.min(times_rest)
        log_tau_init = np.log(max(baseline * 0.1, 1.0))

        soln = optimize.minimize(
            neg_ll,
            [log_sigma_init, log_tau_init],
            method="L-BFGS-B",
            bounds=[(np.log(0.001), np.log(100.0)), (np.log(0.1), np.log(1e4))],
        )

        if not soln.success:
            return None

        log_sigma_opt, log_tau_opt = soln.x
        return {
            "log_sigma": log_sigma_opt,
            "log_tau": log_tau_opt,
            "sigma": np.exp(log_sigma_opt),
            "tau": np.exp(log_tau_opt),
            "success": True,
        }
    except Exception:
        return None


def fit_drw_warm(times_rest, mags_centered, mag_errs, init_log_sigma, init_log_tau):
    """
    Fit DRW with warm start (initial guess from previous fit).
    Useful for repeated fits to similar light curves.
    """
    try:
        times_rest = np.asarray(times_rest)
        mags_centered = np.asarray(mags_centered)
        mag_errs = np.asarray(mag_errs)

        sort_idx = np.argsort(times_rest)
        times_rest = times_rest[sort_idx]
        mags_centered = mags_centered[sort_idx]
        mag_errs = mag_errs[sort_idx]

        def neg_ll(params):
            log_sigma, log_tau = params
            sigma2 = np.exp(2 * log_sigma)
            tau = np.exp(log_tau)
            kernel = terms.RealTerm(a=sigma2, c=1.0 / tau)
            gp = GaussianProcess(kernel)
            try:
                gp.compute(times_rest, yerr=mag_errs)
                ll = gp.log_likelihood(mags_centered)
            except Exception:
                return np.inf
            if not np.isfinite(ll):
                return np.inf
            return -ll

        # Initialise the optimiser from the preceding fit.
        soln = optimize.minimize(
            neg_ll,
            [init_log_sigma, init_log_tau],
            method="L-BFGS-B",
            bounds=[(np.log(0.001), np.log(100.0)), (np.log(0.1), np.log(1e4))],
        )

        if not soln.success and soln.fun == np.inf:
            return None

        log_sigma_opt, log_tau_opt = soln.x
        return {
            "log_sigma": log_sigma_opt,
            "log_tau": log_tau_opt,
            "sigma": np.exp(log_sigma_opt),
            "tau": np.exp(log_tau_opt),
            "success": True,
        }
    except Exception:
        return None


# =============================================================================
# DRW HELPER PRIOR DEFAULTS
# =============================================================================
PRIOR_M_MIN, PRIOR_M_MAX = 10.0, 25.0            # mean apparent magnitude
PRIOR_LN_D_MIN, PRIOR_LN_D_MAX = -20.0, 20.0     # ln(D) where D = sigma^2/tau
PRIOR_LN_TAU_MIN, PRIOR_LN_TAU_MAX = -6.0, 15.0  # ln(tau) [days, rest-frame]
PRIOR_LN_JITTER_MIN, PRIOR_LN_JITTER_MAX = np.log(1e-8), np.log(1.0)  # ln(jitter) [mag]
PRIOR_LN_SIGMA2_MIN, PRIOR_LN_SIGMA2_MAX = -20.0, 20.0  # ln(sigma^2) for fixed-tau fits
# For comparing priors in lnD space: a flat box in lnD that spans the same lnD support
# as the canonical "induced" lnD prior from a flat box in (ln_sigma2, ln_tau).
PRIOR_LN_D_FLAT_MIN = float(PRIOR_LN_SIGMA2_MIN - PRIOR_LN_TAU_MAX)
PRIOR_LN_D_FLAT_MAX = float(PRIOR_LN_SIGMA2_MAX - PRIOR_LN_TAU_MIN)


def _require_emcee():
    try:
        import emcee  # noqa: F401
    except Exception as e:
        raise ImportError(
            "emcee is required for MCMC fits. Activate qsogpmock_env or install emcee."
        ) from e


def _clean_and_restframe(times, mags, mag_errs, z, min_points=10):
    times = np.asarray(times, dtype=float)
    mags = np.asarray(mags, dtype=float)
    mag_errs = np.asarray(mag_errs, dtype=float)

    mask = np.isfinite(times) & np.isfinite(mags) & np.isfinite(mag_errs) & (mag_errs > 0)
    if np.sum(mask) < min_points:
        return None

    times = times[mask]
    mags = mags[mask]
    mag_errs = mag_errs[mask]

    order = np.argsort(times)
    times = times[order]
    mags = mags[order]
    mag_errs = mag_errs[order]

    t0 = times.min()
    times_rest = (times - t0) / (1.0 + z)
    return times_rest, mags, mag_errs


def _build_gp_drw_jitter(m, ln_sigma2, ln_tau, ln_jitter, t_rest, mag_errs):
    tau = np.exp(ln_tau)
    sigma2 = np.exp(ln_sigma2)
    jitter = np.exp(ln_jitter)

    if sigma2 <= 0.0 or tau <= 0.0 or jitter <= 0.0:
        raise ValueError("sigma2, tau, jitter must be positive")

    kernel = terms.RealTerm(a=sigma2, c=1.0 / tau)
    gp = GaussianProcess(kernel, mean=m)
    diag = mag_errs**2 + jitter**2
    gp.compute(t_rest, diag=diag, check_sorted=False)
    return gp


def trapezoid_time_weights(t_rest):
    """
    Return trapezoid-rule weights for a time-average over the observed window.

    Safe for unsorted input: weights are computed in sorted time order and
    returned aligned to the original input order.

    Parameters
    ----------
    t_rest : array-like, shape (N,)
        Rest-frame times (need not be sorted).

    Returns
    -------
    w : ndarray, shape (N,)
        Non-negative weights summing to 1.0, aligned with the input order.
    """
    t = np.asarray(t_rest, dtype=float)
    n = int(t.size)
    if n <= 0:
        raise ValueError("t_rest must be non-empty")
    if n == 1:
        return np.array([1.0], dtype=float)

    order = np.argsort(t)
    t_s = t[order]

    dt = np.diff(t_s)
    dt = np.asarray(dt, dtype=float)
    dt[~np.isfinite(dt)] = 0.0
    dt = np.maximum(dt, 0.0)

    w_s = np.empty(n, dtype=float)
    w_s[0] = 0.5 * dt[0]
    w_s[-1] = 0.5 * dt[-1]
    if n > 2:
        w_s[1:-1] = 0.5 * (dt[:-1] + dt[1:])
    s = float(np.sum(w_s))
    if not np.isfinite(s) or s <= 0:
        return np.full(n, 1.0 / n, dtype=float)

    w = np.empty(n, dtype=float)
    w[order] = w_s / s
    return w


def drw_window_integral_covariances(t_rest, sigma2, tau, a=None, b=None):
    """
    Covariances for the continuous-time DRW window mean.

    For a DRW/OU latent process with covariance
        K(t, t') = sigma2 * exp(-abs(t - t') / tau),
    this returns the covariance vector between
        m_win_cont = (1 / T) * int_a^b f(s) ds
    and the latent values f(t_i), plus Var(m_win_cont).

    Parameters
    ----------
    t_rest : array-like, shape (N,)
        Rest-frame times. They need not be sorted.
    sigma2 : float
        Stationary DRW variance in mag^2.
    tau : float
        DRW damping time in rest-frame days.
    a, b : float or None
        Integration window endpoints. Defaults to min(t_rest), max(t_rest).

    Returns
    -------
    c : ndarray, shape (N,)
        Cov(m_win_cont, f(t_i)), aligned with the input order.
    v : float
        Var(m_win_cont).
    """
    t = np.asarray(t_rest, dtype=float)
    sigma2 = float(sigma2)
    tau = float(tau)
    if t.size == 0:
        return np.array([], dtype=float), np.nan
    if not np.isfinite(sigma2) or not np.isfinite(tau) or sigma2 < 0.0 or tau <= 0.0:
        return np.full(t.shape, np.nan, dtype=float), np.nan

    finite_t = t[np.isfinite(t)]
    if finite_t.size == 0:
        return np.full(t.shape, np.nan, dtype=float), np.nan
    a = float(np.nanmin(finite_t)) if a is None else float(a)
    b = float(np.nanmax(finite_t)) if b is None else float(b)
    if b < a:
        a, b = b, a
    T = float(b - a)

    if not np.isfinite(T) or T <= 0.0:
        anchor = a if np.isfinite(a) else float(finite_t[0])
        c = sigma2 * np.exp(-np.abs(t - anchor) / tau)
        return c, float(sigma2)

    c = np.full(t.shape, np.nan, dtype=float)
    inside = (t >= a) & (t <= b)
    left = t < a
    right = t > b

    pref = sigma2 * tau / T
    if np.any(inside):
        x_left = (t[inside] - a) / tau
        x_right = (b - t[inside]) / tau
        c[inside] = pref * (-np.expm1(-x_left) - np.expm1(-x_right))
    if np.any(left):
        c[left] = pref * np.exp(-(a - t[left]) / tau) * (-np.expm1(-T / tau))
    if np.any(right):
        c[right] = pref * np.exp(-(t[right] - b) / tau) * (-np.expm1(-T / tau))

    x = T / tau
    # The series avoids cancellation when the window is short compared with tau.
    if x < 1.0e-5:
        v = sigma2 * (1.0 - x / 3.0 + x * x / 12.0)
    else:
        v = 2.0 * sigma2 * (x + np.expm1(-x)) / (x * x)
    if not np.isfinite(v) or v < 0.0:
        v = 0.0
    return c, float(v)


def sample_true_mwin_cont(t_rf, m_true, mean_mag, sigma2, tau, rng):
    """
    Draw the true continuous window mean conditional on simulated latent epochs.

    This is used by mock generators.  The simulated latent DRW values at the
    observed epochs are exact noiseless values, so the conditioning covariance is
    the latent DRW covariance K, without photometric noise or jitter.

    Returns
    -------
    draw, mu, var : tuple of floats
        A conditional draw of (1/T) int f(t) dt, its conditional mean, and its
        conditional variance.
    """
    t = np.asarray(t_rf, dtype=float)
    f = np.asarray(m_true, dtype=float)
    mean_mag = float(mean_mag)
    sigma2 = float(sigma2)
    tau = float(tau)
    good = np.isfinite(t) & np.isfinite(f)
    t = t[good]
    f = f[good]

    if t.size == 0:
        return np.nan, np.nan, np.nan
    if t.size == 1:
        val = float(f[0])
        return val, val, 0.0
    if not np.isfinite(sigma2) or sigma2 < 0.0 or not np.isfinite(tau) or tau <= 0.0:
        return np.nan, np.nan, np.nan

    c, v = drw_window_integral_covariances(t, sigma2=sigma2, tau=tau)
    dt = np.abs(t[:, None] - t[None, :])
    K = sigma2 * np.exp(-dt / tau)

    eye = np.eye(t.size)
    nugget0 = 1.0e-10 * max(sigma2, 1.0)
    cf = None
    for scale in (1.0, 1.0e3, 1.0e6):
        try:
            cf = cho_factor(K + (nugget0 * scale) * eye, lower=True, check_finite=False)
            break
        except Exception:
            cf = None
    if cf is None:
        return np.nan, np.nan, np.nan

    alpha = cho_solve(cf, f - mean_mag, check_finite=False)
    beta = cho_solve(cf, c, check_finite=False)
    mu = mean_mag + float(c @ alpha)
    var = float(v - c @ beta)
    if not np.isfinite(var) or var < 0.0:
        var = 0.0
    draw = float(rng.normal(loc=mu, scale=np.sqrt(var)))
    return draw, float(mu), float(var)


def _trapezoid_time_weights(t_rest):
    """
    Alias for `trapezoid_time_weights`.
    """
    return trapezoid_time_weights(t_rest)


def _log_prior_box_jitter_sigma2(m, ln_sigma2, ln_tau, ln_jitter, *, ln_tau_bounds=None, ln_sigma2_bounds=None):
    if ln_tau_bounds is None:
        ln_tau_bounds = (PRIOR_LN_TAU_MIN, PRIOR_LN_TAU_MAX)
    ln_tau_lo, ln_tau_hi = float(ln_tau_bounds[0]), float(ln_tau_bounds[1])
    if ln_sigma2_bounds is None:
        ln_sigma2_bounds = (PRIOR_LN_SIGMA2_MIN, PRIOR_LN_SIGMA2_MAX)
    ln_sigma2_lo, ln_sigma2_hi = float(ln_sigma2_bounds[0]), float(ln_sigma2_bounds[1])
    if not (PRIOR_M_MIN < m < PRIOR_M_MAX):
        return -np.inf
    if not (ln_sigma2_lo < ln_sigma2 < ln_sigma2_hi):
        return -np.inf
    if not (ln_tau_lo < ln_tau < ln_tau_hi):
        return -np.inf
    if not (PRIOR_LN_JITTER_MIN < ln_jitter < PRIOR_LN_JITTER_MAX):
        return -np.inf
    return 0.0


def _log_prior_box_jitter_lnD(m, ln_D, ln_tau, ln_jitter, *, lnD_bounds=None, ln_tau_bounds=None):
    if lnD_bounds is None:
        lnD_bounds = (PRIOR_LN_D_FLAT_MIN, PRIOR_LN_D_FLAT_MAX)
    if ln_tau_bounds is None:
        ln_tau_bounds = (PRIOR_LN_TAU_MIN, PRIOR_LN_TAU_MAX)
    lnD_lo, lnD_hi = float(lnD_bounds[0]), float(lnD_bounds[1])
    ln_tau_lo, ln_tau_hi = float(ln_tau_bounds[0]), float(ln_tau_bounds[1])
    if not (PRIOR_M_MIN < m < PRIOR_M_MAX):
        return -np.inf
    if not (lnD_lo < ln_D < lnD_hi):
        return -np.inf
    if not (ln_tau_lo < ln_tau < ln_tau_hi):
        return -np.inf
    if not (PRIOR_LN_JITTER_MIN < ln_jitter < PRIOR_LN_JITTER_MAX):
        return -np.inf
    return 0.0


def _loglike_from_sigma2_tau_jitter(theta_sigma2, t_rest, mags, mag_errs):
    m, ln_sigma2, ln_tau, ln_jitter = theta_sigma2
    try:
        gp = _build_gp_drw_jitter(m, ln_sigma2, ln_tau, ln_jitter, t_rest, mag_errs)
        ll = gp.log_likelihood(mags)
        return ll if np.isfinite(ll) else -np.inf
    except Exception:
        return -np.inf


def run_drw_mcmc_jitter(
    times,
    mags,
    mag_errs,
    z,
    *,
    parameterization="lnsigma2",
    tau_fixed=None,
    min_points=10,
    nwalkers=32,
    nburn=200,
    nsteps=400,
    nsamples_output=500,
    random_seed=None,
    lnD_prior="induced",
    ln_tau_bounds=None,
    lnD_bounds=None,
    ln_sigma2_bounds=None,
    ln_sigma_bounds=None,
    mwin_mode: str = "none",
    adaptive: bool = False,
    target_thinned: int | None = None,
    step_chunk: int = 500,
    max_steps: int | None = None,
    return_chain: bool = False,
):
    """
    JITTER-ONLY DRW MCMC for a single light curve in magnitude space.

    Supports parameterizations:
      - parameterization='lnsigma2': samples ln(sigma^2) directly
      - parameterization='lnsigma': samples ln(sigma) directly
      - parameterization='lnD': samples ln(D) where D = sigma^2 / tau

    Supports both free-tau and fixed-tau:
      - tau_fixed is None: samples ln_tau (free tau)
      - tau_fixed is float: tau is fixed (rest-frame days)

    Priors (canonical):
      - flat box on (m, ln_sigma2, ln_tau, ln_jitter) using PRIOR_* constants
      - when sampling lnD, the prior is enforced via ln_sigma2 = lnD + ln_tau (or + ln_tau_fixed)
      - for comparisons, lnD_prior='flat' uses a flat box prior directly on lnD
        over [PRIOR_LN_D_FLAT_MIN, PRIOR_LN_D_FLAT_MAX] with the same lnD support
        as the induced prior (and does NOT enforce the ln_sigma2 box in the prior).
      - ln_tau_bounds can override the default [PRIOR_LN_TAU_MIN, PRIOR_LN_TAU_MAX]
      - lnD_bounds can override the default [PRIOR_LN_D_FLAT_MIN, PRIOR_LN_D_FLAT_MAX]
        when lnD_prior='flat'

    Window-mean posterior export:
      - mwin_mode='none': do not derive window-mean latent magnitudes
      - mwin_mode='trap': derive trapezoid/quadrature m_win only
      - mwin_mode='cont': derive continuous-integral m_win_cont only
      - mwin_mode='both': derive both definitions

    Sampling and export
    -------------------
    The IAT-based export stride is applied along the sampler time axis before
    pooling walkers. Stored draws can remain correlated. diag['autocorr_ok'] records
    a successful sampled-parameter IAT calculation, rather than a convergence
    certification. Derived ln_D IAT/N_eff are reported separately after export
    when sampling ln_sigma or ln_sigma2; they do not set stopping or thinning.
    random_seed controls initial positions, emcee transitions, sample
    selection and window-mean draws.

    Returns
    -------
    dict with keys:
      - 't_rest_npts'
      - 'samples' : ndarray (nsamples_output, ndim)
      - 'param_names' : list[str]
      - 'diag' : dict with MCMC diagnostics (IAT, N_eff, acceptance, non-finite log_prob)
    """
    _require_emcee()
    import emcee

    mwin_mode = str(mwin_mode).lower().strip()
    if mwin_mode not in {"none", "trap", "cont", "both"}:
        raise ValueError("mwin_mode must be one of: none, trap, cont, both")

    cleaned = _clean_and_restframe(times, mags, mag_errs, z, min_points=min_points)
    if cleaned is None:
        return None

    t_rest, y, yerr = cleaned
    baseline_rf = float(t_rest.max() - t_rest.min())

    draws_seed, sampler_seed, trap_seed, cont_seed = np.random.SeedSequence(random_seed).spawn(4)
    rng = np.random.default_rng(draws_seed)
    m0 = float(np.median(y))

    if ln_tau_bounds is None:
        ln_tau_bounds = (PRIOR_LN_TAU_MIN, PRIOR_LN_TAU_MAX)
    ln_tau_min, ln_tau_max = float(ln_tau_bounds[0]), float(ln_tau_bounds[1])
    if not np.isfinite(ln_tau_min) or not np.isfinite(ln_tau_max) or not (ln_tau_min < ln_tau_max):
        raise ValueError(f"Invalid ln_tau_bounds={ln_tau_bounds}")

    if ln_sigma2_bounds is None:
        ln_sigma2_bounds = (PRIOR_LN_SIGMA2_MIN, PRIOR_LN_SIGMA2_MAX)
    ln_sigma2_min, ln_sigma2_max = float(ln_sigma2_bounds[0]), float(ln_sigma2_bounds[1])
    if not np.isfinite(ln_sigma2_min) or not np.isfinite(ln_sigma2_max) or not (ln_sigma2_min < ln_sigma2_max):
        raise ValueError(f"Invalid ln_sigma2_bounds={ln_sigma2_bounds}")

    if ln_sigma_bounds is None:
        ln_sigma_bounds = (0.5 * PRIOR_LN_SIGMA2_MIN, 0.5 * PRIOR_LN_SIGMA2_MAX)
    ln_sigma_min, ln_sigma_max = float(ln_sigma_bounds[0]), float(ln_sigma_bounds[1])
    if not np.isfinite(ln_sigma_min) or not np.isfinite(ln_sigma_max) or not (ln_sigma_min < ln_sigma_max):
        raise ValueError(f"Invalid ln_sigma_bounds={ln_sigma_bounds}")

    var_y = float(np.var(y))
    mean_var_err = float(np.mean(yerr * yerr))
    sigma2_init = max(var_y - mean_var_err, 1e-6)
    ln_sigma2_init = float(np.log(sigma2_init))
    ln_sigma2_init = float(np.clip(ln_sigma2_init, ln_sigma2_min + 1e-6, ln_sigma2_max - 1e-6))

    if tau_fixed is None:
        tau_init = max(10.0, 0.3 * baseline_rf)
        ln_tau_init = float(np.log(tau_init))
        ln_tau_init = float(np.clip(ln_tau_init, ln_tau_min + 1e-6, ln_tau_max - 1e-6))
    else:
        tau_fixed = float(tau_fixed)
        if not np.isfinite(tau_fixed) or tau_fixed <= 0:
            raise ValueError("tau_fixed must be positive")
        ln_tau_init = float(np.log(tau_fixed))

    ln_jitter_init = float(np.log(0.02))

    if tau_fixed is None:
        if parameterization == "lnsigma2":
            param_names = ["m", "ln_sigma2", "ln_tau", "ln_jitter"]
            p0 = np.array([m0, ln_sigma2_init, ln_tau_init, ln_jitter_init], dtype=float)

            def log_prob(theta):
                m, ln_sigma2, ln_tau, ln_jitter = theta
                lp = _log_prior_box_jitter_sigma2(
                    m,
                    ln_sigma2,
                    ln_tau,
                    ln_jitter,
                    ln_tau_bounds=(ln_tau_min, ln_tau_max),
                    ln_sigma2_bounds=(ln_sigma2_min, ln_sigma2_max),
                )
                if not np.isfinite(lp):
                    return -np.inf
                ll = _loglike_from_sigma2_tau_jitter(theta, t_rest, y, yerr)
                return lp + ll

        elif parameterization == "lnsigma":
            param_names = ["m", "ln_sigma", "ln_tau", "ln_jitter"]
            ln_sigma_init = float(0.5 * ln_sigma2_init)
            ln_sigma_init = float(np.clip(ln_sigma_init, ln_sigma_min + 1e-6, ln_sigma_max - 1e-6))
            p0 = np.array([m0, ln_sigma_init, ln_tau_init, ln_jitter_init], dtype=float)

            def log_prob(theta):
                m, ln_sigma, ln_tau, ln_jitter = theta
                ln_sigma2 = 2.0 * ln_sigma
                lp = _log_prior_box_jitter_sigma2(
                    m,
                    ln_sigma2,
                    ln_tau,
                    ln_jitter,
                    ln_tau_bounds=(ln_tau_min, ln_tau_max),
                    ln_sigma2_bounds=(2.0 * ln_sigma_min, 2.0 * ln_sigma_max),
                )
                if not np.isfinite(lp):
                    return -np.inf
                ll = _loglike_from_sigma2_tau_jitter(
                    (m, ln_sigma2, ln_tau, ln_jitter), t_rest, y, yerr
                )
                return lp + ll

        elif parameterization == "lnD":
            param_names = ["m", "ln_D", "ln_tau", "ln_jitter"]
            ln_D_init = float(ln_sigma2_init - ln_tau_init)
            p0 = np.array([m0, ln_D_init, ln_tau_init, ln_jitter_init], dtype=float)

            def log_prob(theta):
                m, ln_D, ln_tau, ln_jitter = theta
                ln_sigma2 = ln_D + ln_tau
                if str(lnD_prior).lower() in {"flat", "box"}:
                    lp = _log_prior_box_jitter_lnD(
                        m,
                        ln_D,
                        ln_tau,
                        ln_jitter,
                        lnD_bounds=lnD_bounds,
                        ln_tau_bounds=(ln_tau_min, ln_tau_max),
                    )
                else:
                    lp = _log_prior_box_jitter_sigma2(
                        m,
                        ln_sigma2,
                        ln_tau,
                        ln_jitter,
                        ln_tau_bounds=(ln_tau_min, ln_tau_max),
                        ln_sigma2_bounds=(ln_sigma2_min, ln_sigma2_max),
                    )
                if not np.isfinite(lp):
                    return -np.inf
                ll = _loglike_from_sigma2_tau_jitter(
                    (m, ln_sigma2, ln_tau, ln_jitter), t_rest, y, yerr
                )
                return lp + ll

        else:
            raise ValueError("parameterization must be 'lnsigma2', 'lnsigma', or 'lnD'")

    else:
        ln_tau_fixed = float(np.log(float(tau_fixed)))
        if parameterization == "lnsigma2":
            param_names = ["m", "ln_sigma2", "ln_jitter"]
            p0 = np.array([m0, ln_sigma2_init, ln_jitter_init], dtype=float)

            def log_prob(theta):
                m, ln_sigma2, ln_jitter = theta
                lp = _log_prior_box_jitter_sigma2(
                    m,
                    ln_sigma2,
                    ln_tau_fixed,
                    ln_jitter,
                    ln_tau_bounds=(ln_tau_min, ln_tau_max),
                    ln_sigma2_bounds=(ln_sigma2_min, ln_sigma2_max),
                )
                if not np.isfinite(lp):
                    return -np.inf
                ll = _loglike_from_sigma2_tau_jitter(
                    (m, ln_sigma2, ln_tau_fixed, ln_jitter), t_rest, y, yerr
                )
                return lp + ll

        elif parameterization == "lnD":
            param_names = ["m", "ln_D", "ln_jitter"]
            ln_D_init = float(ln_sigma2_init - ln_tau_fixed)
            p0 = np.array([m0, ln_D_init, ln_jitter_init], dtype=float)

            def log_prob(theta):
                m, ln_D, ln_jitter = theta
                ln_sigma2 = ln_D + ln_tau_fixed
                if str(lnD_prior).lower() in {"flat", "box"}:
                    lp = _log_prior_box_jitter_lnD(
                        m,
                        ln_D,
                        ln_tau_fixed,
                        ln_jitter,
                        lnD_bounds=lnD_bounds,
                        ln_tau_bounds=(ln_tau_min, ln_tau_max),
                    )
                else:
                    lp = _log_prior_box_jitter_sigma2(
                        m,
                        ln_sigma2,
                        ln_tau_fixed,
                        ln_jitter,
                        ln_tau_bounds=(ln_tau_min, ln_tau_max),
                        ln_sigma2_bounds=(ln_sigma2_min, ln_sigma2_max),
                    )
                if not np.isfinite(lp):
                    return -np.inf
                ll = _loglike_from_sigma2_tau_jitter(
                    (m, ln_sigma2, ln_tau_fixed, ln_jitter), t_rest, y, yerr
                )
                return lp + ll

        else:
            raise ValueError("parameterization must be 'lnsigma2' or 'lnD'")

    ndim = len(param_names)
    p_init = p0 + 0.1 * np.abs(p0 + 1e-4) * rng.normal(size=(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.random_state = np.random.RandomState(sampler_seed.generate_state(4)).get_state()

    pos, _, _ = sampler.run_mcmc(p_init, nburn, progress=False)
    burn_chain = None
    burn_log_prob = None
    if return_chain:
        try:
            burn_chain = np.asarray(sampler.get_chain(), dtype=float)
        except Exception:
            burn_chain = None
        try:
            burn_log_prob = np.asarray(sampler.get_log_prob(), dtype=float)
        except Exception:
            burn_log_prob = None
    sampler.reset()

    # Optionally run adaptively in chunks after burn-in to reach a target number
    # of stored candidates estimated from the export-stride selection.
    target = int(nsamples_output) if target_thinned is None else int(target_thinned)
    target = max(1, target)
    if max_steps is None:
        max_steps = int(nsteps)
    max_steps = max(1, int(max_steps))
    if int(step_chunk) <= 0:
        raise ValueError("step_chunk must be > 0")

    steps_run = 0
    pos0 = pos
    if adaptive:
        while steps_run < max_steps:
            n_run = min(int(step_chunk), max_steps - steps_run)
            pos0, _, _ = sampler.run_mcmc(pos0, n_run, progress=False)
            steps_run += int(n_run)

            # Try to estimate IAT and see if we'd have enough thinned draws.
            try:
                tau_est = np.asarray(sampler.get_autocorr_time(tol=0), dtype=float)
                stride_iats = []
                for name, t in zip(param_names, tau_est):
                    if name in {"m", "ln_D", "ln_sigma2", "ln_jitter", "ln_tau"} and np.isfinite(t) and t > 0:
                        stride_iats.append(float(t))
                if stride_iats:
                    stride = max(1, int(np.ceil(max(stride_iats))))
                    # Count the least populated time offset so export cannot stop short.
                    n_thin_avail = (steps_run // stride) * nwalkers
                    if n_thin_avail >= target:
                        break
            except Exception:
                pass
    else:
        pos0, _, _ = sampler.run_mcmc(pos0, int(nsteps), progress=False)
        steps_run = int(nsteps)

    chain_full = sampler.get_chain()  # (nsteps_eff, nwalkers, ndim)
    if chain_full.size == 0:
        return None
    chain_full_arr = None
    log_prob_full_arr = None
    if return_chain:
        try:
            chain_full_arr = np.asarray(chain_full, dtype=float)
        except Exception:
            chain_full_arr = None
        try:
            log_prob_full_arr = np.asarray(sampler.get_log_prob(), dtype=float)
        except Exception:
            log_prob_full_arr = None

    chain_flat = sampler.get_chain(flat=True)
    n_chain = int(chain_flat.shape[0])

    # Diagnostics computed from the full post-burn chain, per parameter.
    # Note: emcee does not have "divergences" (HMC/NUTS concept). We report the
    # fraction of non-finite log_prob values as a proxy for numerical/pathology.
    diag = {
        "nwalkers": int(nwalkers),
        "nsteps": int(steps_run),
        "n_post": int(chain_full.shape[0] * chain_full.shape[1]),
    }

    try:
        logp = sampler.get_log_prob()  # (nsteps_eff, nwalkers)
        nonfinite = ~np.isfinite(logp)
        diag["logprob_nonfinite_count"] = int(np.count_nonzero(nonfinite))
        diag["logprob_nonfinite_frac"] = float(np.mean(nonfinite)) if logp.size else 0.0
    except Exception:
        diag["logprob_nonfinite_count"] = -1
        diag["logprob_nonfinite_frac"] = np.nan

    try:
        acc = np.asarray(sampler.acceptance_fraction, dtype=float)
        diag["accept_mean"] = float(np.nanmean(acc))
        diag["accept_min"] = float(np.nanmin(acc))
        diag["accept_max"] = float(np.nanmax(acc))
    except Exception:
        diag["accept_mean"] = np.nan
        diag["accept_min"] = np.nan
        diag["accept_max"] = np.nan

    iat = {}
    neff = {}
    try:
        tau = np.asarray(sampler.get_autocorr_time(tol=0), dtype=float)
        for name, t in zip(param_names, tau):
            iat[name] = float(t)
            neff[name] = float(diag["n_post"] / t) if np.isfinite(t) and t > 0 else np.nan
        diag["autocorr_ok"] = True
        diag["autocorr_error"] = ""
    except Exception as e:
        diag["autocorr_ok"] = False
        diag["autocorr_error"] = f"{type(e).__name__}: {e}"
        for name in param_names:
            iat[name] = np.nan
            neff[name] = np.nan

    diag["iat"] = iat
    diag["neff"] = neff

    # -------------------------------------------------------------------------
    # Thin post-burn time steps before pooling walkers using the estimated IAT stride.
    # -------------------------------------------------------------------------
    ns_out = int(nsamples_output)
    samples = np.full((ns_out, ndim), np.nan, dtype=float)
    thin_stride = np.nan
    n_thinned_available = -1
    n_thinned_used = -1

    def _stride_params(names: list[str]) -> list[str]:
        wanted: list[str] = []
        for k in ("m", "ln_D", "ln_sigma2", "ln_jitter"):
            if k in names:
                wanted.append(k)
        # If ln_tau is present (free-τ runs), include it as a conservative stride component.
        if "ln_tau" in names:
            wanted.append("ln_tau")
        return wanted

    if bool(diag.get("autocorr_ok", False)):
        params_for_stride = _stride_params(param_names)
        stride_iats = []
        for k in params_for_stride:
            v = iat.get(k, np.nan)
            try:
                v = float(v)
            except Exception:
                v = np.nan
            if np.isfinite(v) and v > 0:
                stride_iats.append(v)

        if stride_iats:
            stride = int(np.ceil(float(np.max(stride_iats))))
            stride = max(1, stride)
            thin_stride = float(stride)
            offset = int(rng.integers(0, stride)) if stride > 1 else 0
            thinned = np.asarray(chain_full[offset::stride].reshape(-1, ndim), dtype=float)
            n_thinned_available = int(thinned.shape[0])
            n_thinned_used = int(min(ns_out, n_thinned_available))
            if n_thinned_used > 0:
                if n_thinned_available >= ns_out:
                    idx = rng.choice(n_thinned_available, size=ns_out, replace=False)
                    samples = thinned[idx]
                else:
                    idx = np.arange(n_thinned_available)
                    rng.shuffle(idx)
                    samples[:n_thinned_available] = thinned[idx]

    if n_thinned_used < 0:
        # Randomly subsample the correlated chain when IAT thinning is unavailable.
        n_thinned_available = int(n_chain)
        n_thinned_used = int(ns_out)
        replace = n_chain < ns_out
        idx = rng.choice(n_chain, size=ns_out, replace=replace)
        samples = np.asarray(chain_flat[idx], dtype=float)

    diag["thin_stride"] = float(thin_stride)
    diag["n_thinned_available"] = int(n_thinned_available)
    diag["n_thinned_used"] = int(n_thinned_used)

    # Reporting only: derive D diagnostics after stopping and posterior selection.
    if "ln_D" not in param_names:
        amplitude = "ln_sigma" if "ln_sigma" in param_names else "ln_sigma2"
        factor = 2.0 if amplitude == "ln_sigma" else 1.0
        ln_D_chain = factor * chain_full[:, :, param_names.index(amplitude)]
        if "ln_tau" in param_names:
            ln_D_chain = ln_D_chain - chain_full[:, :, param_names.index("ln_tau")]
        else:
            ln_D_chain = ln_D_chain - np.log(float(tau_fixed))
        try:
            tau_D = float(emcee.autocorr.integrated_time(ln_D_chain, tol=0)[0])
            if not np.isfinite(tau_D) or tau_D <= 0:
                raise ValueError("Derived ln_D IAT is not finite and positive")
        except Exception as error:
            iat["ln_D"] = np.nan
            neff["ln_D"] = np.nan
            diag["ln_D_autocorr_error"] = f"{type(error).__name__}: {error}"
        else:
            iat["ln_D"] = tau_D
            neff["ln_D"] = float(diag["n_post"] / tau_D)

    out = {
        "t_rest_npts": int(t_rest.size),
        "baseline_rest": baseline_rf,
        "samples": samples,
        "param_names": param_names,
        "diag": diag,
    }
    if mwin_mode != "none":
        # Compute posterior for a time-average window-mean latent magnitude:
        #   trap: m_win = w^T f(t_i), with quadrature weights at observed epochs
        #   cont: m_win_cont = (1 / T) int f(t) dt over the observed window
        want_trap = mwin_mode in {"trap", "both"}
        want_cont = mwin_mode in {"cont", "both"}
        w = _trapezoid_time_weights(t_rest) if want_trap else None
        diag0 = np.zeros_like(t_rest, dtype=float)

        # Deterministic RNG independent of the sampler thinning RNG.
        rng_mwin = np.random.default_rng(trap_seed) if want_trap else None
        rng_mwin_cont = np.random.default_rng(cont_seed) if want_cont else None
        window_a = float(np.min(t_rest))
        window_b = float(np.max(t_rest))

        idx_m = param_names.index("m") if "m" in param_names else None
        idx_lnjit = param_names.index("ln_jitter") if "ln_jitter" in param_names else None
        idx_lntau = param_names.index("ln_tau") if "ln_tau" in param_names else None
        idx_lns2 = param_names.index("ln_sigma2") if "ln_sigma2" in param_names else None
        idx_lnsig = param_names.index("ln_sigma") if "ln_sigma" in param_names else None
        idx_lnD = param_names.index("ln_D") if "ln_D" in param_names else None
        ln_tau_fixed_local = None if tau_fixed is None else float(np.log(float(tau_fixed)))

        mwin = np.full(ns_out, np.nan, dtype=float) if want_trap else None
        mwin_mu = np.full(ns_out, np.nan, dtype=float) if want_trap else None
        mwin_var = np.full(ns_out, np.nan, dtype=float) if want_trap else None
        mwin_cont = np.full(ns_out, np.nan, dtype=float) if want_cont else None
        mwin_mu_cont = np.full(ns_out, np.nan, dtype=float) if want_cont else None
        mwin_var_cont = np.full(ns_out, np.nan, dtype=float) if want_cont else None

        for i_s in range(ns_out):
            theta = np.asarray(samples[i_s], dtype=float)
            if theta.size != ndim or not np.all(np.isfinite(theta)):
                continue
            if idx_m is None or idx_lnjit is None:
                continue

            m = float(theta[idx_m])
            ln_jitter = float(theta[idx_lnjit])

            if ln_tau_fixed_local is None:
                if idx_lntau is None:
                    continue
                ln_tau = float(theta[idx_lntau])
            else:
                ln_tau = float(ln_tau_fixed_local)

            if idx_lns2 is not None:
                ln_sigma2 = float(theta[idx_lns2])
            elif idx_lnsig is not None:
                ln_sigma2 = float(2.0 * theta[idx_lnsig])
            elif idx_lnD is not None:
                ln_sigma2 = float(theta[idx_lnD] + ln_tau)
            else:
                continue

            try:
                gp = _build_gp_drw_jitter(m, ln_sigma2, ln_tau, ln_jitter, t_rest, yerr)

                # Efficient exact computation for linear functionals of f under
                # the GP posterior f | y, theta, without forming the full N x N
                # posterior covariance.
                #
                # Let K be the latent covariance at t_rest and N = diag(yerr^2 + jitter^2).
                # Then:
                #   mu  = m + c^T (K+N)^{-1} (y - m)
                #   var = v - c^T (K+N)^{-1} c
                #
                # celerite2 provides:
                #   - gp.apply_inverse(v) = (K+N)^{-1} v
                #   - kernel.dot(t, diag, v) = (K + diag) v
                kern = gp.kernel
                alpha = np.asarray(gp.apply_inverse(np.asarray(y, dtype=float) - float(m)), dtype=float)

                if want_trap:
                    Kw = np.asarray(kern.dot(t_rest, diag0, w), dtype=float)  # K @ w
                    Kalpha = np.asarray(kern.dot(t_rest, diag0, alpha), dtype=float)  # K @ alpha
                    mu = float(float(m) + float(w @ Kalpha))

                    beta = np.asarray(gp.apply_inverse(Kw), dtype=float)
                    var = float(float(w @ Kw) - float(Kw @ beta))
                    if not np.isfinite(var) or var < 0:
                        var = 0.0
                    mwin_mu[i_s] = mu
                    mwin_var[i_s] = var
                    mwin[i_s] = float(rng_mwin.normal(loc=mu, scale=np.sqrt(var)))

                if want_cont:
                    c_win, v_win = drw_window_integral_covariances(
                        t_rest,
                        sigma2=float(np.exp(ln_sigma2)),
                        tau=float(np.exp(ln_tau)),
                        a=window_a,
                        b=window_b,
                    )
                    mu_c = float(float(m) + float(c_win @ alpha))
                    beta_c = np.asarray(gp.apply_inverse(c_win), dtype=float)
                    var_c = float(float(v_win) - float(c_win @ beta_c))
                    if not np.isfinite(var_c) or var_c < 0:
                        var_c = 0.0
                    mwin_mu_cont[i_s] = mu_c
                    mwin_var_cont[i_s] = var_c
                    mwin_cont[i_s] = float(rng_mwin_cont.normal(loc=mu_c, scale=np.sqrt(var_c)))
            except Exception:
                continue

        extra_names = []
        extra_cols = []
        if want_trap:
            extra_names.extend(["m_win", "m_win_mu", "m_win_var"])
            extra_cols.extend([mwin, mwin_mu, mwin_var])
        if want_cont:
            extra_names.extend(["m_win_cont", "m_win_mu_cont", "m_win_var_cont"])
            extra_cols.extend([mwin_cont, mwin_mu_cont, mwin_var_cont])
        param_names = list(param_names) + extra_names
        samples = np.column_stack([samples, *extra_cols])
        out["param_names"] = param_names
        out["samples"] = samples
    if return_chain:
        out["burn_chain"] = burn_chain
        out["burn_log_prob"] = burn_log_prob
        out["post_chain"] = chain_full_arr
        out["post_log_prob"] = log_prob_full_arr
    return out
