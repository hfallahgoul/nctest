"""Diebold-Mariano and Clark-West tests, usable on their own.

Both take two forecasts of the same target and need nothing else: no increment weights, no
variance estimate. Use them when you want the classical tests without the noise correction,
or to report them beside it.

    res = nctest.dm_test(y, yhat1, yhat2, horizon=1)
    print(res)              # statistic, p-value, decision
    res.p_value

Sign convention, as everywhere in nctest: model 1 is the benchmark, model 2 the candidate,
and a positive statistic favours the candidate.
"""
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .core import default_bandwidth

_ALTERNATIVES = ("greater", "less", "two-sided")


# --------------------------------------------------------------------------- distributions
def normal_cdf(x):
    """Standard normal distribution function."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _betacf(a, b, x, maxit=500, eps=1e-15):
    """Continued fraction for the regularised incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
           + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_cdf(x, df):
    """Student-t distribution function with df degrees of freedom."""
    tail = 0.5 * _betainc(0.5 * df, 0.5, df / (df + x * x))
    return 1.0 - tail if x > 0 else tail


def _p_value(stat, alternative, cdf):
    if alternative == "greater":
        return 1.0 - cdf(stat)
    if alternative == "less":
        return cdf(stat)
    return min(1.0, 2.0 * min(cdf(stat), 1.0 - cdf(stat)))


# ------------------------------------------------------------------------- long-run variance
def long_run_variance(x, bandwidth, kernel="bartlett"):
    """Long-run variance of x with autocovariances up to lag `bandwidth`, normalised by n.

    kernel "bartlett" weights lag l by 1 - l/(bandwidth + 1) (Newey and West, 1987) and is
    never negative. "uniform" gives every lag weight one, as in Diebold and Mariano (1995);
    it can be negative in small samples. bandwidth 0 gives the sample variance.
    """
    if kernel not in ("bartlett", "uniform"):
        raise ValueError("kernel must be 'bartlett' or 'uniform'")
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    s = x @ x / n
    for l in range(1, int(bandwidth) + 1):
        w = 1.0 - l / (bandwidth + 1.0) if kernel == "bartlett" else 1.0
        s += 2.0 * w * (x[l:] @ x[:-l]) / n
    return float(s)


def _variance(d, horizon, bandwidth, kernel, default_kernel, notes):
    n = len(d)
    if bandwidth is None:
        b, k = horizon - 1, kernel or default_kernel
    elif bandwidth == "auto":
        b, k = default_bandwidth(n), kernel or "bartlett"
    else:
        b, k = int(bandwidth), kernel or "bartlett"
    if b < 0:
        raise ValueError("bandwidth must be non-negative")
    v = long_run_variance(d, b, k)
    if v <= 0 and k == "uniform":
        notes.append("the uniform-kernel long-run variance was not positive; the Bartlett "
                     "kernel was used instead")
        k = "bartlett"
        v = long_run_variance(d, b, k)
    return max(v, 1e-14), b, k


# ----------------------------------------------------------------------------------- results
@dataclass
class TestResult:
    """Outcome of a single test. `statistic` is the studentised mean of the loss
    differential d_t; a positive value favours the candidate (model 2)."""
    name: str
    statistic: float
    p_value: float
    reject: bool
    alternative: str
    level: float
    mean: float                  # mean of d_t
    variance: float              # long-run variance of d_t used to studentise
    n: int
    horizon: int
    bandwidth: int
    kernel: str
    distribution: str            # "N(0,1)" or "t(n-1)"
    notes: list = field(default_factory=list)

    __test__ = False             # keep pytest from collecting this class

    def as_dict(self):
        return asdict(self)

    def __str__(self):
        alt = {"greater": "candidate better", "less": "benchmark better",
               "two-sided": "forecasts differ"}[self.alternative]
        lines = [
            f"{self.name}",
            f"  n = {self.n}, horizon = {self.horizon}, {self.kernel} kernel, "
            f"bandwidth = {self.bandwidth}",
            f"  mean differential {self.mean:.6g}, long-run variance {self.variance:.6g}",
            f"  statistic {self.statistic:.3f}, p-value {self.p_value:.4f} "
            f"({self.alternative}: {alt}; {self.distribution})",
            f"  {'reject' if self.reject else 'do not reject'} H0 at the "
            f"{100 * self.level:g}% level",
        ]
        lines += [f"  note: {s}" for s in self.notes]
        return "\n".join(lines)


def _check(y, yhat1, yhat2, horizon, alternative, level):
    y, yhat1, yhat2 = (np.asarray(v, float).ravel() for v in (y, yhat1, yhat2))
    if not (len(y) == len(yhat1) == len(yhat2)):
        raise ValueError("y, yhat1 and yhat2 must have the same length")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(yhat1))
            and np.all(np.isfinite(yhat2))):
        raise ValueError("inputs contain NaN or infinite values")
    if int(horizon) != horizon or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}")
    if not 0 < level < 1:
        raise ValueError("level must lie in (0, 1)")
    if len(y) < 2 * horizon + 2:
        raise ValueError("too few forecasts for this horizon")
    return y, yhat1, yhat2, int(horizon)


_LOSSES = {"squared": lambda e: e ** 2, "absolute": np.abs}


# ------------------------------------------------------------------------------------- tests
def dm_test(y, yhat1, yhat2, loss="squared", horizon=1, alternative="greater",
            bandwidth=None, kernel=None, hln=False, level=0.05):
    """Diebold-Mariano (1995) test of equal predictive accuracy.

    y, yhat1, yhat2  realised values and the benchmark and candidate forecasts, aligned.
    loss         "squared", "absolute", or a function of the forecast error returning losses.
    horizon      forecast horizon h. Under the null, d_t is serially correlated up to lag h-1.
    alternative  "greater" (candidate more accurate, the default), "less" or "two-sided".
    bandwidth    None uses h-1 lags, the DM choice; "auto" uses the Newey-West (1994) rule
                 ceil(4 (n/100)^(2/9)); an integer sets it directly.
    kernel       None means "uniform" with the default bandwidth and "bartlett" otherwise.
                 A non-positive uniform estimate falls back to Bartlett, with a note.
    hln          apply the small-sample correction of Harvey, Leybourne and Newbold (1997)
                 and compare with t(n-1).
    level        size of the test.

    With nested models DM is undersized against a null of equal population accuracy (Clark
    and McCracken, 2001); use cw_test for the population question there, and nctest's
    corrected test for accuracy net of estimation noise.
    """
    y, yhat1, yhat2, h = _check(y, yhat1, yhat2, horizon, alternative, level)
    L = _LOSSES.get(loss, loss) if isinstance(loss, str) else loss
    if isinstance(L, str) or not callable(L):
        raise ValueError("loss must be 'squared', 'absolute' or a callable")
    d = np.asarray(L(y - yhat1), float) - np.asarray(L(y - yhat2), float)
    n, notes = len(d), []
    v, b, k = _variance(d, h, bandwidth, kernel, "uniform", notes)
    stat = math.sqrt(n) * d.mean() / math.sqrt(v)
    if hln:
        stat *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        cdf, dist = (lambda s: t_cdf(s, n - 1)), "t(n-1)"
    else:
        cdf, dist = normal_cdf, "N(0,1)"
    p = _p_value(stat, alternative, cdf)
    name = "Diebold-Mariano test" + (" (HLN-corrected)" if hln else "")
    return TestResult(name, float(stat), float(p), bool(p < level), alternative, level,
                      float(d.mean()), float(v), n, h, b, k, dist, notes)


def cw_test(y, yhat1, yhat2, horizon=1, bandwidth=None, kernel="bartlett", level=0.05):
    """Clark-West (2007) test for nested models.

    The benchmark (model 1) must be nested in the candidate (model 2). The differential is
    d_t = e1_t^2 - [e2_t^2 - (yhat1_t - yhat2_t)^2], and the test is one-sided: it rejects
    when the extra predictors improve the population forecast, a hypothesis on the
    covariance of the forecast increment with the target. It does not ask whether the
    candidate forecasts better at the estimation window in use; for that, see the paper and
    nctest.cc_test.

    horizon    forecast horizon h.
    bandwidth  None uses h-1 lags, so the sample variance at h = 1, as Clark and West
               recommend for one-step forecasts; "auto" or an integer as in dm_test.
    kernel     "bartlett" (default) or "uniform".
    """
    y, yhat1, yhat2, h = _check(y, yhat1, yhat2, horizon, "greater", level)
    d = (y - yhat1) ** 2 - ((y - yhat2) ** 2 - (yhat1 - yhat2) ** 2)
    n, notes = len(d), []
    v, b, k = _variance(d, h, bandwidth, kernel, "bartlett", notes)
    stat = math.sqrt(n) * d.mean() / math.sqrt(v)
    p = 1.0 - normal_cdf(stat)
    return TestResult("Clark-West test", float(stat), float(p), bool(p < level), "greater",
                      level, float(d.mean()), float(v), n, h, b, k, "N(0,1)", notes)
