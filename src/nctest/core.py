"""Building blocks of the noise-corrected test.

Notation follows Fallahgoul (2026). At each forecast origin t the benchmark forecast is
yhat1_t and the candidate forecast is yhat2_t. Both are linear in the T training targets, so
the increment yhat2_t - yhat1_t = a_t' Y_{t-1} for a weight vector a_t fixed by the design and
the penalty. Then

    Psihat_t = ||a_t||^2
    d^DM_t   = e1_t^2 - e2_t^2
    d^CW_t   = e1_t^2 - [e2_t^2 - (yhat1_t - yhat2_t)^2]
    d^CC_t   = e1_t^2 - e2_t^2 + sigma2_t * Psihat_t

and the corrected test rejects H0: 2E[fg] <= E[g^2] when t_CC exceeds the one-sided normal
critical value.
"""
import math

import numpy as np

CRIT_5 = 1.6448536269514722          # one-sided 5% normal critical value


def default_bandwidth(n):
    """Bartlett bandwidth ceil(4 (n/100)^(2/9)), the rule used in the paper."""
    return max(1, int(math.ceil(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def bartlett_lrv(x, bandwidth):
    """Bartlett long-run variance of x, with sample autocovariances normalised by n."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    s = x @ x / n
    for l in range(1, int(bandwidth) + 1):
        s += 2.0 * (1.0 - l / (bandwidth + 1.0)) * (x[l:] @ x[:-l]) / n
    return s


def psihat(a):
    """Psihat_t = ||a_t||^2 for an (n, T) array of increment weights."""
    a = np.atleast_2d(np.asarray(a, float))
    return np.einsum("ij,ij->i", a, a)


def ridge_increment_weights(X, s, z, H=None):
    """Weights a_t of a ridge increment over a benchmark that projects on H.

    X (T, P) training features, s (P,) the feature vector at the forecast origin, z the ridge
    penalty in eigenvalue units of X'X/T, H (T, q) benchmark regressors or None for the zero
    benchmark. The ridge is solved on the raw design and its weights are residualised on H,
    a_t = M_t w_{t-1}. This is the route of the paper's Monte Carlo (Section 5). For a ridge
    fitted to the benchmark's residuals, as in Section 6, use nctest.rolling_ridge.
    """
    X = np.asarray(X, float)
    T, P = X.shape
    A = np.linalg.solve(z * np.eye(P) + X.T @ X / T, np.asarray(s, float))
    w = X @ A / T
    if H is None:
        return w
    H = np.asarray(H, float).reshape(T, -1)
    return w - H @ np.linalg.lstsq(H, w, rcond=None)[0]


def sigma2_window(Ywin, Hwin=None):
    """Window-residual estimator ||M_t Y_{t-1}||^2 / (T - q).

    Ywin (n, T) training targets for each origin; Hwin (n, T, q) benchmark regressors, or
    None for a rolling-mean benchmark (q = 1).
    """
    Y = np.atleast_2d(np.asarray(Ywin, float))
    n, T = Y.shape
    if Hwin is None:
        r = Y - Y.mean(axis=1, keepdims=True)
        return np.einsum("ij,ij->i", r, r) / (T - 1)
    out = np.empty(n)
    for t in range(n):
        H = np.asarray(Hwin[t], float).reshape(T, -1)
        r = Y[t] - H @ np.linalg.lstsq(H, Y[t], rcond=None)[0]
        out[t] = r @ r / (T - H.shape[1])
    return out


def sigma2_difference(y):
    """Difference estimator (1/2n) sum (y_t - y_{t-1})^2 and the per-period terms
    h_t = (y_t - y_{t-1})^2 / 2 that enter the long-run variance. The first h_t is set to the
    mean of the rest."""
    y = np.asarray(y, float)
    h = 0.5 * np.r_[0.0, np.diff(y) ** 2]
    h[0] = h[1:].mean()
    return 0.5 * np.mean(np.diff(y) ** 2), h


def cc_test(y, yhat1, yhat2, psi, sigma2, bandwidth=None, influence=None, crit=CRIT_5):
    """The noise-corrected test.

    y, yhat1, yhat2, psi  length-n arrays aligned on forecast origin.
    sigma2     scalar (a global estimator or a known value) or length-n array (window
               estimator).
    bandwidth  Bartlett bandwidth; None uses default_bandwidth(n).
    influence  per-period terms h_t of a GLOBAL variance estimator. When given, the long-run
               variance is that of d^CC_t - mean + mean(Psihat)(h_t - mean(h)). Leave None for
               a known sigma2 or for the window estimator, whose estimation error is already
               part of d^CC_t.
    Returns the mean differential, its long-run variance, t_CC and the decision.
    """
    y, yhat1, yhat2, psi = (np.asarray(v, float) for v in (y, yhat1, yhat2, psi))
    if bandwidth is None:
        bandwidth = default_bandwidth(len(y))
    e1, e2 = y - yhat1, y - yhat2
    d = e1 ** 2 - e2 ** 2 + np.asarray(sigma2, float) * psi
    score = d if influence is None else (
        d - d.mean() + psi.mean() * (np.asarray(influence, float) - np.mean(influence)))
    lrv = max(bartlett_lrv(score, bandwidth), 1e-14)
    t = math.sqrt(len(d)) * d.mean() / math.sqrt(lrv)
    return dict(mean=float(d.mean()), lrv=float(lrv), t=float(t), reject=bool(t > crit))


def dm_cw(y, yhat1, yhat2, bandwidth=None):
    """Diebold-Mariano and Clark-West t-statistics. With bandwidth=None the sample variance
    studentises (one-step forecasts); otherwise the Bartlett long-run variance."""
    y, yhat1, yhat2 = (np.asarray(v, float) for v in (y, yhat1, yhat2))
    e1, e2 = y - yhat1, y - yhat2
    dDM = e1 ** 2 - e2 ** 2
    dCW = dDM + (yhat1 - yhat2) ** 2
    n = len(y)

    def t(x):
        v = x.var() if bandwidth is None else bartlett_lrv(x, bandwidth)
        return math.sqrt(n) * x.mean() / math.sqrt(max(v, 1e-14))
    return dict(t_DM=t(dDM), t_CW=t(dCW))


def sanity_ratio(yhat1, yhat2, psi, sigma2):
    """R = mean(Delta_t^2) / mean(sigma2_t Psihat_t). It cannot fall below one in population;
    a realised value below one says the correction is mis-scaled."""
    D = np.asarray(yhat2, float) - np.asarray(yhat1, float)
    return float(np.mean(D ** 2) / np.mean(np.asarray(sigma2, float) * np.asarray(psi, float)))


def intersection(test_a, test_b):
    """Reject only if both tests reject."""
    return bool(test_a["reject"] and test_b["reject"])
