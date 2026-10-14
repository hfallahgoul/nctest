"""Design matrices for the HAR comparison of the paper's Section 6."""
import numpy as np


def har_design(x, extra_lags=0):
    """HAR benchmark and own-lag candidate from a series x (for example log realized variance).

    Returns (y, H, X) aligned for nctest.compare, with y[t] = x[t] and
        H[t] = [1, x[t-1], mean(x[t-5:t]), mean(x[t-22:t])]      the HAR of Corsi (2009)
        X[t] = [x[t-2], ..., x[t-extra_lags-1]]                    lags 2..P+1
    Rows without a full history are dropped, so y starts at x[max(22, extra_lags + 1)].
    """
    x = np.asarray(x, float).ravel()
    N, P = len(x), int(extra_lags)
    start = max(22, P + 1)
    t = np.arange(start, N)
    c = np.r_[0.0, np.cumsum(x)]
    mean_k = lambda k: (c[t] - c[t - k]) / k
    H = np.column_stack([np.ones(len(t)), x[t - 1], mean_k(5), mean_k(22)])
    X = np.column_stack([x[t - k] for k in range(2, P + 2)]) if P else np.empty((len(t), 0))
    return x[t], H, X
