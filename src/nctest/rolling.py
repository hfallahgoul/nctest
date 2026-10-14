"""Rolling-window forecasts with the increment weights the test needs."""
from dataclasses import dataclass

import numpy as np


@dataclass
class ForecastPath:
    """Out-of-sample forecasts aligned on forecast origin.

    origin   row index of each forecast in the input arrays
    y        realised targets
    yhat1    benchmark forecasts
    yhat2    candidate forecasts
    psi      Psihat_t = ||a_t||^2
    sigma2_w window-residual variance of the benchmark, ||M_t Y||^2 / (T - q)
    """
    origin: np.ndarray
    y: np.ndarray
    yhat1: np.ndarray
    yhat2: np.ndarray
    psi: np.ndarray
    sigma2_w: np.ndarray
    window: int
    n_features: int
    penalty: float


def _benchmark_matrix(benchmark, N):
    if isinstance(benchmark, str):
        if benchmark == "mean":
            return np.ones((N, 1))
        if benchmark == "zero":
            return None
        raise ValueError("benchmark must be 'mean', 'zero' or an (N, q) array")
    B = np.asarray(benchmark, float)
    B = B.reshape(N, -1) if B.ndim == 1 else B
    if B.shape[0] != N:
        raise ValueError(f"benchmark has {B.shape[0]} rows, y has {N}")
    return B


def rolling_ridge(y, X, benchmark="mean", window=250, penalty=1.0, standardize=True):
    """Benchmark fitted by least squares, candidate = benchmark + ridge on its residuals.

    Row t of X (and of the benchmark matrix) holds the predictors used to forecast y[t], so
    lag the predictors before calling. At each origin t >= window the models are fitted on
    rows t - window, ..., t - 1 and forecast y[t].

    y          (N,) target.
    X          (N, P) the candidate's additional predictors.
    benchmark  'mean' (rolling mean), 'zero' (forecast zero), or an (N, q) array of benchmark
               regressors fitted by least squares on each window. Include a column of ones
               if the benchmark has an intercept.
    window     estimation window T.
    penalty    ridge penalty z in the objective (1/T) sum (u_s - x_s' g)^2 + z ||g||^2, where
               u_s are the benchmark's in-window residuals and x_s the predictors residualised
               on the benchmark regressors. The paper reports ztil = z T / P.
    standardize  scale X with each window's own mean and standard deviation (no look-ahead).

    The candidate forecast is yhat2_t = yhat1_t + a_t' Y with a_t = M Z (Z'MZ/T + zI)^{-1} s_t / T,
    so the two models are nested and Psihat_t = ||a_t||^2 is exact. This is the design of
    Section 6 of the paper.
    """
    y = np.asarray(y, float).ravel()
    N = len(y)
    X = np.asarray(X, float)
    X = X.reshape(N, -1) if X.ndim == 1 else X
    if X.shape[0] != N:
        raise ValueError(f"X has {X.shape[0]} rows, y has {N}")
    B = _benchmark_matrix(benchmark, N)
    T, P, z = int(window), X.shape[1], float(penalty)
    if N <= T:
        raise ValueError(f"need more than window={T} observations, got {N}")
    if z <= 0:
        raise ValueError("penalty must be positive")
    q = 0 if B is None else B.shape[1]
    bad = ~np.isfinite(y) | ~np.isfinite(X).all(1)
    if B is not None:
        bad |= ~np.isfinite(B).all(1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows contain NaN or inf; drop them first")

    n = N - T
    yb, yh, psi, s2w = (np.empty(n) for _ in range(4))
    for i in range(n):
        Y, Xe, s = y[i:i + T], X[i:i + T], X[i + T]
        if standardize:
            mu, sd = Xe.mean(0), Xe.std(0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            Xe, s = (Xe - mu) / sd, (s - mu) / sd
        if B is None:
            yb[i], resid, Z = 0.0, Y, Xe
        else:
            Q, R = np.linalg.qr(B[i:i + T])
            beta = np.linalg.lstsq(R, Q.T @ Y, rcond=None)[0]
            yb[i] = B[i + T] @ beta
            resid = Y - Q @ (Q.T @ Y)
            Z = Xe - Q @ (Q.T @ Xe)
        s2w[i] = resid @ resid / max(T - q, 1)
        if P <= T:
            a = Z @ np.linalg.solve(Z.T @ Z / T + z * np.eye(P), s) / T
        else:
            a = np.linalg.solve(Z @ Z.T / T + z * np.eye(T), Z @ s) / T
        yh[i] = yb[i] + a @ Y
        psi[i] = a @ a
    return ForecastPath(origin=np.arange(T, N), y=y[T:], yhat1=yb, yhat2=yh, psi=psi,
                        sigma2_w=s2w, window=T, n_features=P, penalty=z)
