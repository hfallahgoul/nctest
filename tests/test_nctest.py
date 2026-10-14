"""Tests for nctest. Reference values come from the paper's replication code
(ccpa.py and rv_application.rolling), which is itself checked against the simulation engine.

    pytest -q
"""
import numpy as np
import pytest

import nctest


def _mc_path():
    """Rolling-mean benchmark with the Monte Carlo ridge increment, as in the paper's
    Section 5."""
    g = np.random.default_rng(2026)
    N, T, P, z = 700, 24, 12, 0.5
    S = g.standard_normal((N, P))
    y = 0.3 * S @ np.r_[0.4, np.zeros(P - 1)] + g.standard_normal(N)
    yt, yh1, yh2, psi, Yw = [], [], [], [], []
    for i in range(N - T - 1):
        X, Y, s = S[i:i + T], y[i + 1:i + T + 1], S[T + i]
        a = nctest.ridge_increment_weights(X, s, z, H=np.ones(T))
        yt.append(y[T + 1 + i]); yh1.append(Y.mean()); yh2.append(Y.mean() + a @ Y)
        psi.append(a @ a); Yw.append(Y)
    return (*map(np.array, (yt, yh1, yh2, psi)), np.array(Yw))


def test_core_matches_replication_code():
    yt, yh1, yh2, psi, Yw = _mc_path()
    b = 18
    oracle = nctest.cc_test(yt, yh1, yh2, psi, 1.0, b)
    s2d, h = nctest.sigma2_difference(yt)
    diff = nctest.cc_test(yt, yh1, yh2, psi, s2d, b, influence=h)
    win = nctest.cc_test(yt, yh1, yh2, psi, nctest.sigma2_window(Yw), b)
    dc = nctest.dm_cw(yt, yh1, yh2)
    ref = dict(oracle=0.26333294823869136, diff=0.18819163230051383,
               win=-0.10692959345199678, dm=-5.596553336671523, cw=-0.21361578131011633,
               ratio=0.9241379789486317)
    got = dict(oracle=oracle["t"], diff=diff["t"], win=win["t"], dm=dc["t_DM"],
               cw=dc["t_CW"], ratio=nctest.sanity_ratio(yh1, yh2, psi, 1.0))
    for k in ref:
        assert got[k] == pytest.approx(ref[k], rel=1e-9, abs=1e-12), k


def _ar_series(n=600, seed=3):
    g = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + g.standard_normal()
    return x


def test_rolling_matches_replication_code():
    y, H, X = nctest.har_design(_ar_series(), extra_lags=9)
    p = nctest.rolling_ridge(y, X, benchmark=H, window=120, penalty=0.05)
    assert len(p.y) == 458
    ref = {0: (0.22340009498858307, -0.33996805240219363, 0.09855704230582726, 1.1051096659985664),
           100: (-2.7542981492724374, -3.199812453471863, 0.2265592203837426, 0.8901201179850589),
           457: (1.8124423020180225, 1.8011824998522026, 0.021501936715729326, 0.9214238190423084)}
    for i, (yb, yh, psi, s2) in ref.items():
        assert p.yhat1[i] == pytest.approx(yb, rel=1e-9)
        assert p.yhat2[i] == pytest.approx(yh, rel=1e-9)
        assert p.psi[i] == pytest.approx(psi, rel=1e-8)
        assert p.sigma2_w[i] == pytest.approx(s2, rel=1e-9)


def test_dual_branch_equals_primal():
    """P > T uses the T x T system; it must give the same forecasts as the P x P one."""
    g = np.random.default_rng(1)
    N, P = 140, 60
    X, y = g.standard_normal((N, P)), g.standard_normal(N)
    wide = nctest.rolling_ridge(y, X, window=40, penalty=0.3)          # P > T: dual
    narrow = nctest.rolling_ridge(y, X, window=100, penalty=0.3)       # P < T: primal
    # same windows solved both ways, through the core helper on raw features
    for p in (wide, narrow):
        assert np.all(np.isfinite(p.yhat2)) and np.all(p.psi > 0)
    i = 0
    T = 40
    Xe = X[i:i + T]; mu, sd = Xe.mean(0), Xe.std(0)
    Z = (Xe - mu) / sd; Z = Z - Z.mean(0); s = (X[T] - mu) / sd
    a_primal = Z @ np.linalg.solve(Z.T @ Z / T + 0.3 * np.eye(P), s) / T
    assert wide.psi[0] == pytest.approx(a_primal @ a_primal, rel=1e-9)
    assert wide.yhat2[0] - wide.yhat1[0] == pytest.approx(a_primal @ y[:T], rel=1e-9)


def test_har_design_alignment():
    x = np.arange(60.0)
    y, H, X = nctest.har_design(x, extra_lags=9)
    assert y[0] == 22.0
    assert H[0].tolist() == [1.0, 21.0, np.mean(x[17:22]), np.mean(x[0:22])]
    assert X[0].tolist() == [20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0]


def test_compare_runs_and_reports():
    y, H, X = nctest.har_design(_ar_series(), extra_lags=24)
    res = nctest.compare(y, X, benchmark=H, window=120, penalty=0.01)
    assert res.n == len(y) - 120
    assert res.reject == (res.reject_window and res.reject_difference)
    text = str(res)
    for word in ("Diebold-Mariano", "Clark-West", "CC, window variance", "sanity ratio"):
        assert word in text


def test_benchmarks_and_errors():
    g = np.random.default_rng(0)
    y, X = g.standard_normal(300), g.standard_normal((300, 5))
    zero = nctest.rolling_ridge(y, X, benchmark="zero", window=50)
    assert np.all(zero.yhat1 == 0.0)
    mean = nctest.rolling_ridge(y, X, benchmark="mean", window=50)
    assert mean.yhat1[0] == pytest.approx(y[:50].mean())
    with pytest.raises(ValueError):
        nctest.rolling_ridge(y, X, window=400)
    with pytest.raises(ValueError):
        nctest.rolling_ridge(np.r_[y[:-1], np.nan], X, window=50)
