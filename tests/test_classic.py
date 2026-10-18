"""Tests for the standalone Diebold-Mariano and Clark-West tests (nctest 0.2)."""
import math

import numpy as np
import pytest

import nctest


def test_distributions_match_reference_values():
    # reference values from scipy.stats
    assert nctest.t_cdf(2.0, 10) == pytest.approx(0.9633059826146299, abs=1e-12)
    assert nctest.t_cdf(-1.5, 3) == pytest.approx(0.11529193262241147, abs=1e-12)
    assert nctest.t_cdf(0.3, 250) == pytest.approx(0.6177867754919752, abs=1e-12)
    assert nctest.t_cdf(4.0, 1) == pytest.approx(0.9220208696226306, abs=1e-12)
    assert 1 - nctest.normal_cdf(1.645) == pytest.approx(0.049984905539121376, abs=1e-13)


def _forecasts(n=400, seed=7):
    g = np.random.default_rng(seed)
    x = g.standard_normal(n)
    y = 0.3 * x + g.standard_normal(n)
    return y, np.zeros(n), 0.25 * x + 0.1 * g.standard_normal(n)


def test_one_step_agrees_with_dm_cw():
    y, f1, f2 = _forecasts()
    old = nctest.dm_cw(y, f1, f2)
    assert nctest.dm_test(y, f1, f2).statistic == pytest.approx(old["t_DM"], rel=1e-12)
    assert nctest.cw_test(y, f1, f2).statistic == pytest.approx(old["t_CW"], rel=1e-12)


def test_multistep_uniform_and_hln_by_hand():
    y, f1, f2 = _forecasts()
    h = 3
    d = (y - f1) ** 2 - (y - f2) ** 2
    n, x = len(d), d - d.mean()
    v = x @ x / n + 2 * sum(x[l:] @ x[:-l] / n for l in range(1, h))
    t = math.sqrt(n) * d.mean() / math.sqrt(v)
    res = nctest.dm_test(y, f1, f2, horizon=h)
    assert (res.kernel, res.bandwidth) == ("uniform", h - 1)
    assert res.statistic == pytest.approx(t, rel=1e-12)
    hln = nctest.dm_test(y, f1, f2, horizon=h, hln=True)
    k = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    assert hln.statistic == pytest.approx(t * k, rel=1e-12)
    assert hln.p_value == pytest.approx(1 - nctest.t_cdf(t * k, n - 1), rel=1e-12)
    assert hln.distribution == "t(n-1)"


def test_alternatives_are_consistent():
    y, f1, f2 = _forecasts()
    g = nctest.dm_test(y, f1, f2, alternative="greater").p_value
    l = nctest.dm_test(y, f1, f2, alternative="less").p_value
    two = nctest.dm_test(y, f1, f2, alternative="two-sided").p_value
    assert g + l == pytest.approx(1.0)
    assert two == pytest.approx(2 * min(g, l))


def test_losses_and_bandwidth_options():
    y, f1, f2 = _forecasts()
    a = nctest.dm_test(y, f1, f2, loss="absolute")
    b = nctest.dm_test(y, f1, f2, loss=lambda e: np.abs(e))
    assert a.statistic == pytest.approx(b.statistic)
    auto = nctest.cw_test(y, f1, f2, bandwidth="auto")
    assert auto.bandwidth == nctest.default_bandwidth(len(y)) and auto.kernel == "bartlett"
    assert auto.variance == pytest.approx(
        nctest.bartlett_lrv((y - f1) ** 2 - (y - f2) ** 2 + (f1 - f2) ** 2, auto.bandwidth))


def test_uniform_kernel_falls_back_when_negative():
    n = 200
    y = np.where(np.arange(n) % 2 == 0, 0.0, math.sqrt(2.0))     # d alternates 0, 2
    res = nctest.dm_test(y, np.zeros(n), y, horizon=2)
    assert res.kernel == "bartlett" and res.notes and res.variance > 0


def test_input_checks():
    y, f1, f2 = _forecasts()
    with pytest.raises(ValueError):
        nctest.dm_test(y, f1[:-1], f2)
    with pytest.raises(ValueError):
        nctest.dm_test(y, f1, f2, alternative="bigger")
    with pytest.raises(ValueError):
        nctest.cw_test(np.r_[y[:-1], np.nan], f1, f2)
    with pytest.raises(ValueError):
        nctest.dm_test(y, f1, f2, loss="quartic")
    with pytest.raises(ValueError):
        nctest.dm_test(y, f1, f2, horizon=0)


def test_dm_size_under_equal_accuracy():
    """Two non-nested forecasts of equal accuracy: DM rejects about 5% of the time."""
    g = np.random.default_rng(11)
    rej = 0
    reps, n = 2000, 200
    for _ in range(reps):
        u, a, b = g.standard_normal((3, n))
        rej += nctest.dm_test(u, a, b, alternative="two-sided").reject
    assert 0.035 < rej / reps < 0.065


def test_result_prints_and_exports():
    y, f1, f2 = _forecasts()
    r = nctest.cw_test(y, f1, f2)
    assert "Clark-West" in str(r) and "p-value" in str(r)
    assert set(("statistic", "p_value", "reject")) <= set(r.as_dict())


def test_compare_reports_p_values():
    y, f1, f2 = _forecasts()
    res = nctest.from_forecasts(y, f1, f2, psi=np.full(len(y), 0.05),
                                sigma2_w=np.ones(len(y)))
    assert res.p_DM == pytest.approx(nctest.dm_test(y, f1, f2).p_value)
    assert res.p_CW == pytest.approx(nctest.cw_test(y, f1, f2).p_value)
    assert 0 <= res.p_CC_window <= 1 and "p-value" in str(res)
