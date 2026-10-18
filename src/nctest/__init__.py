"""nctest: noise-corrected tests of predictive ability for high-dimensional forecasts.

    import nctest
    res = nctest.compare(y, X, benchmark=H, window=250, penalty=0.1)
    print(res)

    # the classical tests on their own, on any pair of forecasts
    print(nctest.dm_test(y, yhat1, yhat2, horizon=1))
    print(nctest.cw_test(y, yhat1, yhat2))

Reference: Fallahgoul, H. (2026). Noise-Corrected Tests for High-Dimensional Forecasting
Models.
"""
from .core import (CRIT_5, bartlett_lrv, cc_test, default_bandwidth, dm_cw, intersection,
                   psihat, ridge_increment_weights, sanity_ratio, sigma2_difference,
                   sigma2_window)
from .classic import TestResult, cw_test, dm_test, long_run_variance, normal_cdf, t_cdf
from .rolling import ForecastPath, rolling_ridge
from .compare import Result, compare, from_forecasts
from .har import har_design

__version__ = "0.2.0"
__all__ = ["compare", "from_forecasts", "dm_test", "cw_test", "TestResult",
           "long_run_variance", "normal_cdf", "t_cdf", "rolling_ridge", "har_design", "Result",
           "ForecastPath", "cc_test", "dm_cw", "psihat", "sigma2_window",
           "sigma2_difference", "sanity_ratio", "intersection", "ridge_increment_weights",
           "bartlett_lrv", "default_bandwidth", "CRIT_5"]
