"""nctest: noise-corrected tests of predictive ability for high-dimensional forecasts.

    import nctest
    res = nctest.compare(y, X, benchmark=H, window=250, penalty=0.1)
    print(res)

Reference: Fallahgoul, H. (2026). Noise-Corrected Tests for High-Dimensional Forecasting
Models.
"""
from .core import (CRIT_5, bartlett_lrv, cc_test, default_bandwidth, dm_cw, intersection,
                   psihat, ridge_increment_weights, sanity_ratio, sigma2_difference,
                   sigma2_window)
from .rolling import ForecastPath, rolling_ridge
from .compare import Result, compare, from_forecasts
from .har import har_design

__version__ = "0.1.0"
__all__ = ["compare", "from_forecasts", "rolling_ridge", "har_design", "Result",
           "ForecastPath", "cc_test", "dm_cw", "psihat", "sigma2_window",
           "sigma2_difference", "sanity_ratio", "intersection", "ridge_increment_weights",
           "bartlett_lrv", "default_bandwidth", "CRIT_5"]
