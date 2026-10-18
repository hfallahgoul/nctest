"""Diebold-Mariano and Clark-West on two forecasts made elsewhere.

    python examples/dm_cw_example.py
"""
import numpy as np

import nctest

# A benchmark that forecasts zero and a candidate that uses a predictor x_t known at the
# forecast origin, at a five-step horizon. The target is an h-period sum, so its forecast
# errors are MA(h-1), which is what the horizon argument accounts for.
g = np.random.default_rng(1)
n, h = 600, 5
x = g.standard_normal(n)
u = g.standard_normal(n + h - 1)
y = 0.3 * x + np.convolve(u, np.ones(h), "valid")
yhat1 = np.zeros(n)
yhat2 = 0.25 * x

print(nctest.dm_test(y, yhat1, yhat2, horizon=h, hln=True))
print()
print(nctest.cw_test(y, yhat1, yhat2, horizon=h))
