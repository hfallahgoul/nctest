"""HAR benchmark against HAR plus 24 extra lags, on a simulated long-memory series.

Replace x with the log of your own daily realized variance.
    python examples/har_example.py
"""
import numpy as np

import nctest

# A HAR cascade (Corsi, 2009) for log realized variance, so the HAR is the true model and
# the extra lags carry no signal.
g = np.random.default_rng(0)
n, burn = 4000, 300
x = np.zeros(n + burn)
for t in range(22, n + burn):
    x[t] = (0.36 * x[t - 1] + 0.34 * x[t - 5:t].mean() + 0.25 * x[t - 22:t].mean()
            + 0.42 * g.standard_normal())
x = x[burn:]

y, H, X = nctest.har_design(x, extra_lags=24)
res = nctest.compare(y, X, benchmark=H, window=250, penalty=0.01)
print(res)
