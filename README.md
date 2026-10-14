# nctest

Noise-corrected tests of predictive ability for high-dimensional forecasting models.

When a benchmark forecast is enlarged with many predictors and fitted by ridge, the
Clark–West test asks whether the extra predictors carry any signal. A forecaster usually wants
to know something else: whether the enlarged model forecasts better once its estimation noise
is paid for. The two questions have different answers when the model is large relative to
the estimation window. `nctest` implements the noise-corrected (CC) test of Fallahgoul (2026),
which targets the second question, and reports it next to Diebold–Mariano and Clark–West.

## Install

```
pip install nctest
```

The only dependency is NumPy.

## Example: a HAR benchmark enlarged with extra lags

```python
import numpy as np
import nctest

x = np.log(rv)                                   # daily realized variance, your data
y, H, X = nctest.har_design(x, extra_lags=24)    # HAR regressors and lags 2..25
res = nctest.compare(y, X, benchmark=H, window=250, penalty=0.01)
print(res)
```

On a simulated series where the HAR is the true model (`examples/har_example.py`):

```
Noise-corrected test of predictive ability
  forecasts n = 3725, Bartlett bandwidth = 9
  window T = 250, features P = 24, penalty z = 0.01 (ztil = zT/P = 0.104167)
  MSPE benchmark 0.179374, candidate 0.197627, gap -0.1018
  mean Psihat 0.1042, sanity ratio R 0.96

  statistic                            t
  Diebold-Mariano                  -9.80
  Clark-West                       -0.49
  CC, window variance              -0.05
  CC, difference variance          -3.06

  * exceeds the one-sided 5% critical value 1.645
  decision (reject only if both CC tests reject): do not reject H0
```

The summary also prints notes on the sanity ratio and the bandwidth condition when they apply.
`res.reject` is the decision of the intersection rule (both CC statistics reject at 5%).
The other fields of `res` hold every statistic, the sanity ratio and the out-of-sample
mean-square gap.

## What `compare` does

Row `t` of `X` and of the benchmark matrix `H` holds the predictors used to forecast `y[t]`,
so lag them before the call. At each origin the benchmark is fitted by least squares on the
last `window` observations. The candidate adds a ridge fit of the benchmark's residuals on
the extra predictors, residualised on the benchmark regressors, with penalty `z` in

$$\frac1T\sum_s (\hat u_s - \tilde x_s'\gamma)^2 + z\|\gamma\|^2 .$$

The two models are nested, and the candidate's increment is linear in the training targets,
$\hat y_{2,t}-\hat y_{1,t}=a_t'Y_{t-1}$. The test statistic is built from

$$d^{CC}_t = e_{1,t}^2 - e_{2,t}^2 + \hat\sigma^2_t\,\hat\Psi_t, \qquad \hat\Psi_t = \|a_t\|^2,$$

studentised by a Bartlett long-run variance. `benchmark` may also be `"mean"` (rolling mean)
or `"zero"`.

## Your own forecasts

If your models are linear smoothers of the training targets but not ridge, compute the
weights $a_t$ yourself and call

```python
res = nctest.from_forecasts(y, yhat1, yhat2, psi=nctest.psihat(A), sigma2_w=s2w)
```

The building blocks (`cc_test`, `dm_cw`, `sigma2_window`, `sigma2_difference`,
`sanity_ratio`, `intersection`, `bartlett_lrv`) are exported for custom designs.

## Reading the output

- **Sanity ratio R.** In population R cannot fall below one. A value below one means the
  correction exceeds the whole forecast increment, and the test is mis-scaled for that design.
- **Bandwidth condition.** The paper's sufficient condition is $T=o(n^{1/3})$. Most
  applications violate it; Section 5.6 of the paper measures size in that regime.
- **Own-lag designs.** When the extra predictors are lags of the target, as in the HAR
  example, the correction is accurate only to the extent the paper's scope condition holds;
  Section 4 of the paper measures it for designs of this kind.

## Reference

Fallahgoul, H. (2026). Noise-Corrected Tests for High-Dimensional Forecasting Models.
Replication package: https://github.com/hfallahgoul/noise-corrected-tests

## Licence

MIT.
