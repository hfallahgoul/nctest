# nctest

Noise-corrected tests of predictive ability for high-dimensional forecasting models.

When a benchmark forecast is enlarged with many predictors and fitted by ridge, the
Clark–West test asks whether the extra predictors carry any signal. A forecaster usually wants
to know something else: whether the enlarged model forecasts better once its estimation noise
is paid for. The two questions have different answers when the model is large relative to
the estimation window. `nctest` implements the noise-corrected (CC) test of Fallahgoul (2026),
which targets the second question, and reports it next to Diebold–Mariano and Clark–West.
The Diebold–Mariano and Clark–West tests are also available on their own, for any pair of
forecasts.

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

  statistic                            t   p-value
  Diebold-Mariano                  -9.80    1.0000
  Clark-West                       -0.49    0.6872
  CC, window variance              -0.05    0.5200
  CC, difference variance          -3.06    0.9989

  * exceeds the one-sided 5% critical value 1.645; p-values are one-sided
  decision (reject only if both CC tests reject): do not reject H0
```

The summary also prints notes on the sanity ratio and the bandwidth condition when they apply.
`res.reject` is the decision of the intersection rule (both CC statistics reject at 5%).
The other fields of `res` hold every statistic and its one-sided p-value, the sanity ratio
and the out-of-sample mean-square gap.

## Diebold–Mariano and Clark–West on their own

`dm_test` and `cw_test` need only the realised values and the two forecasts, so they work
for any forecasting method, nested or not, at any horizon:

```python
res = nctest.dm_test(y, yhat1, yhat2, horizon=5, hln=True)
print(res)
res.statistic, res.p_value, res.reject

nctest.cw_test(y, yhat1, yhat2, horizon=5)
```

```
Diebold-Mariano test (HLN-corrected)
  n = 600, horizon = 5, uniform kernel, bandwidth = 4
  mean differential 0.0832501, long-run variance 1.16755
  statistic 1.873, p-value 0.0308 (greater: candidate better; t(n-1))
  reject H0 at the 5% level
```

Model 1 is the benchmark and model 2 the candidate; a positive statistic favours the
candidate. The options are:

| argument | `dm_test` | `cw_test` |
|---|---|---|
| `horizon` | $h$; the long-run variance uses $h-1$ lags by default | same |
| `bandwidth` | `None` ($h-1$), `"auto"` (Newey–West 1994 rule), or an integer | same |
| `kernel` | uniform at the default bandwidth (Diebold–Mariano 1995), Bartlett otherwise; a non-positive uniform estimate falls back to Bartlett | Bartlett |
| `alternative` | `"greater"` (default), `"less"`, `"two-sided"` | one-sided by construction |
| `loss` | `"squared"`, `"absolute"` or a function of the error | squared |
| `hln` | Harvey–Leybourne–Newbold correction with $t_{n-1}$ critical values | – |

At $h=1$ the defaults studentise by the sample variance, which reproduces `dm_cw` from
version 0.1. Each call returns a `TestResult` with the statistic, p-value, decision, the mean
differential and the variance used; `res.as_dict()` exports it. Clark–West requires the
benchmark to be nested in the candidate. It tests whether the extra predictors improve the
population forecast, which is not the same as whether the candidate forecasts better at the
estimation window in use; the noise-corrected test answers the second question.
`examples/dm_cw_example.py` runs both tests at a five-step horizon.

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

The building blocks (`cc_test`, `sigma2_window`, `sigma2_difference`,
`sanity_ratio`, `intersection`, `bartlett_lrv`) are exported for custom designs.

## Reading the output

- **Sanity ratio R.** With exogenous benchmark regressors R cannot fall below one in
  population. When the benchmark regresses on the target's own past, as a HAR does, its null
  value is $\kappa_a$, about 0.94, and R should be compared with that. A value well below
  the relevant reference means the correction exceeds the forecast increment, which biases
  $t_{CC}$ upward. Where $\hat\Psi$ is close to zero, R carries little information.
- **Bandwidth condition.** The paper's sufficient condition is $T=o(n^{1/3})$. Most
  applications violate it; the paper's simulations measure size in that regime.
- **Own-lag designs.** When the extra predictors are lags of the target, as in the HAR
  example, the correction is accurate only to the extent the paper's scope condition holds;
  the paper measures it for designs of this kind. A HAR benchmark enlarged with predictors
  from outside the target's history is the design to avoid.

## Reference

Fallahgoul, H. (2026). Noise-Corrected Tests for High-Dimensional Forecasting Models.
Replication package: https://github.com/hfallahgoul/noise-corrected-tests

## Licence

MIT.
