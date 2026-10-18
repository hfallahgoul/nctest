# Changelog

## 0.2.0

- `dm_test` and `cw_test` run the Diebold–Mariano and Clark–West tests on their own, on any
  pair of forecasts: p-values, forecast horizon, Bartlett or uniform kernel, automatic or
  fixed bandwidth, squared, absolute or user-supplied loss, one- or two-sided alternatives
  for DM, and the Harvey–Leybourne–Newbold correction. They return a `TestResult`.
- `cc_test` also returns a one-sided p-value; `compare` and `from_forecasts` report
  p-values for all four statistics.
- The sanity-ratio note now says that in designs whose benchmark regresses on the target's
  own past the reference value is kappa_a, about 0.94, not one.
- New helpers `long_run_variance`, `normal_cdf` and `t_cdf`; still no dependency beyond NumPy.
- `dm_cw` is unchanged and kept for compatibility.

## 0.1.0

First release. `compare` runs Diebold–Mariano, Clark–West and both noise-corrected
statistics on a rolling ridge comparison; `from_forecasts` runs them on forecasts made
elsewhere; `har_design` builds the HAR benchmark and own-lag predictors used in the paper.
