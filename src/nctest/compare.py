"""One call that runs DM, CW and the noise-corrected test side by side."""
from dataclasses import dataclass, field

import numpy as np

from . import classic, core
from .rolling import ForecastPath, rolling_ridge


@dataclass
class Result:
    n: int
    bandwidth: int
    t_DM: float
    t_CW: float
    t_CC_window: float
    t_CC_difference: float
    reject_window: bool
    reject_difference: bool
    reject: bool                 # intersection rule: both CC tests reject
    ratio: float                 # sanity ratio R with the window variance
    mean_psi: float
    mspe_benchmark: float
    mspe_candidate: float
    window: int = None
    n_features: int = None
    penalty: float = None
    notes: list = field(default_factory=list)
    p_DM: float = None           # one-sided p-values, candidate better
    p_CW: float = None
    p_CC_window: float = None
    p_CC_difference: float = None

    @property
    def gap(self):
        """(MSPE_1 - MSPE_2) / MSPE_1, the out-of-sample R^2 of the candidate over the
        benchmark."""
        return 1.0 - self.mspe_candidate / self.mspe_benchmark

    def summary(self):
        crit = core.CRIT_5
        mark = lambda t: "*" if t > crit else " "

        def row(label, t, p):
            ps = f"{p:>9.4f}" if p is not None else f"{'':>9}"
            return f"  {label:<30}{t:>8.2f}{mark(t)}{ps}"
        lines = [
            "Noise-corrected test of predictive ability",
            f"  forecasts n = {self.n}, Bartlett bandwidth = {self.bandwidth}",
        ]
        if self.window is not None:
            lines.append(f"  window T = {self.window}, features P = {self.n_features}, "
                         f"penalty z = {self.penalty:g} (ztil = zT/P = "
                         f"{self.penalty * self.window / self.n_features:g})")
        lines += [
            f"  MSPE benchmark {self.mspe_benchmark:.6g}, candidate {self.mspe_candidate:.6g}, "
            f"gap {self.gap:.4f}",
            f"  mean Psihat {self.mean_psi:.4f}, sanity ratio R {self.ratio:.2f}",
            "",
            f"  {'statistic':<30}{'t':>8} {'p-value':>9}",
            row("Diebold-Mariano", self.t_DM, self.p_DM),
            row("Clark-West", self.t_CW, self.p_CW),
            row("CC, window variance", self.t_CC_window, self.p_CC_window),
            row("CC, difference variance", self.t_CC_difference, self.p_CC_difference),
            "",
            f"  * exceeds the one-sided 5% critical value {crit:.3f}; p-values are one-sided",
            f"  decision (reject only if both CC tests reject): "
            f"{'reject H0, the candidate adds predictive ability' if self.reject else 'do not reject H0'}",
        ]
        lines += [f"  note: {s}" for s in self.notes]
        return "\n".join(lines)

    def __str__(self):
        return self.summary()


def from_forecasts(y, yhat1, yhat2, psi, sigma2_w, bandwidth=None, window=None):
    """Run all tests on forecasts you have produced yourself.

    psi      Psihat_t = ||a_t||^2, where yhat2_t - yhat1_t = a_t' Y_{t-1} is the candidate's
             increment written as weights on the T training targets.
    sigma2_w window-residual variance of the benchmark at each origin.
    window   the estimation window T, used only for the bandwidth-condition note.
    """
    y, yhat1, yhat2, psi, s2w = (np.asarray(v, float) for v in (y, yhat1, yhat2, psi, sigma2_w))
    n = len(y)
    b = core.default_bandwidth(n) if bandwidth is None else int(bandwidth)
    dm = classic.dm_test(y, yhat1, yhat2)
    cw = classic.cw_test(y, yhat1, yhat2)
    win = core.cc_test(y, yhat1, yhat2, psi, s2w, b)
    s2d, h = core.sigma2_difference(y)
    diff = core.cc_test(y, yhat1, yhat2, psi, s2d, b, influence=h)
    ratio = core.sanity_ratio(yhat1, yhat2, psi, s2w)
    notes = []
    if ratio < 1:
        notes.append(f"R = {ratio:.2f} is below one. With exogenous benchmark regressors "
                     "R >= 1 in population, with equality when the extra predictors carry no "
                     "signal. When the benchmark regresses on the target's own past (a HAR, "
                     "say) its null value is kappa_a, about 0.94, so compare R with that. A "
                     "value well below the relevant reference means the correction exceeds "
                     "the forecast increment and biases t_CC upward.")
    if window is not None and window > n ** (1 / 3):
        notes.append(f"T = {window} exceeds n^(1/3) = {n ** (1 / 3):.1f}, so the sufficient "
                     "bandwidth condition T = o(n^(1/3)) does not hold; the paper's simulations "
                     "measure size in this regime.")
    return Result(
        n=n, bandwidth=b, t_DM=dm.statistic, t_CW=cw.statistic,
        t_CC_window=win["t"], t_CC_difference=diff["t"],
        reject_window=win["reject"], reject_difference=diff["reject"],
        reject=core.intersection(win, diff), ratio=ratio, mean_psi=float(psi.mean()),
        mspe_benchmark=float(np.mean((y - yhat1) ** 2)),
        mspe_candidate=float(np.mean((y - yhat2) ** 2)), window=window, notes=notes,
        p_DM=dm.p_value, p_CW=cw.p_value, p_CC_window=win["p_value"],
        p_CC_difference=diff["p_value"])


def compare(y, X, benchmark="mean", window=250, penalty=1.0, standardize=True,
            bandwidth=None):
    """Fit the rolling benchmark and ridge candidate, then run DM, CW and both CC tests.

    See rolling_ridge for the arguments. Returns a Result; print it for a summary.
    """
    path = rolling_ridge(y, X, benchmark=benchmark, window=window, penalty=penalty,
                         standardize=standardize)
    res = from_forecasts(path.y, path.yhat1, path.yhat2, path.psi, path.sigma2_w,
                         bandwidth=bandwidth, window=path.window)
    res.n_features, res.penalty = path.n_features, path.penalty
    return res
