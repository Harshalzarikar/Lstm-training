import numpy as np
import pandas as pd
from arch import arch_model


def fit_garch11(returns: pd.Series, p: int = 1, q: int = 1):
    """Fit a GARCH(p,q) model (default 1,1) on returns and return fitted model.

    returns: typically daily returns (can be log returns)
    """
    returns = returns.dropna() * 100  # scale returns to percent to stabilize fit
    am = arch_model(returns, vol="Garch", p=p, q=q, rescale=False)
    res = am.fit(disp="off")
    return res


def forecast_volatility(garch_res, horizon: int = 1) -> np.ndarray:
    """Forecast conditional variances for `horizon` steps ahead.

    Returns an array of variances (not stddev).
    """
    f = garch_res.forecast(horizon=horizon, reindex=False)
    # The forecasted variance is in "variance" key for most arch versions
    try:
        var = f.variance.values[-1]
    except Exception:
        # fallback: use mean variance
        var = np.repeat(np.nanmean(garch_res.conditional_volatility ** 2), horizon)
    return var
