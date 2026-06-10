"""
Regime layer (the spine) — walk-forward Gaussian HMM throttle.

Look-ahead-free recipe: refit on an expanding past-only window, standardise on the train
window only, decode causally, relabel states to fixed economic codes (0 calm, 1 normal,
2 stress) from emission means so labels are stable across refits. The stress flag throttles
the short-correlation book BEFORE correlation gaps to 1 — the strategy's tail risk.
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

CALM, NORMAL, STRESS = 0, 1, 2
FEATURES = ["idx_rv20", "COR3M", "cor3m_chg5", "term_slope"]


def feature_panel(cor: pd.DataFrame, spy: pd.Series) -> pd.DataFrame:
    """Build the regime feature panel from the COR term structure and SPY realized vol."""
    df = pd.DataFrame(index=cor.index)
    spy = spy.reindex(cor.index)
    spy_ret = np.log(spy / spy.shift(1))
    df["idx_rv20"] = spy_ret.rolling(20).std() * np.sqrt(252) * 100
    df["COR3M"] = cor["COR3M"]
    df["cor3m_chg5"] = cor["COR3M"].diff(5)
    df["term_slope"] = cor["COR1M"] - cor["COR6M"]   # >0 inversion = stress
    return df


def _label_map(model: GaussianHMM, cols: list[str]) -> dict:
    means = pd.DataFrame(model.means_, columns=cols)
    z = (means - means.mean()) / means.std(ddof=0).replace(0, 1.0)
    score = z["idx_rv20"] + z["COR3M"] + z["cor3m_chg5"] + z["term_slope"]
    order = score.sort_values().index.tolist()
    return {raw: code for raw, code in zip(order, [CALM, NORMAL, STRESS])}


def walk_forward(df: pd.DataFrame, n_states: int = 3, train_min: int = 756,
                 refit_every: int = 63, decode_window: int = 504, seed: int = 42) -> pd.Series:
    X = df[FEATURES].dropna()
    idx = X.index
    out = pd.Series(index=idx, dtype="float")
    model, label_map, mu, sd = None, {}, None, None
    for i in range(train_min, len(idx)):
        if model is None or (i - train_min) % refit_every == 0:
            train = X.iloc[:i]
            mu, sd = train.mean(), train.std(ddof=0).replace(0, 1.0)
            model = GaussianHMM(n_components=n_states, covariance_type="full",
                                n_iter=200, tol=1e-3, random_state=seed)
            model.fit(((train - mu) / sd).values)
            label_map = _label_map(model, FEATURES)
        lo = max(0, i - decode_window + 1)
        Z = ((X.iloc[lo:i + 1] - mu) / sd).values
        out.iloc[i] = label_map[model.predict(Z)[-1]]
    return out.dropna().rename("regime")


def confirm(regime: pd.Series, min_hold: int = 10) -> pd.Series:
    """Minimum-holding hysteresis to damp throttle whipsaw."""
    out = regime.copy()
    cur, hold = regime.iloc[0], 0
    for i, r in enumerate(regime.values):
        if i and (r != cur and hold >= min_hold):
            cur, hold = r, 0
        else:
            hold += 1
        out.iloc[i] = cur
    return out


def throttle_factor(regime: pd.Series, stress_cut: float = 0.0) -> pd.Series:
    """Gross multiplier: 1.0 in calm/normal, stress_cut in confirmed stress."""
    return regime.map({CALM: 1.0, NORMAL: 1.0, STRESS: stress_cut}).rename("throttle")
