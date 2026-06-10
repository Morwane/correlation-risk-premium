"""
Portfolio + risk layer — Correlation Risk Premium Engine.

A dispersion / short-correlation book is structurally SHORT the implied-correlation index:
index options are dear (investors overpay for correlation as crash insurance), so the seller
earns a premium but takes the crash risk when correlation gaps to 1.

We trade the tradable thing directly — the MtM of a short position in COR3M:
    gross_t = size[t-1] * ( -(COR3M_t - COR3M_{t-1}) ) / 100
(implied-realized is NOT a tradable spread across universes; only the index MtM is. The
 +0.81 convergent validity justifies the premium interpretation, not a spread trade.)

Size leans into richness (z of COR3M). Sharpe and Calmar are scale-invariant, so a single
constant vol-target scalar is applied only to make AnnRet/MaxDD readable at ~10% vol.
"""
import numpy as np
import pandas as pd

ANN = 252


def signal_size(z: pd.Series, base: float = 0.5, slope: float = 0.5, cap: float = 1.5) -> pd.Series:
    """Short-correlation size: a base short, leaning in when implied corr is rich (high z)."""
    return (base + slope * z).clip(lower=0.0, upper=cap).rename("size")


def short_corr_pnl(cor3m: pd.Series, size: pd.Series, cost_per_turnover: float = 5e-4) -> pd.Series:
    """Daily net P&L of a short implied-correlation book. Size decided on prior info (shift 1)."""
    s = size.reindex(cor3m.index).shift(1)
    gross = s * (-(cor3m.diff())) / 100.0
    turnover = s.diff().abs()
    net = gross - turnover * cost_per_turnover
    first = s.dropna().index[0]
    net.loc[:first] = np.nan
    return net.rename("ret")


def vol_scaled(ret: pd.Series, target_vol: float = 0.10) -> pd.Series:
    """Constant full-sample scalar to display at target vol (does NOT change Sharpe/Calmar)."""
    r = ret.dropna()
    sc = target_vol / (r.std() * np.sqrt(ANN))
    return (r * sc).rename("ret")


def build_returns(cor3m: pd.Series, z: pd.Series, cost_per_turnover: float = 5e-4,
                  target_vol: float = 0.10) -> dict:
    """naive (constant unit short) vs signal-scaled (lean into richness), net of costs."""
    naive_size = pd.Series(1.0, index=cor3m.index)
    sig_size = signal_size(z.reindex(cor3m.index))
    out = {}
    for name, sz in [("naive", naive_size), ("signal", sig_size)]:
        net = short_corr_pnl(cor3m, sz, cost_per_turnover)
        out[name] = vol_scaled(net, target_vol)
    return out
