"""
Signal layer — Correlation Risk Premium Engine.
Builds the richness signal that scales the short-correlation position.
All features use only past data; the strategy lags them by one day before trading.
"""
import numpy as np
import pandas as pd


def richness_zscore(cor3m: pd.Series, lookback: int = 252) -> pd.Series:
    """
    Expanding-then-rolling z-score of implied correlation level.
    High z = correlation is RICH = attractive to be short. Causal (uses trailing window only).
    """
    mu = cor3m.rolling(lookback, min_periods=126).mean()
    sd = cor3m.rolling(lookback, min_periods=126).std()
    return ((cor3m - mu) / sd).rename("richness_z")


def term_slope(cor1m: pd.Series, cor6m: pd.Series) -> pd.Series:
    """Correlation term-structure slope (short minus long). Positive = front rich."""
    return (cor1m - cor6m).rename("term_slope")


def cor_vix_ratio(cor3m: pd.Series, vix: pd.Series) -> pd.Series:
    """Correlation relative to VIX — the CSI orthogonality bridge to the thesis."""
    return (cor3m / vix).rename("cor_vix_ratio")


def short_corr_pnl(cor3m: pd.Series) -> pd.Series:
    """
    Daily P&L of a unit short-correlation position = -Δ COR3M / 100.
    (Trading the implied-correlation index MtM directly — the honest, replication-aware proxy.
     implied-realized is NOT a tradable spread; only the index MtM is.)
    """
    return (-cor3m.diff() / 100.0).rename("short_corr_pnl")
