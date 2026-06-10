"""
Data layer — Correlation Risk Premium Engine.
Loads CBOE implied-correlation indices, sector ETFs and VIX from local LSEG caches,
builds returns and realized cross-sector correlation. No look-ahead: everything is
computed causally and the caller is responsible for lagging signals.
"""
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw_prices"

# 9-sector thesis set (XLC/XLRE excluded: late start / unusable in audit)
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]


def _load(name: str) -> pd.Series:
    """Load one cached price series. Index RICs are stored with a leading underscore."""
    fname = name if (RAW / f"{name}.csv").exists() else f"_{name}"
    df = pd.read_csv(RAW / f"{fname}.csv", index_col=0)
    s = df.iloc[:, 0].copy()
    s.index = pd.to_datetime(s.index)
    s.name = name
    return s.sort_index()


def load_panel() -> dict:
    """Return aligned panel: COR term structure, sector prices/returns, realized corr, VIX."""
    cor = pd.DataFrame({k: _load(k) for k in ["COR1M", "COR3M", "COR6M"]})
    vix = _load("VIX")
    sect = pd.DataFrame({s: _load(s) for s in SECTORS}).dropna()
    rets = np.log(sect).diff()
    return {"cor": cor, "vix": vix, "sector_prices": sect, "sector_rets": rets}


def realized_avg_corr(sector_rets: pd.DataFrame, window: int = 63) -> pd.Series:
    """Rolling average pairwise correlation of sector returns, scaled 0-100 to match COR."""
    cols = sector_rets.columns
    iu = np.triu_indices(len(cols), k=1)
    r = sector_rets.dropna()
    idx, vals = [], []
    for i in range(window, len(r) + 1):
        c = r.iloc[i - window:i].corr().values
        idx.append(r.index[i - 1])
        vals.append(c[iu].mean())
    return pd.Series(vals, index=idx, name="realized_corr") * 100.0
