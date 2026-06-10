"""Performance metrics — additive (vol-targeted) returns."""
import numpy as np
import pandas as pd

ANN = 252


def stats(ret: pd.Series) -> dict:
    ret = ret.dropna()
    eq = ret.cumsum()  # additive equity for vol-targeted streams
    sharpe = ret.mean() / ret.std() * np.sqrt(ANN) if ret.std() > 0 else np.nan
    dd = (eq - eq.cummax())
    maxdd = dd.min()
    cagr = ret.mean() * ANN
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    return {
        "Sharpe": round(sharpe, 2),
        "AnnRet": round(cagr * 100, 1),
        "AnnVol": round(ret.std() * np.sqrt(ANN) * 100, 1),
        "MaxDD": round(maxdd * 100, 1),
        "Calmar": round(calmar, 2),
        "n": len(ret),
    }
