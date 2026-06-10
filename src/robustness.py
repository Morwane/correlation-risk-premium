"""
Robustness layer — the tests a skeptical PM demands before funding the idea.
Block bootstrap Sharpe CI, transaction-cost grid, throttle-strength sensitivity,
and crisis-period attribution. All causal; no re-tuning on the full sample.
"""
import numpy as np
import pandas as pd

ANN = 252


def _sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(ANN) if len(r) > 2 and r.std() > 0 else np.nan


def block_bootstrap_sharpe(ret: pd.Series, block: int = 21, n: int = 2000, seed: int = 0) -> dict:
    """Stationary-ish block bootstrap CI for the annualized Sharpe."""
    r = ret.dropna().values
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(len(r) / block))
    out = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, len(r) - block, nblocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:len(r)]
        out[i] = sample.mean() / sample.std() * np.sqrt(ANN) if sample.std() > 0 else np.nan
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return {"sharpe": round(_sharpe(ret), 2), "ci_low": round(lo, 2), "ci_high": round(hi, 2),
            "p_sharpe_gt_0": round(float(np.mean(out > 0)), 3)}


def cost_grid(cor3m, size, short_corr_pnl, vol_scaled, bps_list=(0, 2, 5, 10, 20)) -> pd.DataFrame:
    """Net Sharpe vs transaction cost (bps per unit turnover)."""
    rows = []
    for bps in bps_list:
        net = vol_scaled(short_corr_pnl(cor3m, size, cost_per_turnover=bps / 1e4))
        rows.append({"cost_bps": bps, "Sharpe": round(_sharpe(net), 2)})
    return pd.DataFrame(rows)


def throttle_sensitivity(cor3m, sig_size, thr_series, short_corr_pnl, vol_scaled, metrics_stats,
                         cuts=(0.0, 0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    """
    Sensitivity to stress_cut. thr_series is the confirmed regime (0/1/2); we rebuild the throttle
    factor per cut. cut=1.0 == no throttle (full signal). The regime states themselves are fixed
    (walk-forward, not re-estimated), so this is a clean sensitivity, not a re-fit.
    """
    from src.regime import CALM, NORMAL, STRESS
    rows = []
    for cut in cuts:
        fac = thr_series.map({CALM: 1.0, NORMAL: 1.0, STRESS: cut})
        net = vol_scaled(short_corr_pnl(cor3m, sig_size * fac.reindex(cor3m.index)))
        s = metrics_stats(net)
        rows.append({"stress_cut": cut, "Sharpe": s["Sharpe"], "MaxDD%": s["MaxDD"], "Calmar": s["Calmar"]})
    return pd.DataFrame(rows)


CRISES = {
    "EU debt 2011":   ("2011-07-01", "2011-12-31"),
    "China/oil 2015": ("2015-08-01", "2016-02-29"),
    "Volmageddon 18": ("2018-01-15", "2018-03-31"),
    "COVID 2020":     ("2020-02-15", "2020-04-30"),
    "Rates 2022":     ("2022-01-01", "2022-10-31"),
    "2024-25 spikes": ("2024-07-01", "2025-06-30"),
}


def crisis_table(ret_signal: pd.Series, ret_throttle: pd.Series) -> pd.DataFrame:
    """Total return (%) of signal vs throttle over each crisis window."""
    rows = []
    for name, (a, b) in CRISES.items():
        s = ret_signal.loc[a:b].sum() * 100
        t = ret_throttle.loc[a:b].sum() * 100
        rows.append({"crisis": name, "signal%": round(s, 1), "throttle%": round(t, 1),
                     "delta%": round(t - s, 1)})
    return pd.DataFrame(rows)
