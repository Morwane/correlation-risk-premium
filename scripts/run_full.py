"""
Full engine: data -> signal -> portfolio -> REGIME throttle (the spine) -> metrics.
Tests the Phase-11 thesis: the throttle removes the crash tail (lifts Calmar) at equal Sharpe.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data, signals, strategy, regime, metrics  # noqa: E402

P = data.load_panel()
cor = P["cor"]
cor3m = cor["COR3M"]

# --- signal + regime ---
z = signals.richness_zscore(cor3m)
spy = data._load("SPY")                       # SPY realized vol feeds the regime features
feats = regime.feature_panel(cor, spy)
reg = regime.confirm(regime.walk_forward(feats), min_hold=10)
thr = regime.throttle_factor(reg, stress_cut=0.0)

# --- variants ---
naive_size = pd.Series(1.0, index=cor3m.index)
sig_size = strategy.signal_size(z.reindex(cor3m.index))
thr_size = sig_size * thr.reindex(cor3m.index)   # throttle the same signal book

variants = {
    "naive":           strategy.vol_scaled(strategy.short_corr_pnl(cor3m, naive_size)),
    "signal":          strategy.vol_scaled(strategy.short_corr_pnl(cor3m, sig_size)),
    "signal+throttle": strategy.vol_scaled(strategy.short_corr_pnl(cor3m, thr_size)),
}

# common OOS window where regime exists, for a fair signal-vs-throttle comparison
oos = variants["signal+throttle"].dropna().index
print("=" * 70)
print("CORRELATION RISK PREMIUM — full engine (net 5bps/turnover, @10% vol)")
print("=" * 70)
print(f"Regime OOS window: {oos.min().date()} -> {oos.max().date()}  ({len(oos)} days)")
print(f"Stress days (confirmed): {(reg==regime.STRESS).sum()} "
      f"({(reg==regime.STRESS).mean()*100:.1f}%)  | cut to 0% gross")
print("-" * 70)
print(f"{'variant':>16} | {'Sharpe':>6} | {'AnnRet%':>7} | {'MaxDD%':>7} | {'Calmar':>6}  (full sample)")
for name, ret in variants.items():
    s = metrics.stats(ret)
    print(f"{name:>16} | {s['Sharpe']:>6} | {s['AnnRet']:>7} | {s['MaxDD']:>7} | {s['Calmar']:>6}")

print("-" * 70)
print("Apples-to-apples on the SAME regime-OOS window (signal vs +throttle):")
for name in ["signal", "signal+throttle"]:
    s = metrics.stats(variants[name].reindex(oos).dropna())
    print(f"{name:>16} | {s['Sharpe']:>6} | {s['AnnRet']:>7} | {s['MaxDD']:>7} | {s['Calmar']:>6}")

# by-regime P&L of the un-throttled signal book (where does the edge / pain live?)
print("-" * 70)
print("Signal book P&L by regime state (un-throttled, OOS):")
sig_oos = variants["signal"].reindex(oos)
for code, lbl in [(regime.CALM, "calm"), (regime.NORMAL, "normal"), (regime.STRESS, "stress")]:
    mask = reg.reindex(oos) == code
    r = sig_oos[mask].dropna()
    if len(r):
        ann = r.mean() * 252 * 100
        print(f"  {lbl:>7}: {mask.sum():>5} days | mean ann ret {ann:>6.1f}% | "
              f"hit {(r>0).mean()*100:>4.0f}%")
