"""
Robustness + reporting: bootstrap Sharpe CI, cost grid, throttle sensitivity, crisis table,
and the tearsheet figures. Regime is computed once and cached.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import data, signals, strategy, regime, metrics, robustness  # noqa: E402

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---- build everything once ----
P = data.load_panel()
cor3m = P["cor"]["COR3M"]
z = signals.richness_zscore(cor3m)
spy = data._load("SPY")

reg_cache = ROOT / "data" / "regime_states.csv"
if reg_cache.exists():
    reg = pd.read_csv(reg_cache, index_col=0).iloc[:, 0]
    reg.index = pd.to_datetime(reg.index)
else:
    feats = regime.feature_panel(P["cor"], spy)
    reg = regime.confirm(regime.walk_forward(feats), min_hold=10)
    reg.to_csv(reg_cache)

sig_size = strategy.signal_size(z.reindex(cor3m.index))
thr = regime.throttle_factor(reg, stress_cut=0.0).reindex(cor3m.index)
thr_size = sig_size * thr

ret_naive = strategy.vol_scaled(strategy.short_corr_pnl(cor3m, pd.Series(1.0, index=cor3m.index)))
ret_signal = strategy.vol_scaled(strategy.short_corr_pnl(cor3m, sig_size))
ret_thr = strategy.vol_scaled(strategy.short_corr_pnl(cor3m, thr_size))

print("=" * 64)
print("ROBUSTNESS — Correlation Risk Premium Engine")
print("=" * 64)

# 1) bootstrap Sharpe CI
print("\n[1] Block-bootstrap Sharpe 95% CI (signal+throttle, OOS):")
bs = robustness.block_bootstrap_sharpe(ret_thr.reindex(reg.index).dropna())
print(f"    Sharpe {bs['sharpe']}  CI95 [{bs['ci_low']}, {bs['ci_high']}]  "
      f"P(Sharpe>0)={bs['p_sharpe_gt_0']}")

# 2) cost grid
print("\n[2] Net Sharpe vs transaction cost (signal+throttle):")
cg = robustness.cost_grid(cor3m, thr_size, strategy.short_corr_pnl, strategy.vol_scaled)
print(cg.to_string(index=False))

# 3) throttle sensitivity (walk-forward states fixed; only the cut varies)
print("\n[3] Throttle-strength sensitivity (1.0 = no throttle):")
ts = robustness.throttle_sensitivity(cor3m, sig_size, reg, strategy.short_corr_pnl,
                                     strategy.vol_scaled, metrics.stats)
print(ts.to_string(index=False))

# 4) crisis table
print("\n[4] Crisis-window total return (%): signal vs throttle:")
ct = robustness.crisis_table(ret_signal, ret_thr)
print(ct.to_string(index=False))

# ---- figures ----
# F1 equity curves
plt.figure(figsize=(10, 5))
for r, lbl in [(ret_naive, "naive short"), (ret_signal, "signal-scaled"), (ret_thr, "signal+throttle")]:
    plt.plot(r.dropna().cumsum() * 100, label=lbl)
plt.title("Correlation Risk Premium — cumulative return (vol-target 10%, net 5bps)")
plt.ylabel("cumulative return (%)"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(FIG / "f1_equity_curves.png", dpi=150); plt.close()

# F2 regime-shaded drawdown of the throttled book
eq = ret_thr.dropna().cumsum()
dd = (eq - eq.cummax()) * 100
plt.figure(figsize=(10, 4))
plt.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.5)
stress_idx = reg.reindex(dd.index) == regime.STRESS
plt.fill_between(dd.index, dd.min(), 0, where=stress_idx.values, color="grey", alpha=0.2,
                 label="HMM stress (throttled)")
plt.title("Drawdown (signal+throttle) with HMM stress regime shaded")
plt.ylabel("drawdown (%)"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(FIG / "f2_drawdown_regime.png", dpi=150); plt.close()

# F3 cost grid + throttle sensitivity
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(cg["cost_bps"], cg["Sharpe"], "o-"); ax[0].set_xlabel("cost (bps/turnover)")
ax[0].set_ylabel("net Sharpe"); ax[0].set_title("Cost sensitivity"); ax[0].grid(alpha=0.3)
ax[1].plot(ts["stress_cut"], ts["Calmar"], "o-", label="Calmar")
ax[1].plot(ts["stress_cut"], ts["Sharpe"], "s--", label="Sharpe")
ax[1].set_xlabel("stress_cut (1.0 = no throttle)"); ax[1].set_title("Throttle sensitivity")
ax[1].legend(); ax[1].grid(alpha=0.3); plt.tight_layout()
plt.savefig(FIG / "f3_sensitivity.png", dpi=150); plt.close()

# F4 COR3M with richness z and percentile context
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(cor3m, color="navy", lw=0.8, label=".COR3M (implied corr)")
ax1.set_ylabel(".COR3M", color="navy"); ax1.grid(alpha=0.3)
ax2 = ax1.twinx(); ax2.plot(z, color="darkorange", lw=0.6, alpha=0.7, label="richness z")
ax2.axhline(0, color="grey", lw=0.5); ax2.set_ylabel("richness z", color="darkorange")
plt.title("Implied correlation .COR3M and richness signal"); plt.tight_layout()
plt.savefig(FIG / "f4_signal.png", dpi=150); plt.close()

print(f"\nFigures saved -> {FIG}")
for f in sorted(FIG.glob("*.png")):
    print(f"  {f.name}")
