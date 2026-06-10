"""
End-to-end baseline: data -> signal -> cost-aware short-correlation returns.
Proves the premise before adding the regime spine and walk-forward validation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data, signals, strategy, metrics  # noqa: E402

P = data.load_panel()
cor3m = P["cor"]["COR3M"]

# convergent-validity gate (the pre-code test, embedded)
realized = data.realized_avg_corr(P["sector_rets"], window=63)
gate = cor3m.reindex(realized.index).corr(realized)

# signal + returns
z = signals.richness_zscore(cor3m)
out = strategy.build_returns(cor3m, z, cost_per_turnover=5e-4, target_vol=0.10)

print("=" * 64)
print("CORRELATION RISK PREMIUM — baseline (net 5bps/turnover, displayed @10% vol)")
print("=" * 64)
print(f"Convergent-validity gate corr(COR3M, realized 63d) = {gate:.3f}  "
      f"({'PASS' if gate >= 0.7 else 'CHECK'})")
print(f"Sample: {out['signal'].index.min().date()} -> {out['signal'].index.max().date()}  "
      f"({len(out['signal'])} days)")
print("-" * 64)
print(f"{'variant':>10} | {'Sharpe':>6} | {'AnnRet%':>7} | {'Vol%':>5} | {'MaxDD%':>7} | {'Calmar':>6}")
for name, ret in out.items():
    s = metrics.stats(ret)
    print(f"{name:>10} | {s['Sharpe']:>6} | {s['AnnRet']:>7} | {s['AnnVol']:>5} | {s['MaxDD']:>7} | {s['Calmar']:>6}")
print("-" * 64)
print("naive  = constant short corr (secular decline, but crashes eat it).")
print("signal = short scaled by richness z (sell when rich, base 0.5 + 0.5z, cap 1.5).")
print("NEXT layer: walk-forward HMM regime throttle (the spine) -> removes the crash tail.")
