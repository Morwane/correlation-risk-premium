"""Tests — correctness & no-look-ahead guarantees for the Correlation Risk Premium Engine."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import data, signals, strategy, metrics  # noqa: E402

ANN = 252


@pytest.fixture(scope="module")
def panel():
    return data.load_panel()


def test_convergent_validity_gate(panel):
    """The premise: implied and realized correlation co-move (gate >= 0.7)."""
    realized = data.realized_avg_corr(panel["sector_rets"], window=63)
    gate = panel["cor"]["COR3M"].reindex(realized.index).corr(realized)
    assert gate >= 0.7, f"gate {gate:.3f} below 0.7 — premium framing unsafe"


def test_signal_is_lagged(panel):
    """P&L on day t must not use the size decided on day t (no look-ahead)."""
    cor3m = panel["cor"]["COR3M"]
    size = pd.Series(np.arange(len(cor3m), dtype=float), index=cor3m.index)
    pnl = strategy.short_corr_pnl(cor3m, size)
    # reconstruct: gross_t should use size[t-1]
    expected = size.shift(1) * (-cor3m.diff()) / 100.0
    # ignore cost term by setting cost=0
    pnl0 = strategy.short_corr_pnl(cor3m, size, cost_per_turnover=0.0)
    aligned = pnl0.dropna()
    assert np.allclose(aligned.values, expected.reindex(aligned.index).values, equal_nan=True)


def test_size_only_shorts_when_rich():
    """Size is non-negative and capped (never buys correlation, bounded leverage)."""
    z = pd.Series(np.linspace(-5, 5, 100))
    s = strategy.signal_size(z)
    assert (s >= 0).all() and (s <= 1.5).all()
    assert s.iloc[0] == 0.0          # very cheap -> flat
    assert s.iloc[-1] == 1.5         # very rich -> capped


def test_costs_reduce_sharpe(panel):
    """Higher transaction costs cannot increase the Sharpe."""
    cor3m = panel["cor"]["COR3M"]
    z = signals.richness_zscore(cor3m)
    size = strategy.signal_size(z.reindex(cor3m.index))
    sharpes = []
    for bps in (0, 5, 20):
        r = strategy.short_corr_pnl(cor3m, size, cost_per_turnover=bps / 1e4).dropna()
        sharpes.append(r.mean() / r.std() * np.sqrt(ANN))
    assert sharpes[0] >= sharpes[1] >= sharpes[2]


def test_vol_scaling_hits_target(panel):
    """Display vol-scaling produces ~10% annualized vol."""
    cor3m = panel["cor"]["COR3M"]
    z = signals.richness_zscore(cor3m)
    r = strategy.vol_scaled(strategy.short_corr_pnl(cor3m, strategy.signal_size(z.reindex(cor3m.index))))
    realized_vol = r.dropna().std() * np.sqrt(ANN)
    assert abs(realized_vol - 0.10) < 0.005


def test_signal_beats_naive(panel):
    """The economic claim: richness-scaling beats a constant short."""
    cor3m = panel["cor"]["COR3M"]
    z = signals.richness_zscore(cor3m)
    out = strategy.build_returns(cor3m, z)
    assert metrics.stats(out["signal"])["Sharpe"] > metrics.stats(out["naive"])["Sharpe"] + 0.5
