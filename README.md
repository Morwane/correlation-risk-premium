# Correlation Risk Premium Engine

Systematic strategy that harvests the **implied-correlation risk premium** — shorting CBOE implied
correlation when it is rich, with a **walk-forward Hidden-Markov-Model regime throttle** that caps the
crash tail. Look-ahead-free, cost-aware, fully reproducible on LSEG/Refinitiv daily data.

It is the tradable counterpart of an MSc thesis on *Correlation Synchronicity* (realized correlation as
a stress indicator): the same latent object, measured implied (CBOE `.COR`) instead of realized.

## Result (net 5 bps, 10% vol target, 2010–2026; regime OOS 2013+)

| Variant | Sharpe | MaxDD % | Calmar |
|---|---|---|---|
| Naive constant short | 0.05 | −19.3 | 0.03 |
| Signal-scaled (sell when rich) | **1.03** | −21.5 | 0.48 |
| Signal + HMM throttle | 1.02 | **−16.5** | **0.62** |

Bootstrap Sharpe 95% CI **[0.59, 1.48]**; survives costs to 20 bps. Full tearsheet:
[`reports/tearsheet.md`](reports/tearsheet.md).

**Honest headline:** the correlation premium is real but conditional and negatively-skewed; the regime
throttle is *tail insurance* (it trades return for a smaller drawdown), not a free lunch or a uniform
crisis hedge. See the crisis table — it helps in persistent stress (2011, 2022), lags fast gaps (COVID).

## Convergent-validity gate

`corr(.COR3M implied, realized 9-sector correlation) = +0.808` — run `python validate_corr_gate.py`
before trusting the premium framing (kill/downgrade if < 0.5).

## Architecture

```
src/
  data.py        load COR term structure, sectors, VIX; realized correlation
  signals.py     richness z-score, term slope, COR/VIX, short-corr P&L
  strategy.py    signal sizing, cost-aware P&L, vol display scaling
  regime.py      walk-forward Gaussian HMM (causal), economic relabel, throttle
  metrics.py     Sharpe / Calmar / drawdown
  robustness.py  bootstrap CI, cost grid, throttle sensitivity, crisis table
scripts/
  run_baseline.py    data -> signal -> cost-aware returns
  run_full.py        + regime throttle + by-regime attribution
  run_robustness.py  full robustness suite + figures
```

## Run

```bash
pip install -r requirements.txt          # or use the shared ../.venv
python scripts/run_baseline.py
python scripts/run_full.py
python scripts/run_robustness.py
pytest -q
```

## Data

CBOE `.COR1M/.COR3M/.COR6M`, 9 SPDR sector ETFs, SPY, `.VIX` — daily 2010–2026, from LSEG/Refinitiv
Workspace. No option-chain or intraday data required.

**Proprietary vendor data is not redistributed.** Reproduce it with an active LSEG session:

```bash
python scripts/fetch_data.py --start 2010-01-01 --end today   # writes data/raw_prices/*.csv
```

## Limitations

`.COR` is an index (not directly tradable); P&L is the MtM of a short-correlation position, real
implementation (options dispersion) carries replication error. `implied − realized` is **not** a
tradable spread — only the index MtM is traded; convergent validity justifies the premium, not a spread.

_Author: Morwane Mavrothalassitis. Built as the flagship of a multi-strategy systematic research book._
