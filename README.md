# btc-cent-forecast-01 — BTC 4-hour point forecaster

Entry for make.fun's **btc-cent-forecast-01** ("Forecast Bitcoin's price 4 hours ahead, 12 times over 48 hours"). The program reads one case at a time — BTC's 1-minute US-dollar closes for the 72 hours up to a checkpoint — and writes one number: its guess of the price 240 minutes later. Score is the mean absolute error in cents, so the right point forecast is the conditional **median** of the price 4 hours ahead.

## Method (stdlib only, deterministic, no network)

`main.py` blends (geometric mean) two mean-reversion signals, both estimated only from the case's own prices:

1. **Short reversion** — pull the checkpoint price 25% of the way back toward its trailing 6-hour mean in log space. BTC over-reacts to intraday moves; the 6-hour mean is the fastest anchor that still beats persistence.
2. **Extreme reversion** — when the price sits more than one 4-hour volatility (24h 1-minute sigma × √240) away from its trailing 24-hour mean, pull it 50% of the way back; otherwise leave it alone. Big dislocations revert; ordinary drift does not get fought.

Every constant in `PARAMS` is wired into the computation. Runtime ~0.01 s per case (cap is 8 s).

## Local backtest (practice recording 2026-11-07..11, 1-minute closes)

| Evaluation | Model MAE | Persistence MAE | vs baseline |
|---|---|---|---|
| Rolling checkpoints, 15-min spacing, 24h history (256 cases) | 38,668 cents | 41,248 cents | **−6.3%** |
| Rolling checkpoints, 15-min spacing, 72h history (64 cases) | 47,824 cents | 51,573 cents | **−7.3%** |
| Practice cases p00–p04 (the dev-phase score) | 45,687 cents | 47,055 cents | **−2.9%** |

The blend beats persistence on all three sets; each component alone also beats it on all three, and the blend beats both components everywhere. Models rejected on the same data: raw momentum (−4% to −10%), empirical-median drift (wins only on trend days, loses −6% on rolling), AR(1) on 4h returns (+32% on the 24h rolling set — unstable), and 24h-mean reversion (loses on the trending practice day).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Writes `/output/<id>.out` as `{"forecasts":[{"case":"<id>","price":<number>}]}` for each `/input/<id>.in`.
