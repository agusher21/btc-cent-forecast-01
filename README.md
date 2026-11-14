# btc-cent-forecast-01

BTC 4-hour point forecaster for the make.fun competition
**btc-cent-forecast-01** — predict the BTC price 4 hours ahead, scored
in cents of absolute error.

## How to run

```sh
MAKEFUN_INPUT_DIR=/path/to/cases MAKEFUN_OUTPUT_DIR=/path/to/out python3 main.py
```

* Python 3, standard library only — no dependencies, no network.
* Reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (default `/input`).
* Writes one `*.out` per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`).
* Each case JSON has a `markets` list with exactly one market; the
  forecast uses that market's `prices` — 72 hours of 1-minute closes,
  oldest first, the last close being the checkpoint price.
* Each output file is named `<case-id>.out` and contains
  `{"forecasts": [{"case": "<id>", "price": <number>}]}` with compact
  separators and a trailing newline.

## The v2 model

All forecasting is done in log space, blending two mean-reversion
signals:

1. **Short pull (6-hour).** The last price is pulled back toward its
   trailing 6-hour log-mean. The pull is asymmetric: **0.45** when the
   price is above the mean, **0.05** when it is at or below it.
   Upward spikes in these windows tend to fade within 4 hours, while
   downward moves tend to persist.
2. **Extreme gate (24-hour).** The last price is compared with its
   trailing 24-hour log-mean. The pull toward that mean (strength
   **0.6**) fires only when the deviation is large relative to recent
   volatility — an EWMA of squared 1-minute log returns (halflife
   480 minutes), threshold `1.15 * sigma * sqrt(240)`. Otherwise the
   extreme signal contributes nothing.
3. **Blend.** The two guesses are combined in log space:
   `guess = exp(0.45 * log(short) + 0.55 * log(extreme))`.
4. The final price is clamped to `[0.00001, 9999999.99999]` and
   rounded to 5 decimals.

Edge cases: an empty price series raises `ValueError`; a single
price returns that price (clamped and rounded).

## Backtest results

MAE in cents, v2 vs v1 (blended mean-reversion without the
asymmetric pull and EWMA-sigma gate):

| Set | v2 | v1 | Improvement |
| --- | --- | --- | --- |
| Practice set (kit p00–p04) | 43,973 | 45,687 | −3.75% |
| Rolling 24h | 37,654 | 38,652 | −2.58% |
| Rolling 72h | 45,904 | 47,822 | −4.01% |

v2 beats v1 on every set. On the practice set it also beats
persistence (47,055 cents) by −6.55%.

## License

MIT — see `LICENSE`.
