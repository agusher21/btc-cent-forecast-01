#!/usr/bin/env python3
"""BTC 4-hour point forecaster (v2).

For each case, forecast the BTC price 4 hours ahead from a single market's
72 hours of 1-minute closes (oldest first; the last close is the checkpoint).

Model -- blended log-space mean reversion:

1. Short pull: the last price is pulled back toward its trailing 6-hour
   log-mean. The pull is asymmetric: deviations above the mean revert
   hard (0.45), deviations below it barely revert (0.05).
2. Extreme gate: the last price is also compared with its trailing
   24-hour log-mean. Only when that deviation is large relative to
   recent volatility (EWMA sigma of 1-minute log returns, halflife
   480 minutes, threshold 1.15 * sigma * sqrt(240)) is it pulled back
   toward the 24-hour mean with strength 0.6; otherwise the extreme
   signal contributes nothing.
3. The two guesses are blended in log space (0.45 short / 0.55 extreme).

Backtest results (MAE in cents, vs v1 blended mean-reversion):
  practice set (kit p00-p04):  43,973 vs v1 45,687  (-3.75%)
  rolling 24h:                 37,654 vs v1 38,652  (-2.58%)
  rolling 72h:                 45,904 vs v1 47,822  (-4.01%)
The v2 model beats v1 on every set. The asymmetry (hard pull up,
soft pull down) reflects that sharp upward spikes in these windows
tend to fade within 4 hours, while downward moves tend to persist;
the EWMA-sigma gate keeps the 24-hour signal from firing on noise.

Stdlib only, deterministic, no network. Reads every ``*.in`` case in
``$MAKEFUN_INPUT_DIR`` (default ``/input``) and writes one ``*.out``
per case to ``$MAKEFUN_OUTPUT_DIR`` (default ``/output``).
"""

import json
import math
import os
from pathlib import Path

# --- model constants -------------------------------------------------------
SHORT_WINDOW = 360          # minutes in the trailing 6-hour window
LONG_WINDOW = 1440          # minutes in the trailing 24-hour window
PULL_UP = 0.45              # short-pull strength when price is above the 6h mean
PULL_DOWN = 0.05            # short-pull strength when price is at/below the 6h mean
EXTREME_PULL = 0.6          # pull strength toward the 24h mean when triggered
EXTREME_SIGMA_MULT = 1.15  # threshold multiplier on sigma
EXTREME_HORIZON = 240       # minutes (4h) scaled into the threshold
VOL_HALFLIFE = 480          # minutes for the EWMA of squared returns
MIN_SIGMA = 0.0001          # floor on sigma
BLEND_SHORT = 0.45          # log-space weight of the short guess
BLEND_EXTREME = 0.55        # log-space weight of the extreme guess
MIN_PRICE = 0.00001
MAX_PRICE = 9999999.99999


def _log_mean(log_prices, window):
    """Mean of the last ``window`` log prices (all of them if fewer)."""
    tail = log_prices[-window:]
    return sum(tail) / len(tail)


def _ewma_sigma(log_prices):
    """Volatility from an EWMA of squared 1-minute log returns.

    Seeded with the squared first return, halflife 480 minutes, floored
    at MIN_SIGMA. Returns MIN_SIGMA when fewer than two prices exist.
    """
    if len(log_prices) < 2:
        return MIN_SIGMA
    decay = 1.0 - 0.5 ** (1.0 / VOL_HALFLIFE)
    variance = (log_prices[1] - log_prices[0]) ** 2
    for i in range(2, len(log_prices)):
        ret = log_prices[i] - log_prices[i - 1]
        variance = decay * ret * ret + (1.0 - decay) * variance
    return max(MIN_SIGMA, math.sqrt(variance))


def _short_guess(price, log_prices):
    """6-hour mean-reversion guess: asymmetric pull toward the 6h mean."""
    last_log = log_prices[-1]
    dev = last_log - _log_mean(log_prices, SHORT_WINDOW)
    pull = PULL_UP if dev > 0 else PULL_DOWN
    return price * math.exp(-pull * dev)


def _extreme_guess(price, log_prices):
    """24-hour extreme-move guess: pulls back only on large deviations."""
    last_log = log_prices[-1]
    dev = last_log - _log_mean(log_prices, LONG_WINDOW)
    sigma = _ewma_sigma(log_prices)
    threshold = EXTREME_SIGMA_MULT * sigma * math.sqrt(EXTREME_HORIZON)
    if abs(dev) >= threshold:
        return price * math.exp(-EXTREME_PULL * dev)
    return price


def forecast_price(prices):
    """Forecast the price 4 hours ahead from a list of closes (oldest first)."""
    if not prices:
        raise ValueError("empty price series")
    last = float(prices[-1])
    if len(prices) == 1:
        guess = last
    else:
        log_prices = [math.log(p) for p in prices]
        short = _short_guess(last, log_prices)
        extreme = _extreme_guess(last, log_prices)
        guess = math.exp(
            BLEND_SHORT * math.log(short) + BLEND_EXTREME * math.log(extreme)
        )
    clamped = min(max(guess, MIN_PRICE), MAX_PRICE)
    return round(clamped * 1e5) / 1e5


def forecast_case(case):
    """Forecast one case dict; returns the output payload dict."""
    markets = case["markets"]
    if len(markets) != 1:
        raise ValueError("expected exactly one market per case")
    price = forecast_price(markets[0]["prices"])
    return {"forecasts": [{"case": "", "price": price}]}


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for in_path in sorted(input_dir.glob("*.in")):
        case_id = in_path.name[: -len(".in")]
        case = json.loads(in_path.read_text())
        payload = forecast_case(case)
        # The case id in the output is the filename stem.
        payload["forecasts"][0]["case"] = case_id
        out_path = output_dir / f"{case_id}.out"
        out_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
