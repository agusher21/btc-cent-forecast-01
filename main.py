#!/usr/bin/env python3
"""BTC 4-hour point forecaster (btc-cent-forecast-01), v2.
Post-opening re-authored submit version of v2 (re-written in a session
started after the 2026-11-14T20:00:00Z opening): asymmetric 6h reversion
(0.45 up / 0.05 down) + EWMA-sigma extreme gate; backtests in cents —
practice 43,973 vs v1 45,687; rolling 24h 37,654 vs 38,652; rolling 72h
45,904 vs 47,822.

Reads every *.in case in $MAKEFUN_INPUT_DIR (else /input), writes one *.out
per case to $MAKEFUN_OUTPUT_DIR (else /output):
  {"forecasts": [{"case": "<id>", "price": <number>}]}

Score is the mean absolute error in cents, so the right point forecast is the
conditional MEDIAN of the price 240 minutes after the checkpoint. The model
blends (geometric mean) two mean-reversion signals, both estimated only from
the case's own 72 hours of 1-minute closes:

1. Short reversion: pull the checkpoint price back toward its trailing
   6-hour mean in log space, ASYMMETRICALLY: 45% of the way when the price
   sits ABOVE the mean, only 5% when it sits BELOW it. On this recording
   upside spikes revert hard while downside drifts persist (checked on both
   sides of the deviation on the rolling backtests, not just in the one
   downtrend half), so the two directions get different pulls.
2. Extreme reversion: when the price sits more than 1.15x the 4-hour
   volatility away from its trailing 24-hour mean, pull it 60% of the way
   back; otherwise leave it alone. Big dislocations revert; ordinary drift
   does not get fought. The volatility is an EWMA of squared 1-minute
   returns (halflife 480 minutes), which adapts to the current volatility
   regime faster than a plain 24-hour sigma.

Every constant in PARAMS is wired into the computation (no silent no-ops).
Stdlib only, deterministic, no network, ~0.01 s per case.

Local backtest on the practice recording (2026-11-07..11, 1-min closes),
MAE in cents vs v1 (and vs persistence):
  practice cases p00-p04 (the dev-phase score):       43,973 vs 45,687 (-3.75%)
  rolling checkpoints, 15-min spacing, 24h history:  37,654 vs 38,652 (-2.58%)
  rolling checkpoints, 15-min spacing, 72h history:  45,904 vs 47,822 (-4.01%)
  The model also beats v1 on every half of both rolling sets and on the
  non-overlapping 4h-spaced checkpoints. Rolling paired t vs v1: -4.0 (24h,
  n=256, wins 62% of checkpoints) and -2.5 (72h, n=64, wins 72%).
  v1 itself: 45,687 / 38,652 / 47,822 vs persistence 47,055 / 41,248 / 51,573.
"""
import json
import math
import os
from pathlib import Path

PARAMS = {
    # Short reversion: trailing window (minutes) and the two pull fractions,
    # one per side of the trailing mean.
    "rev6_window_min": 360,
    "rev6_k_up": 0.45,    # price above the 6h mean: pull down hard
    "rev6_k_down": 0.05,  # price below the 6h mean: barely fight the drift
    # Extreme reversion: trailing window, pull fraction, threshold in units
    # of 4-hour volatility.
    "ext_window_min": 1440,
    "ext_k": 0.6,
    "ext_thresh_mult": 1.15,
    # EWMA halflife (minutes) for the 1-minute return volatility used by
    # the extreme gate.
    "sigma_ewma_halflife_min": 480,
    # Floor on the 1-minute log-return sigma (avoids a zero threshold on
    # flat stretches of carried-forward prices).
    "sigma_floor": 0.0001,
    # Blend weight on the short-reversion guess (geometric mean).
    "w_rev6": 0.45,
}

MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999


def _price5(x):
    """x kept on the grid's 5 decimals and inside (0, 10,000,000)."""
    return round(min(MAX_PRICE, max(MIN_PRICE, x)) * 1e5) / 1e5


def _mean_log(logp, window):
    n = min(window, len(logp))
    return sum(logp[len(logp) - n:]) / n


def _sigma_ewma(logp, halflife):
    """EWMA sigma of 1-minute log returns over the whole series given.

    Seeded with the first squared return and updated in order, so the
    result depends only on this case's own prices. Converges within the
    4,320 minutes of a case (initial weight ~0.5**(4320/halflife)).
    """
    if len(logp) < 2:
        return PARAMS["sigma_floor"]
    a = 1.0 - 0.5 ** (1.0 / halflife)
    v = (logp[1] - logp[0]) ** 2
    for j in range(2, len(logp)):
        r = logp[j] - logp[j - 1]
        v = a * r * r + (1.0 - a) * v
    return max(PARAMS["sigma_floor"], math.sqrt(v))


def forecast(prices):
    n = len(prices)
    if n == 0:
        raise ValueError("empty price series")
    x = prices[-1]
    if n < 2:
        return _price5(x)
    logp = [math.log(p) for p in prices]
    last = logp[-1]

    # 1. Short reversion toward the trailing 6-hour mean, asymmetric by side.
    mean6 = _mean_log(logp, PARAMS["rev6_window_min"])
    dev6 = last - mean6
    k6 = PARAMS["rev6_k_up"] if dev6 > 0 else PARAMS["rev6_k_down"]
    g6 = x * math.exp(-k6 * dev6)

    # 2. Extreme reversion toward the trailing 24-hour mean, gated by
    # EWMA 4-hour volatility.
    mean24 = _mean_log(logp, PARAMS["ext_window_min"])
    dev = last - mean24
    thresh = (PARAMS["ext_thresh_mult"]
              * _sigma_ewma(logp, PARAMS["sigma_ewma_halflife_min"])
              * math.sqrt(240.0))
    if abs(dev) >= thresh:
        ge = x * math.exp(-PARAMS["ext_k"] * dev)
    else:
        ge = x

    # Geometric-mean blend of the two guesses.
    w = PARAMS["w_rev6"]
    guess = math.exp(w * math.log(g6) + (1.0 - w) * math.log(ge))
    return _price5(guess)


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in sorted(p.name for p in in_dir.glob("*.in")):
        case = json.loads((in_dir / fname).read_text())
        markets = case["markets"]
        if len(markets) != 1:
            raise ValueError(f"{fname} does not hold exactly one market")
        price = forecast(markets[0]["prices"])
        out = json.dumps(
            {"forecasts": [{"case": fname[:-3], "price": price}]},
            separators=(",", ":")) + "\n"
        (out_dir / (fname[:-3] + ".out")).write_text(out)


if __name__ == "__main__":
    main()
