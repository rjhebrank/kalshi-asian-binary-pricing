# Kalshi Asian Binary Pricing

Kalshi settles its Bitcoin contracts on a **60-second arithmetic average** of the
underlying, not on a point-in-time price at the close. That makes them Asian
binary options rather than the vanilla binaries most participants price them as.

Averaging reduces the variance of the settlement value, so a standard
Black-Scholes binary overstates volatility as expiry approaches. This repository
contains the pricer that corrects for it, the volatility estimator that feeds it,
and the test suite that pins down its behavior.

## The math

For an arithmetic average over a window of length `tau` ending at `T`, the
variance of the log settlement value is approximately

```
var[ln A] = sigma^2 * (T - 2*tau/3)
```

Two limits make the approximation legible. The first is asserted in the tests;
the second is stated here for reference:

| Case | Variance | Meaning |
|---|---|---|
| `tau -> 0` | `sigma^2 * T` | Plain Black-Scholes, no averaging |
| `T = tau` | `sigma^2 * tau/3` | Fully inside the averaging window |

So variance falls to a third of its unaveraged value at the limit, which means
volatility falls by a factor of `sqrt(3)`, roughly 1.73. That distinction matters:
it is the variance that drops threefold, not the volatility.

The pricer also handles **partial observation**. Once inside the averaging window,
part of the settlement value is already known, so the remaining path only has to
clear an effective strike:

```
effective_strike = (K - w_obs * observed_avg) / w_rem
```

## Quick start

```bash
pip install -r requirements.txt
python example.py
pytest tests/ -q        # 30 tests
```

## What the example shows

A near-the-money contract five minutes from settlement, priced both ways:

```
  spot            $67,250
  strike          $67,260   (+10 vs spot)
  time to close   300s
  annualized vol  55%

  Asian binary (Turnbull-Wakeman)   46¢
  Vanilla Black-Scholes             46.5¢
  Market quote (YES)                42¢

  Edge against the correct model    +4¢
```

The two models agree far from settlement and separate sharply near it:

```
   time to close     Asian   vanilla BS      gap
      30 seconds    31.56¢       39.08¢    7.52¢
        1 minute    36.72¢       42.22¢    5.50¢
       5 minutes    46.26¢       46.48¢    0.23¢
          1 hour    49.02¢       48.91¢   -0.11¢
           1 day    49.96¢       49.38¢   -0.58¢
```

The 7.5¢ gap at 30 seconds is where the edge lives, and it is what the averaging
correction is for. The small negative gaps past an hour are not an averaging
effect: with a 60-second window the correction is negligible at that horizon, and
what remains is a difference in how the two pricers handle drift and discounting.
Treat the sub-minute column as the meaningful one.

## Live results

The full system traded this model on Kalshi from **28 February to 8 April 2026**
on a $500 bankroll, capped at 5 contracts per market.

| | Overall | Daily BTC | 15-min BTC |
|---|---|---|---|
| Trades | 54 | 21 | 33 |
| Win rate | 37.0% | 52.4% | 27.3% |
| P&L | +$68.08 | +$70.25 | -$2.17 |
| Profit factor | 1.40 | 2.62 | 0.98 |
| Sharpe | 3.04 | 5.60 | 0.07 |
| Sortino | 7.03 | 19.36 | 0.13 |
| Max drawdown | -$49.50 | -$11.50 | -$69.50 |

That is **13.6% on the bankroll with a 9.9% maximum drawdown**. The 15-minute
book alone drew down $69.50, or 13.9% of the bankroll, which is larger than the
total return.

*A personal account funded with $500 of my own money, 54 trades over six weeks.
I am an undergraduate student, not a licensed investment adviser. Fifty-four trades is not a
statistical sample: the Sharpe and Sortino figures describe this period only,
are not annualized, and should not be read as an estimate of anything
forward-looking. Raw trade records are not published here. Nothing on this page
is investment advice or a solicitation, and past results do not indicate future
results.*

**The split is the finding.** Daily contracts worked. Fifteen-minute contracts
round-tripped to breakeven at a profit factor below 1. The model was not the
problem: attribution pointed at liquidity and fill speed. The pricer kept finding
mispriced short-dated contracts that could not be filled at size, which is
precisely where the tables above say the edge is largest.

## Risk framework

Trading ran under 31 rules across signal generation, execution, and risk, each
justified by a specific failure mode. A sample:

| Rule | Why |
|---|---|
| Third-Kelly sizing, drawdown-tiered | Full Kelly is ruinous when the model is wrong |
| Max total exposure $200 | Caps deployed capital at 40% of bankroll |
| 8 consecutive losses halts trading | Losing streaks signal regime change, not bad luck |
| Skip if trend z-score > 3.0 | Directional shocks break the mean-reversion assumption |
| Time gate: last 25% of contract life | Earlier than that, duration dominates edge |
| Idempotent client order IDs (UUID) | Retries cannot double-fire |
| Write-ahead DB before every API call | A crash mid-submit stays reconcilable |

Realized max drawdown was $49.50 against a $50 hard limit, so the circuit breaker
was never tripped.

## What is in this repository

```
src/asian_bs.py        Turnbull-Wakeman pricer, partial-observation logic, range contracts
src/black_scholes.py   Vanilla binary pricer, used as the comparison baseline
src/vol_estimator.py   Realized and EWMA volatility from streaming tick data
tests/test_asian_bs.py 30 tests: convergence, monotonicity, edge cases, partial observation
example.py             The worked example above
```

**The full trading system is private** while the approach is still in use. Not
included: the Kalshi API client, the signal gate stack, order execution and
reconciliation, position sizing, the circuit breakers, and the data pipeline.

## Notes

Built with Claude Code. The ideas and the implementation both came out of that
collaboration rather than from me alone: I set the direction, worked through the
tradeoffs, and made the final call on what shipped and what got cut.

MIT licensed.
