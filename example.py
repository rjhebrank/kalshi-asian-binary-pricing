"""
Worked example: pricing a Kalshi BTC binary, and the gap against vanilla
Black-Scholes that the strategy traded against.

    python example.py
"""

from src.asian_bs import AsianBinaryPricer
from src.black_scholes import BlackScholesBinary

R = 0.045
AVERAGING_SECONDS = 60.0


def main() -> None:
    asian = AsianBinaryPricer(risk_free_rate=R, averaging_seconds=AVERAGING_SECONDS)
    vanilla = BlackScholesBinary(risk_free_rate=R)

    # A near-the-money contract five minutes from settlement.
    spot, strike, t_seconds, vol = 67_250.0, 67_260.0, 300.0, 0.55
    market_quote_cents = 42

    fair = asian.fair_value_cents(spot, strike, t_seconds, vol)
    bs = vanilla.fair_value_cents(spot, strike, t_seconds / 3600.0, vol)

    print("=" * 66)
    print("A single contract, five minutes to settlement")
    print("=" * 66)
    print(f"  spot            ${spot:,.0f}")
    print(f"  strike          ${strike:,.0f}   ({strike - spot:+,.0f} vs spot)")
    print(f"  time to close   {t_seconds:.0f}s")
    print(f"  annualized vol  {vol:.0%}")
    print()
    print(f"  Asian binary (Turnbull-Wakeman)   {fair}¢   <- settles on a 60s average")
    print(f"  Vanilla Black-Scholes             {bs:.1f}¢   <- assumes a point-in-time close")
    print(f"  Market quote (YES)                {market_quote_cents}¢")
    print()
    print(f"  Edge against the correct model    {fair - market_quote_cents:+d}¢")
    print(f"  Black-Scholes overstates fair value by {bs - fair:.1f}¢")

    print()
    print("=" * 66)
    print("Where the two models diverge")
    print("=" * 66)
    print(f"{'time to close':>16}  {'Asian':>8}  {'vanilla BS':>11}  {'gap':>7}")
    for label, secs in [
        ("30 seconds", 30.0),
        ("1 minute", 60.0),
        ("5 minutes", 300.0),
        ("1 hour", 3_600.0),
        ("1 day", 86_400.0),
    ]:
        a = asian.fair_value(spot, strike, secs, vol) * 100
        v = vanilla.fair_value(spot, strike, secs / 3600.0, vol) * 100
        print(f"{label:>16}  {a:>7.2f}¢  {v:>10.2f}¢  {v - a:>6.2f}¢")

    print()
    print("The averaging window only matters near settlement. Far out, the two")
    print("converge, which is the sanity check the test suite asserts. Close in,")
    print("Black-Scholes overstates volatility because it ignores that Kalshi")
    print("settles on a 60-second average rather than a single print.")


if __name__ == "__main__":
    main()
