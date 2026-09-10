"""Tests for the Asian Binary Option pricer.

Validates the AsianBinaryPricer against the standard BlackScholesBinary model
and checks correctness of partial-observation logic, edge cases, and
mathematical properties (monotonicity, symmetry, convergence).
"""

import math
import pytest

from src.asian_bs import AsianBinaryPricer
from src.black_scholes import BlackScholesBinary


@pytest.fixture
def asian():
    return AsianBinaryPricer(risk_free_rate=0.045, averaging_seconds=60.0)


@pytest.fixture
def bs():
    return BlackScholesBinary(risk_free_rate=0.045)


# ------------------------------------------------------------------
# Convergence: as averaging_seconds -> 0, Asian -> standard BS
# ------------------------------------------------------------------

class TestConvergence:
    def test_convergence_to_bs(self, bs):
        """As averaging window shrinks to near-zero, Asian price approaches BS."""
        spot = 100_000.0
        strike = 100_500.0
        vol = 0.60
        T_hours = 1.0  # 1 hour to settlement
        T_seconds = T_hours * 3600

        bs_price = bs.fair_value(spot, strike, T_hours, vol)

        # Use a tiny averaging window so it converges to point-in-time
        pricer_tiny = AsianBinaryPricer(risk_free_rate=0.045, averaging_seconds=0.001)
        asian_price = pricer_tiny.fair_value(spot, strike, T_seconds, vol)

        # Should be very close (within 1% absolute)
        assert abs(asian_price - bs_price) < 0.01, (
            f"Asian ({asian_price:.6f}) should converge to BS ({bs_price:.6f}) "
            f"with tiny averaging window"
        )

    def test_convergence_multiple_strikes(self, bs):
        """Convergence holds across ITM, ATM, and OTM strikes."""
        spot = 100_000.0
        vol = 0.60
        T_hours = 2.0
        T_seconds = T_hours * 3600

        pricer_tiny = AsianBinaryPricer(risk_free_rate=0.045, averaging_seconds=0.001)

        for strike in [95_000, 100_000, 105_000]:
            bs_price = bs.fair_value(spot, strike, T_hours, vol)
            asian_price = pricer_tiny.fair_value(spot, strike, T_seconds, vol)
            assert abs(asian_price - bs_price) < 0.01, (
                f"Strike={strike}: Asian={asian_price:.6f}, BS={bs_price:.6f}"
            )


# ------------------------------------------------------------------
# Partial observation tests
# ------------------------------------------------------------------

class TestPartialObservation:
    def test_partial_observation_high(self, asian):
        """59/60 seconds observed, avg well above strike -> ~1.0."""
        # Observed avg is 101,000, strike is 100,000
        # Only 1 second left — almost certain YES
        fv = asian.fair_value(
            spot=101_000.0,
            strike=100_000.0,
            T_seconds=1.0,  # 1 second left
            vol_annual=0.60,
            observed_avg=101_000.0,
            observed_seconds=59.0,
        )
        assert fv > 0.99, f"Expected ~1.0, got {fv:.6f}"

    def test_partial_observation_low(self, asian):
        """59/60 seconds observed, avg well below strike -> ~0.0."""
        fv = asian.fair_value(
            spot=99_000.0,
            strike=100_000.0,
            T_seconds=1.0,
            vol_annual=0.60,
            observed_avg=99_000.0,
            observed_seconds=59.0,
        )
        assert fv < 0.01, f"Expected ~0.0, got {fv:.6f}"

    def test_partial_effective_strike_negative(self, asian):
        """Observed avg so high that effective strike goes negative -> 1.0."""
        # observed_avg = 200,000 for 50/60s, strike = 100,000
        # effective_strike = (100000 - (50/60)*200000) / (10/60) ~ very negative
        fv = asian.fair_value(
            spot=200_000.0,
            strike=100_000.0,
            T_seconds=10.0,
            vol_annual=0.60,
            observed_avg=200_000.0,
            observed_seconds=50.0,
        )
        assert fv > 0.99, f"Expected ~1.0 (effective strike negative), got {fv:.6f}"

    def test_partial_nearly_done(self, asian):
        """w_rem < 0.01 — deterministic based on observed avg."""
        # 59.5 seconds observed out of 60 => w_rem = 0.5/60 = 0.0083 < 0.01
        fv_yes = asian.fair_value(
            spot=100_500.0,
            strike=100_000.0,
            T_seconds=0.5,
            vol_annual=0.60,
            observed_avg=100_500.0,
            observed_seconds=59.5,
        )
        # With T_seconds < 1, hits the T < 1 edge case which checks observed_avg vs strike
        assert fv_yes > 0.99, f"Expected ~1.0, got {fv_yes:.6f}"

        fv_no = asian.fair_value(
            spot=99_500.0,
            strike=100_000.0,
            T_seconds=0.5,
            vol_annual=0.60,
            observed_avg=99_500.0,
            observed_seconds=59.5,
        )
        assert fv_no < 0.01, f"Expected ~0.0, got {fv_no:.6f}"


# ------------------------------------------------------------------
# ATM symmetry
# ------------------------------------------------------------------

class TestATMSymmetry:
    def test_atm_roughly_half(self, asian):
        """Fair value at the money with large T should be roughly 0.50."""
        fv = asian.fair_value(
            spot=100_000.0,
            strike=100_000.0,
            T_seconds=3600 * 24,  # 24 hours
            vol_annual=0.60,
        )
        # Should be close to 0.50 (drift is small for 24h)
        assert 0.40 < fv < 0.60, f"ATM fair value should be ~0.50, got {fv:.6f}"

    def test_atm_short_duration(self, asian):
        """ATM with short duration should also be ~0.50."""
        fv = asian.fair_value(
            spot=100_000.0,
            strike=100_000.0,
            T_seconds=600,  # 10 minutes
            vol_annual=0.60,
        )
        assert 0.40 < fv < 0.60, f"ATM short-duration: {fv:.6f}"


# ------------------------------------------------------------------
# Monotonicity
# ------------------------------------------------------------------

class TestMonotonicity:
    def test_higher_spot_higher_value(self, asian):
        """Higher spot price -> higher fair value (call option)."""
        strike = 100_000.0
        T_seconds = 1800.0  # 30 min
        vol = 0.60

        fv_low = asian.fair_value(spot=99_000.0, strike=strike, T_seconds=T_seconds, vol_annual=vol)
        fv_mid = asian.fair_value(spot=100_000.0, strike=strike, T_seconds=T_seconds, vol_annual=vol)
        fv_high = asian.fair_value(spot=101_000.0, strike=strike, T_seconds=T_seconds, vol_annual=vol)

        assert fv_low < fv_mid < fv_high, (
            f"Monotonicity violated: low={fv_low:.6f}, mid={fv_mid:.6f}, high={fv_high:.6f}"
        )

    def test_higher_vol_wider_distribution(self, asian):
        """For ATM option, higher vol shouldn't move fair value much (stays ~0.5).
        For OTM option, higher vol should increase fair value."""
        strike = 102_000.0  # OTM
        spot = 100_000.0
        T_seconds = 3600.0

        fv_low_vol = asian.fair_value(spot=spot, strike=strike, T_seconds=T_seconds, vol_annual=0.20)
        fv_high_vol = asian.fair_value(spot=spot, strike=strike, T_seconds=T_seconds, vol_annual=0.80)

        assert fv_high_vol > fv_low_vol, (
            f"OTM: higher vol should increase value. low_vol={fv_low_vol:.6f}, high_vol={fv_high_vol:.6f}"
        )


# ------------------------------------------------------------------
# Vol = 0 (deterministic)
# ------------------------------------------------------------------

class TestVolZero:
    def test_vol_zero_above(self, asian):
        """Vol=0, spot > strike -> deterministic 1.0 (clipped to 0.999)."""
        fv = asian.fair_value(
            spot=101_000.0,
            strike=100_000.0,
            T_seconds=3600.0,
            vol_annual=0.0,
        )
        assert fv > 0.99, f"Vol=0, spot > strike: expected ~1.0, got {fv:.6f}"

    def test_vol_zero_below(self, asian):
        """Vol=0, spot < strike -> deterministic 0.0 (clipped to 0.001)."""
        fv = asian.fair_value(
            spot=99_000.0,
            strike=100_000.0,
            T_seconds=3600.0,
            vol_annual=0.0,
        )
        assert fv < 0.01, f"Vol=0, spot < strike: expected ~0.0, got {fv:.6f}"


# ------------------------------------------------------------------
# Boundary: T = 0
# ------------------------------------------------------------------

class TestBoundaryTZero:
    def test_t_zero_with_observed_avg_above(self, asian):
        """T=0, observed_avg > strike -> YES."""
        fv = asian.fair_value(
            spot=100_000.0,
            strike=99_000.0,
            T_seconds=0.0,
            vol_annual=0.60,
            observed_avg=100_000.0,
            observed_seconds=60.0,
        )
        assert fv > 0.99, f"T=0, avg > strike: {fv:.6f}"

    def test_t_zero_with_observed_avg_below(self, asian):
        """T=0, observed_avg < strike -> NO."""
        fv = asian.fair_value(
            spot=100_000.0,
            strike=101_000.0,
            T_seconds=0.0,
            vol_annual=0.60,
            observed_avg=100_000.0,
            observed_seconds=60.0,
        )
        assert fv < 0.01, f"T=0, avg < strike: {fv:.6f}"

    def test_t_zero_no_observation(self, asian):
        """T=0, no observation -> use spot vs strike."""
        fv = asian.fair_value(
            spot=101_000.0,
            strike=100_000.0,
            T_seconds=0.0,
            vol_annual=0.60,
        )
        assert fv > 0.99, f"T=0, spot > strike: {fv:.6f}"


# ------------------------------------------------------------------
# Output range
# ------------------------------------------------------------------

class TestRange:
    def test_output_always_in_range(self, asian):
        """Output is always in [0.001, 0.999] for non-degenerate inputs."""
        test_cases = [
            # (spot, strike, T_seconds, vol)
            (100_000, 100_000, 3600, 0.60),     # ATM
            (100_000, 200_000, 3600, 0.60),     # deep OTM
            (200_000, 100_000, 3600, 0.60),     # deep ITM
            (100_000, 100_000, 60, 0.10),       # low vol, short time
            (100_000, 100_000, 86400, 2.0),     # high vol, long time
            (100_000, 100_001, 10, 0.60),       # near ATM, very short
            (50_000, 100_000, 3600, 0.30),      # very deep OTM
            (100_000, 50_000, 3600, 0.30),      # very deep ITM
        ]

        for spot, strike, T, vol in test_cases:
            fv = asian.fair_value(spot=spot, strike=strike, T_seconds=T, vol_annual=vol)
            assert 0.001 <= fv <= 0.999, (
                f"Out of range: spot={spot}, strike={strike}, T={T}, vol={vol} -> {fv}"
            )


# ------------------------------------------------------------------
# Asian model gives MORE extreme probabilities near settlement
# ------------------------------------------------------------------

class TestAsianMoreExtreme:
    def test_asian_more_extreme_than_bs_near_settlement(self, asian, bs):
        """Near settlement, Asian model should give more extreme probs than BS.

        Because averaged volatility < spot volatility, the Asian model is
        more "confident" about the outcome. For an OTM option, this means
        a LOWER probability (closer to 0). For an ITM option, a HIGHER
        probability (closer to 1).

        We use slightly OTM/ITM parameters that keep both models in the
        non-clipped region (away from 0.001/0.999 floors).
        """
        strike = 100_000.0
        vol = 0.60
        T_seconds = 120.0  # 2 minutes to settlement (close to averaging window)
        T_hours = T_seconds / 3600.0

        # OTM case: spot slightly below strike
        spot_otm = 99_900.0
        asian_fv = asian.fair_value(spot=spot_otm, strike=strike, T_seconds=T_seconds, vol_annual=vol)
        bs_fv = bs.fair_value(spot=spot_otm, strike=strike, time_to_settlement_hours=T_hours, volatility=vol)

        # Asian OTM should be LOWER (more extreme toward 0) than BS
        assert asian_fv < bs_fv, (
            f"OTM near settlement: Asian ({asian_fv:.6f}) should be < BS ({bs_fv:.6f}) "
            f"because averaging reduces vol"
        )

        # ITM case: spot slightly above strike
        spot_itm = 100_100.0
        asian_fv_itm = asian.fair_value(spot=spot_itm, strike=strike, T_seconds=T_seconds, vol_annual=vol)
        bs_fv_itm = bs.fair_value(spot=spot_itm, strike=strike, time_to_settlement_hours=T_hours, volatility=vol)

        # Asian ITM should be HIGHER (more extreme toward 1) than BS
        assert asian_fv_itm > bs_fv_itm, (
            f"ITM near settlement: Asian ({asian_fv_itm:.6f}) should be > BS ({bs_fv_itm:.6f}) "
            f"because averaging reduces vol"
        )


# ------------------------------------------------------------------
# fair_value_cents convenience method
# ------------------------------------------------------------------

class TestFairValueCents:
    def test_cents_range(self, asian):
        """fair_value_cents returns integer in [1, 99]."""
        cents = asian.fair_value_cents(
            spot=100_000.0,
            strike=100_000.0,
            T_seconds=3600.0,
            vol_annual=0.60,
        )
        assert isinstance(cents, int)
        assert 1 <= cents <= 99

    def test_cents_matches_probability(self, asian):
        """Cents should be round(probability * 100) clamped to [1, 99]."""
        prob = asian.fair_value(
            spot=100_000.0,
            strike=99_000.0,
            T_seconds=3600.0,
            vol_annual=0.60,
        )
        cents = asian.fair_value_cents(
            spot=100_000.0,
            strike=99_000.0,
            T_seconds=3600.0,
            vol_annual=0.60,
        )
        expected_cents = max(1, min(99, round(prob * 100)))
        assert cents == expected_cents


# ------------------------------------------------------------------
# Vol estimator tests
# ------------------------------------------------------------------

class TestVolEstimator:
    def test_basic_vol_estimation(self):
        """Basic vol estimation from synthetic tick data."""
        from src.vol_estimator import RealtimeVolEstimator

        est = RealtimeVolEstimator()

        # Generate 600 ticks, 1 second apart, with small random-ish moves
        # (need enough ticks to produce >=5 subsampled 60s bars for a 300s window)
        import random
        random.seed(42)
        price = 100_000.0
        ts = 1_000_000.0

        for i in range(600):
            ts += 1.0
            price *= math.exp(0.001 * (random.gauss(0, 1)))
            est.update(ts, price)

        vol = est.get_vol(window_seconds=300.0)
        assert vol is not None, "Should have enough ticks for 300s window"
        assert 0.10 <= vol <= 5.0, f"Vol out of bounds: {vol}"

    def test_insufficient_ticks(self):
        """Returns None when fewer than MIN_TICKS."""
        from src.vol_estimator import RealtimeVolEstimator

        est = RealtimeVolEstimator()
        est.update(1.0, 100_000.0)
        est.update(2.0, 100_001.0)

        vol = est.get_vol(window_seconds=60.0)
        assert vol is None, "Should be None with only 2 ticks"

    def test_ewma_vol(self):
        """EWMA vol estimator produces a result with enough data."""
        from src.vol_estimator import RealtimeVolEstimator

        est = RealtimeVolEstimator()

        import random
        random.seed(123)
        price = 50_000.0
        ts = 0.0

        for _ in range(200):
            ts += 1.0
            price *= math.exp(0.0005 * random.gauss(0, 1))
            est.update(ts, price)

        vol = est.get_vol(method="ewma")
        assert vol is not None
        assert 0.10 <= vol <= 5.0

    def test_multi_window(self):
        """get_vol_multi returns dict with expected keys."""
        from src.vol_estimator import RealtimeVolEstimator

        est = RealtimeVolEstimator(max_history_seconds=7200)

        import random
        random.seed(99)
        price = 100_000.0
        ts = 0.0

        for _ in range(500):
            ts += 1.0
            price *= math.exp(0.0003 * random.gauss(0, 1))
            est.update(ts, price)

        result = est.get_vol_multi()
        assert "60s" in result
        assert "300s" in result
        assert "3600s" in result
        assert "ewma" in result

        # 60s window returns None due to subsampling (60s subsample interval
        # means <5 bars in a 60s window), but 300s should have a value
        assert result["300s"] is not None
        # 3600s window — we only have 500 seconds of data, might be None
        # depending on how many ticks fall in window


# ------------------------------------------------------------------
# Range bucket pricing
# ------------------------------------------------------------------

class TestRangeBucket:
    """Tests for range bucket pricing."""

    def test_range_probabilities_sum_to_one(self):
        """All range buckets for an event should sum to ~1.0."""
        pricer = AsianBinaryPricer()
        spot = 97000.0
        vol = 0.60
        T = 3600.0  # 1 hour

        # Simulate buckets: ...-96500, 96500-97000, 97000-97500, 97500-...
        # Below bucket: P(avg < 96500) = 1 - P(avg > 96500)
        p_below = 1.0 - pricer.fair_value(spot, 96500.0, T, vol)

        # Range buckets
        p_96500_97000 = pricer.range_fair_value(spot, 96500.0, 97000.0, T, vol)
        p_97000_97500 = pricer.range_fair_value(spot, 97000.0, 97500.0, T, vol)

        # Above bucket: P(avg > 97500)
        p_above = pricer.fair_value(spot, 97500.0, T, vol)

        total = p_below + p_96500_97000 + p_97000_97500 + p_above
        assert abs(total - 1.0) < 0.01, f"Sum = {total}, expected ~1.0"

    def test_range_probabilities_sum_many_buckets(self):
        """Full set of buckets with $500 width summing to ~1.0."""
        pricer = AsianBinaryPricer()
        spot = 97000.0
        vol = 0.60
        T = 3600.0

        # Create buckets from 94000 to 100000 in $500 steps
        boundaries = list(range(94000, 100500, 500))

        # Below first boundary
        total = 1.0 - pricer.fair_value(spot, float(boundaries[0]), T, vol)

        # Range buckets
        for i in range(len(boundaries) - 1):
            p = pricer.range_fair_value(spot, float(boundaries[i]), float(boundaries[i+1]), T, vol)
            total += p

        # Above last boundary
        total += pricer.fair_value(spot, float(boundaries[-1]), T, vol)

        assert abs(total - 1.0) < 0.02, f"Sum = {total}, expected ~1.0"

    def test_narrower_range_lower_probability(self):
        """Narrower range -> lower probability."""
        pricer = AsianBinaryPricer()
        spot = 97000.0
        vol = 0.60
        T = 3600.0

        p_wide = pricer.range_fair_value(spot, 96500.0, 97500.0, T, vol)   # $1000 range
        p_narrow = pricer.range_fair_value(spot, 96750.0, 97250.0, T, vol)  # $500 range

        assert p_wide > p_narrow, f"Wide ({p_wide}) should be > narrow ({p_narrow})"

    def test_atm_bucket_highest_probability(self):
        """The bucket containing spot should have the highest probability."""
        pricer = AsianBinaryPricer()
        spot = 97000.0
        vol = 0.60
        T = 3600.0

        p_atm = pricer.range_fair_value(spot, 96750.0, 97250.0, T, vol)
        p_otm_up = pricer.range_fair_value(spot, 97750.0, 98250.0, T, vol)
        p_otm_down = pricer.range_fair_value(spot, 95750.0, 96250.0, T, vol)

        assert p_atm > p_otm_up, f"ATM ({p_atm}) should be > OTM up ({p_otm_up})"
        assert p_atm > p_otm_down, f"ATM ({p_atm}) should be > OTM down ({p_otm_down})"

    def test_convergence_at_settlement(self):
        """As T->0, only the bucket containing spot should have probability ~1."""
        pricer = AsianBinaryPricer()
        spot = 97000.0
        vol = 0.60
        T = 1.0  # 1 second to settlement

        p_containing = pricer.range_fair_value(spot, 96500.0, 97500.0, T, vol)
        p_away = pricer.range_fair_value(spot, 98000.0, 98500.0, T, vol)

        assert p_containing > 0.9, f"Containing bucket should be ~1.0, got {p_containing}"
        assert p_away < 0.1, f"Away bucket should be ~0.0, got {p_away}"

    def test_range_fair_value_cents(self):
        """Cents version returns integer in 1-99."""
        pricer = AsianBinaryPricer()
        cents = pricer.range_fair_value_cents(97000.0, 96500.0, 97500.0, 3600.0, 0.60)
        assert isinstance(cents, int)
        assert 1 <= cents <= 99

    def test_range_with_partial_observation(self):
        """Range bucket with partial observation."""
        pricer = AsianBinaryPricer()
        spot = 97000.0
        vol = 0.60
        T = 30.0  # 30 seconds left

        # Observed average right in the middle of the range
        p = pricer.range_fair_value(spot, 96500.0, 97500.0, T, vol,
                                     observed_avg=97000.0, observed_seconds=30.0)
        assert 0.001 <= p <= 0.999
