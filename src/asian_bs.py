"""Asian Binary Option pricer for Kalshi markets.

Kalshi settles using a 60-second average of the underlying asset price,
not a point-in-time snapshot. This makes their contracts effectively
Asian binary options. This module prices them correctly using:

1. Turnbull-Wakeman approximation for the averaging window's reduced vol
2. Partial-observation logic when we're inside the settlement window

The standard Black-Scholes binary model overstates volatility (and thus
misprices) for near-settlement contracts because it ignores the averaging.
"""

import math
from collections.abc import Sequence

from scipy.stats import norm


class AsianBinaryPricer:
    """Pricer for Kalshi's Asian-style binary options.

    Kalshi settles based on a 60-second arithmetic average of the underlying
    asset price. This reduces effective volatility near settlement compared
    to a point-in-time binary option (standard BS).

    Returns fair value as a probability [0, 1].
    """

    SECONDS_PER_YEAR = 365.25 * 24 * 3600  # ~31,557,600

    def __init__(
        self,
        risk_free_rate: float = 0.045,
        averaging_seconds: float = 60.0,
    ):
        """
        Args:
            risk_free_rate: Annualized risk-free rate (default 4.5%).
            averaging_seconds: Duration of the settlement averaging window
                in seconds (default 60 for Kalshi).
        """
        self.r = risk_free_rate
        self.averaging_seconds = averaging_seconds

    def fair_value(
        self,
        spot: float,
        strike: float,
        T_seconds: float,
        vol_annual: float,
        observed_avg: float | None = None,
        observed_seconds: float = 0.0,
    ) -> float:
        """Compute fair probability that the averaged asset price exceeds strike.

        Args:
            spot: Current asset price in USD.
            strike: Strike price in USD.
            T_seconds: Seconds until the END of the averaging window (i.e.,
                seconds until settlement).
            vol_annual: Annualized volatility (e.g. 0.60 for 60%).
            observed_avg: If inside the averaging window, the arithmetic
                average of prices observed so far. None if not yet in window.
            observed_seconds: Number of seconds of observation already
                collected within the averaging window.

        Returns:
            Probability [0, 1] that the averaged price > strike.
        """
        # --- Edge case: T essentially zero ---
        if T_seconds < 1.0:
            if observed_avg is not None:
                raw = 1.0 if observed_avg >= strike else 0.0
            else:
                raw = 1.0 if spot >= strike else 0.0
            return self._clip(raw)

        # --- Edge case: vol = 0 ---
        if vol_annual <= 0.0:
            T_yr = T_seconds / self.SECONDS_PER_YEAR
            forward = spot * math.exp(self.r * T_yr)
            if observed_avg is not None and observed_seconds > 0:
                w_obs = observed_seconds / self.averaging_seconds
                w_rem = 1.0 - w_obs
                if w_rem < 0.01:
                    raw = 1.0 if observed_avg >= strike else 0.0
                else:
                    effective_strike = (strike - w_obs * observed_avg) / w_rem
                    raw = 1.0 if forward >= effective_strike else 0.0
            else:
                raw = 1.0 if forward >= strike else 0.0
            return self._clip(raw)

        # --- In settlement window with partial observation ---
        if observed_seconds > 0 and observed_avg is not None:
            return self._price_partial(
                spot, strike, T_seconds, vol_annual,
                observed_avg, observed_seconds,
            )

        # --- Far from settlement: Turnbull-Wakeman style ---
        return self._price_asian(spot, strike, T_seconds, vol_annual)

    def fair_value_cents(
        self,
        spot: float,
        strike: float,
        T_seconds: float,
        vol_annual: float,
        observed_avg: float | None = None,
        observed_seconds: float = 0.0,
    ) -> int:
        """Same as fair_value but returns integer cents (1-99).

        Convenient for direct comparison with Kalshi yes_price.
        """
        prob = self.fair_value(
            spot, strike, T_seconds, vol_annual,
            observed_avg, observed_seconds,
        )
        return max(1, min(99, round(prob * 100)))

    # ------------------------------------------------------------------
    # Internal pricing methods
    # ------------------------------------------------------------------

    def _price_asian(
        self,
        spot: float,
        strike: float,
        T_seconds: float,
        vol_annual: float,
    ) -> float:
        """Turnbull-Wakeman approximation for arithmetic Asian binary option.

        The averaging window [T - tau, T] has reduced volatility compared
        to spot because the arithmetic average smooths out fluctuations.

        Total variance of ln(average) decomposes into:
        1. Variance of spot reaching time (T - tau): vol^2 * (T - tau)
        2. Variance of the average over [T-tau, T]: vol^2 * tau/3
        Combined: vol^2 * (T - 2*tau/3)

        When tau -> 0 this converges to vol^2 * T (standard BS).
        When T = tau (fully inside averaging window): vol^2 * tau/3.
        """
        tau = self.averaging_seconds / self.SECONDS_PER_YEAR  # averaging window in years
        T = T_seconds / self.SECONDS_PER_YEAR  # time to settlement in years

        # Effective tau: can't be larger than T (if T < averaging window,
        # we're partially inside it from the start)
        tau_eff = min(tau, T)

        # Total variance of ln(average):
        # var = sigma^2 * (T - 2*tau_eff/3)
        var_factor = T - 2.0 * tau_eff / 3.0
        if var_factor < 0:
            var_factor = tau_eff / 3.0  # safety floor

        sigma_A = vol_annual * math.sqrt(var_factor)

        # Drift: expected log of average price at settlement
        # E[ln(A)] ~ ln(S) + (r - 0.5 * sigma_A^2) * T
        # Must use the Asian-adjusted variance (sigma_A^2), not raw vol^2,
        # because the averaging reduces effective variance of the log-average.
        mu_A = math.log(spot) + (self.r - 0.5 * sigma_A ** 2) * T

        # Avoid division by zero for very small sigma_A
        if sigma_A < 1e-15:
            forward_avg = math.exp(mu_A)
            raw = 1.0 if forward_avg >= strike else 0.0
            return self._clip(raw)

        # d2 for binary option: P(avg > K) = N(d2)
        d2 = (mu_A - math.log(strike)) / sigma_A

        raw = norm.cdf(d2)
        return self._clip(raw)

    def _price_partial(
        self,
        spot: float,
        strike: float,
        T_seconds: float,
        vol_annual: float,
        observed_avg: float,
        observed_seconds: float,
    ) -> float:
        """Price when partially inside the averaging window.

        We decompose: final_avg = w_obs * observed_avg + w_rem * remaining_avg.
        The remaining_avg must exceed an effective strike for the contract
        to settle YES.
        """
        w_obs = observed_seconds / self.averaging_seconds
        w_rem = 1.0 - w_obs

        # Nearly done observing — deterministic
        if w_rem < 0.01:
            raw = 1.0 if observed_avg >= strike else 0.0
            return self._clip(raw)

        # Effective strike the remaining average must exceed
        effective_strike = (strike - w_obs * observed_avg) / w_rem

        # Already won: observed portion is so high the remaining can't lose
        if effective_strike <= 0.0:
            return self._clip(1.0)

        # Already lost: observed portion so low that even infinite remaining
        # price can't save it — but we don't clip to 0 here, we let the
        # BS formula handle extreme OTM.

        # Price remaining as standard binary option on spot vs effective_strike
        # with time = remaining seconds in the window.
        remaining_seconds = self.averaging_seconds - observed_seconds
        if remaining_seconds < 1.0:
            raw = 1.0 if spot >= effective_strike else 0.0
            return self._clip(raw)

        T_rem = remaining_seconds / self.SECONDS_PER_YEAR
        vol_rem = vol_annual  # spot vol for the remaining period

        d2 = (math.log(spot / effective_strike) + (self.r - 0.5 * vol_rem ** 2) * T_rem) / (
            vol_rem * math.sqrt(T_rem / 3.0)
        )

        raw = norm.cdf(d2)
        return self._clip(raw)

    def range_fair_value(self, spot: float, strike_lower: float, strike_upper: float,
                         T_seconds: float, vol_annual: float,
                         observed_avg: float = None, observed_seconds: float = 0.0) -> float:
        """Fair value for a range bucket contract: P(strike_lower <= avg_price < strike_upper).

        Computed as P(avg > strike_lower) - P(avg > strike_upper).

        Args:
            spot: Current spot price.
            strike_lower: Lower bound of the range bucket.
            strike_upper: Upper bound of the range bucket.
            T_seconds: Time to settlement in seconds.
            vol_annual: Annualized volatility.
            observed_avg: Observed average price so far (for partial observation).
            observed_seconds: Seconds of the averaging window already observed.

        Returns:
            Fair value probability in [0.001, 0.999].
        """
        p_above_lower = self.fair_value(spot, strike_lower, T_seconds, vol_annual,
                                         observed_avg=observed_avg, observed_seconds=observed_seconds)
        p_above_upper = self.fair_value(spot, strike_upper, T_seconds, vol_annual,
                                         observed_avg=observed_avg, observed_seconds=observed_seconds)
        return max(0.001, min(0.999, p_above_lower - p_above_upper))

    def range_fair_value_cents(self, spot: float, strike_lower: float, strike_upper: float,
                                T_seconds: float, vol_annual: float,
                                observed_avg: float = None, observed_seconds: float = 0.0) -> int:
        """Fair value for range bucket in cents (1-99)."""
        p = self.range_fair_value(spot, strike_lower, strike_upper, T_seconds, vol_annual,
                                   observed_avg=observed_avg, observed_seconds=observed_seconds)
        return max(1, min(99, round(p * 100)))

    @staticmethod
    def _clip(p: float) -> float:
        """Clip probability to [0.001, 0.999]."""
        return max(0.001, min(0.999, p))
