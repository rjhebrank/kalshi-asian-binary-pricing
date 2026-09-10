"""Black-Scholes Binary Option pricer for Kalshi markets.

Implements the cash-or-nothing binary call pricing model for Kalshi's
binary options (e.g., "Will BTC be above $X at settlement?").

Kalshi conventions:
    - yes_price is in CENTS (integers 1-99). yes_price=65 means 65% implied prob.
    - Strike prices are in USD (e.g., 69749.99 for BTC).
    - Strikes are spaced $250 apart for BTC.
    - Settlement uses a 60-second average, not point-in-time.
    - Most trading happens in the last ~25 minutes before settlement.
"""

import math

import numpy as np
from scipy.stats import norm


class BlackScholesBinary:
    """Black-Scholes pricer for Kalshi binary options.

    Returns fair value as a probability [0, 1].
    To compare with Kalshi prices: fair_value * 100 = cents.

    Uses the cash-or-nothing binary call formula:
        P(above) = e^(-rT) * N(d2)
    where:
        d2 = [ln(S/K) + (r - sigma^2/2) * T] / (sigma * sqrt(T))
    """

    # Hours per year for annualization of hourly data
    HOURS_PER_YEAR = 8760

    # Volatility computation windows in hours
    VOL_WINDOWS = {
        "4h": 4,
        "24h": 24,
        "168h": 168,   # 7 days
        "720h": 720,   # 30 days
    }

    def __init__(self, risk_free_rate: float = 0.045):
        """
        Args:
            risk_free_rate: Annualized risk-free rate (default 4.5%).
        """
        self.r = risk_free_rate

    def fair_value(
        self,
        spot: float,
        strike: float,
        time_to_settlement_hours: float,
        volatility: float,
    ) -> float:
        """Compute fair probability that asset will be ABOVE strike at settlement.

        This is the cash-or-nothing binary call price under Black-Scholes:
            P = e^(-rT) * N(d2)

        For Kalshi's short-duration markets (hourly), the discount factor
        is negligible (~0.9999995 for 1 hour at 4.5%), but we include it
        for correctness.

        Args:
            spot: Current asset price in USD.
            strike: Strike price in USD.
            time_to_settlement_hours: Hours until settlement (can be fractional).
            volatility: Annualized volatility (e.g., 0.60 for 60%).

        Returns:
            Probability [0, 1] that asset > strike at settlement.

        Raises:
            ValueError: If spot or strike <= 0.
        """
        # --- Input validation ---
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")
        if strike <= 0:
            raise ValueError(f"strike must be positive, got {strike}")
        if volatility < 0:
            raise ValueError(f"volatility must be non-negative, got {volatility}")
        if time_to_settlement_hours < 0:
            raise ValueError(
                f"time_to_settlement_hours must be non-negative, got {time_to_settlement_hours}"
            )

        # --- Edge case: T = 0 or very small (< 1 second = 1/3600 hours) ---
        if time_to_settlement_hours < 1.0 / 3600.0:
            return 1.0 if spot > strike else 0.0

        # --- Edge case: volatility = 0 ---
        if volatility == 0.0:
            # With zero vol, the asset drifts deterministically at rate r.
            # Forward price = S * e^(rT)
            T = time_to_settlement_hours / self.HOURS_PER_YEAR
            forward = spot * math.exp(self.r * T)
            return 1.0 if forward > strike else 0.0

        # --- Standard Black-Scholes binary call ---
        T = time_to_settlement_hours / self.HOURS_PER_YEAR

        d2 = (math.log(spot / strike) + (self.r - 0.5 * volatility**2) * T) / (
            volatility * math.sqrt(T)
        )

        # Cash-or-nothing binary call: e^(-rT) * N(d2)
        fair = math.exp(-self.r * T) * norm.cdf(d2)

        return float(fair)

    def fair_value_cents(
        self,
        spot: float,
        strike: float,
        time_to_settlement_hours: float,
        volatility: float,
    ) -> float:
        """Same as fair_value but returns price in Kalshi cents [0, 100].

        Convenient for direct comparison with Kalshi yes_price.
        """
        return self.fair_value(spot, strike, time_to_settlement_hours, volatility) * 100.0

    def compute_vol(
        self, prices: np.ndarray | list[float], window_hours: int = 24
    ) -> float:
        """Compute annualized historical volatility from hourly prices.

        Uses log-return standard deviation, annualized by sqrt(8760) since
        the input data is hourly.

        Args:
            prices: Array-like of hourly prices (chronological order).
            window_hours: Number of hours (prices) to use. Uses the last
                `window_hours` prices. If len(prices) < window_hours, uses
                all available prices.

        Returns:
            Annualized volatility as a decimal (e.g., 0.60 for 60%).

        Raises:
            ValueError: If fewer than 2 prices provided.
        """
        prices = np.asarray(prices, dtype=np.float64)

        if len(prices) < 2:
            raise ValueError(f"Need at least 2 prices, got {len(prices)}")

        # Use the last `window_hours` prices (we need window_hours+1 prices
        # to get window_hours returns, but if we don't have enough, use all).
        if len(prices) > window_hours + 1:
            prices = prices[-(window_hours + 1) :]

        # Log returns
        log_returns = np.diff(np.log(prices))

        # Remove any NaN/inf that could arise from zero or negative prices
        log_returns = log_returns[np.isfinite(log_returns)]

        if len(log_returns) < 1:
            raise ValueError("No valid log returns computed")

        # Annualize: hourly std * sqrt(hours_per_year)
        hourly_std = np.std(log_returns, ddof=1)
        annualized_vol = hourly_std * math.sqrt(self.HOURS_PER_YEAR)

        return float(annualized_vol)

    def compute_vol_multiple(
        self, prices: np.ndarray | list[float]
    ) -> dict[str, float | None]:
        """Return dict of annualized volatilities at different time windows.

        Windows: 4h, 24h, 168h (7 days), 720h (30 days).

        Args:
            prices: Array-like of hourly prices (chronological order).

        Returns:
            Dict mapping window label to annualized vol, or None if
            insufficient data for that window.
        """
        prices = np.asarray(prices, dtype=np.float64)
        result = {}

        for label, hours in self.VOL_WINDOWS.items():
            # Need at least hours+1 prices for `hours` returns,
            # but compute_vol handles shorter arrays gracefully.
            # We need at least 2 prices minimum.
            if len(prices) < 2:
                result[label] = None
                continue
            try:
                result[label] = self.compute_vol(prices, window_hours=hours)
            except ValueError:
                result[label] = None

        return result
