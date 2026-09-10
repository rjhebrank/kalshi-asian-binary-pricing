"""Real-time volatility estimator from tick-level price data.

Maintains a rolling buffer of (timestamp, price) observations and computes
annualized realized volatility over configurable windows, for use where vol
must be estimated from a streaming price feed rather than from daily bars.

Two estimators are available:
1. Realized variance (simple sum of squared log returns)
2. EWMA variance (exponentially weighted, lambda=0.94 by default)
"""

import math
from collections import deque


class RealtimeVolEstimator:
    """Estimate annualized realized volatility from streaming tick data.

    Usage:
        vol_est = RealtimeVolEstimator()
        for ts, price in tick_stream:
            vol_est.update(ts, price)
            vol = vol_est.get_vol(window_seconds=1800)

    Notes:
        - MIN_TICKS (30) acts as a sparse-tick guard: if the vol window has
          fewer than 30 raw ticks, get_vol returns None (unreliable estimate).
          This guards against thin or intermittent feeds.
        - MIN_SUBSAMPLED (5) further guards against too few data points after
          subsampling to 60s bars.
    """

    SECONDS_PER_YEAR = 365.25 * 24 * 3600  # ~31,557,600

    # Safeguards
    MIN_TICKS = 30          # minimum ticks to produce an estimate
    MIN_SUBSAMPLED = 5      # minimum subsampled bars to produce an estimate
    VOL_FLOOR = 0.05        # 5% annualized
    VOL_CAP = 5.0           # 500% annualized

    def __init__(
        self,
        max_history_seconds: float = 7200.0,
        ewma_lambda: float = 0.94,
        subsample_interval: float = 60.0,    # seconds between subsampled ticks
        prior_vol: float = 0.25,             # static prior, acts as a safety net
        blend_realized: float = 0.70,        # realized vol carries primary weight
        blend_prior: float = 0.30,           # prior weight
    ):
        """
        Args:
            max_history_seconds: Maximum seconds of tick data to retain
                (default 2 hours). Older ticks are discarded.
            ewma_lambda: Decay factor for EWMA variance (default 0.94).
            subsample_interval: Seconds between subsampled ticks (default 60).
            prior_vol: Long-term volatility prior (default 0.25 = 25%).
            blend_realized: Weight on realized vol estimate (default 0.70).
            blend_prior: Weight on prior vol (default 0.30).
        """
        self.max_history_seconds = max_history_seconds
        self.ewma_lambda = ewma_lambda
        self._subsample_interval = subsample_interval
        self._prior_vol = prior_vol
        self._blend_realized = blend_realized
        self._blend_prior = blend_prior

        # Rolling buffer of (timestamp_seconds, price)
        self._ticks: deque[tuple[float, float]] = deque()

        # EWMA state
        self._ewma_var: float | None = None
        self._last_price: float | None = None
        self._last_ts: float | None = None

    def update(self, timestamp: float, price: float) -> None:
        """Process a new tick.

        Args:
            timestamp: Unix timestamp in seconds (float).
            price: Asset price at this tick.
        """
        if price <= 0:
            return  # skip invalid prices

        # Update EWMA variance
        if self._last_price is not None and self._last_price > 0:
            log_ret = math.log(price / self._last_price)
            if self._ewma_var is None:
                self._ewma_var = log_ret ** 2
            else:
                self._ewma_var = (
                    self.ewma_lambda * self._ewma_var
                    + (1.0 - self.ewma_lambda) * log_ret ** 2
                )

        self._last_price = price
        self._last_ts = timestamp

        # Append to buffer
        self._ticks.append((timestamp, price))

        # Evict old ticks
        cutoff = timestamp - self.max_history_seconds
        while self._ticks and self._ticks[0][0] < cutoff:
            self._ticks.popleft()

    def get_vol(self, window_seconds: float = 1800.0, method: str = "realized") -> float | None:
        """Get annualized realized vol from the last `window_seconds` of ticks.

        Args:
            window_seconds: Lookback window in seconds (default 300 = 5 min).
            method: "realized" for simple realized variance, "ewma" for
                exponentially weighted.

        Returns:
            Annualized volatility as a decimal (e.g. 0.60 for 60%), or None
            if insufficient data. Clamped to [VOL_FLOOR, VOL_CAP].
        """
        if method == "ewma":
            return self._get_vol_ewma()

        return self._get_vol_realized(window_seconds)

    def _get_vol_realized(self, window_seconds: float) -> float | None:
        """Realized variance estimator over a rolling window."""
        if not self._ticks:
            return None

        latest_ts = self._ticks[-1][0]
        cutoff = latest_ts - window_seconds

        # Collect ticks within window
        ticks_in_window: list[tuple[float, float]] = []
        for ts, price in reversed(self._ticks):
            if ts < cutoff:
                break
            ticks_in_window.append((ts, price))

        ticks_in_window.reverse()

        # Subsample to one tick per subsample_interval bucket
        if len(ticks_in_window) < 2:
            return None
        subsampled = [ticks_in_window[0]]
        bucket_end = ticks_in_window[0][0] + self._subsample_interval
        for ts, price in ticks_in_window[1:]:
            if ts >= bucket_end:
                subsampled.append((ts, price))
                bucket_end = ts + self._subsample_interval

        # IMPORTANT: Do NOT use `is` identity check here — the loop creates new
        # tuple objects, so `is` would almost always be False even for the same tick.
        # Use timestamp comparison instead. Also skip if the gap to last subsampled
        # tick is less than half the subsample interval (avoids biasing vol downward
        # with a short final return).
        last_sub_ts = subsampled[-1][0]
        last_tick_ts = ticks_in_window[-1][0]
        if last_tick_ts - last_sub_ts >= self._subsample_interval * 0.5:
            subsampled.append(ticks_in_window[-1])

        n = len(subsampled)
        if n < self.MIN_SUBSAMPLED:
            return None

        # Compute sum of squared log returns
        sum_sq = 0.0
        for i in range(1, n):
            p_prev = subsampled[i - 1][1]
            p_curr = subsampled[i][1]
            if p_prev > 0 and p_curr > 0:
                lr = math.log(p_curr / p_prev)
                sum_sq += lr * lr

        n_returns = n - 1
        if n_returns < 1:
            return None

        # Realized variance per tick interval
        realized_var = sum_sq / n_returns

        # Average tick interval in seconds
        total_span = subsampled[-1][0] - subsampled[0][0]
        if total_span <= 0:
            return None
        avg_tick_interval = total_span / n_returns

        # Annualize: ticks_per_year = seconds_per_year / avg_tick_interval
        ticks_per_year = self.SECONDS_PER_YEAR / avg_tick_interval
        annualized_vol = math.sqrt(realized_var * ticks_per_year)

        # Blend with prior before clamping
        annualized_vol = self._blend_realized * annualized_vol + self._blend_prior * self._prior_vol

        return self._clamp_vol(annualized_vol)

    def _get_vol_ewma(self) -> float | None:
        """EWMA variance estimator."""
        if self._ewma_var is None or self._last_ts is None:
            return None

        # Need enough ticks for EWMA to have warmed up
        if len(self._ticks) < self.MIN_TICKS:
            return None

        # Estimate average tick interval from recent data
        n = len(self._ticks)
        total_span = self._ticks[-1][0] - self._ticks[0][0]
        if total_span <= 0:
            return None
        avg_tick_interval = total_span / (n - 1)

        ticks_per_year = self.SECONDS_PER_YEAR / avg_tick_interval
        annualized_vol = math.sqrt(self._ewma_var * ticks_per_year)

        # Blend with prior before clamping
        annualized_vol = self._blend_realized * annualized_vol + self._blend_prior * self._prior_vol

        return self._clamp_vol(annualized_vol)

    def _clamp_vol(self, vol: float) -> float:
        """Clamp volatility to [VOL_FLOOR, VOL_CAP]."""
        if math.isnan(vol) or math.isinf(vol):
            return self.VOL_FLOOR
        return max(self.VOL_FLOOR, min(self.VOL_CAP, vol))

    @property
    def tick_count(self) -> int:
        """Number of ticks currently in the buffer."""
        return len(self._ticks)

    @property
    def ewma_vol(self) -> float | None:
        """Current EWMA vol estimate (convenience property)."""
        return self.get_vol(method="ewma")

    def get_vol_multi(self) -> dict[str, float | None]:
        """Return vol estimates at standard windows: 60s, 300s, 3600s.

        Returns:
            Dict mapping window label to annualized vol (or None).
        """
        return {
            "60s": self.get_vol(window_seconds=60.0),
            "300s": self.get_vol(window_seconds=300.0),
            "1800s": self.get_vol(window_seconds=1800.0),
            "3600s": self.get_vol(window_seconds=3600.0),
            "ewma": self.get_vol(method="ewma"),
        }
