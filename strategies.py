from typing import List, Dict, Any, Optional

class StrategyBase:
    def ingest(self, bar: Dict[str, Any]) -> None:
        raise NotImplementedError

    def signal(self) -> Dict[str, Any]:
        # {action: 'enter_long'|'enter_short'|'exit'|'hold'}
        raise NotImplementedError

    # Optional: strategy-specific debug state for console rendering
    def debug_state(self) -> Dict[str, Any]:
        return {}

class EmaCross(StrategyBase):
    """
    EMA Cross strategy with SMA-seeded EMA initialization to match chart platforms.

    Behavior:
      - Collect closes until we can compute SMA(fast) and SMA(slow).
      - Seed EMA_fast with SMA(fast), EMA_slow with SMA(slow).
      - After seeding, update each bar with recursive EMA:
           EMA_t = k * Price_t + (1 - k) * EMA_{t-1}, k = 2 / (period + 1)
      - Signals:
           * Exit long when (fast - slow) flips from >= 0 to < 0
           * Exit short when (fast - slow) flips from <= 0 to > 0
           * Enter long when flat and (fast - slow) flips from <= 0 to > 0
           * Enter short when flat and (fast - slow) flips from >= 0 to < 0
      - No signals until both EMAs are initialized from SMA.
    """
    def __init__(self, fast: int = 9, slow: int = 21):
        if fast >= slow:
            raise ValueError("fast EMA must be < slow EMA")
        self.fast = fast
        self.slow = slow

        # Price buffer (closed bars only; app.py feeds backfill first)
        self.prices: List[float] = []

        # Position and EMA state
        self.pos_dir: int = 0                 # +1 long, -1 short, 0 flat
        self.ema_fast: Optional[float] = None
        self.ema_slow: Optional[float] = None
        self.prev_rel: Optional[int] = None   # previous sign of (fast - slow)

        # Cached multipliers
        self.k_fast = 2.0 / (self.fast + 1.0)
        self.k_slow = 2.0 / (self.slow + 1.0)

    @staticmethod
    def _sign(x: float) -> int:
        return 1 if x > 0 else (-1 if x < 0 else 0)

    @staticmethod
    def _sma(values: List[float]) -> float:
        return sum(values) / float(len(values))

    def _update_ema(self, prev: float, price: float, k: float) -> float:
        # Standard EMA update
        return (price * k) + (prev * (1.0 - k))

    def _maybe_seed_emas(self) -> None:
        """
        If we have enough bars, seed EMA_fast with SMA(fast) and EMA_slow with SMA(slow).
        """
        n = len(self.prices)
        if self.ema_fast is None and n >= self.fast:
            self.ema_fast = self._sma(self.prices[-self.fast:])
        if self.ema_slow is None and n >= self.slow:
            self.ema_slow = self._sma(self.prices[-self.slow:])

    def ingest(self, bar: Dict[str, Any]) -> None:
        px = float(bar["c"])
        self.prices.append(px)

        # Seed EMAs with SMA once enough bars have accumulated
        self._maybe_seed_emas()

        # IMPORTANT: Do NOT update on the same bar as seeding.
        # Only start recursive EMA updates AFTER the seed bar.
        n = len(self.prices)
        if self.ema_fast is not None and n > self.fast:
            self.ema_fast = self._update_ema(self.ema_fast, px, self.k_fast)
        if self.ema_slow is not None and n > self.slow:
            self.ema_slow = self._update_ema(self.ema_slow, px, self.k_slow)

    def signal(self) -> Dict[str, Any]:
        # Hold until both EMAs are initialized from SMA
        if self.ema_fast is None or self.ema_slow is None:
            return {"action": "hold"}

        rel_now = self._sign(self.ema_fast - self.ema_slow)

        # Initialize previous relationship once
        if self.prev_rel is None:
            self.prev_rel = rel_now
            return {"action": "hold"}

        # Immediate exits on flip against current position
        if self.pos_dir > 0 and rel_now < 0 and self.prev_rel >= 0:
            self.pos_dir = 0
            self.prev_rel = rel_now
            return {"action": "exit"}
        if self.pos_dir < 0 and rel_now > 0 and self.prev_rel <= 0:
            self.pos_dir = 0
            self.prev_rel = rel_now
            return {"action": "exit"}

        # Entries only on flip when flat
        if self.pos_dir == 0:
            if rel_now > 0 and self.prev_rel <= 0:
                self.pos_dir = +1
                self.prev_rel = rel_now
                return {"action": "enter_long"}
            if rel_now < 0 and self.prev_rel >= 0:
                self.pos_dir = -1
                self.prev_rel = rel_now
                return {"action": "enter_short"}

        # No action this bar
        self.prev_rel = rel_now
        return {"action": "hold"}

    def debug_state(self) -> Dict[str, Any]:
        bars_seen = len(self.prices)
        ready = (self.ema_fast is not None and self.ema_slow is not None)
        bars_until_ready = max(0, max(self.fast - bars_seen, self.slow - bars_seen))
        return {
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "rel": (None if self.ema_fast is None or self.ema_slow is None
                    else (1 if self.ema_fast > self.ema_slow else -1 if self.ema_fast < self.ema_slow else 0)),
            "pos_dir": self.pos_dir,
            "bars_seen": bars_seen,
            "ready": ready,
            "bars_until_ready": bars_until_ready,
        }

def get_strategy(name: str):
    REG = {
        "ema_cross": EmaCross,
    }
    if name not in REG:
        raise KeyError(f"Unknown strategy: {name}")
    return REG[name]
