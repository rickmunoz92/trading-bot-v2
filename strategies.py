from typing import List, Dict, Any, Optional, Type

class StrategyBase:
    def ingest(self, bar: Dict[str, Any]) -> None:
        raise NotImplementedError

    def signal(self) -> Dict[str, Any]:
        # {action: 'enter_long'|'enter_short'|'exit'|'hold'}
        raise NotImplementedError

    # Optional: strategy-specific debug state for console rendering
    def debug_state(self) -> Dict[str, Any]:
        return {}

    def set_position_dir(self, dir: int) -> None:
        """
        dir: +1 (broker net long), -1 (broker net short), 0 (flat)
        Default no-op; strategies override if they track pos_dir.
        """
        pass


class EmaCross(StrategyBase):
    """EMA Cross strategy with SMA-seeded initialization and live preview EMAs."""
    def __init__(self, fast: int = 9, slow: int = 21):
        if fast >= slow:
            raise ValueError("fast EMA must be < slow EMA")
        self.fast = fast
        self.slow = slow
        self.prices: List[float] = []
        self.pos_dir: int = 0
        self.ema_fast: Optional[float] = None
        self.ema_slow: Optional[float] = None
        self.prev_rel: Optional[int] = None
        # Live preview (intrabar)
        self.live_ema_fast: Optional[float] = None
        self.live_ema_slow: Optional[float] = None
        # Multipliers
        self.k_fast = 2.0 / (self.fast + 1.0)
        self.k_slow = 2.0 / (self.slow + 1.0)

    def set_position_dir(self, dir: int) -> None:
        if dir in (-1, 0, 1):
            self.pos_dir = dir

    @staticmethod
    def _sign(x: float) -> int:
        return 1 if x > 0 else (-1 if x < 0 else 0)

    @staticmethod
    def _sma(values: List[float]) -> float:
        return sum(values) / float(len(values)) if values else 0.0

    def _update_ema(self, prev: float, price: float, k: float) -> float:
        return (price * k) + (prev * (1.0 - k))

    def _maybe_seed_emas(self) -> None:
        n = len(self.prices)
        if self.ema_fast is None and n >= self.fast:
            self.ema_fast = self._sma(self.prices[-self.fast:])
        if self.ema_slow is None and n >= self.slow:
            self.ema_slow = self._sma(self.prices[-self.slow:])

    def ingest_live(self, price: float) -> None:
        """Non-committing intrabar EMAs derived from the latest tick price."""
        if self.ema_fast is None or self.ema_slow is None:
            self.live_ema_fast = None
            self.live_ema_slow = None
            return
        try:
            p = float(price)
        except Exception:
            return
        self.live_ema_fast = self._update_ema(self.ema_fast, p, self.k_fast)
        self.live_ema_slow = self._update_ema(self.ema_slow, p, self.k_slow)

    def ingest(self, bar: Dict[str, Any]) -> None:
        px = float(bar["c"])
        self.prices.append(px)
        self._maybe_seed_emas()
        n = len(self.prices)
        if self.ema_fast is not None and n > self.fast:
            self.ema_fast = self._update_ema(self.ema_fast, px, self.k_fast)
        if self.ema_slow is not None and n > self.slow:
            self.ema_slow = self._update_ema(self.ema_slow, px, self.k_slow)
        # Clear preview after committing
        self.live_ema_fast = None
        self.live_ema_slow = None

    def signal(self) -> Dict[str, Any]:
        if self.ema_fast is None or self.ema_slow is None:
            return {"action": "hold"}
        rel_now = self._sign(self.ema_fast - self.ema_slow)
        if self.prev_rel is None:
            self.prev_rel = rel_now
            return {"action": "hold"}
        # exits first
        if self.pos_dir > 0 and rel_now < 0 and self.prev_rel >= 0:
            self.pos_dir = 0
            self.prev_rel = rel_now
            return {"action": "exit"}
        if self.pos_dir < 0 and rel_now > 0 and self.prev_rel <= 0:
            self.pos_dir = 0
            self.prev_rel = rel_now
            return {"action": "exit"}
        # entries
        if self.pos_dir == 0:
            if rel_now > 0 and self.prev_rel <= 0:
                self.pos_dir = +1
                self.prev_rel = rel_now
                return {"action": "enter_long"}
            if rel_now < 0 and self.prev_rel >= 0:
                self.pos_dir = -1
                self.prev_rel = rel_now
                return {"action": "enter_short"}
        self.prev_rel = rel_now
        return {"action": "hold"}

    def debug_state(self) -> Dict[str, Any]:
        # Prefer live preview if available
        ef = self.live_ema_fast if self.live_ema_fast is not None else self.ema_fast
        es = self.live_ema_slow if self.live_ema_slow is not None else self.ema_slow
        return {
            "ema_fast": ef,
            "ema_slow": es,
            "pos_dir": self.pos_dir,
            "ready": (self.ema_fast is not None and self.ema_slow is not None),
            "bars_seen": len(self.prices),
            "bars_until_ready": max(0, self.slow - len(self.prices)) if self.ema_slow is None else 0,
        }

# ---------------- Strategy registry & factory ----------------
REG: Dict[str, Type[StrategyBase]] = {
    "ema_cross": EmaCross,
}

def get_strategy(name: str) -> Type[StrategyBase]:
    key = (name or "").strip().lower()
    if key not in REG:
        raise ValueError(f"Unknown strategy '{name}'. Available: {', '.join(sorted(REG.keys()))}")
    return REG[key]
