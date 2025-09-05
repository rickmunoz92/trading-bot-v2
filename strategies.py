from typing import List, Dict, Any, Optional

class StrategyBase:
    def ingest(self, bar: Dict[str, Any]) -> None:
        raise NotImplementedError

    def signal(self) -> Dict[str, Any]:
        # {action: 'enter_long'|'enter_short'|'exit'|'hold'}
        raise NotImplementedError

class EmaCross(StrategyBase):
    """
    Symmetric EMA cross strategy that enters ONLY on the bar where the cross occurs.

    Logic:
      - Track the sign of (EMA_fast - EMA_slow) as rel = +1, 0, or -1.
      - On each bar:
          * If currently LONG and rel flips to negative  -> exit (close long)
          * If currently SHORT and rel flips to positive -> exit (close short)
          * If FLAT and rel flips from non-positive -> positive -> enter_long
          * If FLAT and rel flips from non-negative -> negative -> enter_short
      - While rel stays the same, we do not fire repeated signals.
      - On startup, we wait until we have at least one prior rel value before entering,
        so we don't jump into a trend that started before the bot was launched.
    """
    def __init__(self, fast: int = 9, slow: int = 21):
        if fast >= slow:
            raise ValueError("fast EMA must be < slow EMA")
        self.fast = fast
        self.slow = slow
        self.prices: List[float] = []
        self.pos_dir: int = 0          # +1 long, -1 short, 0 flat
        self.ema_fast: Optional[float] = None
        self.ema_slow: Optional[float] = None
        self.prev_rel: Optional[int] = None  # sign of (fast - slow) from previous bar: +1, 0, or -1

    @staticmethod
    def _ema(prev, price, period):
        k = 2 / (period + 1)
        return price if prev is None else (price * k + prev * (1 - k))

    @staticmethod
    def _sign(x: float) -> int:
        return 1 if x > 0 else (-1 if x < 0 else 0)

    def ingest(self, bar: Dict[str, Any]) -> None:
        px = float(bar["c"])
        self.prices.append(px)
        self.ema_fast = self._ema(self.ema_fast, px, self.fast)
        self.ema_slow = self._ema(self.ema_slow, px, self.slow)

    def signal(self) -> Dict[str, Any]:
        if self.ema_fast is None or self.ema_slow is None:
            return {"action": "hold"}

        rel_now = self._sign(self.ema_fast - self.ema_slow)

        # No action until we have a prior relationship to compare against.
        if self.prev_rel is None:
            self.prev_rel = rel_now
            return {"action": "hold"}

        # Exits: only when the relationship actually flips through zero.
        if self.pos_dir > 0 and rel_now < 0 and self.prev_rel >= 0:
            self.pos_dir = 0
            self.prev_rel = rel_now
            return {"action": "exit"}
        if self.pos_dir < 0 and rel_now > 0 and self.prev_rel <= 0:
            self.pos_dir = 0
            self.prev_rel = rel_now
            return {"action": "exit"}

        # Entries only on the flip bar when flat.
        if self.pos_dir == 0:
            if rel_now > 0 and self.prev_rel <= 0:
                self.pos_dir = +1
                self.prev_rel = rel_now
                return {"action": "enter_long"}
            if rel_now < 0 and self.prev_rel >= 0:
                self.pos_dir = -1
                self.prev_rel = rel_now
                return {"action": "enter_short"}

        # Update prev_rel when no action fired.
        self.prev_rel = rel_now
        return {"action": "hold"}

def get_strategy(name: str):
    REG = {
        "ema_cross": EmaCross,
    }
    if name not in REG:
        raise KeyError(f"Unknown strategy: {name}")
    return REG[name]
