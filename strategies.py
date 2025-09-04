from typing import List, Dict, Any

class StrategyBase:
    def ingest(self, bar: Dict[str, Any]) -> None:
        raise NotImplementedError

    def signal(self) -> Dict[str, Any]:
            # {action: 'enter_long'|'enter_short'|'exit'|'hold'}
        raise NotImplementedError

class EmaCross(StrategyBase):
    def __init__(self, fast: int = 9, slow: int = 21):
        if fast >= slow:
            raise ValueError("fast EMA must be < slow EMA")
        self.fast = fast
        self.slow = slow
        self.prices: List[float] = []
        self.pos_dir = 0
        self.ema_fast = None
        self.ema_slow = None

    @staticmethod
    def _ema(prev, price, period):
        k = 2 / (period + 1)
        return price if prev is None else (price * k + prev * (1 - k))

    def ingest(self, bar: Dict[str, Any]) -> None:
        px = float(bar["c"])
        self.prices.append(px)
        self.ema_fast = self._ema(self.ema_fast, px, self.fast)
        self.ema_slow = self._ema(self.ema_slow, px, self.slow)

    def signal(self) -> Dict[str, Any]:
        if self.ema_fast is None or self.ema_slow is None:
            return {"action": "hold"}
        if self.pos_dir <= 0 and self.ema_fast > self.ema_slow:
            self.pos_dir = +1
            return {"action": "enter_long"}
        if self.pos_dir > 0 and self.ema_fast < self.ema_slow:
            self.pos_dir = 0
            return {"action": "exit"}
        return {"action": "hold"}

def get_strategy(name: str):
    REG = {
        "ema_cross": EmaCross,
    }
    if name not in REG:
        raise KeyError(f"Unknown strategy: {name}")
    return REG[name]
