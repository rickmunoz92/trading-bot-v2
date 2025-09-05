import re

import argparse
import json
import signal
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo
except Exception:
    from backports.zoneinfo import ZoneInfo  # type: ignore
MT = ZoneInfo('America/Denver')
ET = ZoneInfo('America/New_York')

from typing import Optional, Dict, Any

from io_utils import R, Journal
from strategies import StrategyBase, get_strategy
from adapters import BrokerBase, LocalPaperBroker, make_broker

@dataclass
class Config:
    timeframe: str
    symbol: str
    risk_pct: float
    tp_pct: float
    sl_pct: float
    poll: int
    strategy_name: str
    trade_mode: str
    broker: str
    equity: float
    fast: int
    slow: int
    side: Optional[str]  # "long" | "short" | "both" | None (None => dynamic default)

@dataclass
class OrderPlan:
    side: str
    qty: float
    entry: float
    take_profit: float
    stop_loss: float
    meta: Dict[str, Any]


def format_strategy_label(name: str, fast: int, slow: int) -> str:
    names = {
        "ema_cross": "EMA Cross",
    }
    label = names.get(name, name.replace("_", " ").title())
    return f"{label} ({fast}/{slow})"

def human_ts(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        dt = dt.astimezone()  # convert UTC to local timezone
        return dt.strftime("%I:%M:%S %p %a %b %d, %Y")
    except Exception:
        return ts

def fmt_mt(dt: datetime) -> str:
    dt_mt = dt.astimezone(MT)
    s = dt_mt.strftime("%a %b %d @ %I:%M%p").replace("AM","am").replace("PM","pm")
    # remove leading zero in hour
    s = re.sub(r"@ 0(\d):", r"@ \1:", s)
    return f"{s} MT"

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Lightweight trading bot")
    p.add_argument("timeframe", type=str, help="e.g. 5m, 15m, 1h, 1d")
    p.add_argument("symbol", type=str, help="e.g. AAPL, BTC/USD")
    p.add_argument("risk_pct", type=float, help="position sizing % of equity at risk per trade")
    p.add_argument("tp_pct", type=float, help="target profit %")
    p.add_argument("sl_pct", type=float, help="stop loss %")

    p.add_argument("--poll", type=int, default=60, help="polling interval seconds")
    p.add_argument("--strategy", nargs="+", default=["ema_cross"], help="strategy and optional params, e.g.: --strategy ema_cross 9 21")
    p.add_argument("--trade-mode", choices=["paper", "live"], default="paper")
    p.add_argument("--broker", choices=["alpaca", "ibkr", "local"], default="local")
    p.add_argument("--equity", type=float, default=100_000.0, help="only for local paper broker")
    p.add_argument("--fast", type=int, default=9, help="ema fast period")
    p.add_argument("--slow", type=int, default=21, help="ema slow period")
    p.add_argument("--side", choices=["long", "short", "both"], default=None,
                   help="Restrict trades to long only, short only, or both. Default: equities=both, crypto=long.")

    a = p.parse_args()
    # Support forms: "--strategy ema_cross" or "--strategy ema_cross 9 21"
    if isinstance(a.strategy, list):
        strat_parts = a.strategy
        a.strategy = strat_parts[0]
        if len(strat_parts) >= 3:
            # override fast/slow from strategy args
            try:
                a.fast = int(strat_parts[1])
                a.slow = int(strat_parts[2])
            except ValueError:
                pass
    return Config(
        timeframe=a.timeframe,
        symbol=a.symbol,
        risk_pct=a.risk_pct,
        tp_pct=a.tp_pct,
        sl_pct=a.sl_pct,
        poll=a.poll,
        strategy_name=a.strategy,
        trade_mode=a.trade_mode,
        broker=a.broker,
        equity=a.equity,
        fast=a.fast,
        slow=a.slow,
        side=a.side,
    )


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj)
    except TypeError:
        # Fallback: coerce non-serializable objects to strings
        def _default(o):
            try:
                return str(o)
            except Exception:
                return repr(o)
        return json.dumps(obj, default=_default)

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def plan_bracket(side: str, entry: float, tp_pct: float, sl_pct: float, qty: float, meta: Dict[str, Any]) -> OrderPlan:
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")

    if side == "buy":
        take = entry * (1 + tp_pct / 100.0)
        stop = entry * (1 - sl_pct / 100.0)
    else:
        take = entry * (1 - tp_pct / 100.0)
        stop = entry * (1 + sl_pct / 100.0)

    return OrderPlan(side=side, qty=qty, entry=entry, take_profit=round(take, 4), stop_loss=round(stop, 4), meta=meta)

def risk_size_qty(equity: float, risk_pct: float, entry: float, stop: float, lot_size: float = 1.0) -> float:
    risk_dollars = equity * (risk_pct / 100.0)
    per_unit_risk = abs(entry - stop)
    if per_unit_risk <= 0:
        return 0.0
    qty = risk_dollars / per_unit_risk
    steps = int(qty / lot_size)
    return max(0.0, steps * lot_size)

def format_header(cfg: Config, broker: BrokerBase, current_price: float = None, action: str = None) -> str:
    kv = [
        R.kv("timeframe", cfg.timeframe),
        R.kv("symbol", cfg.symbol),
        R.kv("mode", cfg.trade_mode.upper()),
        R.kv("broker", broker.name.upper()),
        R.kv("strategy", format_strategy_label(cfg.strategy_name, cfg.fast, cfg.slow)),
        R.kv("side", (cfg.side or "").upper() or "AUTO"),
        R.kv("Risk", f"${cfg.equity * (cfg.risk_pct/100.0):.2f} ({cfg.risk_pct:.2f}%)"),
        R.kv("poll", f"{cfg.poll}s"),
    ]
    # If we have a price, compute absolute TP/SL for display
    if current_price is not None:
        # Choose side: if actual action is an entry, use it; otherwise assume long for display
        if action in ("enter_long", "enter_short"):
            _side = "buy" if action == "enter_long" else "sell"
        else:
            _side = "buy"
        _plan = plan_bracket(_side, current_price, cfg.tp_pct, cfg.sl_pct, qty=0, meta={})
        kv.append(R.kv("TP", f"${_plan.take_profit:.2f} ({cfg.tp_pct:.2f}%)"))
        kv.append(R.kv("SL", f"${_plan.stop_loss:.2f} ({cfg.sl_pct:.2f}%)"))
    return "  ".join(kv)

def _strategy_hit(action: Optional[str]) -> bool:
    """Return True when the strategy signaled an actionable entry this bar."""
    return action in ("enter_long", "enter_short")

def _apply_side_filter(action: str, side_pref: str) -> str:
    """Force long-only or short-only by converting disallowed entries to 'hold'."""
    if side_pref == "long" and action == "enter_short":
        return "hold"
    if side_pref == "short" and action == "enter_long":
        return "hold"
    return action

def main() -> None:
    cfg = parse_args()

    strategy: StrategyBase = get_strategy(cfg.strategy_name)(fast=cfg.fast, slow=cfg.slow)

    # Broker selection with safe fallback
    if cfg.trade_mode == "paper" and cfg.broker == "local":
        broker: BrokerBase = LocalPaperBroker(equity=cfg.equity)
    else:
        try:
            broker = make_broker(cfg.broker, paper=(cfg.trade_mode == "paper"))
        except Exception as e:
            print(R.warn(f"Falling back to local paper broker: {e}"))
            broker = LocalPaperBroker(equity=cfg.equity)

    # Dynamic default for --side:
    # - crypto => "long"
    # - equities => "both"
    if cfg.side is None:
        asset_cls = broker.asset_class(cfg.symbol)
        cfg.side = "long" if asset_cls == "crypto" else "both"

    # Graceful, responsive shutdown
    stop_event = threading.Event()

    def _sigint(_s, _f):
        # Just set the flag; avoid heavy work in signal handler
        stop_event.set()

    signal.signal(signal.SIGINT, _sigint)

    def responsive_sleep(total_seconds: int, step: float = 0.2):
        """Sleep in small increments so Ctrl-C responds quickly even during long polls."""
        remaining = float(total_seconds)
        while remaining > 0 and not stop_event.is_set():
            time.sleep(min(step, remaining))
            remaining -= step

    journal = Journal(symbol=cfg.symbol)

    last_bar_ts: Optional[str] = None
    open_position = broker.get_position(cfg.symbol)

    try:
        while not stop_event.is_set():
            # --- Trading window guard ---
            try:
                is_open, next_open_et = broker.is_trading_window_open(cfg.symbol, allow_extended=False)
            except TypeError:
                # Backward-compat with brokers that don't implement the 2-arg signature
                state = broker.is_trading_window_open(cfg.symbol) if hasattr(broker, 'is_trading_window_open') else (True, None)
                if isinstance(state, tuple):
                    is_open, next_open_et = state
                else:
                    is_open, next_open_et = bool(state), None
            if not is_open:
                if next_open_et is not None:
                    print(f"{human_ts(now_utc_iso())} " + R.dim(f"Trading window is closed for {cfg.symbol}. It will re-open on {fmt_mt(next_open_et)}"))
                else:
                    print(f"{human_ts(now_utc_iso())} " + R.dim(f"Trading window is closed for {cfg.symbol}."))
                responsive_sleep(cfg.poll)
                continue

            # --- Fetch market data ---
            bar = broker.get_latest_bar(cfg.symbol, cfg.timeframe)
            if not bar:
                print(R.warn("No market data yet."))
                responsive_sleep(cfg.poll)
                continue
            last_bar_ts = bar["t"]

            # --- Strategy ---
            strategy.ingest(bar)
            signal_out = strategy.signal()
            action = signal_out.get("action", "hold")

            # Apply side preference
            action = _apply_side_filter(action, cfg.side)

            # --- Header (each poll with TP/SL) ---
            print(f"{human_ts(now_utc_iso())} " + format_header(cfg, broker, current_price=bar["c"], action=action))

            # --- Status line ---
            line = [
                R.kv("price", f"{bar['c']:.4f}"),
                R.kv("sig", action),
                R.kv("hit", "yes" if _strategy_hit(action) else "no"),
            ]
            if open_position:
                line.append(R.kv("pos_avg", f"{open_position.avg_price:.4f}"))
                line.append(R.kv("pos_qty", f"{open_position.qty}"))
            print(" ".join(line))

            # --- Execute ---
            if action in ("enter_long", "enter_short") and not open_position:
                side = "buy" if action == "enter_long" else "sell"
                entry = bar["c"]
                tmp_plan = plan_bracket(side, entry, cfg.tp_pct, cfg.sl_pct, qty=0, meta={})
                qty = risk_size_qty(broker.get_equity(), cfg.risk_pct, entry, tmp_plan.stop_loss, lot_size=broker.min_lot(cfg.symbol))
                if qty <= 0:
                    print(R.warn("qty computed to 0; skipping."))
                else:
                    plan = plan_bracket(side, entry, cfg.tp_pct, cfg.sl_pct, qty=qty, meta={"strategy": cfg.strategy_name})
                    order_id = broker.submit_bracket(cfg.symbol, plan)
                    open_position = broker.get_position(cfg.symbol)
                    journal.on_entry(
                        when=now_utc_iso(),
                        symbol=cfg.symbol,
                        asset_class=broker.asset_class(cfg.symbol),
                        strategy=format_strategy_label(cfg.strategy_name, cfg.fast, cfg.slow),
                        entry_price=plan.entry,
                        stop_loss=plan.stop_loss,
                        take_profit=plan.take_profit,
                        position_size=qty,
                        notes=json.dumps({"order_id": str(order_id)}),
                    )

            elif action == "exit" and open_position:
                exit_price = bar["c"]
                pnl_abs, pnl_pct = broker.close_position(cfg.symbol, exit_price)
                journal.on_exit(
                    when=now_utc_iso(),
                    exit_price=exit_price,
                    win_loss=("Win" if pnl_abs >= 0 else "Loss"),
                    pnl_abs=pnl_abs,
                    pnl_pct=pnl_pct,
                )
                open_position = None

            responsive_sleep(cfg.poll)

    except KeyboardInterrupt:
        stop_event.set()
    finally:
        try:
            if hasattr(broker, "ib"):
                broker.ib.disconnect()
        except Exception:
            pass
        try:
            if hasattr(broker, "trading") and hasattr(broker.trading, "close"):
                # some clients offer close(); ignore if not present
                broker.trading.close()
        except Exception:
            pass
        print("\n" + R.dim("Stopped."))

if __name__ == "__main__":
    main()
