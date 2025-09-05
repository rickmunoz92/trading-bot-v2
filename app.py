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

from typing import Optional, Dict, Any, Tuple, Callable
import random
from queue import Queue, Empty

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
    if isinstance(a.strategy, list):
        strat_parts = a.strategy
        a.strategy = strat_parts[0]
        if len(strat_parts) >= 3:
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
    if current_price is not None:
        if action in ("enter_long", "enter_short"):
            _side = "buy" if action == "enter_long" else "sell"
        else:
            _side = "buy"
        _plan = plan_bracket(_side, current_price, cfg.tp_pct, cfg.sl_pct, qty=0, meta={})
        kv.append(R.kv("TP", f"${_plan.take_profit:.2f} ({cfg.tp_pct:.2f}%)"))
        kv.append(R.kv("SL", f"${_plan.stop_loss:.2f} ({cfg.sl_pct:.2f}%)"))
    return "  ".join(kv)

def _strategy_hit(action: Optional[str]) -> bool:
    return action in ("enter_long", "enter_short")

def _apply_side_filter(action: str, side_pref: str) -> str:
    if side_pref == "long" and action == "enter_short":
        return "hold"
    if side_pref == "short" and action == "enter_long":
        return "hold"
    return action

_TF_RE = re.compile(r"^\s*(\d+)\s*([mhd])\s*$", re.IGNORECASE)

def _parse_timeframe_seconds(tf: str) -> int:
    m = _TF_RE.match(tf or "")
    if not m:
        return 60
    n = int(m.group(1))
    unit = m.group(2).lower()
    if unit == "m":
        return n * 60
    if unit == "h":
        return n * 3600
    if unit == "d":
        return n * 86400
    return 60

def _iso_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

# ---------- TP/SL reconciliation & enforcement helpers ----------

def _infer_side_from_position(pos) -> str:
    try:
        s = getattr(pos, "side", None)
        if s in ("long", "short"):
            return s
    except Exception:
        pass
    try:
        qty = float(getattr(pos, "qty", 0.0))
        return "long" if qty >= 0 else "short"
    except Exception:
        return "long"

def _compute_tp_sl_for_side(avg_price: float, side: str, tp_pct: float, sl_pct: float) -> Tuple[float, float]:
    if side == "long":
        tp = avg_price * (1 + tp_pct / 100.0)
        sl = avg_price * (1 - sl_pct / 100.0)
    else:
        tp = avg_price * (1 - tp_pct / 100.0)
        sl = avg_price * (1 + sl_pct / 100.0)
    return (tp, sl)

def _breach_for_side(side: str, price: float, tp: float, sl: float) -> Optional[str]:
    if side == "long":
        if price >= tp:
            return "tp"
        if price <= sl:
            return "sl"
    else:
        if price <= tp:
            return "tp"
        if price >= sl:
            return "sl"
    return None

# ---------------- Broker call timeout + retry (with budget) -------------------

def _call_with_timeout(label: str, fn: Callable, args: tuple, kwargs: dict, timeout: float):
    q: Queue = Queue(maxsize=1)
    def _worker():
        try:
            q.put(("ok", fn(*args, **kwargs)))
        except Exception as e:
            q.put(("err", e))
    t = threading.Thread(target=_worker, name=f"{label}-worker", daemon=True)
    t.start()
    try:
        kind, payload = q.get(timeout=timeout)
        if kind == "ok":
            return payload
        else:
            raise payload
    except Empty:
        raise TimeoutError(f"{label} timed out after {timeout:.1f}s")

def _retry_broker(
    stop_event: threading.Event,
    sleep_fn: Callable[[float], None],
    label: str,
    fn: Callable,
    *args,
    tries: int = 5,
    base: float = 0.5,
    cap: float = 5.0,
    timeout: float = 5.0,
    max_total_seconds: Optional[float] = None,
    **kwargs
):
    """
    Retry a broker call with jittered exponential backoff and per-attempt timeout.
    If max_total_seconds is set, abort retries once that budget is exceeded.
    """
    last_err = None
    start = time.monotonic()
    for attempt in range(1, tries + 1):
        if stop_event.is_set():
            break
        if max_total_seconds is not None and (time.monotonic() - start) >= max_total_seconds:
            raise TimeoutError(f"{label} budget exceeded after {time.monotonic() - start:.1f}s")
        try:
            return _call_with_timeout(label, fn, args, kwargs, timeout=timeout)
        except Exception as e:
            last_err = e
            if attempt >= tries:
                break
            # honor budget before sleeping
            delay = min(cap, base * (2 ** (attempt - 1)))
            delay *= 1.0 + 0.25 * random.random()  # jitter
            # If the remaining budget is smaller than delay, trim it
            if max_total_seconds is not None:
                remaining = max_total_seconds - (time.monotonic() - start)
                if remaining <= 0:
                    break
                delay = max(0.0, min(delay, remaining))
            print(R.warn(f"{label} failed: {e} — retrying in {delay:.1f}s (attempt {attempt}/{tries})"))
            sleep_fn(delay)
    if last_err:
        raise last_err

# --------------------------------------------------------------------

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

    # Dynamic default for --side
    if cfg.side is None:
        asset_cls = broker.asset_class(cfg.symbol)
        cfg.side = "long" if asset_cls == "crypto" else "both"

    # Graceful, responsive shutdown
    stop_event = threading.Event()
    def _sigint(_s, _f):
        stop_event.set()
    signal.signal(signal.SIGINT, _sigint)

    def responsive_sleep(total_seconds: float, step: float = 0.2):
        remaining = float(total_seconds)
        while remaining > 0 and not stop_event.is_set():
            time.sleep(min(step, remaining))
            remaining -= step

    def pace_sleep(loop_started_at: float, poll_seconds: float):
        elapsed = time.monotonic() - loop_started_at
        remaining = poll_seconds - elapsed
        if remaining > 0:
            responsive_sleep(remaining)

    journal = Journal(symbol=cfg.symbol)

    tf_secs = _parse_timeframe_seconds(cfg.timeframe)
    last_bucket_sec: Optional[int] = None
    prev_bar: Optional[Dict[str, Any]] = None

    # Quick/budgeted fetch helpers
    def _market_call_budget() -> float:
        # Use up to 60% of poll interval for live data calls (never < 2s, never > 6s)
        return max(2.0, min(6.0, cfg.poll * 0.6))

    def price_fetch() -> Optional[float]:
        try:
            bar = _retry_broker(
                stop_event, responsive_sleep, "get_latest_bar",
                broker.get_latest_bar, cfg.symbol, cfg.timeframe,
                tries=2, base=0.25, cap=1.0, timeout=2.0,
                max_total_seconds=_market_call_budget()
            )
            if bar:
                return float(bar.get("c", None))
        except Exception:
            return None
        return None

    def get_position_safe() -> Optional[Dict[str, Any]]:
        try:
            return _retry_broker(
                stop_event, responsive_sleep, "get_position",
                broker.get_position, cfg.symbol,
                tries=3, base=0.25, cap=1.0, timeout=2.5,
                max_total_seconds=_market_call_budget()
            )
        except Exception as e:
            print(R.warn(f"get_position failed: {e}"))
            return None

    def is_open_safe() -> Tuple[bool, Optional[datetime]]:
        try:
            return _retry_broker(
                stop_event, responsive_sleep, "is_trading_window_open",
                broker.is_trading_window_open, cfg.symbol, allow_extended=False,
                tries=2, base=0.25, cap=1.0, timeout=2.0,
                max_total_seconds=_market_call_budget()
            )
        except TypeError:
            state = _retry_broker(
                stop_event, responsive_sleep, "is_trading_window_open",
                broker.is_trading_window_open, cfg.symbol,
                tries=2, base=0.25, cap=1.0, timeout=2.0,
                max_total_seconds=_market_call_budget()
            )
            if isinstance(state, tuple):
                return state
            return bool(state), None

    # ---------- Reconciliation & enforcement (use price_fetch) ----------
    def _reconcile_on_start(cfg: Config, broker: BrokerBase, journal: Journal) -> Optional[Dict[str, Any]]:
        pos = get_position_safe()
        if not pos:
            return None
        avg = float(getattr(pos, "avg_price", 0.0))
        if avg <= 0:
            return pos
        side = _infer_side_from_position(pos)
        tp = avg * (1 + cfg.tp_pct / 100.0) if side == "long" else avg * (1 - cfg.tp_pct / 100.0)
        sl = avg * (1 - cfg.sl_pct / 100.0) if side == "long" else avg * (1 + cfg.sl_pct / 100.0)
        price = price_fetch()
        if price is None:
            return pos
        hit = _breach_for_side(side, price, tp, sl)
        if hit:
            exit_price = price
            try:
                pnl_abs, pnl_pct = _retry_broker(
                    stop_event, responsive_sleep, "close_position",
                    broker.close_position, cfg.symbol, exit_price,
                    tries=2, base=0.25, cap=1.0, timeout=2.0,
                    max_total_seconds=_market_call_budget()
                )
            except Exception as e:
                print(R.warn(f"close_position failed during reconcile: {e}"))
                return pos
            journal.on_exit(
                when=now_utc_iso(),
                exit_price=exit_price,
                win_loss=("Win" if pnl_abs >= 0 else "Loss"),
                pnl_abs=pnl_abs,
                pnl_pct=pnl_pct,
            )
            return None
        return pos

    def _enforce_tp_sl_and_maybe_exit(cfg: Config, broker: BrokerBase, journal: Journal, open_position) -> Optional[Dict[str, Any]]:
        if not open_position:
            return None
        avg = float(getattr(open_position, "avg_price", 0.0))
        if avg <= 0:
            return open_position
        price = price_fetch()
        if price is None:
            return open_position
        side = _infer_side_from_position(open_position)
        tp = avg * (1 + cfg.tp_pct / 100.0) if side == "long" else avg * (1 - cfg.tp_pct / 100.0)
        sl = avg * (1 - cfg.sl_pct / 100.0) if side == "long" else avg * (1 + cfg.sl_pct / 100.0)
        hit = _breach_for_side(side, price, tp, sl)
        if hit:
            exit_price = price
            try:
                pnl_abs, pnl_pct = _retry_broker(
                    stop_event, responsive_sleep, "close_position",
                    broker.close_position, cfg.symbol, exit_price,
                    tries=2, base=0.25, cap=1.0, timeout=2.0,
                    max_total_seconds=_market_call_budget()
                )
            except Exception as e:
                print(R.warn(f"close_position failed: {e}"))
                return open_position
            journal.on_exit(
                when=now_utc_iso(),
                exit_price=exit_price,
                win_loss=("Win" if pnl_abs >= 0 else "Loss"),
                pnl_abs=pnl_abs,
                pnl_pct=pnl_pct
            )
            return None
        return open_position

    # --- Startup reconcile
    open_position = _reconcile_on_start(cfg, broker, journal)

    try:
        while not stop_event.is_set():
            loop_start = time.monotonic()

            # --- Trading window guard ---
            try:
                is_open, next_open_et = is_open_safe()
            except Exception as e:
                print(R.warn(f"is_trading_window_open failed: {e} — assuming open for now"))
                is_open, next_open_et = True, None
            if not is_open:
                if next_open_et is not None:
                    print(f"{human_ts(now_utc_iso())} " + R.dim(f"Trading window is closed for {cfg.symbol}. It will re-open on {fmt_mt(next_open_et)}"))
                else:
                    print(f"{human_ts(now_utc_iso())} " + R.dim(f"Trading window is closed for {cfg.symbol}."))
                pace_sleep(loop_start, cfg.poll)
                continue

            # --- Fetch market data (budgeted retry) ---
            try:
                bar_now = _retry_broker(
                    stop_event, responsive_sleep, "get_latest_bar",
                    broker.get_latest_bar, cfg.symbol, cfg.timeframe,
                    tries=2, base=0.25, cap=1.0, timeout=2.0,
                    max_total_seconds=_market_call_budget()
                )
            except Exception as e:
                print(R.warn(f"get_latest_bar failed: {e}"))
                pace_sleep(loop_start, cfg.poll)
                continue

            if not bar_now:
                print(R.warn("No market data yet."))
                pace_sleep(loop_start, cfg.poll)
                continue

            # Compute timeframe bucket
            try:
                dt_now = _iso_to_dt(bar_now["t"])
                sec_now = int(dt_now.timestamp())
                bucket_now = (sec_now // _parse_timeframe_seconds(cfg.timeframe)) * _parse_timeframe_seconds(cfg.timeframe)
            except Exception:
                bucket_now = None

            action = "hold"
            display_price = float(bar_now.get("c", 0.0))

            # ---------- Enforce TP/SL before strategy ----------
            if open_position:
                updated = _enforce_tp_sl_and_maybe_exit(cfg, broker, journal, open_position)
                open_position = updated
                if not open_position:
                    header = f"{human_ts(now_utc_iso())} " + format_header(cfg, broker, current_price=display_price, action=action)
                    line = [
                        R.kv("price", f"{display_price:.4f}"),
                        R.kv("sig", "exit (tp/sl)"),
                        R.kv("hit", "n/a"),
                    ]
                    print(f"{header} | " + " ".join(line))
                    pace_sleep(loop_start, cfg.poll)
                    continue
            # -----------------------------------------------------------------

            # Bar gating for strategy updates
            if last_bucket_sec is None:
                last_bucket_sec = bucket_now
                prev_bar = bar_now
            else:
                if bucket_now == last_bucket_sec:
                    prev_bar = bar_now
                else:
                    effective_bar = prev_bar or bar_now
                    display_price = float(effective_bar.get("c", display_price))
                    strategy.ingest(effective_bar)
                    signal_out = strategy.signal()
                    action = signal_out.get("action", "hold")
                    action = _apply_side_filter(action, cfg.side)
                    last_bucket_sec = bucket_now
                    prev_bar = bar_now

            # --- Header ---
            header = f"{human_ts(now_utc_iso())} " + format_header(cfg, broker, current_price=display_price, action=action)

            # --- Status line ---
            line = [
                R.kv("price", f"{display_price:.4f}"),
                R.kv("sig", action),
                R.kv("hit", "yes" if _strategy_hit(action) else "no"),
            ]
            if open_position:
                try:
                    line.append(R.kv("pos_avg", f"{float(getattr(open_position, 'avg_price', 0.0)):.4f}"))
                    line.append(R.kv("pos_qty", f"{getattr(open_position, 'qty', 0)}"))
                except Exception:
                    pass
            print(f"{header} | " + " ".join(line))

            # --- Execute (budgeted retry) ---
            if action in ("enter_long", "enter_short") and not open_position:
                side = "buy" if action == "enter_long" else "sell"
                entry = display_price
                tmp_plan = plan_bracket(side, entry, cfg.tp_pct, cfg.sl_pct, qty=0, meta={})
                try:
                    equity = _retry_broker(
                        stop_event, responsive_sleep, "get_equity", broker.get_equity,
                        tries=2, base=0.25, cap=1.0, timeout=2.0,
                        max_total_seconds=_market_call_budget()
                    )
                    lot = _retry_broker(
                        stop_event, responsive_sleep, "min_lot", broker.min_lot, cfg.symbol,
                        tries=2, base=0.25, cap=1.0, timeout=2.0,
                        max_total_seconds=_market_call_budget()
                    )
                except Exception as e:
                    print(R.warn(f"pre-entry sizing failed: {e}"))
                    pace_sleep(loop_start, cfg.poll)
                    continue

                qty = risk_size_qty(equity, cfg.risk_pct, entry, tmp_plan.stop_loss, lot_size=lot)
                if qty <= 0:
                    print(R.warn("qty computed to 0; skipping."))
                else:
                    plan = plan_bracket(side, entry, cfg.tp_pct, cfg.sl_pct, qty=qty, meta={"strategy": cfg.strategy_name})
                    try:
                        order_id = _retry_broker(
                            stop_event, responsive_sleep, "submit_bracket",
                            broker.submit_bracket, cfg.symbol, plan,
                            tries=2, base=0.25, cap=1.0, timeout=2.0,
                            max_total_seconds=_market_call_budget()
                        )
                        open_position = get_position_safe()
                    except Exception as e:
                        print(R.warn(f"submit_bracket failed: {e}"))
                        open_position = None
                        pace_sleep(loop_start, cfg.poll)
                        continue

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
                exit_price = display_price
                try:
                    pnl_abs, pnl_pct = _retry_broker(
                        stop_event, responsive_sleep, "close_position",
                        broker.close_position, cfg.symbol, exit_price,
                        tries=2, base=0.25, cap=1.0, timeout=2.0,
                        max_total_seconds=_market_call_budget()
                    )
                except Exception as e:
                    print(R.warn(f"close_position failed: {e}"))
                    pace_sleep(loop_start, cfg.poll)
                    continue
                journal.on_exit(
                    when=now_utc_iso(),
                    exit_price=exit_price,
                    win_loss=("Win" if pnl_abs >= 0 else "Loss"),
                    pnl_abs=pnl_abs,
                    pnl_pct=pnl_pct,
                )
                open_position = None

            pace_sleep(loop_start, cfg.poll)

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
                broker.trading.close()
        except Exception:
            pass
        print("\n" + R.dim("Stopped."))

if __name__ == "__main__":
    main()
