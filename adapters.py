import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime, timezone, time, timedelta
import random

from dotenv import load_dotenv
load_dotenv()

try:
    from zoneinfo import ZoneInfo
except Exception:
    from backports.zoneinfo import ZoneInfo  # type: ignore

ET = ZoneInfo('America/New_York')
MT = ZoneInfo('America/Denver')

@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float

class BrokerBase:
    name = "base"

    def is_trading_window_open(self, symbol: str, allow_extended: bool = False):
        """Return (is_open_now: bool, next_open_dt_et: datetime|None).
        Base broker assumes always open.
        """
        return True, None

    def min_lot(self, symbol: str) -> float:
        return 1.0

    def get_equity(self) -> float:
        raise NotImplementedError

    def get_position(self, symbol: str) -> Optional[Position]:
        raise NotImplementedError

    def asset_class(self, symbol: str) -> str:
        return "crypto" if "/" in symbol else "equity"

    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def submit_bracket(self, symbol: str, plan) -> str:
        raise NotImplementedError

    def close_position(self, symbol: str, exit_price: float):
        raise NotImplementedError

class LocalPaperBroker(BrokerBase):
    name = "local"

    def is_trading_window_open(self, symbol: str, allow_extended: bool = False):
        # Local paper broker trades 24/7 for simplicity
        return True, None

    def __init__(self, equity: float = 100_000.0):
        self._equity = equity
        self._positions: Dict[str, Position] = {}
        self._last_price: Dict[str, float] = {}

    def min_lot(self, symbol: str) -> float:
        return 1.0 if self.asset_class(symbol) == "equity" else 0.0001

    def get_equity(self) -> float:
        return self._equity

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        last = self._last_price.get(symbol, 100.0)
        new = max(0.01, last + random.uniform(-0.5, 0.5))
        self._last_price[symbol] = new
        iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return {"t": iso, "o": last, "h": max(last, new), "l": min(last, new), "c": new, "v": 1000}

    def submit_bracket(self, symbol: str, plan) -> str:
        qty = plan.qty if plan.side == "buy" else -plan.qty
        self._positions[symbol] = Position(symbol=symbol, qty=qty, avg_price=plan.entry)
        return f"local-{symbol}-{int(datetime.now().timestamp())}"

    def close_position(self, symbol: str, exit_price: float):
        pos = self._positions.pop(symbol, None)
        if not pos:
            return 0.0, 0.0
        side = 1 if pos.qty > 0 else -1
        pnl_abs = side * (exit_price - pos.avg_price) * abs(pos.qty)
        self._equity += pnl_abs
        pnl_pct = (exit_price - pos.avg_price) / pos.avg_price * side * 100.0
        return pnl_abs, pnl_pct

# -------- Alpaca Broker (minimal) --------
class AlpacaBroker(BrokerBase):
    name = "alpaca"

    def __init__(self, paper: bool = True):
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
        except Exception as e:
            raise RuntimeError(f"alpaca-py not installed or import failed: {e}")

        api_key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret:
            raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY in environment")

        self.paper = paper
        self.trading = TradingClient(api_key, secret, paper=paper)
        self.stock_data = StockHistoricalDataClient(api_key, secret)
        self.crypto_data = CryptoHistoricalDataClient(api_key, secret)

    def min_lot(self, symbol: str) -> float:
        return 1.0 if self.asset_class(symbol) == "equity" else 0.0001

    def get_equity(self) -> float:
        acct = self.trading.get_account()
        return float(acct.cash) + float(acct.portfolio_value) - float(acct.cash)  # portfolio_value as equity
        # Alternatively: return float(acct.portfolio_value)

    def get_position(self, symbol: str) -> Optional[Position]:
        try:
            p = self.trading.get_open_position(symbol.replace("/", ""))  # BTCUSD for crypto
        except Exception:
            return None
        try:
            qty = float(p.qty) if hasattr(p, "qty") else float(p.current_qty)
        except Exception:
            qty = float(getattr(p, "qty", 0))
        avg = float(p.avg_entry_price)
        return Position(symbol=symbol, qty=qty, avg_price=avg)

    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        from alpaca.data.requests import StockLatestBarRequest, CryptoLatestBarRequest
        from alpaca.data.timeframe import TimeFrame
        tf_map = {"1m":"Minute", "5m":"Minute", "15m":"Minute", "1h":"Hour", "1d":"Day"}
        # Alpaca latest bar endpoints ignore timeframe; we still poll each period
        is_crypto = "/" in symbol
        if is_crypto:
            symbol_api = symbol.replace("/", "")
            req = CryptoLatestBarRequest(symbol_or_symbols=symbol_api)
            bar = self.crypto_data.get_crypto_latest_bar(req)[symbol_api]
        else:
            req = StockLatestBarRequest(symbol_or_symbols=symbol)
            bar = self.stock_data.get_stock_latest_bar(req)[symbol]
        return {
            "t": bar.timestamp.isoformat(),
            "o": float(bar.open),
            "h": float(bar.high),
            "l": float(bar.low),
            "c": float(bar.close),
            "v": float(bar.volume or 0),
        }

    def submit_bracket(self, symbol: str, plan) -> str:
        from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
        side = OrderSide.BUY if plan.side == "buy" else OrderSide.SELL
        qty = plan.qty
        sym = symbol.replace("/", "")  # BTCUSD
        order = MarketOrderRequest(
            symbol=sym,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(float(plan.take_profit), 2)),
            stop_loss=StopLossRequest(stop_price=round(float(plan.stop_loss), 2)),
        )
        res = self.trading.submit_order(order_data=order)
        return res.id

    def close_position(self, symbol: str, exit_price: float):
        # Market-close position; Alpaca handles price. We'll compute pnl approximately
        sym = symbol.replace("/", "")
        try:
            pos = self.trading.get_open_position(sym)
        except Exception:
            return 0.0, 0.0
        avg = float(pos.avg_entry_price)
        qty = float(getattr(pos, "qty", getattr(pos, "current_qty", 0)))
        side = 1 if qty > 0 else -1
        self.trading.close_position(sym)
        pnl_abs = side * (exit_price - avg) * abs(qty)
        pnl_pct = (exit_price - avg) / avg * side * 100.0
        return pnl_abs, pnl_pct

    # --- Trading window checks ---
    def _equity_extended_ranges_for_day(self, day_et: datetime):
        d = day_et.date()
        pre_open  = datetime.combine(d, time(4, 0), tzinfo=ET)
        pre_close = datetime.combine(d, time(9, 30), tzinfo=ET)
        reg_open  = datetime.combine(d, time(9, 30), tzinfo=ET)
        reg_close = datetime.combine(d, time(16, 0), tzinfo=ET)
        aft_open  = datetime.combine(d, time(16, 0), tzinfo=ET)
        aft_close = datetime.combine(d, time(20, 0), tzinfo=ET)
        return pre_open, pre_close, reg_open, reg_close, aft_open, aft_close

    def is_trading_window_open(self, symbol: str, allow_extended: bool = False):
        """Return (is_open_now, next_open_dt_ET|None).
        Crypto assumed 24/7. For equities, use Alpaca clock+calendar.
        """
        # Crypto path (BTC/USD etc.)
        if "/" in symbol:
            return True, None
        try:
            clock = self.trading.get_clock()
            now_et = datetime.now(tz=ET)
            if not allow_extended:
                if bool(getattr(clock, 'is_open', False)):
                    return True, None
                # Compute next regular session open (9:30 ET) without relying on calendar
                now_et = datetime.now(tz=ET)
                wk = now_et.weekday()
                reg_open = datetime.combine(now_et.date(), time(9,30), tzinfo=ET)
                reg_close = datetime.combine(now_et.date(), time(16,0), tzinfo=ET)
                if wk < 5 and now_et < reg_open:
                    return False, reg_open
                # find next business day (skip Sat/Sun)
                d = now_et.date() + timedelta(days=1)
                while d.weekday() >= 5:
                    d += timedelta(days=1)
                nxt = datetime.combine(d, time(9,30), tzinfo=ET)
                return False, nxt
            else:
                # Extended hours window logic
                cal = self.trading.get_calendar(start=now_et.date().isoformat(), end=(now_et.date() + timedelta(days=10)).isoformat())
                target = None
                for row in cal:
                    if row.date == now_et.date() or row.date > now_et.date():
                        target = row
                        break
                if target is None:
                    return bool(getattr(clock, 'is_open', False)), None
                day_et = datetime.combine(target.date, time(0,0), tzinfo=ET)
                pre_open, pre_close, reg_open, reg_close, aft_open, aft_close = self._equity_extended_ranges_for_day(day_et)
                def in_rng(t,a,b):
                    return a <= t < b
                if target.date == now_et.date():
                    if in_rng(now_et, pre_open, pre_close) or in_rng(now_et, reg_open, reg_close) or in_rng(now_et, aft_open, aft_close):
                        return True, None
                    if now_et < pre_open:
                        return False, pre_open
                    if now_et < reg_open:
                        return False, reg_open
                    if now_et < aft_open:
                        return False, aft_open
                    # after after-hours → use next day's 4:00 ET
                    next_idx = [i for i, r in enumerate(cal) if r.date == target.date][0] + 1
                    if next_idx < len(cal):
                        nd = cal[next_idx].date
                    else:
                        nd = target.date + timedelta(days=1)
                    return False, datetime.combine(nd, time(4,0), tzinfo=ET)
                else:
                    return False, (datetime.combine(target.date, time(4,0), tzinfo=ET) if allow_extended else datetime.combine(target.date, time(9,30), tzinfo=ET))
        except Exception:
            # Fallback to regular hours Mon-Fri 9:30-16:00 ET
            now_et = datetime.now(tz=ET)
            wk = now_et.weekday()
            reg_open = datetime.combine(now_et.date(), time(9,30), tzinfo=ET)
            reg_close = datetime.combine(now_et.date(), time(16,0), tzinfo=ET)
            if wk < 5 and reg_open <= now_et < reg_close:
                return True, None
            if wk < 5 and now_et < reg_open:
                return False, reg_open
            # next business day 9:30 ET
            days_ahead = (7 - wk) % 7 or 1
            nxt = datetime.combine(now_et.date() + timedelta(days=days_ahead), time(9,30), tzinfo=ET)
            return False, nxt

# -------- IBKR Broker (skeleton) --------
class IbkrBroker(BrokerBase):
    name = "ibkr"

    def is_trading_window_open(self, symbol: str, allow_extended: bool = False):
        # Stub: assume open; production should use IBKR trading schedule
        return True, None

    def __init__(self, paper: bool = True):
        # Minimal connection skeleton. You must have TWS/Gateway running.
        try:
            from ib_insync import IB
        except Exception as e:
            raise RuntimeError(f"ib-insync not installed or import failed: {e}")
        self.paper = paper
        self.host = os.getenv("IB_HOST", "127.0.0.1")
        self.port = int(os.getenv("IB_PORT", "7497" if paper else "7496"))
        self.client_id = int(os.getenv("IB_CLIENT_ID", "42"))
        self.ib = IB()
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to IBKR TWS/Gateway: {e}")

    def get_equity(self) -> float:
        acct_vals = {av.tag: av.value for av in self.ib.accountValues()}
        return float(acct_vals.get("NetLiquidation", 0))

    def get_position(self, symbol: str) -> Optional[Position]:
        # For brevity, not fully implemented in this skeleton.
        return None

    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        # You can implement with IBKR's reqMktData or historicalData
        return None

    def submit_bracket(self, symbol: str, plan) -> str:
        raise NotImplementedError("Implement IBKR bracket submission with ib_insync.")

    def close_position(self, symbol: str, exit_price: float):
        raise NotImplementedError("Implement IBKR position close with ib_insync.")

def make_broker(kind: str, paper: bool) -> BrokerBase:
    if kind == "local":
        return LocalPaperBroker()
    if kind == "alpaca":
        return AlpacaBroker(paper=paper)
    if kind == "ibkr":
        return IbkrBroker(paper=paper)
    raise ValueError(f"Unknown broker: {kind}")
