import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timezone, time, timedelta
import random
import time as _time

from dotenv import load_dotenv
load_dotenv()

try:
    from zoneinfo import ZoneInfo
except Exception:
    from backports.zoneinfo import ZoneInfo  # type: ignore

ET = ZoneInfo('America/New_York')
MT = ZoneInfo('America/Denver')

# ---- Symbol normalization helpers ----
def _normalize_symbol(symbol: str) -> str:
    """Uppercase & strip separators, e.g. 'btc/usd' -> 'BTCUSD'."""
    return symbol.replace("/", "").replace("-", "").upper().strip()

_FIAT_SUFFIXES = ("USD", "USDT", "USDC", "EUR")

_COMMON_CRYPTO_BASES = {
    "BTC", "ETH", "SOL", "DOGE", "ADA", "LTC", "BCH", "AVAX", "MATIC", "SHIB",
    "ETC", "XRP", "DOT", "LINK", "ATOM", "NEAR", "APT", "ARB", "OP", "PEPE",
    "TON", "SUI", "ALGO", "FIL", "ICP", "AAVE", "UNI"
}

def _split_crypto(symbol: str) -> Tuple[str, str]:
    """Return (BASE, QUOTE) for crypto-like symbols in 'BASE/QUOTE' or 'BASEQUOTE' forms."""
    s = symbol.strip().upper()
    if "/" in s:
        base, quote = s.split("/", 1)
        return base.strip(), quote.strip()
    sn = _normalize_symbol(s)
    for q in _FIAT_SUFFIXES:
        if sn.endswith(q) and len(sn) > len(q):
            return sn[:-len(q)], q
    # Fallback (treat last 3 as quote)
    return sn[:-3], sn[-3:]

def _is_crypto_symbol(symbol: str) -> bool:
    """Heuristic to classify crypto pairs."""
    if "/" in symbol:
        return True
    sn = _normalize_symbol(symbol)
    for q in _FIAT_SUFFIXES:
        if sn.endswith(q) and len(sn) > len(q):
            base = sn[:-len(q)]
            return base in _COMMON_CRYPTO_BASES
    return False

def _to_alpaca_crypto_data_symbol(symbol: str) -> str:
    """Alpaca DATA API expects 'BASE/QUOTE' for crypto (e.g., 'BTC/USD')."""
    base, quote = _split_crypto(symbol)
    return f"{base}/{quote}"

def _to_alpaca_crypto_trading_symbol(symbol: str) -> str:
    """Alpaca TRADING API expects 'BASEQUOTE' for crypto (e.g., 'BTCUSD')."""
    base, quote = _split_crypto(symbol)
    return f"{base}{quote}"

@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float

class BrokerBase:
    name = "base"

    def is_trading_window_open(self, symbol: str, allow_extended: bool = False):
        """Return (is_open_now: bool, next_open_dt_et: datetime|None)."""
        return True, None

    def min_lot(self, symbol: str) -> float:
        return 1.0

    def get_equity(self) -> float:
        raise NotImplementedError

    def get_position(self, symbol: str) -> Optional[Position]:
        raise NotImplementedError

    def asset_class(self, symbol: str) -> str:
        return "crypto" if _is_crypto_symbol(symbol) else "equity"

    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    # NEW: most recent trade
    def get_latest_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return {'t': iso_ts, 'p': price, 's': size} for the most recent trade."""
        raise NotImplementedError

    # NEW: most recent quote (bid/ask)
    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return {'t': iso_ts, 'bp': bid_price, 'ap': ask_price} for the most recent quote."""
        raise NotImplementedError

    # NEW: historical backfill
    def get_recent_bars(self, symbol: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
        """Return up to `limit` most recent CLOSED bars, oldest→newest. Each bar is {t,o,h,l,c,v} with t as ISO8601."""
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

    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        # Local broker does not simulate a live order book
        return None

    def _tf_seconds(self, timeframe: str) -> int:
        u = timeframe.strip().lower()
        n = int(u[:-1]); k = u[-1]
        return n * (60 if k == "m" else 3600 if k == "h" else 86400)

    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        last = self._last_price.get(symbol, 100.0)
        new = max(0.01, last + random.uniform(-0.5, 0.5))
        self._last_price[symbol] = new
        iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return {"t": iso, "o": last, "h": max(last, new), "l": min(last, new), "c": new, "v": 1000}

    def get_latest_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        # Use the same generator as bars for a simple latest trade
        bar = self.get_latest_bar(symbol, "1m")
        return {"t": bar["t"], "p": float(bar["c"]), "s": 1.0}

    def get_recent_bars(self, symbol: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
        # Generate synthetic closed bars
        tf = self._tf_seconds(timeframe)
        now = int(datetime.now(timezone.utc).timestamp())
        start_bucket = (now // tf) * tf - (limit * tf)
        bars: List[Dict[str, Any]] = []
        price = self._last_price.get(symbol, 100.0)
        for i in range(limit):
            t0 = start_bucket + i * tf
            # simple random walk
            close = max(0.01, price + random.uniform(-1.0, 1.0))
            high = max(price, close) + random.uniform(0.0, 0.3)
            low = min(price, close) - random.uniform(0.0, 0.3)
            bars.append({
                "t": datetime.fromtimestamp(t0 + tf, tz=timezone.utc).isoformat(timespec="seconds"),
                "o": float(price),
                "h": float(high),
                "l": float(low),
                "c": float(close),
                "v": 1000.0
            })
            price = close
        # set last price to last close so get_latest_bar continues smoothly
        if bars:
            self._last_price[symbol] = float(bars[-1]["c"])
        return bars

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

# -------- Alpaca Broker (with IEX default feed for equities) --------
class AlpacaBroker(BrokerBase):
    name = "alpaca"

    def __init__(self, paper: bool = True):
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
            from alpaca.data.enums import DataFeed
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

        # Choose stock feed: default to IEX to avoid SIP subscription errors,
        # allow override via ALPACA_STOCK_FEED in {"IEX","SIP"}.
        feed_env = (os.getenv("ALPACA_STOCK_FEED") or "IEX").strip().upper()
        self.stock_feed = DataFeed.SIP if feed_env == "SIP" else DataFeed.IEX

    def get_bars_range(self, symbol: str, timeframe: str, start_iso: str, end_iso: str, max_bars: int = 2000, chunk: int = 500) -> List[Dict[str, Any]]:
        """Fetch CLOSED bars between [start_iso, end_iso], chunking requests to avoid short-window/limit issues.
        Returns oldest→newest bars (up to max_bars)."""
        tf = self._to_alpaca_tf(timeframe)
        bars: List[Dict[str, Any]] = []
        current_end = end_iso
        while len(bars) < max_bars:
            want = min(chunk, max_bars - len(bars))
            if _is_crypto_symbol(symbol):
                from alpaca.data.requests import CryptoBarsRequest
                sym = _to_alpaca_crypto_data_symbol(symbol)
                req = CryptoBarsRequest(symbol_or_symbols=sym, timeframe=tf, start=start_iso, end=current_end, limit=want)
                out = self.crypto_data.get_crypto_bars(req).data.get(sym, [])
            else:
                from alpaca.data.requests import StockBarsRequest
                sym = _normalize_symbol(symbol)
                req = StockBarsRequest(symbol_or_symbols=sym, timeframe=tf, start=start_iso, end=current_end, limit=want, feed=self.stock_feed)
                out = self.stock_data.get_stock_bars(req).data.get(sym, [])
            chunk_bars: List[Dict[str, Any]] = []
            for b in out:
                chunk_bars.append({
                    "t": b.timestamp.isoformat(),
                    "o": float(b.open),
                    "h": float(b.high),
                    "l": float(b.low),
                    "c": float(b.close),
                    "v": float(b.volume or 0),
                })
            if not chunk_bars:
                break
            # Ensure chronological, then append
            chunk_bars.sort(key=lambda x: x["t"])
            bars = chunk_bars + bars if (bars and chunk_bars[0]["t"] < bars[0]["t"]) else bars + chunk_bars
            # Move the end cursor to just before the earliest we've got
            earliest = chunk_bars[0]["t"]
            try:
                from datetime import datetime, timezone, timedelta
                earliest_dt = datetime.fromisoformat(earliest.replace("Z","+00:00"))
                current_end = (earliest_dt - timedelta(seconds=1)).isoformat()
            except Exception:
                break
        # Final chronological order oldest→newest
        bars.sort(key=lambda x: x["t"])
        return bars

    def min_lot(self, symbol: str) -> float:
        return 1.0 if self.asset_class(symbol) == "equity" else 0.0001

    def get_equity(self) -> float:
        acct = self.trading.get_account()
        return float(acct.portfolio_value)

    def get_position(self, symbol: str) -> Optional[Position]:
        try:
            sym = _to_alpaca_crypto_trading_symbol(symbol) if _is_crypto_symbol(symbol) else _normalize_symbol(symbol)
            p = self.trading.get_open_position(sym)
        except Exception:
            return None
        try:
            qty = float(p.qty) if hasattr(p, "qty") else float(p.current_qty)
        except Exception:
            qty = float(getattr(p, "qty", 0))
        avg = float(p.avg_entry_price)
        return Position(symbol=symbol, qty=qty, avg_price=avg)

    def _to_alpaca_tf(self, timeframe: str):
        # Map like "5m","15m","1h","1d" → Alpaca TimeFrame
        try:
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            tf = timeframe.strip().lower()
            n = int(tf[:-1]); u = tf[-1]
            unit = TimeFrameUnit.Minute if u == "m" else (TimeFrameUnit.Hour if u == "h" else TimeFrameUnit.Day)
            return TimeFrame(amount=n, unit=unit)
        except Exception as e:
            raise RuntimeError(f"Unsupported timeframe '{timeframe}': {e}")

    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        from alpaca.data.requests import StockLatestBarRequest, CryptoLatestBarRequest
        if _is_crypto_symbol(symbol):
            symbol_api = _to_alpaca_crypto_data_symbol(symbol)   # e.g., BTC/USD
            req = CryptoLatestBarRequest(symbol_or_symbols=symbol_api)
            bar = self.crypto_data.get_crypto_latest_bar(req)[symbol_api]
        else:
            sym = _normalize_symbol(symbol)  # e.g., AAPL
            req = StockLatestBarRequest(symbol_or_symbols=sym, feed=self.stock_feed)
            bar = self.stock_data.get_stock_latest_bar(req)[sym]
        return {
            "t": bar.timestamp.isoformat(),
            "o": float(bar.open),
            "h": float(bar.high),
            "l": float(bar.low),
            "c": float(bar.close),
            "v": float(bar.volume or 0),
        }

    def get_latest_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        # Return the most recent trade (matches Alpaca UI "Last Trade" better than bar close)
        if _is_crypto_symbol(symbol):
            from alpaca.data.requests import CryptoLatestTradeRequest
            sym = _to_alpaca_crypto_data_symbol(symbol)  # e.g., BTC/USD
            req = CryptoLatestTradeRequest(symbol_or_symbols=sym)
            tr = self.crypto_data.get_crypto_latest_trade(req)[sym]
        else:
            from alpaca.data.requests import StockLatestTradeRequest
            sym = _normalize_symbol(symbol)              # e.g., AAPL
            req = StockLatestTradeRequest(symbol_or_symbols=sym, feed=self.stock_feed)
            tr = self.stock_data.get_stock_latest_trade(req)[sym]
        return {"t": tr.timestamp.isoformat(), "p": float(tr.price), "s": float(getattr(tr, "size", 0) or 0)}

    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        # Return the most recent quote (bid/ask). For crypto this often updates more frequently
        # than trades; we use it for a better live display match with Alpaca UI mid price.
        if _is_crypto_symbol(symbol):
            from alpaca.data.requests import CryptoLatestQuoteRequest
            sym = _to_alpaca_crypto_data_symbol(symbol)
            req = CryptoLatestQuoteRequest(symbol_or_symbols=sym)
            qt = self.crypto_data.get_crypto_latest_quote(req)[sym]
            return {
                't': qt.timestamp.isoformat(),
                'bp': float(getattr(qt, 'bid_price', 0.0) or 0.0),
                'ap': float(getattr(qt, 'ask_price', 0.0) or 0.0),
            }
        else:
            from alpaca.data.requests import StockLatestQuoteRequest
            sym = _normalize_symbol(symbol)
            req = StockLatestQuoteRequest(symbol_or_symbols=sym, feed=self.stock_feed)
            qt = self.stock_data.get_stock_latest_quote(req)[sym]
            return {
                't': qt.timestamp.isoformat(),
                'bp': float(getattr(qt, 'bid_price', 0.0) or 0.0),
                'ap': float(getattr(qt, 'ask_price', 0.0) or 0.0),
            }

    def get_recent_bars(self, symbol: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
        # Fetch most recent CLOSED bars, oldest→newest
        tf = self._to_alpaca_tf(timeframe)
        if _is_crypto_symbol(symbol):
            from alpaca.data.requests import CryptoBarsRequest
            sym = _to_alpaca_crypto_data_symbol(symbol)
            req = CryptoBarsRequest(symbol_or_symbols=sym, timeframe=tf, limit=limit)
            out = self.crypto_data.get_crypto_bars(req).data.get(sym, [])
        else:
            from alpaca.data.requests import StockBarsRequest
            sym = _normalize_symbol(symbol)
            req = StockBarsRequest(symbol_or_symbols=sym, timeframe=tf, limit=limit, feed=self.stock_feed)
            out = self.stock_data.get_stock_bars(req).data.get(sym, [])
        bars: List[Dict[str, Any]] = []
        for b in out:
            bars.append({
                "t": b.timestamp.isoformat(),
                "o": float(b.open),
                "h": float(b.high),
                "l": float(b.low),
                "c": float(b.close),
                "v": float(b.volume or 0),
            })
        # Ensure chronological order (oldest→newest)
        bars.sort(key=lambda x: x["t"])
        return bars

    def _cancel_open_orders_for_symbol(self, sym_norm: str, timeout_sec: float = 5.0) -> None:
        """
        Cancel any open orders for the given normalized trading symbol (e.g., 'AAPL' or 'BTCUSD').
        We try multiple client methods to stay compatible across alpaca-py versions.
        """
        # Gather open orders for the symbol
        orders: List[Any] = []
        try:
            # Preferred: filter by status & symbol(s)
            from alpaca.trading.requests import GetOrdersRequest
            req = GetOrdersRequest(status='open', nested=False, symbols=[sym_norm])
            orders = list(self.trading.get_orders(filter=req) or [])
        except Exception:
            try:
                # Fallback #1: status arg
                orders = list(self.trading.get_orders(status='open') or [])
            except Exception:
                # Fallback #2: no args
                orders = list(self.trading.get_orders() or [])
        # Filter by symbol just in case
        orders = [o for o in orders if getattr(o, "symbol", "") == sym_norm]

        # Cancel each order; try cancel_order_by_id first
        for o in orders:
            oid = getattr(o, "id", None) or getattr(o, "client_order_id", None)
            if not oid:
                continue
            try:
                self.trading.cancel_order_by_id(oid)
            except Exception:
                try:
                    self.trading.cancel_order(oid)
                except Exception:
                    pass  # ignore

        # Wait until no open orders remain for the symbol (or timeout)
        end = _time.time() + max(0.0, timeout_sec)
        while _time.time() < end:
            try:
                refreshed: List[Any] = []
                try:
                    from alpaca.trading.requests import GetOrdersRequest
                    req = GetOrdersRequest(status='open', nested=False, symbols=[sym_norm])
                    refreshed = list(self.trading.get_orders(filter=req) or [])
                except Exception:
                    refreshed = list(self.trading.get_orders(status='open') or [])
            except Exception:
                refreshed = []
            refreshed = [o for o in refreshed if getattr(o, "symbol", "") == sym_norm]
            if not refreshed:
                break
            _time.sleep(0.2)

    def submit_bracket(self, symbol: str, plan) -> str:
        from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
        side = OrderSide.BUY if plan.side == "buy" else OrderSide.SELL
        qty = plan.qty

        if _is_crypto_symbol(symbol):
            # Crypto does NOT support advanced/otoco (bracket) orders on Alpaca.
            # Submit a simple market order and let the bot handle exits.
            sym = _to_alpaca_crypto_trading_symbol(symbol)  # e.g., BTCUSD
            order = MarketOrderRequest(
                symbol=sym,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC,
            )
            res = self.trading.submit_order(order_data=order)
            return res.id
        else:
            sym = _normalize_symbol(symbol)  # e.g., AAPL
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
        """
        Safely flatten a position even if bracket child orders (TP/SL) are holding qty.
        Strategy:
          1) Cancel any open orders for the symbol (TP/SL legs).
          2) Wait briefly until they are gone.
          3) Call close_position on the trading client.
        Returns (pnl_abs, pnl_pct) computed from avg_entry_price and provided exit_price.
        """
        # TRADING API uses 'BASEQUOTE' for crypto
        sym = _to_alpaca_crypto_trading_symbol(symbol) if _is_crypto_symbol(symbol) else _normalize_symbol(symbol)
        try:
            pos = self.trading.get_open_position(sym)
        except Exception:
            return 0.0, 0.0

        # Cancel any open orders (e.g., bracket OCO legs) to free held qty
        try:
            self._cancel_open_orders_for_symbol(sym_norm=sym, timeout_sec=5.0)
        except Exception:
            pass  # best-effort; proceed either way

        # Re-fetch to compute PnL & ensure we still have a position
        try:
            pos = self.trading.get_open_position(sym)
        except Exception:
            return 0.0, 0.0

        avg = float(pos.avg_entry_price)
        # qty is needed for PnL calc; API may name it qty or current_qty
        try:
            qty = float(getattr(pos, "qty", getattr(pos, "current_qty", 0)))
        except Exception:
            qty = float(getattr(pos, "qty", 0))

        side = 1 if qty > 0 else -1

        # Attempt to close; some client versions support cancel_orders flag
        try:
            try:
                self.trading.close_position(sym, cancel_orders=True)  # type: ignore[arg-type]
            except Exception:
                self.trading.close_position(sym)
        except Exception as e:
            # As a last resort, submit a market order to flatten (prefer reduce-only when supported)
            try:
                from alpaca.trading.requests import MarketOrderRequest
                from alpaca.trading.enums import OrderSide, TimeInForce
                s = OrderSide.SELL if qty > 0 else OrderSide.BUY

                # Try with reduce_only first (some alpaca-py versions support it)
                try:
                    order = MarketOrderRequest(
                        symbol=sym,
                        qty=abs(qty),
                        side=s,
                        time_in_force=TimeInForce.GTC,
                        client_order_id=f"close-reduce-{sym}-{int(datetime.now().timestamp())}",
                        reduce_only=True  # type: ignore[arg-type]
                    )
                except Exception:
                    # Fallback without reduce_only
                    order = MarketOrderRequest(
                        symbol=sym,
                        qty=abs(qty),
                        side=s,
                        time_in_force=TimeInForce.GTC,
                        client_order_id=f"close-{sym}-{int(datetime.now().timestamp())}"
                    )

                self.trading.submit_order(order_data=order)
            except Exception:
                # If even that fails, re-raise the original error so caller can log it
                raise e

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
        """Return (is_open_now, next_open_dt_ET|None).  Crypto is 24/7."""
        if _is_crypto_symbol(symbol):
            return True, None
        try:
            clock = self.trading.get_clock()
            now_et = datetime.now(tz=ET)
            if not allow_extended:
                if bool(getattr(clock, 'is_open', False)):
                    return True, None
                # Compute next regular session open (9:30 ET)
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
        return None

    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        return None

    def get_latest_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        return None

    def get_recent_bars(self, symbol: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
        return []

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
