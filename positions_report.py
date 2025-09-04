
#!/usr/bin/env python3
import argparse
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from app import make_broker, LocalPaperBroker  # type: ignore
except Exception as e:
    print(f"Error: could not import broker helpers from app.py: {e}", file=sys.stderr)
    sys.exit(1)

def human_local(ts_iso_utc: str) -> str:
    try:
        dt = datetime.fromisoformat(ts_iso_utc.replace("Z", "+00:00")).astimezone()
        return dt.strftime("%I:%M:%S %p %a %b %d, %Y")
    except Exception:
        return ts_iso_utc

def fmt_money(x: Optional[float]) -> str:
    if x is None or x == "—":
        return "—"
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)

def fmt_num(x: Optional[float]) -> str:
    if x is None or x == "—":
        return "—"
    try:
        xv = float(x)
    except Exception:
        return str(x)
    if abs(xv) >= 1:
        return f"{xv:,.4f}".rstrip("0").rstrip(".")
    return f"{xv:.6f}".rstrip("0").rstrip(".")

def detect_side(qty: float) -> str:
    if qty > 0:
        return "long"
    if qty < 0:
        return "short"
    return "flat"

def print_table(headers: List[str], rows: List[List[str]]) -> None:
    w = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            w[i] = max(w[i], len(cell))
    line = "  ".join(h.ljust(w[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * w[i] for i in range(len(headers))))
    for r in rows:
        print("  ".join(r[i].ljust(w[i]) for i in range(len(headers))))

def extract_pos_fields(pos: Any, default_symbol: Optional[str]) -> Dict[str, Any]:
    symbol = getattr(pos, "symbol", getattr(pos, "asset", default_symbol or "—"))
    qty = getattr(pos, "qty", getattr(pos, "quantity", None))
    avg_price = getattr(pos, "avg_price", None)
    if avg_price in (None, "—"):
        avg_price = getattr(pos, "avg_entry_price", avg_price)
    if symbol in (None, "—"):
        symbol = getattr(pos, "asset_symbol", symbol)
    take_profit = getattr(pos, "take_profit", None)
    stop_loss = getattr(pos, "stop_loss", None)
    unrl = getattr(pos, "unrealized_pnl", getattr(pos, "unrealized_pl", None))
    return {
        "symbol": symbol,
        "qty": qty,
        "avg_price": avg_price,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "unrealized_pnl": unrl,
    }

def list_positions_generic(broker) -> List[Any]:
    for name in ["list_positions", "get_positions", "positions"]:
        if hasattr(broker, name):
            try:
                res = getattr(broker, name)()
                if isinstance(res, list):
                    return res
            except Exception:
                pass
    try:
        trading = getattr(broker, "trading", None)
        if trading and hasattr(trading, "get_all_positions"):
            res = trading.get_all_positions()
            if isinstance(res, list):
                return res
    except Exception:
        pass
    return []

def list_orders_alpaca(broker, symbol: Optional[str], status: str = "open", nested: bool = True) -> List[Any]:
    try:
        trading = getattr(broker, "trading", None)
        if not trading or not hasattr(trading, "get_orders"):
            return []
        kwargs = {"status": status}
        if nested:
            kwargs["nested"] = True
        if symbol:
            kwargs["symbols"] = [symbol]
        return trading.get_orders(**kwargs)  # type: ignore
    except Exception:
        return []

def derive_tp_sl_from_orders(orders: List[Any]) -> Dict[str, Optional[float]]:
    tp = None
    sl = None
    for o in orders or []:
        legs = getattr(o, "legs", None) or getattr(o, "children", None) or []
        for leg in legs:
            lt = getattr(leg, "type", getattr(leg, "order_type", "")).lower()
            lp = getattr(leg, "limit_price", getattr(leg, "limit", None))
            sp = getattr(leg, "stop_price", getattr(leg, "stop", None))
            if lt == "limit" and lp is not None:
                tp = float(lp) if tp is None else tp
            if lt == "stop" and sp is not None:
                sl = float(sp) if sl is None else sl
        ot = getattr(o, "type", getattr(o, "order_type", "")).lower()
        if ot == "limit" and getattr(o, "limit_price", None) is not None:
            tp = float(getattr(o, "limit_price"))
        if ot == "stop" and getattr(o, "stop_price", None) is not None:
            sl = float(getattr(o, "stop_price"))
    return {"tp": tp, "sl": sl}

def main():
    ap = argparse.ArgumentParser(description="Show open positions and open orders in tables.")
    ap.add_argument("symbol", nargs="?", default=None, help="Symbol to inspect (optional, but recommended)")
    ap.add_argument("--broker", choices=["alpaca", "ibkr", "local"], default="local")
    ap.add_argument("--trade-mode", choices=["paper", "live"], default="paper")
    ap.add_argument("--equity", type=float, default=100000.0, help="Equity (used only for local paper broker)")
    ap.add_argument("--tp-pct", type=float, default=None, help="Hypothetical TP% if no bracket found (e.g., 2.0)")
    ap.add_argument("--sl-pct", type=float, default=None, help="Hypothetical SL% if no bracket found (e.g., 1.0)")
    args = ap.parse_args()

    if args.trade_mode == "paper" and args.broker == "local":
        broker = LocalPaperBroker(equity=args.equity)
    else:
        broker = make_broker(args.broker, paper=(args.trade_mode == "paper"))

    print("POSITIONS")
    rows_pos: List[List[str]] = []
    headers_pos = ["Symbol", "Side", "Qty", "Avg Price", "Take Profit", "Stop Loss", "Unrl PnL"]

    listed = list_positions_generic(broker)
    positions: List[Dict[str, Any]] = []
    if listed:
        for pos in listed:
            positions.append(extract_pos_fields(pos, args.symbol))
    elif args.symbol:
        try:
            p = broker.get_position(args.symbol)
        except Exception:
            p = None
        if p:
            positions.append(extract_pos_fields(p, args.symbol))

    for pinfo in positions:
        tp = pinfo.get("take_profit")
        sl = pinfo.get("stop_loss")
        avg = pinfo.get("avg_price")
        sym = pinfo.get("symbol")
        if tp is None or sl is None:
            orders = list_orders_alpaca(broker, sym, status="open", nested=True) or                      list_orders_alpaca(broker, sym, status="all", nested=True)
            derived = derive_tp_sl_from_orders(orders)
            tp = tp or derived["tp"]
            sl = sl or derived["sl"]
        if (tp is None or sl is None) and avg is not None and args.tp_pct is not None and args.sl_pct is not None:
            try:
                avgf = float(avg)
                tp = tp or avgf * (1 + args.tp_pct / 100.0)
                sl = sl or avgf * (1 - args.sl_pct / 100.0)
            except Exception:
                pass
        rows_pos.append([
            str(sym),
            detect_side(float(pinfo.get("qty") or 0.0)),
            fmt_num(pinfo.get("qty")),
            fmt_money(avg),
            fmt_money(tp),
            fmt_money(sl),
            fmt_money(pinfo.get("unrealized_pnl")),
        ])

    if rows_pos:
        print_table(headers_pos, rows_pos)
    else:
        print("(none)  Hint: pass a symbol, e.g. ./positions.sh AAPL --broker alpaca --trade-mode paper")

    print("\nOPEN ORDERS")
    orders = list_orders_alpaca(broker, args.symbol, status="open", nested=True)
    if not orders:
        print("(none)")
    else:
        headers_ord = ["Symbol", "Side", "Type", "Qty", "Limit/Stop", "TP", "SL", "Status", "Submitted"]
        rows_ord: List[List[str]] = []
        for o in orders:
            sym = getattr(o, "symbol", getattr(o, "asset", "—"))
            side = getattr(o, "side", "—")
            otype = getattr(o, "type", getattr(o, "order_type", "—"))
            qty = getattr(o, "qty", getattr(o, "quantity", None))
            limit_price = getattr(o, "limit_price", getattr(o, "limit", None))
            stop_price = getattr(o, "stop_price", getattr(o, "stop", None))
            d = derive_tp_sl_from_orders([o])
            status = getattr(o, "status", "—")
            submitted = getattr(o, "submitted_at", getattr(o, "created_at", None))
            if isinstance(submitted, str):
                submitted = human_local(submitted)
            rows_ord.append([
                str(sym),
                str(side),
                str(otype),
                fmt_num(qty),
                fmt_money(limit_price or stop_price),
                fmt_money(d["tp"]),
                fmt_money(d["sl"]),
                str(status),
                str(submitted or "—"),
            ])
        print_table(headers_ord, rows_ord)

if __name__ == "__main__":
    main()
