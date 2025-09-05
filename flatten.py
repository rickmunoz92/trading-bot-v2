#!/usr/bin/env python3
import argparse
import sys
import os
import json

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()  # auto-detect .env in cwd/parents
except ImportError:
    pass

try:
    from alpaca.trading.client import TradingClient
except Exception:
    TradingClient = None  # type: ignore

import requests

PAPER_BASE = "https://paper-api.alpaca.markets"
LIVE_BASE = "https://api.alpaca.markets"


def auth_headers(api_key: str, api_secret: str):
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
        "accept": "application/json",
        "content-type": "application/json",
    }


def get_api_credentials():
    """Return (key, secret, source_label) from env, supporting multiple names."""
    # Preferred (official) names
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    source = "APCA_*"

    # Common alternates in some repos/configs
    if not key:
        key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY") or os.getenv("APCA_KEY_ID")
        if key:
            source = "ALPACA_API_KEY/ALPACA_KEY"
    if not secret:
        secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_SECRET_KEY")
        if secret:
            source = "ALPACA_SECRET_KEY/ALPACA_API_SECRET"

    return key, secret, source


def close_all_positions_rest(base: str, api_key: str, api_secret: str, cancel_orders: bool) -> None:
    url = f"{base}/v2/positions"
    params = {"cancel_orders": "true" if cancel_orders else "false"}
    r = requests.delete(url, headers=auth_headers(api_key, api_secret), params=params, timeout=30)
    try:
        body = r.json()
    except Exception:
        body = r.text
    print(f"[REST] DELETE {url} -> {r.status_code}")
    if body:
        try:
            print(json.dumps(body, indent=2))
        except Exception:
            print(body)


def cancel_all_open_orders_sdk(tc) -> None:
    orders = tc.get_orders(status="open")
    if not orders:
        print("No open orders to cancel.")
        return
    print(f"Cancelling {len(orders)} open order(s)...")
    for o in orders:
        try:
            tc.cancel_order_by_id(o.id)
        except Exception as e:
            print(f"  ! Failed to cancel {o.id}: {e}")


def close_all_positions_sdk(tc, cancel_orders: bool) -> None:
    if hasattr(tc, "close_all_positions"):
        try:
            tc.close_all_positions(cancel_orders=cancel_orders)
            print("Requested close_all_positions via SDK.")
            return
        except TypeError:
            if cancel_orders:
                cancel_all_open_orders_sdk(tc)
            tc.close_all_positions()
            print("Requested close_all_positions via SDK (no cancel_orders kwarg).")
            return
    if cancel_orders:
        cancel_all_open_orders_sdk(tc)
    positions = tc.get_all_positions()
    if not positions:
        print("No open positions.")
        return
    print(f"Closing {len(positions)} position(s) via market close...")
    for p in positions:
        try:
            tc.close_position(p.symbol)
        except Exception as e:
            print(f"  ! Failed to close {p.symbol}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Flatten all positions and optionally cancel orders (Alpaca).")
    parser.add_argument("--cancel-orders", action="store_true",
                        help="Also cancel all open orders (before liquidating).")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper",
                        help="Trading mode (paper or live). Default: paper")
    args = parser.parse_args()

    api_key, api_secret, source = get_api_credentials()

    # Debug print to confirm env load
    print(f"DEBUG creds source: {source}")
    print("DEBUG API_KEY loaded:", "yes" if api_key else "no")
    print("DEBUG API_SECRET loaded:", "yes" if api_secret else "no")

    if not api_key or not api_secret:
        print("Error: Missing Alpaca API credentials.")
        print("Looked for APCA_API_KEY_ID / APCA_API_SECRET_KEY, and fallbacks:")
        print("  ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_API_SECRET, ALPACA_KEY, APCA_KEY_ID, APCA_SECRET_KEY")
        sys.exit(1)

    base = PAPER_BASE if args.mode == "paper" else LIVE_BASE

    if TradingClient is not None:
        try:
            tc = TradingClient(api_key, api_secret, paper=(args.mode == "paper"))
            close_all_positions_sdk(tc, cancel_orders=args.cancel_orders)
            print("Done.")
            return
        except Exception as e:
            print(f"[SDK] Fell back to REST due to error: {e}")

    close_all_positions_rest(base, api_key, api_secret, cancel_orders=args.cancel_orders)
    print("Done. (REST)")


if __name__ == "__main__":
    main()
