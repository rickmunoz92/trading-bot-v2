
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from datetime import datetime

class R:
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    @classmethod
    def kv(cls, key: str, val: str) -> str:
        return f"{cls.CYAN}{key}{cls.WHITE}={val}{cls.RESET}"

    @classmethod
    def bold(cls, s: str) -> str:
        return f"{cls.BOLD}{s}{cls.RESET}"

    @classmethod
    def dim(cls, s: str) -> str:
        return f"{cls.DIM}{s}{cls.RESET}"

    @classmethod
    def err(cls, s: str) -> str:
        return f"{cls.RED}{s}{cls.RESET}"

    @classmethod
    def warn(cls, s: str) -> str:
        return f"{cls.YELLOW}{s}{cls.RESET}"

    @classmethod
    def title(cls, s: str) -> str:
        line = "─" * max(12, len(s) + 2)
        return f"\n{cls.BOLD}{s}{cls.RESET}\n{cls.DIM}{line}{cls.RESET}"

@dataclass
class _Pending:
    when: str
    symbol: str
    asset_class: str
    strategy: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    notes: str

class Journal:
    FIELDS = [
        "Entry Date", "Entry Time", "Exit Date", "Exit Time",
        "Symbol", "Asset Class", "Strategy/Setup",
        "Entry Price", "Stop Loss", "Take Profit", "Exit Price",
        "Position Size", "Result (Win/Loss)", "PnL ($)", "PnL (%)", "Notes"
    ]

    def __init__(self, symbol: str):
        safe = symbol.replace("/", "-")
        self.csv_path = Path(f"{safe}.csv")
        self.json_path = Path(f"{safe}.json")
        self.pending: Optional[_Pending] = None
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                csv.writer(f).writerow(self.FIELDS)
        if not self.json_path.exists():
            with self.json_path.open("w") as f:
                json.dump([], f)

    def on_entry(self, when: str, symbol: str, asset_class: str, strategy: str,
                 entry_price: float, stop_loss: float, take_profit: float,
                 position_size: float, notes: str = ""):
        self.pending = _Pending(when, symbol, asset_class, strategy,
                                entry_price, stop_loss, take_profit, position_size, notes)

    def on_exit(self, when: str, exit_price: float, win_loss: str,
                pnl_abs: float, pnl_pct: float, notes: str = ""):
        if not self.pending:
            return

        # Entry timestamp (local, human-readable)
        dt_entry = datetime.fromisoformat(self.pending.when.replace("Z", "+00:00"))
        dt_entry_local = dt_entry.astimezone()
        entry_date = dt_entry_local.strftime("%a %b %d, %Y")
        entry_time = dt_entry_local.strftime("%I:%M:%S %p")

        # Exit timestamp (local, human-readable)
        dt_exit = datetime.fromisoformat(when.replace("Z", "+00:00"))
        dt_exit_local = dt_exit.astimezone()
        exit_date = dt_exit_local.strftime("%a %b %d, %Y")
        exit_time = dt_exit_local.strftime("%I:%M:%S %p")

        row = [
            entry_date,
            entry_time,
            exit_date,
            exit_time,
            self.pending.symbol,
            self.pending.asset_class,
            self.pending.strategy,
            f"{self.pending.entry_price:.4f}",
            f"{self.pending.stop_loss:.4f}",
            f"{self.pending.take_profit:.4f}",
            f"{exit_price:.4f}",
            f"{self.pending.position_size}",
            "Win" if pnl_abs >= 0 else "Loss" if not win_loss else win_loss,
            f"{pnl_abs:.2f}",
            f"{pnl_pct:.2f}",
            notes or self.pending.notes,
        ]

        # Append to CSV
        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)

        # Mirror to JSON (same order/values as CSV)
        entry = {k: v for k, v in zip(Journal.FIELDS, row)}
        try:
            with self.json_path.open("r") as f:
                data = json.load(f)
        except Exception:
            data = []
        data.append(entry)
        with self.json_path.open("w") as f:
            json.dump(data, f, indent=2)

        self.pending = None
