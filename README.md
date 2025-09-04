# Lightweight Trading Bot (Alpaca / IBKR / Local)

Minimal, scalable skeleton with **EMA Cross** strategy, clean console output, and automatic CSV/JSON trade journaling per symbol.

## Files
- `app.py` — CLI, loop, risk sizing, bracket planning, journaling hooks
- `adapters.py` — `BrokerBase`, working `LocalPaperBroker`, minimal **AlpacaBroker**, IBKR skeleton
- `strategies.py` — `StrategyBase`, `EmaCross`
- `io_utils.py` — renderer (cyan keys / white values), journal writer
- `requirements.txt` — deps
- `.env.example` — env template

## Install
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Configure Brokers

### Alpaca (Paper or Live)
1. Create keys in Alpaca dashboard.
2. Edit `.env` with:
   ```
   ALPACA_API_KEY=your_key
   ALPACA_SECRET_KEY=your_secret
   ALPACA_PAPER=True
   ```
   - Use `--trade-mode paper` to paper trade; `--trade-mode live` for live.
   - Paper/live selection is primarily driven by the CLI flag; `.env` is a safeguard.
3. Symbols:
   - Stocks: `AAPL`, `NVDA`, etc.
   - Crypto: `BTC/USD` (internally mapped to `BTCUSD`).

### IBKR (Skeleton)
- Install and run TWS or IB Gateway.
- In TWS: enable API connections (Configure → API → Settings).
- Set `.env`:
  ```
  IB_HOST=127.0.0.1
  IB_PORT=7497   # 7497 paper, 7496 live
  IB_CLIENT_ID=42
  ```
- Note: this skeleton stubs most IBKR functions. You’ll flesh out symbol mapping, market data, and bracket orders using `ib_insync`.

## Run (Local Paper)
```bash
python app.py 5m AAPL 0.1 2 1 --poll 60 --strategy ema_cross --trade-mode paper --broker local
```

## Run (Alpaca Paper)
```bash
# after setting .env with keys
python app.py 5m AAPL 0.1 2 1 --poll 60 --strategy ema_cross --trade-mode paper --broker alpaca
```
Crypto example:
```bash
python app.py 15m BTC/USD 0.05 0.5 0.25 --poll 30 --strategy ema_cross --trade-mode paper --broker alpaca
```

## CLI (positional + flags)
```
timeframe  symbol   risk%  tp%  sl%  [--poll SECONDS] [--strategy ema_cross] [--trade-mode paper|live] [--broker alpaca|ibkr|local] [--equity 100000] [--fast 9] [--slow 21]
```

## Journaling
- Creates `<SYMBOL>.csv` and `<SYMBOL>.json` in the working folder (e.g., `AAPL.csv`, `BTC-USD.json`).
- CSV headers: `Date,Time,Symbol,Asset Class,Strategy/Setup,Entry Price,Stop Loss,Take Profit,Exit Price,Position Size,Result (Win/Loss),PnL ($),PnL (%),Notes`.

## Notes
- The **Alpaca** adapter is minimal: latest bar, equity, position, market order with TP/SL (bracket-style). You may want to add order-idempotency, partial fills, and exact TP/SL fill tracking.
- The **Local** broker simulates prices via a tiny random walk and instantly fills orders (good for dry runs).
- Exit logic is strategy-driven; if you want hard TP/SL exits regardless of signal, add broker-side checks each poll.

## Troubleshooting
- `alpaca-py import failed`: ensure `pip install alpaca-py==0.42.0` and Python 3.10+.
- `Missing ALPACA_API_KEY/SECRET`: set in `.env` or environment.
- IBKR: verify TWS is running, API enabled, correct port, and `ib-insync` installed.
