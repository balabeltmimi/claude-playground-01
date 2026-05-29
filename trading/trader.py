"""
Automated US Stock Trader
Strategy: RSI + Alpha Vantage news sentiment
Broker:   Alpaca Paper Trading (no real money)
"""

import os
import csv
import time
import logging
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import requests
import yfinance as yf

# ── Configuration ─────────────────────────────────────────────────────────────

WATCHLIST = ["AAPL", "NVDA", "MSFT"]

RISK = {
    "max_per_trade":      100.0,   # USD per position
    "max_open_positions": 2,
    "stop_loss_pct":      0.02,    # 2% below entry
    "take_profit_pct":    0.04,    # 4% above entry
}

SIGNAL = {
    "rsi_buy":            25,      # RSI must be below this to buy
    "rsi_sell":           65,      # RSI must be above this to sell
    "sentiment_min":      0.3,     # Alpha Vantage score must exceed this
}

MARKET_OPEN  = dtime(10, 0)        # EST — avoid open volatility
MARKET_CLOSE = dtime(15, 30)       # EST — avoid close volatility

AUTO_EXECUTE = False               # Set True to enable real paper-trade orders

SIGNALS_CSV  = "signals.csv"
EST          = ZoneInfo("America/New_York")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Alpaca client ──────────────────────────────────────────────────────────────

ALPACA_BASE = "https://paper-api.alpaca.markets/v2"

def _alpaca_headers() -> dict:
    key    = os.environ["ALPACA_API_KEY"]
    secret = os.environ["ALPACA_SECRET_KEY"]
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def get_open_positions() -> dict[str, dict]:
    r = requests.get(f"{ALPACA_BASE}/positions", headers=_alpaca_headers(), timeout=10)
    r.raise_for_status()
    return {p["symbol"]: p for p in r.json()}


def place_order(symbol: str, side: str, qty: int) -> dict:
    payload = {
        "symbol":        symbol,
        "qty":           str(qty),
        "side":          side,
        "type":          "market",
        "time_in_force": "day",
    }
    r = requests.post(f"{ALPACA_BASE}/orders", json=payload,
                      headers=_alpaca_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def close_position(symbol: str) -> dict:
    r = requests.delete(f"{ALPACA_BASE}/positions/{symbol}",
                        headers=_alpaca_headers(), timeout=10)
    r.raise_for_status()
    return r.json()

# ── Market data ────────────────────────────────────────────────────────────────

def get_rsi(symbol: str, period: int = 14) -> float | None:
    df = yf.download(symbol, period="30d", interval="1d", progress=False, auto_adjust=True)
    if df.empty or len(df) < period + 1:
        return None
    close = df["Close"].squeeze()
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float("nan"))
    rsi   = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def get_price(symbol: str) -> float | None:
    info = yf.Ticker(symbol).fast_info
    return float(info.last_price) if info.last_price else None

# ── News sentiment ─────────────────────────────────────────────────────────────

def get_sentiment(symbol: str) -> float | None:
    """Returns Alpha Vantage overall_sentiment_score (-1 to +1), or None on error."""
    api_key = os.environ["ALPHAVANTAGE_API_KEY"]
    url = (
        "https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT&tickers={symbol}&limit=10&apikey={api_key}"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    feed = data.get("feed", [])
    if not feed:
        return None
    scores = [
        float(ts["ticker_sentiment_score"])
        for article in feed
        for ts in article.get("ticker_sentiment", [])
        if ts["ticker"] == symbol
    ]
    return sum(scores) / len(scores) if scores else None

# ── Signal logic ───────────────────────────────────────────────────────────────

def should_buy(symbol: str, rsi: float, sentiment: float | None) -> bool:
    if sentiment is None:
        return False
    return rsi < SIGNAL["rsi_buy"] and sentiment > SIGNAL["sentiment_min"]


def should_sell(symbol: str, rsi: float, position: dict | None) -> bool:
    if position is None:
        return False
    if rsi > SIGNAL["rsi_sell"]:
        return True
    entry = float(position["avg_entry_price"])
    current = float(position["current_price"])
    pct_change = (current - entry) / entry
    if pct_change <= -RISK["stop_loss_pct"]:
        log.info("%s  STOP-LOSS triggered  entry=%.2f  current=%.2f", symbol, entry, current)
        return True
    if pct_change >= RISK["take_profit_pct"]:
        log.info("%s  TAKE-PROFIT triggered  entry=%.2f  current=%.2f", symbol, entry, current)
        return True
    return False

# ── CSV logging ────────────────────────────────────────────────────────────────

def log_signal(symbol: str, action: str, price: float | None,
               rsi: float | None, sentiment: float | None, executed: bool) -> None:
    file_exists = os.path.isfile(SIGNALS_CSV)
    with open(SIGNALS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "action", "price", "rsi", "sentiment", "executed"])
        writer.writerow([
            datetime.now(EST).isoformat(),
            symbol,
            action,
            f"{price:.2f}" if price else "",
            f"{rsi:.2f}"   if rsi is not None else "",
            f"{sentiment:.4f}" if sentiment is not None else "",
            executed,
        ])

# ── Market hours check ─────────────────────────────────────────────────────────

def is_market_window() -> bool:
    now = datetime.now(EST).time()
    return MARKET_OPEN <= now <= MARKET_CLOSE

# ── Main loop ──────────────────────────────────────────────────────────────────

def run_once() -> None:
    if not is_market_window():
        log.info("Outside trading window (10:00–15:30 EST). Skipping.")
        return

    positions = get_open_positions()
    open_count = len(positions)

    for symbol in WATCHLIST:
        rsi       = get_rsi(symbol)
        price     = get_price(symbol)
        sentiment = get_sentiment(symbol)

        if rsi is None or price is None:
            log.warning("%s  Could not fetch data, skipping.", symbol)
            continue

        log.info("%s  RSI=%.1f  sentiment=%s  price=%.2f",
                 symbol, rsi, f"{sentiment:.3f}" if sentiment is not None else "n/a", price)

        in_position = symbol in positions

        # ── Sell check ──
        if in_position and should_sell(symbol, rsi, positions[symbol]):
            executed = False
            if AUTO_EXECUTE:
                close_position(symbol)
                executed = True
            log.info("SELL signal  %s  executed=%s", symbol, executed)
            log_signal(symbol, "SELL", price, rsi, sentiment, executed)

        # ── Buy check ──
        elif not in_position and open_count < RISK["max_open_positions"]:
            if should_buy(symbol, rsi, sentiment):
                qty = max(1, int(RISK["max_per_trade"] / price))
                executed = False
                if AUTO_EXECUTE:
                    place_order(symbol, "buy", qty)
                    executed = True
                    open_count += 1
                log.info("BUY signal  %s  qty=%d  executed=%s", symbol, qty, executed)
                log_signal(symbol, "BUY", price, rsi, sentiment, executed)

        time.sleep(1)   # respect API rate limits between tickers


def run_loop(interval_seconds: int = 300) -> None:
    log.info("Trader started. AUTO_EXECUTE=%s  Watchlist=%s", AUTO_EXECUTE, WATCHLIST)
    while True:
        try:
            run_once()
        except Exception as e:
            log.error("Cycle error: %s", e)
        log.info("Sleeping %ds until next cycle.", interval_seconds)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    run_loop()
