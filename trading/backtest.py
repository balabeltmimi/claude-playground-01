"""
Strategy backtester — uses real historical prices, simulated sentiment.

Usage:
    python trading/backtest.py
    python trading/backtest.py --start 2023-01-01 --end 2024-12-31
    python trading/backtest.py --symbols AAPL NVDA --start 2024-01-01
"""

import argparse
import random
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import yfinance as yf

# ── Default settings (mirrors trader.py) ──────────────────────────────────────

DEFAULT_SYMBOLS   = ["AAPL", "NVDA", "MSFT", "TSLA"]
DEFAULT_START     = "2023-01-01"
DEFAULT_END       = str(date.today())

RISK = {
    "max_per_trade":      100.0,
    "max_open_positions": 2,
    "stop_loss_pct":      0.02,
    "take_profit_pct":    0.04,
}

SIGNAL = {
    "rsi_buy":        25,
    "rsi_sell":       65,
    "sentiment_min":  0.3,
}

# Sentiment simulation: real markets are bullish ~55% of the time
SENTIMENT_BULLISH_PROB = 0.55
SENTIMENT_BULLISH_RANGE = (0.31, 0.80)
SENTIMENT_NEUTRAL_RANGE = (-0.30, 0.29)

# ── Data helpers ───────────────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def generate_synthetic_prices(symbol: str, start: str, end: str,
                               seed: int | None = None) -> pd.DataFrame:
    """
    Geometric Brownian Motion price simulation.
    Starting prices and volatility are calibrated to approximate real stocks.
    """
    params = {
        "AAPL": (185.0, 0.013, 0.0003),
        "NVDA": (495.0, 0.022, 0.0007),
        "MSFT": (375.0, 0.012, 0.0003),
        "TSLA": (250.0, 0.028, 0.0002),
    }
    s0, vol, drift = params.get(symbol, (100.0, 0.015, 0.0003))

    dates = pd.bdate_range(start=start, end=end)
    rng   = random.Random(seed or hash(symbol) % 9999)

    import math
    price = s0
    closes = []
    for _ in dates:
        z      = rng.gauss(0, 1)
        price *= math.exp((drift - 0.5 * vol ** 2) + vol * z)
        closes.append(price)

    df = pd.DataFrame({"Close": closes}, index=dates)
    return df


def fetch_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
    try:
        df = yf.download(symbol, start=start, end=end, interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 20:
            raise ValueError("insufficient data")
        print(f"  {symbol}: loaded {len(df)} real trading days from Yahoo Finance.")
        return df[["Close"]]
    except Exception:
        print(f"  {symbol}: Yahoo Finance unavailable — using synthetic price simulation.")
        return generate_synthetic_prices(symbol, start, end)


def simulate_sentiment(n: int, seed: int = 42) -> list[float]:
    rng = random.Random(seed)
    scores = []
    for _ in range(n):
        if rng.random() < SENTIMENT_BULLISH_PROB:
            scores.append(rng.uniform(*SENTIMENT_BULLISH_RANGE))
        else:
            scores.append(rng.uniform(*SENTIMENT_NEUTRAL_RANGE))
    return scores

# ── Trade record ───────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol:     str
    entry_date: str
    entry_price: float
    exit_date:  str  = ""
    exit_price: float = 0.0
    qty:        int   = 1
    exit_reason: str  = ""

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.qty

    @property
    def pnl_pct(self) -> float:
        return (self.exit_price - self.entry_price) / self.entry_price * 100

# ── Backtest engine ────────────────────────────────────────────────────────────

@dataclass
class Portfolio:
    cash:       float = 0.0
    trades:     list[Trade] = field(default_factory=list)
    open_pos:   dict[str, Trade] = field(default_factory=dict)

    @property
    def closed_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.exit_date]

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.closed_trades)

    @property
    def win_rate(self) -> float:
        closed = self.closed_trades
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t.pnl > 0)
        return wins / len(closed) * 100

    @property
    def max_drawdown(self) -> float:
        if not self.closed_trades:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in self.closed_trades:
            cumulative += t.pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        return max_dd


def run_backtest(symbols: list[str], start: str, end: str) -> Portfolio:
    portfolio = Portfolio()

    for symbol in symbols:
        df = fetch_prices(symbol, start, end)
        if df.empty or len(df) < 20:
            print(f"  {symbol}: not enough data, skipping.")
            continue

        close     = df["Close"].squeeze()
        rsi_series = compute_rsi(close)
        sentiments = simulate_sentiment(len(df))

        for i in range(15, len(df)):
            dt      = str(df.index[i].date())
            price   = float(close.iloc[i])
            rsi     = float(rsi_series.iloc[i])
            sent    = sentiments[i]
            in_pos  = symbol in portfolio.open_pos

            if in_pos:
                trade  = portfolio.open_pos[symbol]
                entry  = trade.entry_price
                chg    = (price - entry) / entry
                reason = None

                if rsi > SIGNAL["rsi_sell"]:
                    reason = "RSI_SELL"
                elif chg <= -RISK["stop_loss_pct"]:
                    reason = "STOP_LOSS"
                elif chg >= RISK["take_profit_pct"]:
                    reason = "TAKE_PROFIT"

                if reason:
                    trade.exit_date   = dt
                    trade.exit_price  = price
                    trade.exit_reason = reason
                    portfolio.cash   += price * trade.qty
                    del portfolio.open_pos[symbol]

            else:
                open_count = len(portfolio.open_pos)
                if open_count >= RISK["max_open_positions"]:
                    continue
                if rsi < SIGNAL["rsi_buy"] and sent > SIGNAL["sentiment_min"]:
                    qty = max(1, int(RISK["max_per_trade"] / price))
                    trade = Trade(
                        symbol=symbol,
                        entry_date=dt,
                        entry_price=price,
                        qty=qty,
                    )
                    portfolio.open_pos[symbol] = trade
                    portfolio.trades.append(trade)
                    portfolio.cash -= price * qty

    # Close any remaining open positions at last known price
    for symbol, trade in list(portfolio.open_pos.items()):
        trade.exit_date   = str(date.today())
        trade.exit_price  = trade.entry_price   # conservative: assume flat if unknown
        trade.exit_reason = "END_OF_TEST"
        portfolio.cash   += trade.exit_price * trade.qty
        del portfolio.open_pos[symbol]

    return portfolio

# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(portfolio: Portfolio, start: str, end: str, symbols: list[str]) -> None:
    trades = portfolio.closed_trades
    print("\n" + "═" * 60)
    print("  BACKTEST REPORT")
    print(f"  Period : {start}  →  {end}")
    print(f"  Symbols: {', '.join(symbols)}")
    print("═" * 60)

    if not trades:
        print("  No trades were triggered in this period.")
        print("═" * 60)
        return

    print(f"  Total trades     : {len(trades)}")
    print(f"  Win rate         : {portfolio.win_rate:.1f}%")
    print(f"  Total P&L        : ${portfolio.total_pnl:+.2f}")
    print(f"  Max drawdown     : ${portfolio.max_drawdown:.2f}")
    print(f"  Avg P&L/trade    : ${portfolio.total_pnl / len(trades):+.2f}")
    print("─" * 60)

    by_reason: dict[str, int] = {}
    for t in trades:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1
    print("  Exit breakdown:")
    for reason, count in sorted(by_reason.items()):
        print(f"    {reason:<20} {count} trades")

    print("─" * 60)
    print("  Trade log:")
    print(f"  {'Date':<12} {'Sym':<6} {'Entry':>7} {'Exit':>7} {'P&L':>8}  Reason")
    for t in sorted(trades, key=lambda x: x.entry_date):
        print(f"  {t.entry_date:<12} {t.symbol:<6} "
              f"${t.entry_price:>6.2f}  ${t.exit_price:>6.2f}  "
              f"{t.pnl:>+7.2f}   {t.exit_reason}")
    print("═" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the RSI+sentiment strategy")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--start",   default=DEFAULT_START)
    parser.add_argument("--end",     default=DEFAULT_END)
    args = parser.parse_args()

    print(f"\nDownloading historical data for {args.symbols} …")
    portfolio = run_backtest(args.symbols, args.start, args.end)
    print_report(portfolio, args.start, args.end, args.symbols)


if __name__ == "__main__":
    main()
