from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "swingdesk.sqlite"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# Default to Opus 4.7 (most capable). Used for the per-holding AI investment
# thesis, where nuance matters. Override via env if you want cheaper inference.
CLAUDE_MODEL = os.getenv("SWINGDESK_CLAUDE_MODEL", "claude-opus-4-7")
# Bulk news-sentiment tagging is a simple, high-volume classification task with a
# stable cached rubric — Haiku 4.5 is the fast/cheap fit and keeps the "Analyze
# news" button responsive (Opus made it take minutes per click). Separate knob so
# it doesn't affect the thesis model above.
SENTIMENT_MODEL = os.getenv("SWINGDESK_SENTIMENT_MODEL", "claude-haiku-4-5")
SENTIMENT_BATCH_SIZE = int(os.getenv("SWINGDESK_SENTIMENT_BATCH", "15"))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ---- Portfolio / risk sizing ----
# All in INR. Override via .env if needed.
ACCOUNT_CAPITAL = float(os.getenv("SWINGDESK_CAPITAL", "100000"))     # ₹1 lakh default
RISK_PER_TRADE_PCT = float(os.getenv("SWINGDESK_RISK_PCT", "1.0"))    # 1% of capital per trade
MAX_OPEN_POSITIONS = int(os.getenv("SWINGDESK_MAX_POSITIONS", "8"))   # concurrent positions cap

# Paper-autotrader risk limits.
# Portfolio "heat" = total open risk (Σ (entry−stop)×qty) as a % of capital. At
# 1% risk/trade, 6% heat ≈ six full-risk trades on at once — the book-level cap.
MAX_PORTFOLIO_HEAT_PCT = float(os.getenv("SWINGDESK_MAX_HEAT_PCT", "6.0"))
# Kill-switch: halt NEW entries when equity drawdown from its peak hits this %.
KILL_SWITCH_DD_PCT = float(os.getenv("SWINGDESK_KILL_DD_PCT", "10.0"))

# Block opening trades within this many days of an earnings announcement
# (default 3 — covers the result day + 2 days of price reaction).
EARNINGS_BLACKOUT_DAYS = int(os.getenv("SWINGDESK_EARNINGS_BLACKOUT", "3"))

# Round-trip transaction cost (brokerage + STT + charges + slippage) as a % of
# notional, applied to backtest expectancy so it reflects real-world returns.
# ~0.30% is a reasonable NSE delivery-swing default (STT alone is ~0.2% round
# trip). Backtest stats and the signal edge-gate are reported NET of this.
BACKTEST_COST_PCT = float(os.getenv("SWINGDESK_BACKTEST_COST_PCT", "0.30"))

# Trailing-stop policy. When the position moves to +N*R, move stop to entry
# (breakeven). Beyond +2R, trail the stop at high - (ATR * mult) below new highs.
TRAIL_BREAKEVEN_R = float(os.getenv("SWINGDESK_TRAIL_BE_R", "1.0"))
TRAIL_ATR_MULT    = float(os.getenv("SWINGDESK_TRAIL_ATR_MULT", "1.5"))

# ---- Groww charges (Indian equity) + capital-gains tax ----
# Used by analyze.charges to itemise the real cost of a trade. Every rate is
# overridable because Groww's brokerage and India's CG-tax rates have changed
# over time — modelled charges are INDICATIVE; for exact figures use the actual
# charge columns from a Groww contract-note / Tax-P&L export. Percentages are in
# "percent of turnover" units (0.10 == 0.10%). Sources: Groww brokerage
# calculator + NSE/SEBI rate cards (FY2024-25).
GROWW_BROKERAGE_PCT_DELIVERY = float(os.getenv("SWINGDESK_BROK_DELIV_PCT", "0.10"))  # min(cap, 0.10%)
GROWW_BROKERAGE_PCT_INTRADAY = float(os.getenv("SWINGDESK_BROK_INTRA_PCT", "0.05"))
GROWW_BROKERAGE_CAP = float(os.getenv("SWINGDESK_BROK_CAP", "20.0"))   # ₹ per executed order
STT_PCT_DELIVERY = 0.10        # % per leg (buy AND sell)
STT_PCT_INTRADAY = 0.025       # % on the sell leg only
EXCHANGE_TXN_PCT_NSE = 0.00297  # % of turnover
EXCHANGE_TXN_PCT_BSE = 0.00375
SEBI_TXN_PCT = 0.0001          # % (₹10 per crore)
STAMP_PCT_DELIVERY = 0.015     # % on the buy leg only
STAMP_PCT_INTRADAY = 0.003
GST_PCT = 18.0                 # 18% on (brokerage + exchange txn + SEBI fee)
DP_CHARGE_PER_SELL = float(os.getenv("SWINGDESK_DP_CHARGE", "15.93"))  # ₹/scrip on a delivery sell (incl GST)

# Capital-gains tax on listed equity (post 2024-07-23 budget). Indicative only —
# not tax advice. Pre-2024-07-23 sales used 15% STCG / 10% LTCG; override if
# analysing older years.
LTCG_HOLDING_DAYS = 365
STCG_TAX_PCT = float(os.getenv("SWINGDESK_STCG_PCT", "20.0"))   # Sec 111A
LTCG_TAX_PCT = float(os.getenv("SWINGDESK_LTCG_PCT", "12.5"))   # Sec 112A
LTCG_EXEMPTION = float(os.getenv("SWINGDESK_LTCG_EXEMPTION", "125000"))  # ₹1.25L/year, applied portfolio-wide

# Default watchlist — large-cap + popular swing-trading names on NSE.
# yfinance suffix for NSE: ".NS". For BSE: ".BO".
# Edit freely; this is also editable in the Streamlit UI.
DEFAULT_WATCHLIST: list[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "ITC.NS",
    "HINDUNILVR.NS",
    "BHARTIARTL.NS",
    "MARUTI.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    "HINDALCO.NS",
    "ONGC.NS",
    "COALINDIA.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "ASIANPAINT.NS",
    "TITAN.NS",
    "NESTLEIND.NS",
    "WIPRO.NS",
    "HCLTECH.NS",
    "TECHM.NS",
    "SUNPHARMA.NS",
    "DRREDDY.NS",
    "CIPLA.NS",
    "DIVISLAB.NS",
    "APOLLOHOSP.NS",
    "ULTRACEMCO.NS",
    "GRASIM.NS",
    "BRITANNIA.NS",
    "EICHERMOT.NS",
    "BAJAJ-AUTO.NS",
    "HEROMOTOCO.NS",
    "M&M.NS",
    "INDUSINDBK.NS",
    "SBILIFE.NS",
    "HDFCLIFE.NS",
    "BPCL.NS",
    "IOC.NS",
    "VEDL.NS",
    "DLF.NS",
]

# Indian market hours (IST). Used by scheduler.
MARKET_OPEN = (9, 15)
MARKET_CLOSE = (15, 30)
TIMEZONE = "Asia/Kolkata"
