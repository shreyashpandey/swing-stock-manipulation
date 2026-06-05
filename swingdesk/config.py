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
# Default to Opus 4.7 (most capable). Override via env if you want cheaper inference.
CLAUDE_MODEL = os.getenv("SWINGDESK_CLAUDE_MODEL", "claude-opus-4-7")
SENTIMENT_BATCH_SIZE = int(os.getenv("SWINGDESK_SENTIMENT_BATCH", "15"))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ---- Portfolio / risk sizing ----
# All in INR. Override via .env if needed.
ACCOUNT_CAPITAL = float(os.getenv("SWINGDESK_CAPITAL", "100000"))     # ₹1 lakh default
RISK_PER_TRADE_PCT = float(os.getenv("SWINGDESK_RISK_PCT", "1.0"))    # 1% of capital per trade
MAX_OPEN_POSITIONS = int(os.getenv("SWINGDESK_MAX_POSITIONS", "8"))   # concurrent positions cap

# Block opening trades within this many days of an earnings announcement
# (default 3 — covers the result day + 2 days of price reaction).
EARNINGS_BLACKOUT_DAYS = int(os.getenv("SWINGDESK_EARNINGS_BLACKOUT", "3"))

# Trailing-stop policy. When the position moves to +N*R, move stop to entry
# (breakeven). Beyond +2R, trail the stop at high - (ATR * mult) below new highs.
TRAIL_BREAKEVEN_R = float(os.getenv("SWINGDESK_TRAIL_BE_R", "1.0"))
TRAIL_ATR_MULT    = float(os.getenv("SWINGDESK_TRAIL_ATR_MULT", "1.5"))

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
