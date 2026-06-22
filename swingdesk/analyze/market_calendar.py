"""A year's worth of dates that historically tilt the Indian market bullish or
bearish — so you can position *ahead* of them instead of reacting.

Two kinds of dates live here:

  * **Deterministic** — computable exactly for any year: the Union Budget
    (Feb 1), monthly F&O expiry (last Thursday), the four earnings seasons,
    fixed national-holiday closes, and the well-documented seasonality windows
    ("Sell in May", the Santa rally, festive-consumption season, FY-end window
    dressing). These are always right.

  * **Floating** — tied to the lunar calendar or to a regulator's schedule
    (Diwali / Muhurat trading, Holi, Akshaya Tritiya, RBI MPC, US FOMC). Their
    exact dates are *announced*, not derived, so they're kept in ``CURATED``
    per year and flagged ``approximate=True``. Treat those as "confirm the
    date" markers, not gospel — edit ``CURATED`` when the official dates drop.

Bias is a *historical tendency*, not a guarantee. "Sell in May" is a tilt, not
a rule — use these to size and time, not to bet the farm.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, timedelta

# Bias buckets (UI maps these to colours).
BULLISH = "Bullish"
BEARISH = "Bearish"
VOLATILE = "Volatile"
NEUTRAL = "Neutral"

# Categories for filtering.
CAT_POLICY = "Policy"
CAT_EARNINGS = "Earnings"
CAT_EXPIRY = "Expiry"
CAT_SEASON = "Seasonality"
CAT_FESTIVE = "Festive"
CAT_GLOBAL = "Global"
CAT_HOLIDAY = "Holiday"


@dataclass
class MarketEvent:
    name: str
    category: str
    bias: str
    start: date
    end: date | None = None          # None => single day
    note: str = ""                   # rationale + how to position
    approximate: bool = False        # exact date needs official confirmation
    sectors: list[str] = field(default_factory=list)  # yfinance sectors this event tilts (empty = broad)

    @property
    def is_range(self) -> bool:
        return self.end is not None and self.end != self.start

    def to_row(self) -> dict:
        return {
            "start": self.start,
            "end": self.end or self.start,
            "name": self.name,
            "category": self.category,
            "bias": self.bias,
            "approximate": self.approximate,
            "sectors": self.sectors,
            "note": self.note,
        }


# yfinance sector buckets, for the event sector tilts below.
SEC_INDUSTRIALS = "Industrials"
SEC_FINANCIALS = "Financial Services"
SEC_TECH = "Technology"
SEC_CONS_CYC = "Consumer Cyclical"
SEC_CONS_DEF = "Consumer Defensive"
SEC_MATERIALS = "Basic Materials"
SEC_REALTY = "Real Estate"


# --- date helpers -------------------------------------------------------------
def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Last `weekday` (Mon=0 … Sun=6) of a month — e.g. last Thursday for expiry."""
    d = date(year, month, calendar.monthrange(year, month)[1])
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


# --- deterministic events -----------------------------------------------------
def _deterministic(year: int) -> list[MarketEvent]:
    ev: list[MarketEvent] = []

    # Union Budget — single biggest scheduled volatility event of the year.
    ev.append(MarketEvent(
        "Union Budget", CAT_POLICY, VOLATILE, date(year, 2, 1),
        note="Biggest scheduled-volatility day of the year. Sharp two-way swings; "
             "sectors move on tax/allocation news. Avoid fresh leveraged bets into it; "
             "let the knee-jerk move settle before positioning.",
    ))
    ev.append(MarketEvent(
        "Economic Survey", CAT_POLICY, VOLATILE, date(year, 1, 31),
        note="Tabled the day before the Budget — sets the growth/fiscal tone.",
    ))

    # Monthly F&O expiry — last Thursday of each month.
    for m in range(1, 13):
        ev.append(MarketEvent(
            f"Monthly F&O expiry ({calendar.month_abbr[m]})", CAT_EXPIRY, VOLATILE,
            _last_weekday(year, m, 3),
            note="Expiry-day churn and rollovers — intraday whippy. Don't read a single "
                 "expiry candle as a trend; wait for the next session to confirm.",
        ))

    # Four earnings seasons (approx windows — stock-specific volatility).
    earnings = [
        ("Q3 (Oct–Dec) earnings season", (1, 10), (2, 14)),
        ("Q4 + full-year earnings season", (4, 10), (5, 30)),
        ("Q1 (Apr–Jun) earnings season", (7, 10), (8, 14)),
        ("Q2 (Jul–Sep) earnings season", (10, 10), (11, 14)),
    ]
    for name, (sm, sd), (em, ed) in earnings:
        ev.append(MarketEvent(
            name, CAT_EARNINGS, VOLATILE, date(year, sm, sd), date(year, em, ed),
            note="Stock-specific gaps on results. Hold winners through the print only with "
                 "a plan; avoid carrying weak hands into their result date.",
        ))

    # Seasonality windows (the documented calendar effects).
    ev.append(MarketEvent(
        "New-year / January effect", CAT_SEASON, BULLISH, date(year, 1, 1), date(year, 1, 15),
        note="Fresh allocations + small-cap bid early in the year. Mild positive tilt.",
    ))
    ev.append(MarketEvent(
        "Pre-Budget run-up", CAT_SEASON, BULLISH, date(year, 1, 20), date(year, 1, 31),
        note="Sector rotation on Budget expectations (rail, infra, defence, capex). "
             "Theme trades run up — book before the event, not after.",
        sectors=[SEC_INDUSTRIALS, SEC_MATERIALS],
    ))
    ev.append(MarketEvent(
        "FY-end window dressing", CAT_SEASON, BULLISH, date(year, 3, 20), date(year, 3, 31),
        note="Funds mark up holdings into March 31 NAV. Quality large-caps often firm; "
             "beware early-April give-back.",
    ))
    ev.append(MarketEvent(
        "'Sell in May and go away'", CAT_SEASON, BEARISH, date(year, 5, 1), date(year, 5, 31),
        note="Start of the historically weaker May–Sep stretch. Tighten stops, trim "
             "over-extended positions, lower leverage rather than chase.",
    ))
    ev.append(MarketEvent(
        "September seasonal soft patch", CAT_SEASON, BEARISH, date(year, 9, 1), date(year, 9, 30),
        note="Historically one of the weakest months globally. Stay selective; keep cash ready "
             "for the festive-season setups that follow.",
    ))
    ev.append(MarketEvent(
        "Festive-consumption season", CAT_SEASON, BULLISH, date(year, 9, 25), date(year, 11, 15),
        note="Navratri→Diwali demand lifts auto, consumer-durables, retail, jewellery. "
             "Position in consumption names ahead of the festive run.",
        sectors=[SEC_CONS_CYC, SEC_CONS_DEF],
    ))
    ev.append(MarketEvent(
        "Santa Claus rally", CAT_SEASON, BULLISH, date(year, 12, 20), date(year, 12, 31),
        note="Light-volume year-end drift higher, globally positive seasonality. "
             "Tends to favour large-caps and the prior year's winners.",
    ))

    # US non-farm payrolls — first Friday each month (global risk pivot).
    for m in range(1, 13):
        d = date(year, m, 1)
        while d.weekday() != 4:  # Friday
            d += timedelta(days=1)
        ev.append(MarketEvent(
            f"US jobs report (NFP, {calendar.month_abbr[m]})", CAT_GLOBAL, VOLATILE, d,
            note="US payrolls move global rate expectations → gaps in IT, banks, and the rupee.",
        ))

    # Fixed national-holiday closes (NSE shut — fully deterministic dates).
    for (mm, dd, label) in [
        (1, 26, "Republic Day"), (4, 14, "Dr. Ambedkar Jayanti"),
        (5, 1, "Maharashtra Day / May Day"), (8, 15, "Independence Day"),
        (10, 2, "Gandhi Jayanti"), (12, 25, "Christmas"),
    ]:
        ev.append(MarketEvent(
            f"NSE holiday — {label}", CAT_HOLIDAY, NEUTRAL, date(year, mm, dd),
            note="Market closed. No trades; F&O carry/theta still accrues over the break.",
        ))

    return ev


# --- floating / announced events (edit per year) ------------------------------
# Lunar-calendar festivals and regulator meeting dates are ANNOUNCED, not derived.
# Best-effort dates below are flagged approximate=True — confirm against the NSE
# holiday list, the RBI MPC schedule, and the Fed calendar, then correct here.
def _curated(year: int) -> list[MarketEvent]:
    table = CURATED.get(year, {})
    out: list[MarketEvent] = []

    for d, name, note in table.get("rbi_mpc", []):
        out.append(MarketEvent(f"RBI MPC decision — {name}", CAT_POLICY, VOLATILE,
                               d, note=note, approximate=True,
                               sectors=[SEC_FINANCIALS, SEC_REALTY, SEC_CONS_CYC]))
    for d, name, note in table.get("fomc", []):
        out.append(MarketEvent(f"US Fed FOMC — {name}", CAT_GLOBAL, VOLATILE,
                               d, note=note, approximate=True, sectors=[SEC_TECH]))
    for d, name, bias, note in table.get("festive", []):
        # Auspicious buying days tilt toward consumption / jewellery names.
        secs = [SEC_CONS_CYC] if bias == BULLISH else []
        out.append(MarketEvent(name, CAT_FESTIVE, bias, d, note=note,
                               approximate=True, sectors=secs))

    return out


# Per-year curated data. Replace the approximate dates with the official ones.
# (RBI publishes the MPC calendar; the Fed publishes FOMC dates; NSE publishes
#  the festival/holiday list each December for the following year.)
CURATED: dict[int, dict] = {
    2026: {
        "rbi_mpc": [
            (date(2026, 2, 6), "Feb", "First MPC of the calendar year — sets the rate/liquidity tone for H1."),
            (date(2026, 4, 8), "Apr", "Start-of-FY policy; guidance on the rate cycle for the new year."),
            (date(2026, 6, 5), "Jun", "Mid-year check on inflation vs growth."),
            (date(2026, 8, 7), "Aug", "Post-monsoon read on food inflation."),
            (date(2026, 10, 8), "Oct", "Festive-season liquidity + festive-demand inflation."),
            (date(2026, 12, 4), "Dec", "Year-end stance into the next Budget."),
        ],
        "fomc": [
            (date(2026, 1, 28), "Jan", "Sets the global rate tone for the year — IT & banks gap on the dot-plot."),
            (date(2026, 3, 18), "Mar", "Quarterly projections (SEP) — high global volatility."),
            (date(2026, 4, 29), "Apr", "Inter-meeting; statement-only."),
            (date(2026, 6, 17), "Jun", "SEP meeting — watch USD/INR and FII flows."),
            (date(2026, 7, 29), "Jul", "Statement-only."),
            (date(2026, 9, 16), "Sep", "SEP meeting — coincides with the seasonal soft patch."),
            (date(2026, 10, 28), "Oct", "Statement-only."),
            (date(2026, 12, 16), "Dec", "Final SEP of the year — sets the H1-next tone."),
        ],
        "festive": [
            (date(2026, 3, 4), "Holi (market closed)", NEUTRAL,
             "Lunar date — confirm. Market typically closed; thin liquidity around it."),
            (date(2026, 4, 19), "Akshaya Tritiya", BULLISH,
             "Auspicious gold-buying day — tailwind for jewellery & gold-financier stocks. Confirm date."),
            (date(2026, 11, 6), "Dhanteras", BULLISH,
             "Start of the Diwali buying window — consumer-durables, autos, jewellery. Confirm date."),
            (date(2026, 11, 8), "Diwali — Muhurat trading", BULLISH,
             "Symbolic one-hour evening session; new Samvat. Sentiment-positive, token buying. Confirm date."),
        ],
    },
}


# --- public API ---------------------------------------------------------------
def build_calendar(year: int) -> list[MarketEvent]:
    """All market-tilting events for `year`, sorted by start date."""
    events = _deterministic(year) + _curated(year)
    events.sort(key=lambda e: (e.start, e.name))
    return events


def upcoming(year: int, today: date, within_days: int = 60) -> list[MarketEvent]:
    """Events whose start falls within the next `within_days` from `today`."""
    horizon = today + timedelta(days=within_days)
    return [e for e in build_calendar(year) if today <= e.start <= horizon]


def upcoming_bullish(year: int, today: date, within_days: int = 75) -> list[MarketEvent]:
    """Upcoming bullish events — the windows worth positioning *into*. Used to
    drive calendar-based stock ideas (pair with sectors.event_picks)."""
    return [e for e in upcoming(year, today, within_days) if e.bias == BULLISH]


def has_curated(year: int) -> bool:
    """Whether floating (festival/MPC/FOMC) dates are filled in for this year."""
    return year in CURATED
