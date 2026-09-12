"""Institutional-flow scanner — "where is the smart money going?"

Two descriptive, public-data views (SEBI-safe — facts, not advice):

  1. recent_institutional_flow(): aggregates NSE/BSE BULK & BLOCK deals (which
     name the buyer/seller) by stock — net institutional buy/sell, the named
     clients, and a flag when a marquee institution (BlackRock, Morgan Stanley,
     JPMorgan, Jefferies, GIC, a big domestic MF, …) is on the tape.

  2. brokerage_actions(): a rule-based scan of recent headlines for sell-side
     calls — "Jefferies initiates Buy", "Morgan Stanley raises target" — so you
     can see which research desks are turning bullish/bearish on what.

Caveats worth remembering (see docs): you only see DISCLOSED trades (post-close),
foreign banks often trade via aggregate FPI flows that never name them, and
prop/market-makers like Jane Street are hedged — their tape is not a directional
signal. Track long-only accumulators, not market-makers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from swingdesk.storage import load_deals, load_news

# Marquee institutions whose appearance on the deal tape is worth flagging. These
# are substrings matched case-insensitively against the deal's client name. Mix
# of foreign long-only / sovereign funds and large domestic mutual funds.
MARQUEE_INSTITUTIONS: tuple[str, ...] = (
    "BLACKROCK", "MORGAN STANLEY", "GOLDMAN SACHS", "JPMORGAN", "JP MORGAN",
    "JEFFERIES", "CITIGROUP", "CITIBANK", "NOMURA", "UBS", "HSBC",
    "GOVERNMENT OF SINGAPORE", "GIC ", "MONETARY AUTHORITY OF SINGAPORE",
    "ABU DHABI", "NORGES", "VANGUARD", "FIDELITY", "CAPITAL GROUP", "T ROWE",
    "SOCIETE GENERALE", "BNP PARIBAS", "GRAVITON", "COPTHALL", "GMO ",
    "SBI MUTUAL", "SBI FUNDS", "ICICI PRUDENTIAL", "HDFC MUTUAL", "HDFC ASSET",
    "NIPPON", "KOTAK MAHINDRA MUTUAL", "AXIS MUTUAL", "MIRAE", "FRANKLIN",
    "DSP ", "WHITEOAK", "QUANT MUTUAL", "MOTILAL OSWAL MUTUAL", "TATA MUTUAL",
    "LIFE INSURANCE CORPORATION", "PLUTUS WEALTH", "SOCIÉTÉ GÉNÉRALE",
)

# Sell-side desks for the brokerage-action scan.
BROKERAGES: tuple[str, ...] = (
    "GOLDMAN SACHS", "MORGAN STANLEY", "JEFFERIES", "JPMORGAN", "JP MORGAN",
    "CITIGROUP", "NOMURA", "UBS", "CLSA", "MACQUARIE", "BERNSTEIN", "HSBC",
    "BANK OF AMERICA", "BOFA", "NUVAMA", "KOTAK INSTITUTIONAL", "KOTAK EQUITIES",
    "MOTILAL OSWAL", "ICICI SECURITIES", "JM FINANCIAL", "AXIS CAPITAL",
    "EMKAY", "ANTIQUE", "INVESTEC", "ELARA", "PHILLIPCAPITAL", "PRABHUDAS",
    "SHAREKHAN", "INCRED", "SYSTEMATIX",
    # Shorter fallbacks last so the fuller name wins when both appear.
    "GOLDMAN", "CITI", "MORGAN",
)

# Direction-explicit words only — ambiguous ones (e.g. "target price", "trigger",
# "raises" alone) are deliberately excluded so they don't force a false direction.
_BULL_WORDS = ("BUY", "OUTPERFORM", "OVERWEIGHT", "ACCUMULATE", "ADD",
               "UPGRADE", "UPGRADES", "INITIATE", "INITIATES", "RAISES TARGET",
               "RAISE TARGET", "HIKES TARGET", "TARGET RAISED", "BULLISH",
               "TOP PICK", "UPSIDE", "POSITIVE", "RE-RATE", "RERATE", "RERATING",
               "BENEFICIARY", "BENEFICIARIES", "REITERATES BUY")
_BEAR_WORDS = ("SELL", "UNDERPERFORM", "UNDERWEIGHT", "REDUCE", "DOWNGRADE",
               "DOWNGRADES", "CUTS TARGET", "CUT TARGET", "LOWERS TARGET",
               "TARGET CUT", "DOWNSIDE", "NEGATIVE", "BEARISH", "CAUTIOUS", "NEUTRAL")


@dataclass
class InstitutionalFlow:
    ticker: str
    security: str | None
    n_deals: int
    buy_qty: float
    sell_qty: float
    net_qty: float
    buy_value: float
    sell_value: float
    net_value: float
    net_side: str                       # "BUY" | "SELL" | "FLAT"
    clients: list[tuple[str, str]] = field(default_factory=list)   # (name, side)
    marquee: list[str] = field(default_factory=list)
    exchanges: list[str] = field(default_factory=list)
    latest_date: str | None = None


@dataclass
class BrokerageAction:
    ticker: str | None
    broker: str
    action: str                         # the matched keyword
    sentiment: str                      # "bullish" | "bearish"
    headline: str
    published: str | None = None
    link: str | None = None


@dataclass
class InvestorActivity:
    client: str
    n_deals: int
    tickers: list[str]
    buy_value: float
    sell_value: float
    net_value: float
    net_side: str
    marquee: str | None = None
    exchanges: list[str] = field(default_factory=list)
    latest_date: str | None = None


def _match_marquee(client: str | None) -> str | None:
    if not client:
        return None
    up = f" {client.upper()} "
    for name in MARQUEE_INSTITUTIONS:
        n = name.strip()
        # Word-boundary so short names ('GIC') don't match inside 'LOGIC' etc.
        if re.search(rf"\b{re.escape(n)}\b", up):
            return n
    return None


def aggregate_flow(deals: pd.DataFrame, only_marquee: bool = False,
                   limit: int = 30) -> list[InstitutionalFlow]:
    """Pure core: roll bulk/block deal rows up per ticker. DB-free for testing.

    `deals` needs columns: date, ticker, security, client, side, qty, price.
    """
    if deals is None or deals.empty:
        return []
    agg: dict[str, dict] = {}
    for r in deals.itertuples():
        tk = getattr(r, "ticker", None)
        if not tk:
            continue
        side = str(getattr(r, "side", "") or "").upper()
        qty = float(getattr(r, "qty", 0) or 0)
        price = float(getattr(r, "price", 0) or 0)
        value = qty * price
        a = agg.setdefault(tk, {"security": getattr(r, "security", None), "n": 0,
                                "buy_qty": 0.0, "sell_qty": 0.0,
                                "buy_val": 0.0, "sell_val": 0.0,
                                "clients": [], "marquee": set(), "exchanges": set(),
                                "latest": None})
        a["n"] += 1
        if side.startswith("B"):
            a["buy_qty"] += qty
            a["buy_val"] += value
        elif side.startswith("S"):
            a["sell_qty"] += qty
            a["sell_val"] += value
        ex = str(getattr(r, "exchange", "") or "").upper()
        if ex:
            a["exchanges"].add(ex)
        client = getattr(r, "client", None)
        if client:
            a["clients"].append((str(client), side))
        mk = _match_marquee(client)
        if mk:
            a["marquee"].add(mk)
        d = getattr(r, "date", None)
        d = str(d.date()) if hasattr(d, "date") else (str(d) if d is not None else None)
        if d and (a["latest"] is None or d > a["latest"]):
            a["latest"] = d

    flows: list[InstitutionalFlow] = []
    for tk, a in agg.items():
        if only_marquee and not a["marquee"]:
            continue
        net_qty = a["buy_qty"] - a["sell_qty"]
        net_val = a["buy_val"] - a["sell_val"]
        net_side = "BUY" if net_qty > 0 else "SELL" if net_qty < 0 else "FLAT"
        flows.append(InstitutionalFlow(
            ticker=tk, security=a["security"], n_deals=a["n"],
            buy_qty=a["buy_qty"], sell_qty=a["sell_qty"], net_qty=net_qty,
            buy_value=round(a["buy_val"], 2), sell_value=round(a["sell_val"], 2),
            net_value=round(net_val, 2), net_side=net_side,
            clients=a["clients"][:12], marquee=sorted(a["marquee"]),
            exchanges=sorted(a["exchanges"]),
            latest_date=a["latest"],
        ))
    # Marquee names first, then by absolute net value traded.
    flows.sort(key=lambda f: (bool(f.marquee), abs(f.net_value)), reverse=True)
    return flows[:limit]


def recent_institutional_flow(days: int = 30, only_marquee: bool = False,
                              limit: int = 30) -> list[InstitutionalFlow]:
    """Live: aggregate the last `days` of stored bulk/block deals. Run
    ``swingdesk nse --all-deals`` first to capture market-wide deals."""
    return aggregate_flow(load_deals(days=days), only_marquee=only_marquee, limit=limit)


def investor_activity(days: int = 30, net_side: str | None = None,
                      limit: int = 30) -> list[InvestorActivity]:
    """Group recent disclosed deals by investor/client across stocks."""
    deals = load_deals(days=days)
    if deals.empty:
        return []
    agg: dict[str, dict] = {}
    for r in deals.itertuples():
        client = str(getattr(r, "client", "") or "").strip()
        if not client:
            continue
        side = str(getattr(r, "side", "") or "").upper()
        qty = float(getattr(r, "qty", 0) or 0)
        price = float(getattr(r, "price", 0) or 0)
        value = qty * price
        a = agg.setdefault(client, {
            "n": 0, "buy_val": 0.0, "sell_val": 0.0, "tickers": set(),
            "latest": None, "marquee": _match_marquee(client), "exchanges": set(),
        })
        a["n"] += 1
        if side.startswith("B"):
            a["buy_val"] += value
        elif side.startswith("S"):
            a["sell_val"] += value
        ticker = str(getattr(r, "ticker", "") or "").strip()
        if ticker:
            a["tickers"].add(ticker)
        ex = str(getattr(r, "exchange", "") or "").upper()
        if ex:
            a["exchanges"].add(ex)
        d = getattr(r, "date", None)
        d = str(d.date()) if hasattr(d, "date") else (str(d) if d is not None else None)
        if d and (a["latest"] is None or d > a["latest"]):
            a["latest"] = d

    out: list[InvestorActivity] = []
    want = (net_side or "").upper().strip()
    for client, a in agg.items():
        net_val = a["buy_val"] - a["sell_val"]
        side = "BUY" if net_val > 0 else "SELL" if net_val < 0 else "FLAT"
        if want and side != want:
            continue
        out.append(InvestorActivity(
            client=client,
            n_deals=a["n"],
            tickers=sorted(a["tickers"])[:8],
            buy_value=round(a["buy_val"], 2),
            sell_value=round(a["sell_val"], 2),
            net_value=round(net_val, 2),
            net_side=side,
            marquee=a["marquee"],
            exchanges=sorted(a["exchanges"]),
            latest_date=a["latest"],
        ))
    out.sort(key=lambda x: (bool(x.marquee), abs(x.net_value), x.n_deals), reverse=True)
    return out[:limit]


def deal_coverage(days: int = 30) -> pd.DataFrame:
    """Simple coverage read: how much disclosed deal tape exists by exchange/type."""
    deals = load_deals(days=days)
    if deals.empty:
        return pd.DataFrame(columns=["exchange", "deal_type", "rows", "latest_date", "stocks", "clients"])
    tmp = deals.copy()
    tmp["date_only"] = tmp["date"].dt.strftime("%Y-%m-%d")
    rows = []
    for (exchange, deal_type), sub in tmp.groupby(["exchange", "deal_type"], dropna=False):
        rows.append({
            "exchange": exchange or "—",
            "deal_type": deal_type or "—",
            "rows": int(len(sub)),
            "latest_date": str(sub["date_only"].max()),
            "stocks": int(sub["ticker"].nunique()),
            "clients": int(sub["client"].fillna("").replace("", pd.NA).dropna().nunique()),
        })
    return pd.DataFrame(rows).sort_values(["exchange", "deal_type"]).reset_index(drop=True)


def extract_brokerage_action(title: str | None,
                             summary: str | None = "") -> tuple[str, str, str] | None:
    """Pure core: detect a (broker, action_keyword, sentiment), or None. Requires
    BOTH a known broker name AND a directional keyword.

    The broker may appear in title OR summary, but the *direction* is read from the
    TITLE only — summaries are full of "should you buy, sell or hold?" boilerplate
    that would otherwise force a false call. That boilerplate is stripped too.
    """
    if not title:
        return None
    broker_text = f" {(title + ' ' + (summary or '')).upper()} "
    broker = next((b for b in BROKERAGES if re.search(rf"\b{re.escape(b)}\b", broker_text)), None)
    if not broker:
        return None
    act = f" {title.upper()} "
    # Drop the "buy, sell (or) hold" question phrasing — it's not a real call.
    act = re.sub(r"\bBUY[\s,/]+SELL[\s,/]+(OR\s+)?HOLD\b", " ", act)
    act = re.sub(r"\bBUY[\s,/]+(OR\s+)?SELL\b", " ", act)
    act = re.sub(r"\bSHOULD YOU (BUY|SELL|HOLD)\b", " ", act)
    # Bearish first: a downgrade / "cautious" / "neutral" is an explicit call that
    # should win over a softer bullish word ("upside") in the same headline.
    for w in _BEAR_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", act):
            return broker, w, "bearish"
    for w in _BULL_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", act):
            return broker, w, "bullish"
    return None


def brokerage_actions(days: int = 7, limit: int = 40) -> list[BrokerageAction]:
    """Live: scan recent headlines for sell-side calls (Jefferies/MS/JPM/…).
    Works on raw titles — no Claude analysis required."""
    news = load_news(limit=1000)
    if news.empty:
        return []
    pub = pd.to_datetime(news["published"], errors="coerce", utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    news = news[pub.isna() | (pub >= cutoff)]
    out: list[BrokerageAction] = []
    for r in news.itertuples():
        title = getattr(r, "title", "") or ""
        summary = getattr(r, "summary", "") or ""
        hit = extract_brokerage_action(title, summary)
        if not hit:
            continue
        broker, action, sentiment = hit
        tickers = str(getattr(r, "tickers", "") or "").split(",")
        ticker = next((t.strip() for t in tickers if t.strip()), None)
        out.append(BrokerageAction(
            ticker=ticker, broker=broker, action=action, sentiment=sentiment,
            headline=title.strip(), published=getattr(r, "published", None),
            link=getattr(r, "link", None),
        ))
        if len(out) >= limit:
            break
    return out
