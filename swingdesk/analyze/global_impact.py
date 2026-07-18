"""Map global market news into likely Indian sector/watchlist impact.

This is intentionally curated, not "fetch everything". The goal is a useful
Global Impact Radar: identify market-moving global cues, map them to Indian
sectors and known tickers, and keep the output descriptive.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd

from swingdesk.storage import load_fundamentals, load_global_news


@dataclass
class GlobalImpact:
    source: str
    title: str
    link: str
    published: str | None
    cue: str
    direction: str
    impact_score: float
    sectors: list[str]
    tickers: list[str]
    rationale: str
    summary: str | None = None


SECTOR_ALIASES = {
    "IT": ("software", "technology", "tech stocks", "nasdaq", "ai ", "cloud", "outsourcing"),
    "Pharma": ("pharma", "drug", "fda", "healthcare", "biotech", "medicine"),
    "Oil & Gas": ("oil", "crude", "brent", "opec", "gas", "energy"),
    "Metals": ("metal", "steel", "copper", "aluminium", "aluminum", "iron ore", "china stimulus"),
    "Autos": ("auto", "vehicle", "ev", "battery", "lithium", "tesla"),
    "Banks": ("bank", "fed", "rates", "yield", "bond", "credit", "liquidity"),
    "Aviation": ("aviation", "airline", "jet fuel"),
    "Chemicals": ("chemical", "specialty chemical", "fertilizer", "fertiliser"),
    "FMCG": ("consumer", "palm oil", "food inflation", "staples"),
    "Real Estate": ("real estate", "mortgage", "housing", "property"),
    "Defense": ("defence", "defense", "war", "missile", "geopolitical"),
}

CUE_RULES = [
    ("crude_oil", ("crude", "brent", "opec", "oil prices", "oil rises", "oil falls"),
     ["Oil & Gas", "Aviation", "Chemicals", "FMCG"]),
    ("us_rates", ("fed", "federal reserve", "us yields", "treasury yields", "rate cut", "rate hike"),
     ["Banks", "IT", "Real Estate"]),
    ("us_tech", ("nasdaq", "nvidia", "microsoft", "apple", "semiconductor", "chip", "ai rally"),
     ["IT", "Autos"]),
    ("china_growth", ("china", "beijing stimulus", "property crisis", "pmi", "industrial output"),
     ["Metals", "Chemicals", "Autos"]),
    ("currency", ("dollar", "dxy", "rupee", "usd/inr", "em currencies"),
     ["IT", "Oil & Gas", "Pharma"]),
    ("geopolitics", ("war", "sanction", "tariff", "red sea", "middle east", "border tension"),
     ["Defense", "Oil & Gas", "Aviation"]),
    ("global_risk", ("s&p 500", "dow", "wall street", "global stocks", "risk-off", "selloff", "vix"),
     ["Banks", "IT", "Metals"]),
]

POSITIVE_WORDS = (
    "rally", "surge", "gain", "rise", "jumps", "stimulus", "rate cut", "eases",
    "beats", "record high", "optimism", "boost",
)
NEGATIVE_WORDS = (
    "fall", "drops", "slump", "selloff", "war", "sanction", "tariff", "rate hike",
    "inflation", "recession", "miss", "cuts outlook", "risk-off", "crisis",
)


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _sector_tickers(sectors: list[str], limit: int = 80) -> list[str]:
    """Map impacted sector names to stored fundamentals tickers."""
    try:
        fund = load_fundamentals()
    except Exception:
        return []
    if fund is None or fund.empty or "sector" not in fund.columns:
        return []
    wanted = [s.lower() for s in sectors]
    out: list[str] = []
    for _, row in fund.iterrows():
        sector = str(row.get("sector") or "").lower()
        industry = str(row.get("industry") or "").lower()
        if any(w in sector or w in industry or sector in w for w in wanted):
            t = row.get("ticker")
            if isinstance(t, str) and t:
                out.append(t)
    return sorted(dict.fromkeys(out))[:limit]


def map_headline(source: str, title: str, link: str, *,
                 published: str | None = None, summary: str | None = None,
                 ticker_resolver=_sector_tickers) -> GlobalImpact | None:
    """Pure-ish headline mapper. Returns None when the headline has no obvious
    global market cue for Indian equities."""
    text = _norm(f"{title} {summary or ''}")
    matched_cues: list[str] = []
    sectors: list[str] = []
    for cue, keys, cue_sectors in CUE_RULES:
        if any(k in text for k in keys):
            matched_cues.append(cue)
            sectors.extend(cue_sectors)

    # Sector-only fallback catches "global pharma"/"copper" style headlines.
    for sec, keys in SECTOR_ALIASES.items():
        if any(k in text for k in keys):
            sectors.append(sec)

    sectors = sorted(dict.fromkeys(sectors))
    if not matched_cues and not sectors:
        return None

    pos = sum(1 for w in POSITIVE_WORDS if w in text)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text)
    direction = "mixed"
    if pos > neg:
        direction = "positive"
    elif neg > pos:
        direction = "negative"
    elif pos == neg == 0:
        direction = "neutral"

    cue = "+".join(matched_cues) if matched_cues else "sector_global"
    impact = min(100.0, 35.0 + 12.0 * len(matched_cues) + 6.0 * len(sectors) + 5.0 * max(pos, neg))
    tickers = ticker_resolver(sectors) if sectors else []
    rationale = f"Matched {cue}; India sectors: {', '.join(sectors) or 'broad market'}."
    return GlobalImpact(
        source=source, title=title, link=link, published=published,
        cue=cue, direction=direction, impact_score=round(impact, 1),
        sectors=sectors, tickers=tickers, rationale=rationale, summary=summary,
    )


def recent_impacts(days: int = 3, limit: int = 50) -> pd.DataFrame:
    return load_global_news(limit=limit, days=days)


def ticker_impact_score(ticker: str, days: int = 3) -> tuple[float, list[str]]:
    """Signed recent global-news pressure for a ticker.

    Positive headline directions add score; negative directions subtract. Mixed
    and neutral count lightly as awareness but not directional conviction.
    """
    df = load_global_news(limit=100, days=days, ticker=ticker)
    if df.empty:
        return 0.0, []
    score = 0.0
    reasons: list[str] = []
    for _, r in df.iterrows():
        imp = float(r.get("impact_score") or 0.0)
        direction = r.get("direction")
        mult = 1.0 if direction == "positive" else -1.0 if direction == "negative" else 0.25
        score += mult * imp / 10.0
        if len(reasons) < 3:
            reasons.append(f"{r.get('cue')}: {str(r.get('title') or '')[:90]}")
    return round(score, 1), reasons
