from __future__ import annotations

import re
from datetime import datetime
from email.utils import parsedate_to_datetime

import feedparser
from rich.console import Console

from swingdesk.storage import combined_universe, insert_news, load_fundamentals

console = Console()

# Indian-equity-focused RSS feeds. Stable, no auth.
FEEDS: list[tuple[str, str]] = [
    # ---- Indian markets (per-stock impact) ----
    ("Moneycontrol-Markets",  "https://www.moneycontrol.com/rss/marketreports.xml"),
    ("Moneycontrol-Business", "https://www.moneycontrol.com/rss/business.xml"),
    ("Moneycontrol-LatestNews", "https://www.moneycontrol.com/rss/latestnews.xml"),
    ("ET-Markets",            "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("ET-Stocks",             "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"),
    ("Livemint-Markets",      "https://www.livemint.com/rss/markets"),
    ("BusinessStandard-Markets", "https://www.business-standard.com/rss/markets-106.rss"),
    ("BusinessStandard-Companies", "https://www.business-standard.com/rss/companies-101.rss"),
    # ---- Small / mid-cap focused (where most of our watchlist lives) ----
    # These carry far more headlines naming smaller names, IPOs and results
    # than the broad-market feeds above.
    ("ET-MidcapStocks",       "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"),
    ("Moneycontrol-Buzzing",  "https://www.moneycontrol.com/rss/buzzingstocks.xml"),
    ("Moneycontrol-Results",  "https://www.moneycontrol.com/rss/results.xml"),
    ("Moneycontrol-MarketEdge", "https://www.moneycontrol.com/rss/marketedge.xml"),
    ("ET-IPO",                "https://economictimes.indiatimes.com/markets/ipos/fpos/rssfeeds/14655708.cms"),
    ("Moneycontrol-IPO",      "https://www.moneycontrol.com/rss/iponews.xml"),
    # ---- Added on request (verified live): NDTV Profit, Hindu BusinessLine,
    # CNBC-TV18, extra Livemint. (LiveSquawk / MarketsMojo / ScoutQuest / BSE
    # don't publish public RSS; NSE/BSE bulk trades are ingested via nse.py.)
    ("NDTVProfit",            "https://feeds.feedburner.com/ndtvprofit-latest"),
    ("HinduBL-Markets",       "https://www.thehindubusinessline.com/markets/feeder/default.rss"),
    ("HinduBL-Companies",     "https://www.thehindubusinessline.com/companies/feeder/default.rss"),
    ("CNBCTV18-Stocks",       "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/market/stocks.xml"),
    ("Livemint-Companies",    "https://www.livemint.com/rss/companies"),
    # ---- Global macro that drives Indian markets ----
    # US monetary policy & macro: directly drives USD/INR + Indian IT + FII flows
    ("CNBC-WorldNews",        "https://www.cnbc.com/id/100727362/device/rss/rss.html"),
    ("CNBC-USMarkets",        "https://www.cnbc.com/id/15839069/device/rss/rss.html"),
    ("Reuters-Business",      "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best"),
    # India-focused macro & global commentary (mostly Indian audience but covers
    # global moves with India lens)
    ("ET-International",      "https://economictimes.indiatimes.com/news/international/rssfeeds/3534138.cms"),
    ("ET-Economy",            "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms"),
    ("Moneycontrol-Economy",  "https://www.moneycontrol.com/rss/economy.xml"),
    ("Moneycontrol-World",    "https://www.moneycontrol.com/rss/world-news.xml"),
]


def _normalize_date(entry) -> str | None:
    for key in ("published", "updated", "pubDate"):
        val = entry.get(key)
        if not val:
            continue
        try:
            return parsedate_to_datetime(val).isoformat()
        except Exception:
            try:
                return datetime.fromisoformat(val).isoformat()
            except Exception:
                pass
    return None


# Curated name->ticker aliases for names that don't appear as the bare symbol in
# prose. build_alias_map() supplements these at runtime with aliases derived from
# fundamentals.short_name (universe-wide, no manual upkeep); the curated entries
# here stay as hand-tuned overrides for tricky names.
_CURATED_ALIASES: dict[str, list[str]] = {
        "RELIANCE.NS": ["RELIANCE", "RIL"],
        "TCS.NS": ["TCS", "TATA CONSULTANCY"],
        "INFY.NS": ["INFOSYS"],
        "HDFCBANK.NS": ["HDFC BANK"],
        "ICICIBANK.NS": ["ICICI BANK"],
        "SBIN.NS": ["SBI", "STATE BANK"],
        "AXISBANK.NS": ["AXIS BANK"],
        "KOTAKBANK.NS": ["KOTAK"],
        "INDUSINDBK.NS": ["INDUSIND BANK", "INDUSIND"],
        "LT.NS": ["LARSEN", "L&T", "L AND T"],
        "HINDUNILVR.NS": ["HUL", "HINDUSTAN UNILEVER"],
        "BHARTIARTL.NS": ["BHARTI AIRTEL", "AIRTEL", "BHARTI"],
        "MARUTI.NS": ["MARUTI", "MARUTI SUZUKI"],
        "TATAMOTORS.NS": ["TATA MOTORS"],
        "TATASTEEL.NS": ["TATA STEEL"],
        "JSWSTEEL.NS": ["JSW STEEL"],
        "SUNPHARMA.NS": ["SUN PHARMA"],
        "DRREDDY.NS": ["DR REDDY", "DR. REDDY", "DR REDDYS"],
        "DIVISLAB.NS": ["DIVI", "DIVIS LAB", "DIVI'S LAB"],
        "CIPLA.NS": ["CIPLA"],
        "APOLLOHOSP.NS": ["APOLLO HOSPITAL"],
        "BAJFINANCE.NS": ["BAJAJ FINANCE"],
        "BAJAJFINSV.NS": ["BAJAJ FINSERV"],
        "BAJAJ-AUTO.NS": ["BAJAJ AUTO"],
        "ASIANPAINT.NS": ["ASIAN PAINTS"],
        "ADANIENT.NS": ["ADANI ENTERPRISES"],
        "ADANIPORTS.NS": ["ADANI PORTS"],
        "EICHERMOT.NS": ["EICHER", "ROYAL ENFIELD"],
        "HEROMOTOCO.NS": ["HERO MOTOCORP", "HERO MOTO"],
        "M&M.NS": ["MAHINDRA & MAHINDRA", "M&M", "MAHINDRA"],
        "NTPC.NS": ["NTPC"],
        "POWERGRID.NS": ["POWER GRID", "POWERGRID"],
        "ONGC.NS": ["ONGC"],
        "BPCL.NS": ["BPCL", "BHARAT PETROLEUM"],
        "IOC.NS": ["IOC", "INDIAN OIL"],
        "COALINDIA.NS": ["COAL INDIA"],
        "VEDL.NS": ["VEDANTA"],
        "HINDALCO.NS": ["HINDALCO"],
        "ITC.NS": ["ITC"],
        "TITAN.NS": ["TITAN"],
        "WIPRO.NS": ["WIPRO"],
        "HCLTECH.NS": ["HCL TECH", "HCL TECHNOLOGIES"],
        "TECHM.NS": ["TECH MAHINDRA"],
        "BRITANNIA.NS": ["BRITANNIA"],
        "NESTLEIND.NS": ["NESTLE"],
        "ULTRACEMCO.NS": ["ULTRATECH"],
        "GRASIM.NS": ["GRASIM"],
        "HDFCLIFE.NS": ["HDFC LIFE"],
        "SBILIFE.NS": ["SBI LIFE"],
        "DLF.NS": ["DLF"],
        # ---- Small-cap aliases (Phase: Small Caps) ----
        # Defense
        "PARAS.NS": ["PARAS DEFENCE", "PARAS DEFENSE"],
        "MTARTECH.NS": ["MTAR TECHNOLOGIES", "MTAR TECH"],
        "ASTRAMICRO.NS": ["ASTRA MICROWAVE"],
        "DATAPATTNS.NS": ["DATA PATTERNS"],
        "ZENTEC.NS": ["ZEN TECHNOLOGIES", "ZEN TECH"],
        # IT services
        "MASTEK.NS": ["MASTEK"],
        "TANLA.NS": ["TANLA PLATFORMS", "TANLA"],
        "NEWGEN.NS": ["NEWGEN SOFTWARE"],
        "CYIENT.NS": ["CYIENT"],
        "HAPPSTMNDS.NS": ["HAPPIEST MINDS"],
        "INTELLECT.NS": ["INTELLECT DESIGN"],
        "ZENSARTECH.NS": ["ZENSAR"],
        "SAKSOFT.NS": ["SAKSOFT"],
        "RSYSTEMS.NS": ["R SYSTEMS"],
        "BIRLASOFT.NS": ["BIRLASOFT", "BIRLA SOFT"],
        "SONATSOFTW.NS": ["SONATA SOFTWARE", "SONATA"],
        # Banking / Financial
        "EQUITASBNK.NS": ["EQUITAS SMALL FINANCE", "EQUITAS"],
        "UJJIVANSFB.NS": ["UJJIVAN SMALL FINANCE", "UJJIVAN"],
        "CSBBANK.NS": ["CSB BANK"],
        "KARURVYSYA.NS": ["KARUR VYSYA"],
        "DCBBANK.NS": ["DCB BANK"],
        "ANANDRATHI.NS": ["ANAND RATHI"],
        "360ONE.NS": ["360 ONE"],
        "IIFLSEC.NS": ["IIFL SECURITIES"],
        "MOTILALOFS.NS": ["MOTILAL OSWAL"],
        "ANGELONE.NS": ["ANGEL ONE", "ANGEL BROKING"],
        "CAMS.NS": ["CAMS", "COMPUTER AGE MANAGEMENT"],
        # Pharma
        "CAPLIPOINT.NS": ["CAPLIN POINT"],
        "SUVENPHAR.NS": ["SUVEN PHARMA", "SUVEN"],
        "LAURUSLABS.NS": ["LAURUS LABS"],
        "IPCALAB.NS": ["IPCA LABORATORIES", "IPCA LAB"],
        "AJANTPHARM.NS": ["AJANTA PHARMA"],
        "JBCHEPHARM.NS": ["JB CHEMICALS"],
        "ERIS.NS": ["ERIS LIFESCIENCES", "ERIS LIFE"],
        "GLENMARK.NS": ["GLENMARK"],
        "BIOCON.NS": ["BIOCON"],
        # Capital goods
        "TEGA.NS": ["TEGA INDUSTRIES", "TEGA"],
        "KSB.NS": ["KSB PUMPS", "KSB LIMITED"],
        "ELECON.NS": ["ELECON ENGINEERING"],
        "ACE.NS": ["ACTION CONSTRUCTION"],
        "KIRLOSENG.NS": ["KIRLOSKAR OIL", "KIRLOSKAR ENGINE"],
        "HONAUT.NS": ["HONEYWELL AUTOMATION"],
        "AIAENG.NS": ["AIA ENGINEERING"],
        # Auto ancillary
        "BALKRISIND.NS": ["BALKRISHNA INDUSTRIES", "BKT"],
        "SUPRAJIT.NS": ["SUPRAJIT ENGINEERING"],
        "ENDURANCE.NS": ["ENDURANCE TECHNOLOGIES"],
        # Real estate
        "SOBHA.NS": ["SOBHA"],
        "BRIGADE.NS": ["BRIGADE ENTERPRISES"],
        "PURVA.NS": ["PURAVANKARA"],
        "MAHLIFE.NS": ["MAHINDRA LIFESPACE"],
        "SUNTECK.NS": ["SUNTECK REALTY"],
        # Consumer
        "INDIGOPNTS.NS": ["INDIGO PAINTS"],
        "SYMPHONY.NS": ["SYMPHONY"],
        "KAJARIACER.NS": ["KAJARIA CERAMICS"],
        "CERA.NS": ["CERA SANITARYWARE"],
        # Chemicals
        "VINATIORGA.NS": ["VINATI ORGANICS"],
        "FINEORG.NS": ["FINE ORGANIC"],
        "ALKYLAMINE.NS": ["ALKYL AMINES"],
        # Renewables
        "INOXWIND.NS": ["INOX WIND"],
        "BORORENEW.NS": ["BOROSIL RENEWABLES"],
        # Misc
        "RAINBOW.NS": ["RAINBOW CHILDREN", "RAINBOW HOSPITALS"],
        "KIMS.NS": ["KRISHNA INSTITUTE", "KIMS HEALTHCARE"],
        "METROPOLIS.NS": ["METROPOLIS HEALTHCARE"],
        "AVANTIFEED.NS": ["AVANTI FEEDS"],
        "POONAWALLA.NS": ["POONAWALLA FINCORP"],
        "MANAPPURAM.NS": ["MANAPPURAM FINANCE"],
}

# Corporate-form tokens trimmed from a company name when deriving an alias.
_NAME_SUFFIXES = {
    "LIMITED", "LTD", "PVT", "PRIVATE", "CORPORATION", "CORP", "CO",
    "COMPANY", "INC", "INDIA",
}


def _normalize_company_name(name: str | None) -> str | None:
    """Turn a fundamentals short_name into a clean prose alias.

    'Bharat Forge Ltd' -> 'BHARAT FORGE'; 'VA Tech Wabag Limited' -> 'VA TECH WABAG';
    'NOCIL Limited' -> 'NOCIL'. Returns None when nothing usable remains (too short).
    """
    if not name:
        return None
    cleaned = re.sub(r"\([^)]*\)", " ", name)            # drop "(India)" etc.
    cleaned = re.sub(r"[^A-Za-z0-9& ]+", " ", cleaned)    # punctuation -> space
    tokens = cleaned.upper().split()
    while tokens and tokens[-1] in _NAME_SUFFIXES:        # strip trailing Ltd / Limited / ...
        tokens.pop()
    alias = " ".join(tokens).strip()
    # Guard: too-short aliases (<4 chars) false-match in prose; let the bare
    # symbol token / curated map handle those.
    return alias if len(alias) >= 4 else None


def build_alias_map(universe: list[str] | set[str]) -> dict[str, list[str]]:
    """Derive a name->ticker alias map from stored fundamentals.short_name,
    scoped to `universe`.

    This is the fix for the news blind spot: every stock we have fundamentals for
    resolves by its company name automatically, so a headline like 'Bharat Forge
    wins defence order' or 'VA Tech Wabag bags contract' tags the right ticker
    with NO hand-maintained alias. Returns {} if fundamentals aren't ingested yet.
    """
    uni = set(universe)
    derived: dict[str, list[str]] = {}
    try:
        fund = load_fundamentals()
    except Exception:
        return derived
    if fund is None or fund.empty or "ticker" not in fund.columns \
            or "short_name" not in fund.columns:
        return derived
    for tkr, short_name in zip(fund["ticker"], fund["short_name"]):
        if tkr not in uni:
            continue
        alias = _normalize_company_name(None if short_name is None else str(short_name))
        if alias:
            derived[tkr] = [alias]
    return derived


def _match_tickers(text: str, universe: list[str] | set[str],
                   alias_map: dict[str, list[str]] | None = None) -> list[str]:
    """Match company / symbol mentions in a headline to tickers in `universe`.

    Three word-boundary-matched passes (a ticker only needs one to hit):
      1. the bare symbol token (e.g. 'NOCIL');
      2. curated aliases (_CURATED_ALIASES) for hand-tuned names;
      3. derived aliases (`alias_map`, from build_alias_map) — the universe-wide
         auto map built from fundamentals.
    Only tickers present in `universe` can be returned, so passing a broad
    `alias_map` is safe.
    """
    if not text:
        return []
    uni = set(universe)
    upper = text.upper()
    hits: set[str] = set()

    # 1. Bare symbol token (word-boundary so 'ITC' won't match inside 'WITCH').
    for t in uni:
        base = t.split(".")[0]
        if re.search(rf"\b{re.escape(base)}\b", upper):
            hits.add(t)

    # 2 + 3. Curated then derived aliases, gated to the universe.
    for source in (_CURATED_ALIASES, alias_map or {}):
        for tkr, names in source.items():
            if tkr in hits or tkr not in uni:
                continue
            for nm in names:
                if re.search(rf"\b{re.escape(nm)}\b", upper):
                    hits.add(tkr)
                    break
    return sorted(hits)


def fetch_feed(name: str, url: str, universe: list[str] | set[str],
               alias_map: dict[str, list[str]] | None = None) -> list[dict]:
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries:
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        if not title or not link:
            continue
        summary = (e.get("summary") or "").strip()
        published = _normalize_date(e)
        tickers = _match_tickers(f"{title} {summary}", universe, alias_map)
        items.append({
            "source": name,
            "title": title,
            "link": link,
            "published": published,
            "summary": summary[:500],
            "tickers": tickers,
        })
    return items


def _tagging_context(universe: list[str] | None) -> tuple[set[str], dict[str, list[str]]]:
    """Build the (universe, alias_map) used for tagging. Tagging ALWAYS runs
    against the full investable universe (watchlist + holdings + discovery +
    small-cap pools) plus the fundamentals-derived alias map — regardless of what
    a caller passes — so a headline about a stock you don't own still gets tagged.
    """
    tag_universe = set(combined_universe(include_smallcaps=True, include_discovery=True,
                                         include_extended=True))
    if universe:
        tag_universe |= set(universe)
    return tag_universe, build_alias_map(tag_universe)


def ingest(universe: list[str] | None = None) -> int:
    tag_universe, alias_map = _tagging_context(universe)
    total = 0
    for name, url in FEEDS:
        try:
            items = fetch_feed(name, url, tag_universe, alias_map)
            n = insert_news(items)
            total += n
            console.print(f"  news: {name:>30} -> {len(items):>3} items ({n} new)")
        except Exception as e:
            console.print(f"[red]news fetch failed for {name}: {e}[/red]")
    # Non-RSS sources (LiveSquawk, MarketsMojo) via HTML scrapers. Lazy import
    # avoids a circular dependency (news_scrape imports _match_tickers from here).
    try:
        from swingdesk.ingest import news_scrape
        total += news_scrape.ingest(tag_universe, alias_map)
    except Exception as e:
        console.print(f"[red]news scrape stage failed: {e}[/red]")
    return total


def retag_news(universe: list[str] | None = None) -> int:
    """Re-run ticker tagging over ALL stored news and update their `tickers`
    column. Run this after expanding the universe / alias logic (or after a fresh
    fundamentals fetch) to surface past headlines that were ingested before a
    stock was covered. Returns the number of rows whose tags changed.
    """
    from swingdesk.storage import load_all_news_for_retag, update_news_tickers
    tag_universe, alias_map = _tagging_context(universe)
    rows = load_all_news_for_retag()
    if rows.empty:
        return 0
    updates = []
    for _, r in rows.iterrows():
        text = f"{r['title']} {r['summary'] or ''}"
        new_tags = ",".join(_match_tickers(text, tag_universe, alias_map))
        if new_tags != (r["tickers"] or ""):
            updates.append({"id": int(r["id"]), "tickers": new_tags})
    return update_news_tickers(updates)
