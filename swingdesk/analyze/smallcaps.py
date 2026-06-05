"""Dedicated small-cap scanner.

Separate from the large/mid-cap discovery universe because small caps need:
  - Tighter liquidity filters (skip ultra-thin names)
  - More aggressive trend confirmation (false signals are more common)
  - Higher quality bar (more fundamentally questionable businesses)

The small-cap universe below is ~95 names covering NSE smallcap-100-style
listings with reasonable trading volume — defense, IT services, banks/NBFCs,
pharma, capital goods, real estate, consumer, renewables, chemicals.

This module shares the Opportunity dataclass and scoring logic from
discovery.py — only the universe and a couple of small-cap-specific filters
differ. Output: ranked list of high-conviction small-cap swing candidates.
"""
from __future__ import annotations

from swingdesk.analyze.discovery import Opportunity, _rank_one
from swingdesk.storage import get_watchlist, holdings_tickers

# ~95 curated NSE small-cap names. Market cap roughly ₹500-15,000 cr.
SMALLCAP_UNIVERSE: list[str] = [
    # Defense / aerospace smallcaps — hottest theme last 18 months
    "PARAS.NS", "MTARTECH.NS", "ASTRAMICRO.NS", "DATAPATTNS.NS", "ZENTEC.NS",

    # IT services smallcaps
    "MASTEK.NS", "TANLA.NS", "NEWGEN.NS", "CYIENT.NS", "HAPPSTMNDS.NS",
    "INTELLECT.NS", "ZENSARTECH.NS", "SAKSOFT.NS", "RSYSTEMS.NS", "BIRLASOFT.NS",
    "SONATSOFTW.NS", "SUBEXLTD.NS",

    # Banking / financial services smallcaps
    "EQUITASBNK.NS", "UJJIVANSFB.NS", "CSBBANK.NS", "KARURVYSYA.NS", "DCBBANK.NS",
    "ANANDRATHI.NS", "360ONE.NS", "IIFLSEC.NS", "MOTILALOFS.NS", "ANGELONE.NS",
    "CAMS.NS", "KFINTECH.NS", "FIVESTAR.NS", "SBFC.NS",

    # Pharma / healthcare smallcaps
    "CAPLIPOINT.NS", "SUVENPHAR.NS", "LAURUSLABS.NS", "IPCALAB.NS", "AJANTPHARM.NS",
    "JBCHEPHARM.NS", "ERIS.NS", "GLAND.NS", "GLENMARK.NS", "BIOCON.NS",
    "RAINBOW.NS", "KIMS.NS", "METROPOLIS.NS",

    # Auto / auto-ancillary smallcaps
    "BALKRISIND.NS", "SUPRAJIT.NS", "ENDURANCE.NS", "JAMNAAUTO.NS", "MINDACORP.NS",
    "EXIDEIND.NS", "SCHAEFFLER.NS", "TIMKEN.NS",

    # Capital goods / industrials smallcaps
    "TEGA.NS", "KSB.NS", "ELECON.NS", "ACE.NS", "KIRLOSENG.NS",
    "HONAUT.NS", "GRINDWELL.NS", "AIAENG.NS", "CARBORUNIV.NS", "TIINDIA.NS",
    "VOLTAS.NS", "BLUESTARCO.NS",

    # Real estate smallcaps
    "SOBHA.NS", "BRIGADE.NS", "PURVA.NS", "MAHLIFE.NS", "SUNTECK.NS",
    "KOLTEPATIL.NS",

    # Consumer / consumer-durables smallcaps
    "INDIGOPNTS.NS", "SYMPHONY.NS", "IFBIND.NS", "KAJARIACER.NS", "CERA.NS",
    "VGUARD.NS", "WHIRLPOOL.NS", "CROMPTON.NS",

    # Specialty chemicals smallcaps
    "VINATIORGA.NS", "GUJALKALI.NS", "DCMSHRIRAM.NS", "CHEMPLASTS.NS", "HEG.NS",
    "GRAPHITE.NS", "ALKYLAMINE.NS", "FINEORG.NS",

    # Renewables / EV-thematic smallcaps
    "INOXWIND.NS", "BORORENEW.NS", "TARC.NS",

    # Logistics smallcaps
    "BLUEDART.NS", "TCIEXP.NS", "MAHLOG.NS", "ALLCARGO.NS",

    # Misc growth smallcaps
    "KRBL.NS", "AVANTIFEED.NS", "DEEPAKFERT.NS", "GNFC.NS",
    "AMARAJABAT.NS", "EIDPARRY.NS", "JKLAKSHMI.NS", "PRINCEPIPE.NS",
    "ASTERDM.NS", "POONAWALLA.NS", "MANAPPURAM.NS",
]


def scan(*, exclude_held: bool = True, exclude_watchlist: bool = True,
         min_volume_x: float = 0.5) -> list[Opportunity]:
    """Rank small-cap candidates. Same scoring as the main discovery scanner
    but uses the small-cap universe and applies a liquidity filter (any ticker
    with today's volume < 0.5× its 20-day average is too thin to swing-trade).
    """
    held = set(holdings_tickers()) if exclude_held else set()
    wl = set(get_watchlist()) if exclude_watchlist else set()
    skip = held | wl

    results: list[Opportunity] = []
    for tk in SMALLCAP_UNIVERSE:
        if tk in skip:
            continue
        opp = _rank_one(tk)
        if not opp:
            continue
        # Smallcap liquidity guard — drop too-thin names
        if opp.volume_x_avg is not None and opp.volume_x_avg < min_volume_x:
            continue
        results.append(opp)

    results.sort(key=lambda o: o.composite_score, reverse=True)
    return results


def high_conviction(opps: list[Opportunity] | None = None,
                    min_score: float = 70.0) -> list[Opportunity]:
    """Same alignment filter as the large/mid-cap scanner but applied to
    small caps. Quality + uptrend + volume-confirmed + positive momentum +
    healthy RSI. Use this when you want the "invest without much thought"
    subset from the small-cap pool."""
    pool = opps if opps is not None else scan()
    return [o for o in pool
            if o.conviction == "high" and o.composite_score >= min_score]


def smallcap_universe() -> list[str]:
    """Public accessor for the curated small-cap list."""
    return list(SMALLCAP_UNIVERSE)
