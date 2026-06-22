"""Tests for the HTML news scrapers (LiveSquawk, MarketsMojo). Uses canned HTML
so they're offline and stable against site changes."""
from __future__ import annotations

from swingdesk.ingest import news_scrape as ns

WL = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

LIVESQUAWK_HTML = """
<html><body>
  <div class="latest__news__each">
    <div class="latest_news_each_text">Reliance Industries Q4 Profit Rises 12% 1 hour ago</div>
  </div>
  <div class="latest__news__each">
    <div class="latest_news_each_text">Brent Crude Futures Settle At $78.96/Bbl, Down 5% 2 hours ago</div>
  </div>
  <div class="latest__news__each">
    <div class="latest_news_each_text">European Central Bank Holds Rates Steady Amid Inflation Concerns Show Detail READ HERE
      <a href="https://external.example/article">link</a></div>
  </div>
  <div class="latest_news_each_text">tiny</div>
</body></html>
"""

MARKETSMOJO_HTML = """
<html><body>
  <a href="/news/stock-market-news">Stock Market News</a>  <!-- nav, too short -->
  <a href="/news/stock-market-news/tcs-wins-deal">TCS Wins Large Deal Worth $500 Million From European Bank</a>
  <a href="/news/stock-market-news/tcs-wins-deal">TCS Wins Large Deal Worth $500 Million From European Bank — summary para duplicate</a>
  <a href="https://www.marketsmojo.com/news/stock-market-news/sensex-rallies">Sensex And Nifty Advance As Realty Leads The Gains Today</a>
  <a href="/other/section/foo">Unrelated long link that should be ignored entirely here</a>
</body></html>
"""


def test_livesquawk_parses_and_cleans():
    items = ns.scrape_livesquawk(WL, html=LIVESQUAWK_HTML)
    titles = [i["title"] for i in items]
    # 'ago' suffix stripped; 'tiny' (<20 chars) dropped.
    assert any(t == "Reliance Industries Q4 Profit Rises 12%" for t in titles)
    assert all("ago" not in t.lower()[-6:] for t in titles)
    assert all(len(t) >= 20 for t in titles)
    # Ticker match worked on the cleaned headline.
    rel = next(i for i in items if i["title"].startswith("Reliance"))
    assert "RELIANCE.NS" in rel["tickers"]
    # Headline-only items get a stable synthetic link; external link preserved.
    assert any(i["link"].startswith("https://www.livesquawk.com/latest-news#") for i in items)
    assert any(i["link"] == "https://external.example/article" for i in items)


def test_livesquawk_links_unique_and_stable():
    a = ns.scrape_livesquawk(WL, html=LIVESQUAWK_HTML)
    b = ns.scrape_livesquawk(WL, html=LIVESQUAWK_HTML)
    links = [i["link"] for i in a]
    assert len(links) == len(set(links))          # unique within a fetch
    assert [i["link"] for i in a] == [i["link"] for i in b]  # stable across fetches


def test_marketsmojo_parses_and_dedupes():
    items = ns.scrape_marketsmojo(WL, html=MARKETSMOJO_HTML)
    # Duplicate href collapses to one; nav/short + unrelated links excluded.
    links = [i["link"] for i in items]
    assert len(links) == len(set(links))
    assert any("tcs-wins-deal" in l for l in links)
    assert all("/news/stock-market-news/" in l for l in links)
    assert all(l.startswith("https://www.marketsmojo.com") for l in links)
    tcs = next(i for i in items if "tcs-wins-deal" in i["link"])
    assert "TCS.NS" in tcs["tickers"]


def test_empty_html_no_crash():
    assert ns.scrape_livesquawk(WL, html="<html></html>") == []
    assert ns.scrape_marketsmojo(WL, html="<html></html>") == []
