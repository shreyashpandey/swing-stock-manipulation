"""Import smoke tests.

These exist to catch packaging/import regressions (e.g. the relative-import
issue with `streamlit run swingdesk/app.py`) before runtime.

If you ever see `ImportError: attempted relative import with no known parent
package`, one of these tests will fail.
"""
from __future__ import annotations

import importlib
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

MODULES = [
    "swingdesk",
    "swingdesk.config",
    "swingdesk.storage",
    "swingdesk.cli",
    "swingdesk.ingest",
    "swingdesk.ingest.prices",
    "swingdesk.ingest.news_rss",
    "swingdesk.analyze",
    "swingdesk.analyze.technicals",
    "swingdesk.analyze.setups",
    "swingdesk.analyze.sentiment",
    "swingdesk.analyze.score",
    "swingdesk.notify",
    "swingdesk.notify.telegram",
    "swingdesk.scheduler",
    "swingdesk.backtest",
    "swingdesk.backtest.engine",
    "swingdesk.backtest.metrics",
    "swingdesk.portfolio",
    "swingdesk.portfolio.positions",
    "swingdesk.portfolio.journal",
    "swingdesk.portfolio.import_groww",
    "swingdesk.portfolio.reconcile",
    "swingdesk.backtest.optimizer",
    "swingdesk.ingest.earnings",
    "swingdesk.ingest.fundamentals",
    "swingdesk.analyze.quality",
    "swingdesk.portfolio.holdings",
    "swingdesk.analyze.exits",
    "swingdesk.analyze.thesis",
    "swingdesk.ingest.macro",
    "swingdesk.analyze.discovery",
    "swingdesk.analyze.chart_signals",
    "swingdesk.analyze.summary",
    "swingdesk.analyze.early_exits",
    "swingdesk.analyze.smallcaps",
]


@pytest.mark.parametrize("mod", MODULES)
def test_module_imports(mod):
    importlib.import_module(mod)


def test_cli_runs_as_script(tmp_path, monkeypatch):
    """`python swingdesk/cli.py --help` must work (no relative-import error)."""
    # Run in a subprocess so the script's sys.path bootstrap is exercised.
    result = subprocess.run(
        [sys.executable, str(ROOT / "swingdesk" / "cli.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=tmp_path,  # run from elsewhere to prove path bootstrap works
        timeout=30,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "swingdesk" in result.stdout.lower()


def test_cli_runs_as_module(tmp_path):
    """`python -m swingdesk.cli --help` must also work."""
    result = subprocess.run(
        [sys.executable, "-m", "swingdesk.cli", "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=30,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"


def test_app_module_imports_without_streamlit_runtime():
    """The Streamlit app module must import cleanly (relative-import bug guard).

    Streamlit will warn about 'missing ScriptRunContext' which is fine — we only
    care that the import itself succeeds.
    """
    # Force a fresh import in case something else loaded it earlier.
    for name in list(sys.modules):
        if name.startswith("swingdesk.app"):
            del sys.modules[name]
    importlib.import_module("swingdesk.app")
