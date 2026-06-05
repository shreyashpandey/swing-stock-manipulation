"""CLI smoke tests — invoke each subcommand programmatically."""
from __future__ import annotations

import pytest

from swingdesk import cli, storage


def test_cli_init(tmp_db, capsys):
    cli.main(["init"])
    out = capsys.readouterr().out
    assert "initialized" in out.lower() or "watchlist" in out.lower()
    assert len(storage.get_watchlist()) > 0


def test_cli_watchlist_set_then_view(tmp_db, capsys):
    cli.main(["watchlist", "--set", "A.NS,B.NS,C.NS"])
    cli.main(["watchlist"])
    out = capsys.readouterr().out
    assert "A.NS" in out and "B.NS" in out and "C.NS" in out


def test_cli_signals_empty(tmp_db, capsys):
    cli.main(["signals"])
    out = capsys.readouterr().out
    assert "no signals" in out.lower()


def test_cli_scan_no_prices_does_not_crash(tmp_db, capsys):
    cli.main(["watchlist", "--set", "FAKE.NS"])
    capsys.readouterr()  # drain
    cli.main(["scan"])
    out = capsys.readouterr().out
    assert "0 signals" in out


def test_cli_help_exits_zero():
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
