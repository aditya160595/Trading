"""Inventory has to be the broker's number, not the loop's guess.

Two bugs are pinned here:

1. KILL set `inv.inventory = 0.0` and sent nothing. The loop believed it
   was flat while the venue still held the position.
2. Nothing ever reconciled local bookkeeping against Alpaca. Only the
   directional leg moved `inventory`, so a resting quote that filled was
   invisible, and a restart began by assuming flat -- meaning every
   dollar limit was policing a number the loop had made up.
"""

import json

import pytest

from jevloop.assets import resolve_symbol
from jevloop.execution.alpaca import (
    AlpacaAPIError,
    AlpacaPaperClient,
    MarketClosedError,
)
from jevloop.limits import Limits
from jevloop.state import InventoryState, reconcile_position

from fakes import AlwaysBuyClient, FakeAlpaca

BTC = resolve_symbol("BTC/USD")


# -- the endpoints ------------------------------------------------------


def test_position_url_percent_encodes_a_crypto_pair():
    """BTC/USD in a URL path has to become BTC%2FUSD, or the request hits
    the wrong route entirely."""
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=BTC)
    assert client._position_url().endswith("/v2/positions/BTC%2FUSD")


def test_equity_position_url_needs_no_encoding():
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=resolve_symbol("AAPL"))
    assert client._position_url().endswith("/v2/positions/AAPL")


def test_get_position_treats_404_as_flat_not_as_an_error():
    """Alpaca 404s a symbol you hold nothing in. That is 'flat', and it
    must not propagate as an error and kill the tick."""
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=BTC)

    def _404(method, url, **kwargs):
        raise AlpacaAPIError(404, "position does not exist")

    client._request = _404
    assert client.get_position() is None


def test_get_position_still_raises_on_a_real_error():
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=BTC)

    def _500(method, url, **kwargs):
        raise AlpacaAPIError(500, "boom")

    client._request = _500
    with pytest.raises(AlpacaAPIError):
        client.get_position()


def test_close_position_treats_404_as_nothing_to_close():
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=BTC)

    def _404(method, url, **kwargs):
        raise AlpacaAPIError(404, "position does not exist")

    client._request = _404
    assert client.close_position() is None


def test_close_position_refused_on_a_closed_equity_market():
    client = AlpacaPaperClient(api_key="x", secret_key="y", spec=resolve_symbol("AAPL"))
    client.is_market_open = lambda: False
    with pytest.raises(MarketClosedError):
        client.close_position()


# -- reconciliation -----------------------------------------------------


def test_reconcile_adopts_the_brokers_number():
    inv = InventoryState(inventory=0.0)
    before, after = reconcile_position(
        inv, {"qty": "0.75", "avg_entry_price": "101.5"}, as_of=1000.0
    )
    assert (before, after) == (0.0, 0.75)
    assert inv.inventory == 0.75
    assert inv.entry_price == 101.5


def test_reconcile_adopts_a_short_position():
    inv = InventoryState(inventory=0.0)
    _, after = reconcile_position(inv, {"qty": "-2"}, as_of=1000.0)
    assert after == -2.0


def test_reconcile_reports_a_correction_so_it_is_never_silent():
    """The return value is what lets the loop print 'your number was
    wrong'. A silent correction would hide exactly the bug this fixes."""
    inv = InventoryState(inventory=5.0)
    before, after = reconcile_position(inv, None, as_of=1000.0)
    assert before == 5.0 and after == 0.0


def test_reconcile_to_flat_clears_the_position_clock():
    inv = InventoryState(inventory=1.0, position_opened_at=500.0)
    reconcile_position(inv, None, as_of=1000.0)
    assert inv.position_opened_at is None


def test_reconcile_starts_the_clock_on_a_newly_seen_position():
    inv = InventoryState(inventory=0.0, position_opened_at=None)
    reconcile_position(inv, {"qty": "1"}, as_of=1000.0)
    assert inv.position_opened_at == 1000.0


def test_reconcile_does_not_restart_the_clock_on_an_existing_position():
    """Otherwise max_inventory_age_s could never fire: every sync would
    reset the age to zero."""
    inv = InventoryState(inventory=1.0, position_opened_at=500.0)
    reconcile_position(inv, {"qty": "1.5"}, as_of=1000.0)
    assert inv.position_opened_at == 500.0


# -- the loop picks up a position it did not open -----------------------


def test_startup_adopts_a_pre_existing_position(tmp_path, monkeypatch, capsys):
    """Restarting into an open position must not begin from 'flat'. This
    is the case where believing your own bookkeeping doubles a position
    that is already at the limit."""
    monkeypatch.setenv("JEV_LOOP_HOME", str(tmp_path))
    monkeypatch.setenv("ALPACA_API_KEY", "PKfake")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

    from jevloop import loop as loopmod

    monkeypatch.setattr(loopmod, "LOG_DIR", tmp_path)
    monkeypatch.setattr(loopmod, "LOG_FILE", tmp_path / "log.jsonl")
    monkeypatch.setattr(loopmod, "LATEST_FILE", tmp_path / "latest.json")

    fake = FakeAlpaca(BTC)
    fake.position_qty = 0.4  # already holding when the loop starts
    monkeypatch.setattr(
        loopmod, "client_from_env", lambda symbol, live, confirmation: fake
    )
    monkeypatch.setattr(
        loopmod, "resolve_decision_client", lambda mock: AlwaysBuyClient()
    )

    limits = Limits()
    limits.tick_seconds = 0.02
    limits.max_position_usd = 10_000.0  # not the thing under test

    loopmod.run(symbol="BTC/USD", ticks=1, mock=True, limits=limits)

    lines = [
        json.loads(x)
        for x in (tmp_path / "log.jsonl").read_text().splitlines()
        if x.strip()
    ]
    # The very first logged tick already knows about the existing position.
    assert lines[0]["inventory"] >= 0.4
    assert "position sync" in capsys.readouterr().out


# -- KILL actually closes the position ----------------------------------


def test_kill_closes_the_position_at_the_venue(tmp_path, monkeypatch):
    """The headline fix: a kill must leave the broker flat, not just the
    loop's variable."""
    monkeypatch.setenv("JEV_LOOP_HOME", str(tmp_path))
    monkeypatch.setenv("ALPACA_API_KEY", "PKfake")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

    from jevloop import loop as loopmod

    monkeypatch.setattr(loopmod, "LOG_DIR", tmp_path)
    monkeypatch.setattr(loopmod, "LOG_FILE", tmp_path / "log.jsonl")
    monkeypatch.setattr(loopmod, "LATEST_FILE", tmp_path / "latest.json")

    fake = FakeAlpaca(BTC)
    monkeypatch.setattr(
        loopmod, "client_from_env", lambda symbol, live, confirmation: fake
    )
    monkeypatch.setattr(
        loopmod, "resolve_decision_client", lambda mock: AlwaysBuyClient()
    )

    limits = Limits()
    limits.tick_seconds = 0.02
    limits.max_position_usd = 0.01  # the first fill breaches

    loopmod.run(symbol="BTC/USD", ticks=20, mock=True, limits=limits)

    assert fake.closes >= 1, "KILL never asked the venue to close the position"
    assert fake.position_qty == 0.0, "venue still holds a position after KILL"
    assert fake.cancels >= 1, "KILL never cancelled resting orders"


def test_kill_that_cannot_flatten_does_not_claim_to_be_flat():
    """If the close fails, the loop must keep the real number and say so.
    Reporting flat on faith is how a live position gets forgotten."""
    from jevloop.loop import _flatten

    fake = FakeAlpaca(BTC)
    fake.position_qty = 1.25

    def _explode():
        raise AlpacaAPIError(500, "venue refused")

    fake.close_position = _explode

    inv = InventoryState(inventory=1.25)
    note = _flatten(fake, inv, as_of=1000.0, dry=False)

    assert "FLATTEN FAILED" in note
    assert inv.inventory == 1.25, "inventory was zeroed despite a failed close"


def test_kill_reports_when_the_venue_still_shows_a_position():
    """close_position() returning cleanly is not proof. Verify, then say
    what is actually there."""
    from jevloop.loop import _flatten

    fake = FakeAlpaca(BTC)
    fake.position_qty = 2.0
    fake.close_position = lambda: {"id": "accepted"}  # accepts, changes nothing

    inv = InventoryState(inventory=2.0)
    note = _flatten(fake, inv, as_of=1000.0, dry=False)

    assert "STILL HOLDING" in note
    assert inv.inventory == 2.0


def test_dry_execution_never_touches_the_venue_on_a_kill():
    from jevloop.loop import _flatten

    fake = FakeAlpaca(BTC)
    fake.position_qty = 1.0
    inv = InventoryState(inventory=1.0)

    note = _flatten(fake, inv, as_of=1000.0, dry=True)

    assert "dry" in note
    assert fake.closes == 0 and fake.cancels == 0
    assert fake.position_qty == 1.0
