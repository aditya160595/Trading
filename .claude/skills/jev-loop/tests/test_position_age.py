"""When was the position actually opened?

Alpaca's position payload says what you hold, never since when. The loop
used to measure age from the moment it first noticed a position, which
made `max_inventory_age_s` -- an age limit -- silently generous by
however long the position had already been held. On a restart into an
open position that is exactly when you least want it to go soft.

The age is now reconstructed by walking filled order history backwards.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from jevloop.assets import resolve_symbol
from jevloop.limits import Limits
from jevloop.state import (
    InventoryState,
    infer_position_opened_at,
    reconcile_position,
)

from fakes import AlwaysBuyClient, FakeAlpaca

BTC = resolve_symbol("BTC/USD")

T0 = datetime(2026, 9, 22, 10, 0, 0, tzinfo=timezone.utc)


def _at(seconds: int) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _fill(side: str, qty: float, seconds: int) -> dict:
    return {
        "side": side,
        "filled_qty": str(qty),
        "status": "filled",
        "filled_at": _at(seconds),
    }


def _epoch(seconds: int) -> float:
    return (T0 + timedelta(seconds=seconds)).timestamp()


# -- walking the history backwards --------------------------------------


def test_single_fill_opened_the_position():
    fills = [_fill("buy", 5, 100)]
    assert infer_position_opened_at(5, fills) == _epoch(100)


def test_several_fills_report_the_first_one():
    """+4 then +6 with nothing before: the position started at the +4."""
    fills = [_fill("buy", 6, 200), _fill("buy", 4, 100)]
    assert infer_position_opened_at(10, fills) == _epoch(100)


def test_a_previously_closed_position_is_not_counted():
    """History: +4, -4 (flat), then +5, +2, +3. The current position began
    at the +5, not at the older +4 that was already closed out."""
    fills = [
        _fill("buy", 3, 500),
        _fill("buy", 2, 400),
        _fill("buy", 5, 300),
        _fill("sell", 4, 200),
        _fill("buy", 4, 100),
    ]
    assert infer_position_opened_at(10, fills) == _epoch(300)


def test_a_flip_through_flat_starts_a_new_position():
    """Long 3, then a sell of 8 flips to short 5. The short was opened by
    that sell, not by anything before it."""
    fills = [_fill("sell", 8, 200), _fill("buy", 3, 100)]
    assert infer_position_opened_at(-5, fills) == _epoch(200)


def test_short_positions_are_handled():
    fills = [_fill("sell", 2, 100)]
    assert infer_position_opened_at(-2, fills) == _epoch(100)


def test_partial_fills_use_filled_qty():
    """An order for 10 that filled 4 moved the position by 4."""
    fills = [
        {"side": "buy", "filled_qty": "4", "qty": "10", "status": "filled",
         "filled_at": _at(100)},
    ]
    assert infer_position_opened_at(4, fills) == _epoch(100)


# -- and refusing to guess ----------------------------------------------


def test_insufficient_history_returns_unknown():
    """The position is older than the orders we can see. Saying 'unknown'
    beats inventing a number that weakens a risk limit."""
    fills = [_fill("buy", 3, 100)]
    assert infer_position_opened_at(10, fills) is None


def test_no_history_at_all_returns_unknown():
    assert infer_position_opened_at(10, []) is None


def test_flat_position_has_no_open_time():
    assert infer_position_opened_at(0, [_fill("buy", 5, 100)]) is None


def test_unparseable_timestamp_returns_unknown_rather_than_crashing():
    fills = [{"side": "buy", "filled_qty": "5", "status": "filled",
              "filled_at": "not a timestamp"}]
    assert infer_position_opened_at(5, fills) is None


def test_nanosecond_precision_is_parsed():
    """Alpaca sometimes reports 9 fractional digits, which fromisoformat
    refuses outright."""
    fills = [{"side": "buy", "filled_qty": "5", "status": "filled",
              "filled_at": "2026-09-22T10:00:00.123456789Z"}]
    result = infer_position_opened_at(5, fills)
    assert result is not None
    assert abs(result - _epoch(0)) < 1.0


def test_malformed_quantities_are_skipped_not_fatal():
    fills = [
        {"side": "buy", "filled_qty": None, "status": "filled", "filled_at": _at(200)},
        _fill("buy", 5, 100),
    ]
    assert infer_position_opened_at(5, fills) == _epoch(100)


# -- reconcile uses it --------------------------------------------------


def test_reconcile_prefers_the_real_open_time():
    inv = InventoryState(inventory=0.0)
    reconcile_position(
        inv, {"qty": "1"}, as_of=_epoch(1000), opened_at=_epoch(100)
    )
    assert inv.position_opened_at == _epoch(100)


def test_reconcile_falls_back_to_now_when_the_open_time_is_unknown():
    inv = InventoryState(inventory=0.0)
    reconcile_position(inv, {"qty": "1"}, as_of=_epoch(1000), opened_at=None)
    assert inv.position_opened_at == _epoch(1000)


# -- the payoff: an age limit that actually fires on a restart ----------


def _run_once(tmp_path, monkeypatch, fake, limits, ticks=1):
    monkeypatch.setenv("JEV_LOOP_HOME", str(tmp_path))
    monkeypatch.setenv("ALPACA_API_KEY", "PKfake")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

    from jevloop import loop as loopmod

    monkeypatch.setattr(loopmod, "LOG_DIR", tmp_path)
    monkeypatch.setattr(loopmod, "LOG_FILE", tmp_path / "log.jsonl")
    monkeypatch.setattr(loopmod, "LATEST_FILE", tmp_path / "latest.json")
    monkeypatch.setattr(
        loopmod, "client_from_env", lambda symbol, live, confirmation: fake
    )
    monkeypatch.setattr(
        loopmod, "resolve_decision_client", lambda mock: AlwaysBuyClient()
    )
    loopmod.run(symbol="BTC/USD", ticks=ticks, mock=True, limits=limits)
    return [
        json.loads(x)
        for x in (tmp_path / "log.jsonl").read_text().splitlines()
        if x.strip()
    ]


def test_restart_into_an_old_position_knows_its_real_age(
    tmp_path, monkeypatch, capsys
):
    """A position opened 20 minutes ago is 20 minutes old on the first
    tick after a restart -- not zero seconds old."""
    fake = FakeAlpaca(BTC)
    fake.seed_position(0.05, opened_seconds_ago=1200)

    limits = Limits()
    limits.tick_seconds = 0.02
    limits.max_position_usd = 10_000.0
    limits.max_inventory_age_s = 100_000.0  # not the thing under test here

    lines = _run_once(tmp_path, monkeypatch, fake, limits)

    # The logged tick is after execution, so inventory is the adopted
    # 0.05 plus whatever this tick added -- never less than what was
    # already held.
    assert lines[0]["inventory"] >= 0.05
    out = capsys.readouterr().out
    assert "per order history" in out
    # ~1200s, not ~0s.
    assert "1200s ago" in out or "1199s ago" in out or "1201s ago" in out


def test_inventory_age_limit_fires_immediately_on_a_stale_position(
    tmp_path, monkeypatch
):
    """The limitation, gone: a position already held far longer than
    max_inventory_age_s is vetoed on the first tick after a restart,
    instead of being granted a fresh full allowance."""
    fake = FakeAlpaca(BTC)
    fake.seed_position(0.05, opened_seconds_ago=3600)  # an hour old

    limits = Limits()
    limits.tick_seconds = 0.02
    limits.max_position_usd = 10_000.0
    limits.max_inventory_age_s = 900.0  # 15 minutes; the position is 60

    lines = _run_once(tmp_path, monkeypatch, fake, limits)

    # The risk engine refuses to add to a position this old, on tick 1:
    # no fill, and the tick did not quietly trade anyway.
    assert lines[0]["fill_qty"] is None, (
        "traded on a position already older than max_inventory_age_s"
    )


def test_unknown_open_time_is_announced_not_hidden(tmp_path, monkeypatch, capsys):
    """When history cannot reach back far enough, the loop says the age
    limit is running soft rather than quietly pretending otherwise."""
    fake = FakeAlpaca(BTC)
    fake.position_qty = 0.05  # a position with NO matching order history

    limits = Limits()
    limits.tick_seconds = 0.02
    limits.max_position_usd = 10_000.0

    _run_once(tmp_path, monkeypatch, fake, limits)

    out = capsys.readouterr().out
    assert "could not determine when it was opened" in out
    assert "max_inventory_age_s is" in out
