"""Regression tests for the kill path reaching the ladder.

The bug these pin: only max_drawdown was wired into select_rung(). The
other four kill-class limits were discovered inside _execute_action(),
where the verdict reached the printed line and nothing else -- so the
loop refused every order, tick after tick, instead of flattening and
stopping. `rung` stayed "run".
"""

import pytest

from jevloop.assets import resolve_symbol
from jevloop.limits import Limits
from jevloop.risk import check, kill_check

from fakes import AlwaysBuyClient, FakeAlpaca

L = Limits()

FLAT = dict(
    drawdown_pct=0.01,
    inventory=0.0,
    mid=85_000.0,
    daily_loss_usd=0.0,
    position_age_s=0.0,
    data_age_s=0.1,
    leverage=1.0,
)


# -- kill_check: every kill-class limit, no order size needed ------------


def test_kill_check_passes_a_healthy_snapshot():
    assert kill_check(FLAT, L, api_error_streak=0).ok


@pytest.mark.parametrize(
    "overrides,streak,expected",
    [
        (dict(drawdown_pct=0.9), 0, "max_drawdown breached"),
        (dict(inventory=1.0), 0, "max_position_usd breached"),
        (dict(daily_loss_usd=999.0), 0, "max_daily_loss breached"),
        (dict(), 99, "max_api_errors breached"),
        (dict(leverage=5.0), 0, "max_leverage breached"),
    ],
)
def test_every_kill_class_limit_is_caught_before_any_order_is_sized(
    overrides, streak, expected
):
    snap = dict(FLAT, **overrides)
    verdict = kill_check(snap, L, api_error_streak=streak)
    assert not verdict.ok
    assert verdict.kill
    assert verdict.veto == expected


def test_kill_check_is_not_concerned_with_order_specific_vetoes():
    """Order notional, inventory age, stale data and latency are vetoes,
    not kills: they must not appear here, or the loop would stop dead on
    a condition that only meant 'skip this one order'."""
    stale_and_old = dict(FLAT, data_age_s=999.0, position_age_s=999_999.0)
    assert kill_check(stale_and_old, L, api_error_streak=0).ok


def test_check_still_catches_every_kill_after_delegating():
    """check() delegates to kill_check() -- confirm nothing was lost."""
    for overrides, streak in [
        (dict(drawdown_pct=0.9), 0),
        (dict(inventory=1.0), 0),
        (dict(daily_loss_usd=999.0), 0),
        (dict(), 99),
        (dict(leverage=5.0), 0),
    ]:
        snap = dict(FLAT, **overrides)
        v = check(snap, 20.0, L, streak, 90.0)
        assert not v.ok and v.kill


def test_kill_dominates_a_plain_veto():
    """A snapshot breaching both a kill limit and an order-size veto must
    report the kill: stopping beats skipping one order."""
    snap = dict(FLAT, inventory=1.0)  # position breach
    v = check(snap, 10_000.0, L, 0, 90.0)  # also over max_order_notional
    assert v.kill
    assert v.veto == "max_position_usd breached"


# -- the loop actually stops -------------------------------------------


def test_position_breach_stops_the_loop_instead_of_looping_forever(
    tmp_path, monkeypatch
):
    """The regression proper: with a position cap small enough that the
    first fill breaches it, the loop must log rung 'kill' and stop -- not
    run out its full tick budget printing KILL every tick."""
    import json

    monkeypatch.setenv("JEV_LOOP_HOME", str(tmp_path))
    monkeypatch.setenv("ALPACA_API_KEY", "PKfake")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

    from jevloop import loop as loopmod

    monkeypatch.setattr(loopmod, "LOG_DIR", tmp_path)
    monkeypatch.setattr(loopmod, "LOG_FILE", tmp_path / "log.jsonl")
    monkeypatch.setattr(loopmod, "LATEST_FILE", tmp_path / "latest.json")

    spec = resolve_symbol("BTC/USD")
    fake = FakeAlpaca(spec)
    monkeypatch.setattr(
        loopmod, "client_from_env", lambda symbol, live, confirmation: fake
    )
    monkeypatch.setattr(
        loopmod, "resolve_decision_client", lambda mock: AlwaysBuyClient()
    )

    limits = Limits()
    limits.tick_seconds = 0.05
    limits.max_position_usd = 0.01  # the first fill breaches this

    budget = 25
    rc = loopmod.run(symbol="BTC/USD", ticks=budget, mock=True, limits=limits)
    assert rc == 0

    lines = [
        json.loads(x)
        for x in (tmp_path / "log.jsonl").read_text().splitlines()
        if x.strip()
    ]
    # It must have stopped early, not burned the whole budget.
    assert len(lines) < budget, "loop did not stop on a position breach"
    # The last tick is the kill, and it is recorded as one.
    assert lines[-1]["rung"] == "kill"
    # And no tick after a kill: the kill is terminal.
    assert [r["rung"] for r in lines].count("kill") == 1


def test_api_error_streak_can_actually_trip(tmp_path, monkeypatch):
    """max_api_errors was unreachable: the error branch incremented the
    streak then `continue`d, and the success branch reset it to 0, so no
    risk check ever saw a non-zero value. The loop must now stop once the
    streak passes the limit."""
    monkeypatch.setenv("JEV_LOOP_HOME", str(tmp_path))
    monkeypatch.setenv("ALPACA_API_KEY", "PKfake")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

    from jevloop import loop as loopmod
    from jevloop.execution.alpaca import AlpacaAPIError

    monkeypatch.setattr(loopmod, "LOG_DIR", tmp_path)
    monkeypatch.setattr(loopmod, "LOG_FILE", tmp_path / "log.jsonl")
    monkeypatch.setattr(loopmod, "LATEST_FILE", tmp_path / "latest.json")

    spec = resolve_symbol("BTC/USD")
    fake = FakeAlpaca(spec)

    attempts = {"n": 0}

    def _always_fails():
        attempts["n"] += 1
        raise AlpacaAPIError(500, "venue down")

    fake.get_orderbook = _always_fails
    monkeypatch.setattr(
        loopmod, "client_from_env", lambda symbol, live, confirmation: fake
    )

    limits = Limits()
    limits.tick_seconds = 0.05
    limits.max_api_errors = 3

    budget = 30
    rc = loopmod.run(symbol="BTC/USD", ticks=budget, mock=True, limits=limits)
    assert rc == 0
    # It must give up once the streak passes max_api_errors, not grind
    # through the whole tick budget against a venue that is plainly down.
    # (Counting attempts, not log lines: the error path never logs a tick,
    # so a log-based assertion would pass vacuously either way.)
    assert attempts["n"] == limits.max_api_errors + 1, (
        f"expected to stop after {limits.max_api_errors + 1} failed reads, "
        f"made {attempts['n']}"
    )
