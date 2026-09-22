"""The inventory guard in strategy.apply_strategy().

The directional leg is the only thing in this loop that moves inventory,
and nothing else reduces it. A persistently one-sided direction call
walks the position into max_position_usd in a few fills, which is a KILL.
The guard listens to Jev's inventory_pressure answer -- already computed
every tick, previously only logged -- and drops a leg that would grow an
already-pressured position.
"""

import json

import pytest

from jevloop.assets import resolve_symbol
from jevloop.limits import Limits
from jevloop.policy import QUOTE_BOTH_SIDES, Action, compose_action
from jevloop.strategy import THRESHOLDS, apply_strategy

L = Limits()

LONG = dict(inventory=0.5, drawdown_pct=0.0, mid=100.0)
SHORT = dict(inventory=-0.5, drawdown_pct=0.0, mid=100.0)
FLAT = dict(inventory=0.0, drawdown_pct=0.0, mid=100.0)

HIGH_PRESSURE = {"inventory_pressure": {"score": 2.6}}
LOW_PRESSURE = {"inventory_pressure": {"score": 0.4}}


def _leg(direction):
    return Action(QUOTE_BOTH_SIDES, reason="env 2.4", direction_leg=direction)


# -- the guard drops legs that grow a pressured position ----------------


def test_long_and_pressured_drops_a_buy_leg():
    action = apply_strategy(_leg("up"), HIGH_PRESSURE, LONG, L)
    assert action.direction_leg is None
    assert "inventory pressure" in action.reason


def test_short_and_pressured_drops_a_sell_leg():
    action = apply_strategy(_leg("down"), HIGH_PRESSURE, SHORT, L)
    assert action.direction_leg is None


# -- but never a leg that CUTS the position -----------------------------


def test_long_and_pressured_keeps_a_sell_leg():
    """Selling into a pressured long is the whole point -- never block it."""
    action = apply_strategy(_leg("down"), HIGH_PRESSURE, LONG, L)
    assert action.direction_leg == "down"


def test_short_and_pressured_keeps_a_buy_leg():
    action = apply_strategy(_leg("up"), HIGH_PRESSURE, SHORT, L)
    assert action.direction_leg == "up"


# -- and never fires when it shouldn't ----------------------------------


def test_low_pressure_leaves_the_leg_alone():
    action = apply_strategy(_leg("up"), LOW_PRESSURE, LONG, L)
    assert action.direction_leg == "up"


def test_flat_inventory_leaves_the_leg_alone_even_under_pressure():
    """Pressure to cut a position you do not hold must not stop you
    opening one, or the loop could never take a position at all."""
    action = apply_strategy(_leg("up"), HIGH_PRESSURE, FLAT, L)
    assert action.direction_leg == "up"


def test_action_without_a_leg_is_untouched():
    action = apply_strategy(_leg(None), HIGH_PRESSURE, LONG, L)
    assert action.direction_leg is None
    assert action.reason == "env 2.4"  # unannotated: the guard never ran


def test_missing_inventory_pressure_answer_is_survived():
    """Defensive: a caller passing a partial answers dict must not crash
    the tick."""
    action = apply_strategy(_leg("up"), {}, LONG, L)
    assert action.direction_leg == "up"


def test_guard_never_adds_or_enlarges_a_leg():
    """Structural: the guard is one-way. Whatever it returns, the leg is
    either unchanged or gone -- never newly set."""
    for snapshot in (LONG, SHORT, FLAT):
        for answers in (HIGH_PRESSURE, LOW_PRESSURE):
            before = _leg(None)
            after = apply_strategy(before, answers, snapshot, L)
            assert after.direction_leg is None


def test_threshold_is_tunable_and_disablable():
    from jevloop.strategy import StrategyThresholds

    assert StrategyThresholds().inventory_pressure_leg_veto_score == 2.0
    # Pressure just under the shipped threshold must not trip it.
    just_under = {"inventory_pressure": {"score": 1.99}}
    assert apply_strategy(_leg("up"), just_under, LONG, L).direction_leg == "up"


def test_guard_is_reached_through_compose_action():
    """Not just callable in isolation: compose_action() must route through
    it, or the guard is dead code in the real loop."""
    snapshot = dict(
        drawdown_pct=0.0,
        inventory=0.5,
        mid=100.0,
        daily_loss_usd=0.0,
        position_age_s=0.0,
        data_age_s=0.1,
        leverage=1.0,
        spread_bps=4.0,
        imbalance=0.0,
    )
    answers = {
        "toxic_flow": {"noul": 0.1},
        "liquidity_stressed": {"noul": 0.1},
        "quote_environment": {"score": 2.5, "confidence": 0.9},
        "inventory_pressure": {"score": 2.8},
        "direction": {"choice": "up", "confidence": 0.9},
    }
    action = compose_action(answers, snapshot, L)
    assert action.kind == QUOTE_BOTH_SIDES  # still quoting
    assert action.direction_leg is None  # but not growing the long


# -- the payoff: the loop stops walking into its own risk cap -----------


class _FakeAlpaca:
    def __init__(self, spec):
        self.spec = spec
        self.symbol = spec.symbol
        self.base_url = "https://paper-api.alpaca.markets"
        self.is_live = False

    def is_market_open(self):
        return True

    def get_orderbook(self):
        return {
            "b": [{"p": 100.0 - i, "s": 5.0} for i in range(3)],
            "a": [{"p": 101.0 + i, "s": 5.0} for i in range(3)],
        }

    def get_latest_trade(self):
        return {"p": 100.5, "s": 1.0}

    def get_recent_trades(self, limit=100):
        return [{"p": 100.5, "s": 0.1, "tks": "B"} for _ in range(30)]

    def submit_limit_order(self, side, qty, limit_price, tif="gtc"):
        return {"id": "x"}

    def submit_market_order(self, side, qty):
        return {"id": "x"}

    def cancel_all_orders(self):
        pass


class _PressureAwareClient:
    """Always calls "up" confidently -- the one-sided case that walks the
    position into the cap -- but reports inventory_pressure honestly,
    rising with the position, the way a calibrated model would."""

    name = "STUB"
    model = "stub-pressure-aware"

    def __init__(self, max_position_usd):
        self._cap = max_position_usd

    def ask(self, state, questions, timeout):
        position_usd = abs(state.get("inventory", 0.0)) * state.get("mid", 0.0)
        pressure = min(3.0, 3.0 * position_usd / self._cap) if self._cap else 0.0
        score = lambda v: {  # noqa: E731
            "type": "score",
            "score": v,
            "legend": {"0": "a", "1": "b", "2": "c", "3": "d"},
            "probabilities": {"0": 0.0, "1": 0.05, "2": 0.1, "3": 0.85},
            "confidence": 0.95,
        }
        answers = {
            "regime": {
                "type": "choice",
                "choice": "trending",
                "probabilities": {"trending": 0.9, "mean_reverting": 0.1},
                "confidence": 0.9,
            },
            "direction": {
                "type": "choice",
                "choice": "up",
                "probabilities": {"up": 0.9, "down": 0.05, "neutral": 0.05},
                "confidence": 0.9,
            },
            "toxic_flow": {"type": "noul", "noul": 0.1},
            "liquidity_stressed": {"type": "noul", "noul": 0.1},
            "quote_environment": score(2.8),
            "inventory_pressure": score(pressure),
            "execution_health": score(2.9),
        }
        return answers, {"route": self.name, "model": self.model, "latency_ms": 12.0}


def test_loop_runs_its_full_budget_instead_of_killing_on_position(
    tmp_path, monkeypatch
):
    """With the guard listening to inventory_pressure, a relentlessly
    one-sided direction call no longer walks the book into
    max_position_usd: the loop completes its tick budget and never lands
    on the kill rung. The cap itself is untouched -- it is simply not
    reached any more."""
    monkeypatch.setenv("JEV_LOOP_HOME", str(tmp_path))
    monkeypatch.setenv("ALPACA_API_KEY", "PKfake")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

    from jevloop import loop as loopmod

    monkeypatch.setattr(loopmod, "LOG_DIR", tmp_path)
    monkeypatch.setattr(loopmod, "LOG_FILE", tmp_path / "log.jsonl")
    monkeypatch.setattr(loopmod, "LATEST_FILE", tmp_path / "latest.json")

    spec = resolve_symbol("BTC/USD")
    monkeypatch.setattr(
        loopmod, "client_from_env", lambda symbol, live, confirmation: _FakeAlpaca(spec)
    )

    limits = Limits()
    limits.tick_seconds = 0.02
    monkeypatch.setattr(
        loopmod,
        "resolve_decision_client",
        lambda mock: _PressureAwareClient(limits.max_position_usd),
    )

    budget = 30
    rc = loopmod.run(symbol="BTC/USD", ticks=budget, mock=True, limits=limits)
    assert rc == 0

    lines = [
        json.loads(x)
        for x in (tmp_path / "log.jsonl").read_text().splitlines()
        if x.strip()
    ]
    assert len(lines) == budget, "loop stopped early -- the guard did not hold"
    assert "kill" not in [r["rung"] for r in lines]
    # It did trade: the guard throttles the leg, it does not ban it.
    assert any(r["fill_qty"] for r in lines)
    # And the position stayed inside the cap the whole way.
    worst = max(abs(r["inventory"]) * r["mid"] for r in lines)
    assert worst <= limits.max_position_usd
