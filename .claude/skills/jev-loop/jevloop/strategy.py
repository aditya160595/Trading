"""strategy.py -- the file you edit to change how this loop trades.

This is the harness the video talks about, not the edge. Everything it
shipped with is a generic strategy pulled out of thin air, wired in only
so the demo has something to trade. Real strategies are hard to build
properly; this file is where yours goes.

Two things live here:

1. `StrategyThresholds`, one number per action in `compose_action()`
   (policy.py): when to pull quotes, when to widen, when the quote
   environment is good enough to quote both sides or just quote wide, how
   much inventory pressure skews sizing, and how confident Jev has to be
   about direction before a directional leg is taken. Change a number,
   restart the loop, see different behaviour on the next tick.

2. `apply_strategy()`, a hook called once per tick with the action
   `compose_action()` already produced from the thresholds above. Return
   a different action to override it, or an action with
   `kind=STAND_DOWN` to veto the tick outright. This is the one function
   a real strategy plugs into. It ships with a single inventory guard,
   described on the function itself.

Shipped default: every threshold below matches what the video ran.
`apply_strategy()` ships with exactly one guard -- it drops a directional
leg that would grow an already-pressured position -- and is otherwise a
pass-through. That guard only ever makes the loop more conservative, and
it can be disabled by raising `inventory_pressure_leg_veto_score` above
3.0 or by returning `action` unchanged.

The hard risk caps (max position, max daily loss, max drawdown, and so
on) do NOT live here: they live in `jevloop/limits.py`, checked in
`risk.py` before every order, and this file cannot raise them. A
strategy can make the loop more conservative than the risk engine
allows; it can never make it less conservative than the risk engine
allows.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StrategyThresholds:
    """One threshold per action in compose_action(). These are exactly the
    numbers the video shipped with."""

    toxic_flow_pull_threshold: float = 0.6  # ans["toxic_flow"] > this -> PULL_QUOTES
    liquidity_stressed_widen_threshold: float = (
        0.7  # ans["liquidity_stressed"] > this -> WIDEN
    )
    quote_env_full_score: float = 2.0  # env >= this and confident -> quote both sides
    quote_env_full_confidence: float = 0.80
    quote_env_wide_score: float = 1.0  # env >= this (but below full) -> quote wide
    inventory_pressure_max_score: float = 3.0  # denominator for the skew calculation

    # the directional leg, bolted on so the demo shows fills, not just quotes
    direction_confidence_threshold: float = (
        0.55  # direction.confidence above this -> take the leg
    )

    # Inventory guard (see apply_strategy below). Jev's inventory_pressure
    # score runs 0 "None" .. 3 "Reduce now"; at or above this, a leg that
    # would GROW the open position is dropped. Raise it toward 3.0 to let
    # inventory run further, or set it above 3.0 to disable the guard and
    # restore the original no-op behaviour.
    inventory_pressure_leg_veto_score: float = 2.0  # "Skew hard" or worse


THRESHOLDS = StrategyThresholds()


def apply_strategy(action, answers: dict, snapshot: dict, limits) -> object:
    """The strategy hook. Called once per tick, after compose_action() has
    already turned Jev's answers into an action using THRESHOLDS above.

    Shipped behaviour: one guard, and otherwise the action is returned
    untouched. The guard drops the directional leg when Jev says inventory
    pressure has reached `inventory_pressure_leg_veto_score` AND the leg
    would grow the position rather than cut it. Quoting is unaffected --
    only the leg is dropped.

    Why it exists: the directional leg is the only thing in this loop that
    moves inventory, and nothing else ever reduces it. A persistently
    one-sided direction call (exactly what a trending regime produces)
    walks the position into `max_position_usd` in a handful of fills, and
    that is a KILL: the loop flattens and stops. Jev is already answering
    "how urgent is it to cut this position" every tick; this listens to
    that answer instead of discarding it, so the risk cap stays an
    emergency backstop rather than a scheduled event.

    This makes the loop MORE conservative, never less: it only ever
    removes a leg, never adds or enlarges one. risk.py still runs after
    this and still holds the final veto.

    To customise, edit this function. You can:
      - inspect `answers` (the seven Jev judgments this tick) or
        `snapshot` (the deterministic state) and return a different
        Action than the one compose_action() chose
      - veto the tick outright by returning
        `Action(STAND_DOWN, reason="my strategy said no")`
      - drop the guard entirely by returning `action` unchanged, which
        restores the behaviour this file originally shipped with
    """
    if action.direction_leg is None:
        return action

    pressure = (answers.get("inventory_pressure") or {}).get("score")
    if pressure is None or pressure < THRESHOLDS.inventory_pressure_leg_veto_score:
        return action

    inventory = snapshot.get("inventory", 0.0)
    if not inventory:
        # Pressure to cut a position you do not hold is not a reason to
        # refuse to open one.
        return action

    grows_position = (action.direction_leg == "up" and inventory > 0) or (
        action.direction_leg == "down" and inventory < 0
    )
    if grows_position:
        action.direction_leg = None
        action.reason = (
            f"{action.reason}; leg dropped, inventory pressure {pressure:.1f}"
        )
    return action
