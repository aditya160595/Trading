"""Shared stand-ins for the Alpaca venue and the decision client.

The venue fake holds a real position: orders move it, get_position()
reports it, and close_position() clears it. That matters because the loop
now reconciles its own bookkeeping against the broker every few ticks --
a fake that always claimed flat would let a reconciliation bug pass.

It also keeps filled-order history, newest first, because the loop
reconstructs when a position was opened from exactly that. seed_position()
plants a position with a chosen open time, which is how a restart into an
existing position gets tested.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _iso(ts: float) -> str:
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


class FakeAlpaca:
    """A flat book, and a position that orders actually move."""

    def __init__(self, spec, entry_price: float = 100.5):
        self.spec = spec
        self.symbol = spec.symbol
        self.base_url = "https://paper-api.alpaca.markets"
        self.is_live = False
        self.orders: list[tuple] = []
        self.cancels = 0
        self.closes = 0
        self.position_qty = 0.0
        self.entry_price = entry_price
        self.fills: list[dict] = []  # newest first, as Alpaca returns them

    # -- market data ----------------------------------------------------
    def is_market_open(self) -> bool:
        return True

    def get_orderbook(self) -> dict:
        return {
            "b": [{"p": 100.0 - i, "s": 5.0} for i in range(3)],
            "a": [{"p": 101.0 + i, "s": 5.0} for i in range(3)],
        }

    def get_latest_trade(self) -> dict:
        return {"p": self.entry_price, "s": 1.0}

    def get_recent_trades(self, limit: int = 100) -> list:
        return [{"p": self.entry_price, "s": 0.1, "tks": "B"} for _ in range(30)]

    # -- positions ------------------------------------------------------
    def get_position(self):
        if not self.position_qty:
            return None
        return {
            "qty": str(self.position_qty),
            "avg_entry_price": str(self.entry_price),
        }

    def close_position(self):
        if not self.position_qty:
            return None
        self._record_fill(
            "sell" if self.position_qty > 0 else "buy", abs(self.position_qty)
        )
        self.position_qty = 0.0
        self.closes += 1
        return {"id": "close-order"}

    def get_filled_orders(self, limit: int = 500) -> list:
        return self.fills[:limit]

    # -- test helpers ---------------------------------------------------
    def _record_fill(self, side: str, qty: float, at: float | None = None) -> None:
        self.fills.insert(
            0,
            {
                "side": side,
                "filled_qty": str(qty),
                "status": "filled",
                "filled_at": _iso(at if at is not None else datetime.now(timezone.utc).timestamp()),
            },
        )

    def seed_position(self, qty: float, opened_seconds_ago: float = 0.0) -> None:
        """Plant a position the loop did not open, with a real open time in
        the order history -- the restart case."""
        opened_at = datetime.now(timezone.utc).timestamp() - opened_seconds_ago
        self.position_qty = qty
        self._record_fill("buy" if qty > 0 else "sell", abs(qty), at=opened_at)

    # -- orders ---------------------------------------------------------
    def submit_limit_order(self, side, qty, limit_price, tif="gtc"):
        self.orders.append(("limit", side, qty))
        return {"id": f"fake-{len(self.orders)}"}

    def submit_market_order(self, side, qty):
        self.orders.append(("market", side, qty))
        self.position_qty += qty if side == "buy" else -qty
        self._record_fill(side, qty)
        return {"id": f"fake-{len(self.orders)}"}

    def cancel_all_orders(self):
        self.cancels += 1


def _score(value: float) -> dict:
    return {
        "type": "score",
        "score": value,
        "legend": {"0": "a", "1": "b", "2": "c", "3": "d"},
        "probabilities": {"0": 0.0, "1": 0.05, "2": 0.1, "3": 0.85},
        "confidence": 0.95,
    }


def _battery(direction: str, pressure: float) -> dict:
    return {
        "regime": {
            "type": "choice",
            "choice": "trending",
            "probabilities": {"trending": 0.9, "mean_reverting": 0.1},
            "confidence": 0.9,
        },
        "direction": {
            "type": "choice",
            "choice": direction,
            "probabilities": {"up": 0.9, "down": 0.05, "neutral": 0.05},
            "confidence": 0.9,
        },
        "toxic_flow": {"type": "noul", "noul": 0.1},
        "liquidity_stressed": {"type": "noul", "noul": 0.1},
        "quote_environment": _score(2.8),
        "inventory_pressure": _score(pressure),
        "execution_health": _score(2.9),
    }


class AlwaysBuyClient:
    """No randomness: a confident quote environment and a confident "up"
    call every tick, with inventory pressure pinned low so the guard in
    strategy.py never intervenes. The one-sided case that walks a position
    into the cap."""

    name = "STUB"
    model = "stub-always-buy"

    def ask(self, state, questions, timeout):
        return _battery("up", pressure=0.2), {
            "route": self.name,
            "model": self.model,
            "latency_ms": 12.0,
        }


class PressureAwareClient:
    """Always calls "up", but reports inventory_pressure honestly, rising
    with the position, the way a calibrated model would."""

    name = "STUB"
    model = "stub-pressure-aware"

    def __init__(self, max_position_usd: float):
        self._cap = max_position_usd

    def ask(self, state, questions, timeout):
        position_usd = abs(state.get("inventory", 0.0)) * state.get("mid", 0.0)
        pressure = min(3.0, 3.0 * position_usd / self._cap) if self._cap else 0.0
        return _battery("up", pressure=pressure), {
            "route": self.name,
            "model": self.model,
            "latency_ms": 12.0,
        }
