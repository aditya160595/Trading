"""Shared stand-ins for the Alpaca venue and the decision client.

The venue fake holds a real position: orders move it, get_position()
reports it, and close_position() clears it. That matters because the loop
now reconciles its own bookkeeping against the broker every few ticks --
a fake that always claimed flat would let a reconciliation bug pass.
"""

from __future__ import annotations


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
        self.position_qty = 0.0
        self.closes += 1
        return {"id": "close-order"}

    # -- orders ---------------------------------------------------------
    def submit_limit_order(self, side, qty, limit_price, tif="gtc"):
        self.orders.append(("limit", side, qty))
        return {"id": f"fake-{len(self.orders)}"}

    def submit_market_order(self, side, qty):
        self.orders.append(("market", side, qty))
        self.position_qty += qty if side == "buy" else -qty
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
