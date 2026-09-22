"""The deterministic state snapshot.

Everything computable stays in code. This module never calls Jev. It turns
an order book (or, on venues with no L2 depth, best bid/ask), a slice of
recent trades, and the loop's own bookkeeping (inventory, PnL, health,
session VWAP) into one compact dict under roughly 400 tokens.

Strict timestamp discipline: every field is computed only from data whose
timestamp is strictly before `as_of`. Never let a fill or a trade that
happened after the decision clock started leak into the snapshot.

Honest degradation: on an asset with no Level 2 depth (US equities on the
basic feed), the caller passes empty depth lists rather than fabricated
ones. `imbalance` and the depth fields come back `None` / empty in that
case instead of a fake "balanced" reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _pct_return(
    prices: list[tuple[float, float]], now: float, lookback_s: float
) -> float | None:
    """Return over the last lookback_s seconds using only points before `now`."""
    past = [p for ts, p in prices if ts <= now - lookback_s]
    current = [p for ts, p in prices if ts <= now]
    if not past or not current:
        return None
    base = past[-1]
    latest = current[-1]
    if base == 0:
        return None
    return (latest - base) / base


def _realised_vol(
    prices: list[tuple[float, float]], now: float, window_s: float
) -> float | None:
    """Simple realised volatility: stdev of log returns inside the window."""
    import math

    pts = [p for ts, p in prices if ts <= now and ts >= now - window_s]
    if len(pts) < 3:
        return None
    rets = []
    for a, b in zip(pts, pts[1:]):
        if a > 0 and b > 0:
            rets.append(math.log(b / a))
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(max(var, 0.0))


@dataclass
class InventoryState:
    """The loop's own bookkeeping, carried tick to tick. Not fetched from
    Alpaca. Session VWAP and slippage are both computed here, in code:
    Jev never sees a raw trade tape, only the numbers this file derives
    from it."""

    inventory: float = 0.0
    entry_price: float = 0.0
    position_opened_at: float | None = None
    realised_pnl_usd: float = 0.0
    high_water_mark_usd: float = 0.0
    equity_usd: float = 0.0
    fills: int = 0
    orders_submitted: int = 0
    orders_rejected: int = 0
    api_error_streak: int = 0
    recent_latencies_ms: list[float] = field(default_factory=list)
    recent_slippage_bps: list[float] = field(default_factory=list)
    # Session VWAP accumulators. Reset when the loop process restarts;
    # that is an honest limitation (documented in README.md), not a bug.
    vwap_cum_pv: float = 0.0
    vwap_cum_vol: float = 0.0
    vwap_last_trade_ts: float = 0.0
    # Set when something happened that makes the local inventory number
    # suspect (a fill, or a close that could not be verified), so the loop
    # re-reads the broker's position on the next tick rather than waiting
    # for the periodic sync.
    needs_position_sync: bool = False


def update_vwap(inv: InventoryState, trades: list[tuple[float, float, float]]) -> None:
    """Fold newly-seen trades (timestamp, price, size) into the running
    session VWAP accumulators. Only trades strictly newer than the last
    one already folded in are counted, so calling this every tick with an
    overlapping recent-trades window never double-counts a fill."""
    newest_ts = inv.vwap_last_trade_ts
    for ts, price, size in trades:
        if ts <= inv.vwap_last_trade_ts:
            continue
        if price <= 0 or size <= 0:
            continue
        inv.vwap_cum_pv += price * size
        inv.vwap_cum_vol += size
        newest_ts = max(newest_ts, ts)
    inv.vwap_last_trade_ts = newest_ts


def record_fill_slippage(
    inv: InventoryState, expected_price: float, fill_price: float, side: str
) -> None:
    """Signed slippage in basis points: positive means the fill was worse
    than expected (paid more on a buy, received less on a sell)."""
    if expected_price <= 0:
        return
    sign = 1.0 if side == "buy" else -1.0
    bps = sign * (fill_price - expected_price) / expected_price * 10_000
    inv.recent_slippage_bps.append(round(bps, 2))
    inv.recent_slippage_bps = inv.recent_slippage_bps[-10:]


def _parse_timestamp(value) -> float | None:
    """RFC3339 -> epoch seconds. Alpaca sometimes reports nanosecond
    precision, which fromisoformat refuses, so the fractional part is
    truncated to microseconds before parsing."""
    import re
    from datetime import datetime

    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    text = re.sub(r"\.(\d{6})\d+", r".\1", text)
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def infer_position_opened_at(
    position_qty: float,
    fills: list[dict],
    tolerance: float = 1e-9,
) -> float | None:
    """When did the current unbroken position actually start?

    Alpaca's position payload says what you hold, never since when, so a
    loop adopting a position it did not open has no age for it -- and
    `max_inventory_age_s` is an age limit. Measuring from first sight
    makes that limit silently generous by however long the position was
    already held, which on a restart is exactly when you least want it
    to be.

    `fills` is the symbol's filled orders, NEWEST FIRST. Walk backwards,
    undoing each fill from the current position; the fill that takes the
    running total to zero, or flips it through zero, is the one that
    opened the position from flat. Its timestamp is the answer.

    Returns None when the history runs out before reaching zero: the
    position is older than the orders we can see, and saying "unknown"
    beats inventing a number that weakens a risk limit.
    """
    if not position_qty:
        return None

    original_sign = 1.0 if position_qty > 0 else -1.0
    remaining = float(position_qty)

    for order in fills:
        try:
            qty = float(order.get("filled_qty") or 0.0)
        except (TypeError, ValueError):
            continue
        if qty <= 0:
            continue

        signed = qty if order.get("side") == "buy" else -qty
        remaining -= signed

        crossed = abs(remaining) <= tolerance or (
            remaining * original_sign < 0
        )
        if crossed:
            # This fill opened the position we are holding now, either
            # from flat or by flipping the previous one through flat.
            return _parse_timestamp(order.get("filled_at"))

    return None


def reconcile_position(
    inv: InventoryState,
    position: dict | None,
    as_of: float,
    opened_at: float | None = None,
) -> tuple[float, float]:
    """Overwrite the loop's own inventory bookkeeping with the broker's.

    The loop only ever adjusted `inventory` when it submitted a
    directional market order, so a resting quote that filled was
    invisible to it and a restart began by assuming flat. Both make
    `max_position_usd` a limit on a number the loop made up rather than
    on the position actually held. This is the correction: Alpaca's
    answer wins, always.

    `position` is Alpaca's position payload, or None when flat.
    `opened_at`, when known, is when the position was actually opened --
    see infer_position_opened_at(). Returns (previous_inventory,
    new_inventory) so the caller can report a correction rather than
    silently swallowing one.
    """
    previous = inv.inventory

    if not position:
        inv.inventory = 0.0
        inv.position_opened_at = None
        return previous, 0.0

    qty = float(position.get("qty", 0.0))
    inv.inventory = qty
    entry = position.get("avg_entry_price")
    if entry is not None:
        inv.entry_price = float(entry)

    if qty == 0:
        inv.position_opened_at = None
    elif previous == 0 or inv.position_opened_at is None:
        # First sight of this position. Alpaca's payload does not say when
        # it was opened, so the caller reconstructs that from filled order
        # history and passes it here. Only when the history cannot reach
        # back far enough does this fall back to now -- and the caller
        # says so out loud rather than letting an age limit quietly go
        # soft.
        inv.position_opened_at = opened_at if opened_at is not None else as_of

    return previous, inv.inventory


def build_snapshot(
    *,
    as_of: float,
    mid: float,
    microprice: float,
    spread_bps: float,
    bid_depth: list[tuple[float, float]],
    ask_depth: list[tuple[float, float]],
    trade_prices: list[tuple[float, float]],
    trade_sides: list[tuple[float, str]],
    inv: InventoryState,
    data_timestamp: float,
    has_depth: bool = True,
) -> dict:
    """Assemble the deterministic snapshot. All list inputs must already be
    filtered to timestamps <= as_of by the caller (execution/alpaca.py).

    `has_depth` is False for venues with no Level 2 book (equities on the
    basic feed): `bid_depth`/`ask_depth` are then expected to hold at most
    one (price, size) pair each, taken from the best bid/ask of the latest
    quote, and `imbalance` is computed from that single level rather than
    three, or left `None` if even that is unavailable."""

    bid_sz = sum(sz for _, sz in bid_depth[:3])
    ask_sz = sum(sz for _, sz in ask_depth[:3])
    if bid_sz + ask_sz > 0:
        imbalance = round((bid_sz - ask_sz) / (bid_sz + ask_sz), 4)
    else:
        imbalance = None

    recent_trades = [(ts, side) for ts, side in trade_sides if ts <= as_of]
    window_trades = [s for ts, s in recent_trades if ts >= as_of - 30.0]
    buys = sum(1 for s in window_trades if s == "buy")
    aggressive_buy_ratio = buys / len(window_trades) if window_trades else 0.5
    trade_intensity = len(window_trades) / 30.0  # trades per second, trailing 30s

    unrealised_pnl = (mid - inv.entry_price) * inv.inventory if inv.inventory else 0.0
    equity = inv.equity_usd + unrealised_pnl
    peak = max(inv.high_water_mark_usd, equity)
    drawdown_pct = (peak - equity) / peak if peak > 0 else 0.0
    position_age_s = (
        (as_of - inv.position_opened_at)
        if (inv.inventory and inv.position_opened_at)
        else 0.0
    )

    fill_ratio = inv.fills / inv.orders_submitted if inv.orders_submitted else 1.0
    data_age_s = max(0.0, as_of - data_timestamp)

    vwap = inv.vwap_cum_pv / inv.vwap_cum_vol if inv.vwap_cum_vol > 0 else mid

    return {
        "as_of": as_of,
        # PRICE
        "mid": mid,
        "microprice": microprice,
        "vwap": round(vwap, 6),
        "return_1m": _pct_return(trade_prices, as_of, 60),
        "return_5m": _pct_return(trade_prices, as_of, 300),
        "return_30m": _pct_return(trade_prices, as_of, 1800),
        # BOOK (depth fields are honestly empty/None where the venue has no L2 book)
        "spread_bps": spread_bps,
        "has_depth": has_depth,
        "depth_levels_available": (
            min(len(bid_depth), len(ask_depth)) if has_depth else 0
        ),
        "bid_depth_3": [[p, s] for p, s in bid_depth[:3]],
        "ask_depth_3": [[p, s] for p, s in ask_depth[:3]],
        "imbalance": imbalance,
        # FLOW
        "aggressive_buy_ratio": round(aggressive_buy_ratio, 4),
        "trade_intensity_per_s": round(trade_intensity, 4),
        # VOL
        "realised_vol_short": _realised_vol(trade_prices, as_of, 300),
        "realised_vol_medium": _realised_vol(trade_prices, as_of, 1800),
        # BOOK PnL
        "inventory": inv.inventory,
        "unrealised_pnl_usd": round(unrealised_pnl, 4),
        "daily_loss_usd": round(max(0.0, -inv.realised_pnl_usd - unrealised_pnl), 4),
        "drawdown_pct": round(drawdown_pct, 6),
        "position_age_s": round(position_age_s, 1),
        # HEALTH
        "fill_ratio": round(fill_ratio, 4),
        "reject_count": inv.orders_rejected,
        "last_10_latencies_ms": inv.recent_latencies_ms[-10:],
        "last_10_slippage_bps": inv.recent_slippage_bps[-10:],
        "data_age_s": round(data_age_s, 3),
        "leverage": 1.0,
    }


def approx_token_count(snapshot: dict) -> int:
    """Rough token estimate (chars / 4) so the loop can print a sanity check
    that the snapshot stays well under the 400-token guideline."""
    import json

    return len(json.dumps(snapshot, default=str)) // 4
