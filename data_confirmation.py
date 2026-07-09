import time

import pandas as pd

import config
from exchange import client, get_klines
from logger import log_warning


_cache = {}


def _safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def _normalise(value, scale):
    scale = abs(scale) or 1
    return _clamp(value / scale, -1, 1)


def _cache_get(symbol):
    cached = _cache.get(symbol)
    ttl = max(float(getattr(config, "DATA_CONFIRMATION_CACHE_SECONDS", 120)), 0)

    if cached and ttl > 0 and time.time() - cached["time"] <= ttl:
        return cached["data"]

    return None


def _cache_set(symbol, data):
    _cache[symbol] = {
        "time": time.time(),
        "data": data,
    }
    return data


def _numeric_series(frame, column):
    if frame is None or column not in frame:
        return pd.Series(dtype="float64")

    return pd.to_numeric(frame[column], errors="coerce").fillna(0)


def _source_frame(symbol, entry_df):
    if entry_df is not None and len(entry_df) >= 20:
        return entry_df.copy()

    return get_klines(
        symbol,
        getattr(config, "DATA_CONFIRMATION_KLINE_INTERVAL", "15m"),
        getattr(config, "DATA_CONFIRMATION_KLINE_LIMIT", 96),
    )


def _order_book_metrics(symbol):
    limit = max(int(getattr(config, "DATA_CONFIRMATION_ORDER_BOOK_LIMIT", 50)), 5)
    depth = max(int(getattr(config, "DATA_CONFIRMATION_ORDER_BOOK_DEPTH", 20)), 1)
    data = client.futures_order_book(symbol=symbol, limit=limit)
    bids = data.get("bids", [])[:depth]
    asks = data.get("asks", [])[:depth]

    bid_notional = sum(_safe_float(price) * _safe_float(qty) for price, qty in bids)
    ask_notional = sum(_safe_float(price) * _safe_float(qty) for price, qty in asks)
    total = bid_notional + ask_notional
    imbalance = (bid_notional - ask_notional) / total if total > 0 else 0

    whale_min = max(
        float(getattr(config, "DATA_CONFIRMATION_WHALE_MIN_NOTIONAL", 50_000)),
        0,
    )
    whale_bids = sum(
        _safe_float(price) * _safe_float(qty)
        for price, qty in bids
        if _safe_float(price) * _safe_float(qty) >= whale_min
    )
    whale_asks = sum(
        _safe_float(price) * _safe_float(qty)
        for price, qty in asks
        if _safe_float(price) * _safe_float(qty) >= whale_min
    )
    whale_total = whale_bids + whale_asks
    whale_imbalance = (
        (whale_bids - whale_asks) / whale_total
        if whale_total > 0
        else 0
    )

    return {
        "order_book_imbalance": _clamp(imbalance, -1, 1),
        "whale_imbalance": _clamp(whale_imbalance, -1, 1),
        "bid_notional": bid_notional,
        "ask_notional": ask_notional,
        "whale_bid_notional": whale_bids,
        "whale_ask_notional": whale_asks,
    }


def _flow_metrics(symbol, entry_df):
    frame = _source_frame(symbol, entry_df)

    if frame is None or len(frame) < 20:
        return {
            "cvd_imbalance": 0,
            "cvd_trend": 0,
            "volume_delta": 0,
            "relative_volume": 1,
            "price_momentum": 0,
        }

    close = _numeric_series(frame, "close")
    quote_volume = _numeric_series(frame, "qav")

    if quote_volume.empty or quote_volume.sum() <= 0:
        quote_volume = _numeric_series(frame, "volume") * close

    taker_buy_quote = _numeric_series(frame, "tbqav")

    if taker_buy_quote.empty or taker_buy_quote.sum() <= 0:
        taker_buy_base = _numeric_series(frame, "tbbav")
        taker_buy_quote = taker_buy_base * close

    taker_sell_quote = (quote_volume - taker_buy_quote).clip(lower=0)
    delta = taker_buy_quote - taker_sell_quote

    short_volume = _safe_float(quote_volume.tail(12).sum())
    long_volume = _safe_float(quote_volume.tail(48).sum())
    latest_volume = _safe_float(quote_volume.iloc[-1])
    recent_volume_mean = _safe_float(quote_volume.tail(48).mean())
    recent_close = _safe_float(close.iloc[-1])
    previous_close = _safe_float(close.iloc[-13]) if len(close) >= 13 else recent_close

    cvd_imbalance = (
        _safe_float(delta.tail(12).sum()) / short_volume
        if short_volume > 0
        else 0
    )
    cvd_trend = (
        _safe_float(delta.tail(48).sum()) / long_volume
        if long_volume > 0
        else 0
    )
    volume_delta = (
        _safe_float(delta.iloc[-1]) / latest_volume
        if latest_volume > 0
        else 0
    )
    relative_volume = latest_volume / recent_volume_mean if recent_volume_mean > 0 else 1
    price_momentum = (
        (recent_close - previous_close) / previous_close
        if previous_close > 0
        else 0
    )

    return {
        "cvd_imbalance": _clamp(cvd_imbalance, -1, 1),
        "cvd_trend": _clamp(cvd_trend, -1, 1),
        "volume_delta": _clamp(volume_delta, -1, 1),
        "relative_volume": _clamp(relative_volume, 0, 5),
        "price_momentum": _clamp(price_momentum, -1, 1),
    }


def _open_interest_bias(symbol, price_momentum):
    period = getattr(config, "DATA_CONFIRMATION_OI_PERIOD", "5m")
    limit = max(int(getattr(config, "DATA_CONFIRMATION_OI_LIMIT", 12)), 2)

    try:
        history = client.futures_open_interest_hist(
            symbol=symbol,
            period=period,
            limit=limit,
        )
    except Exception as exc:
        return 0, None, f"OI:{exc}"

    if not history or len(history) < 2:
        return 0, None, None

    first = _safe_float(history[0].get("sumOpenInterest"))
    last = _safe_float(history[-1].get("sumOpenInterest"))
    change_pct = ((last - first) / first) * 100 if first > 0 else 0
    change_norm = max(_normalise(change_pct, 3.0), 0)
    momentum_norm = _normalise(price_momentum, 0.015)
    bias = change_norm * momentum_norm

    return _clamp(bias, -1, 1), round(change_pct, 4), None


def _funding_bias(symbol):
    try:
        premium = client.futures_mark_price(symbol=symbol)
        funding = _safe_float(premium.get("lastFundingRate"))
    except Exception as exc:
        return 0, None, f"FUNDING:{exc}"

    max_abs = max(
        float(getattr(config, "DATA_CONFIRMATION_MAX_ABS_FUNDING", 0.0015)),
        0.00000001,
    )
    bias = -_normalise(funding, max_abs)
    return _clamp(bias, -1, 1), funding, None


def _liquidation_metrics(symbol):
    if not getattr(config, "DATA_CONFIRMATION_LIQUIDATIONS_ENABLED", True):
        return 0, None

    try:
        orders = client.futures_liquidation_orders(
            symbol=symbol,
            limit=max(int(getattr(config, "DATA_CONFIRMATION_LIQUIDATION_LIMIT", 50)), 1),
        )
    except Exception as exc:
        return 0, f"LIQ:{exc}"

    buy_liq = 0.0
    sell_liq = 0.0

    for item in orders or []:
        side = str(item.get("side", "")).upper()
        price = _safe_float(item.get("price") or item.get("averagePrice"))
        qty = _safe_float(item.get("origQty") or item.get("executedQty"))
        notional = price * qty

        if side == "BUY":
            buy_liq += notional
        elif side == "SELL":
            sell_liq += notional

    total = buy_liq + sell_liq
    imbalance = (buy_liq - sell_liq) / total if total > 0 else 0
    return _clamp(imbalance, -1, 1), None


def _metric_weight(name, default):
    return max(_safe_float(getattr(config, name, default), default), 0)


def _classify_metric(name, value, weight, side, confirmations, conflicts):
    min_weight = float(
        getattr(config, "DATA_CONFIRMATION_MIN_CONFIRMATION_WEIGHT", 0.50)
    )

    if weight < min_weight:
        return

    signed = value if side == "BUY" else -value
    confirm_at = float(getattr(config, "DATA_CONFIRMATION_METRIC_CONFIRM_AT", 0.08))
    conflict_at = -float(getattr(config, "DATA_CONFIRMATION_METRIC_CONFLICT_AT", 0.08))

    if signed >= confirm_at:
        confirmations.append(name)
        return "confirm"
    elif signed <= conflict_at:
        conflicts.append(name)
        return "conflict"

    return None


def confirm_market_data(symbol, chart_side, entry_df=None):
    context = {
        "enabled": bool(getattr(config, "DATA_CONFIRMATION_ENABLED", True)),
        "ok": True,
        "side": "DISABLED",
        "confidence": 0,
        "edge": 0,
        "buy_score": 0,
        "sell_score": 0,
        "reason": "DATA_CONFIRMATION_DISABLED",
        "confirmations": [],
        "conflicts": [],
        "metrics": {},
        "errors": [],
    }

    if not context["enabled"]:
        return context

    cached = _cache_get(symbol)

    if cached:
        context = dict(cached)
        context["ok"] = _decision_ok(context, chart_side)
        context["reason"] = _decision_reason(context, chart_side)
        return context

    try:
        metrics = {}
        metrics.update(_order_book_metrics(symbol))
        metrics.update(_flow_metrics(symbol, entry_df))

        oi_bias, oi_change_pct, oi_error = _open_interest_bias(
            symbol,
            metrics.get("price_momentum", 0),
        )
        funding_bias, funding_rate, funding_error = _funding_bias(symbol)
        liquidation_imbalance, liquidation_error = _liquidation_metrics(symbol)

        metrics["open_interest_bias"] = oi_bias
        metrics["open_interest_change_pct"] = oi_change_pct
        metrics["funding_bias"] = funding_bias
        metrics["funding_rate"] = funding_rate
        metrics["liquidation_imbalance"] = liquidation_imbalance

        errors = [
            error for error in (oi_error, funding_error, liquidation_error)
            if error
        ]
        weighted = {
            "order_book": (
                metrics["order_book_imbalance"],
                _metric_weight("DATA_CONFIRMATION_WEIGHT_ORDER_BOOK", 0.50),
            ),
            "cvd": (
                metrics["cvd_imbalance"],
                _metric_weight("DATA_CONFIRMATION_WEIGHT_CVD", 1.60),
            ),
            "cvd_trend": (
                metrics["cvd_trend"],
                _metric_weight("DATA_CONFIRMATION_WEIGHT_CVD_TREND", 1.10),
            ),
            "volume_delta": (
                metrics["volume_delta"],
                _metric_weight("DATA_CONFIRMATION_WEIGHT_VOLUME_DELTA", 1.10),
            ),
            "open_interest": (
                metrics["open_interest_bias"],
                _metric_weight("DATA_CONFIRMATION_WEIGHT_OPEN_INTEREST", 0.90),
            ),
            "funding": (
                metrics["funding_bias"],
                _metric_weight("DATA_CONFIRMATION_WEIGHT_FUNDING", 0.10),
            ),
            "liquidations": (
                metrics["liquidation_imbalance"],
                _metric_weight("DATA_CONFIRMATION_WEIGHT_LIQUIDATIONS", 0.15),
            ),
            "whale_orders": (
                metrics["whale_imbalance"],
                _metric_weight("DATA_CONFIRMATION_WEIGHT_WHALE_ORDERS", 0.20),
            ),
        }
        weighted = {
            name: item
            for name, item in weighted.items()
            if item[1] > 0
        }
        total_weight = sum(weight for _value, weight in weighted.values())
        if total_weight <= 0:
            total_weight = 1
        directional_score = sum(
            value * weight
            for value, weight in weighted.values()
        ) / total_weight
        directional_score = _clamp(directional_score, -1, 1)
        score_scale = max(
            _safe_float(getattr(config, "DATA_CONFIRMATION_SCORE_SCALE", 0.18), 0.18),
            0.01,
        )
        scaled_directional_score = _clamp(directional_score / score_scale, -1, 1)

        side = "BUY" if directional_score > 0 else "SELL" if directional_score < 0 else "NEUTRAL"
        raw_edge = round(abs(directional_score) * 100, 2)
        edge = round(abs(scaled_directional_score) * 100, 2)
        confidence = round(50 + (edge * 0.5), 2)
        buy_score = round(50 + scaled_directional_score * 50, 2)
        sell_score = round(50 - scaled_directional_score * 50, 2)
        confirmations = []
        conflicts = []
        confirmation_score = 0.0
        conflict_score = 0.0

        for name, (value, weight) in weighted.items():
            classification = _classify_metric(
                name,
                value,
                weight,
                side,
                confirmations,
                conflicts,
            )

            if classification == "confirm":
                confirmation_score += weight
            elif classification == "conflict":
                conflict_score += weight

        context = {
            "enabled": True,
            "ok": False,
            "side": side,
            "confidence": confidence,
            "edge": edge,
            "raw_edge": raw_edge,
            "directional_score": round(directional_score, 5),
            "scaled_directional_score": round(scaled_directional_score, 5),
            "buy_score": buy_score,
            "sell_score": sell_score,
            "reason": "",
            "confirmations": confirmations,
            "conflicts": conflicts,
            "confirmation_score": round(confirmation_score, 2),
            "conflict_score": round(conflict_score, 2),
            "metrics": metrics,
            "weights": {
                name: weight
                for name, (_value, weight) in weighted.items()
            },
            "errors": errors,
        }
        context["ok"] = _decision_ok(context, chart_side)
        context["reason"] = _decision_reason(context, chart_side)
        return _cache_set(symbol, context)

    except Exception as exc:
        log_warning(f"{symbol} data confirmation error: {exc}")
        context.update({
            "ok": bool(getattr(config, "DATA_CONFIRMATION_FAIL_OPEN", False)),
            "side": "ERROR",
            "reason": f"DATA_CONFIRMATION_ERROR:{exc}",
            "errors": [str(exc)],
        })
        return context


def _decision_ok(context, chart_side):
    if not context.get("enabled"):
        return True

    if context.get("side") != chart_side:
        return False

    if context.get("confidence", 0) < getattr(config, "DATA_CONFIRMATION_MIN_CONFIDENCE", 60):
        return False

    if context.get("edge", 0) < getattr(config, "DATA_CONFIRMATION_MIN_EDGE", 12):
        return False

    confirmations = len(context.get("confirmations", []))
    confirmation_score = context.get("confirmation_score", 0)
    min_confirmations = getattr(config, "DATA_CONFIRMATION_MIN_CONFIRMATIONS", 3)
    min_confirmation_score = getattr(
        config,
        "DATA_CONFIRMATION_MIN_CONFIRMATION_SCORE",
        2.40,
    )

    if (
        confirmations < min_confirmations
        and confirmation_score < min_confirmation_score
    ):
        return False

    if len(context.get("conflicts", [])) > getattr(config, "DATA_CONFIRMATION_MAX_CONFLICTS", 1):
        return False

    if context.get("conflict_score", 0) > getattr(config, "DATA_CONFIRMATION_MAX_CONFLICT_SCORE", 1.25):
        return False

    return True


def _decision_reason(context, chart_side):
    if not context.get("enabled"):
        return "DATA_CONFIRMATION_DISABLED"

    side = context.get("side")

    if side != chart_side:
        return f"DATA_SIDE_MISMATCH chart={chart_side} data={side}"

    confidence = context.get("confidence", 0)
    min_confidence = getattr(config, "DATA_CONFIRMATION_MIN_CONFIDENCE", 60)

    if confidence < min_confidence:
        return f"DATA_CONFIDENCE_LOW {confidence} < {min_confidence}"

    edge = context.get("edge", 0)
    min_edge = getattr(config, "DATA_CONFIRMATION_MIN_EDGE", 12)

    if edge < min_edge:
        return f"DATA_EDGE_LOW {edge} < {min_edge}"

    confirmations = len(context.get("confirmations", []))
    min_confirmations = getattr(config, "DATA_CONFIRMATION_MIN_CONFIRMATIONS", 3)
    confirmation_score = context.get("confirmation_score", 0)
    min_confirmation_score = getattr(
        config,
        "DATA_CONFIRMATION_MIN_CONFIRMATION_SCORE",
        2.40,
    )

    if (
        confirmations < min_confirmations
        and confirmation_score < min_confirmation_score
    ):
        return (
            f"DATA_CONFIRMATIONS_LOW count={confirmations}/{min_confirmations} "
            f"score={confirmation_score}/{min_confirmation_score}"
        )

    conflicts = len(context.get("conflicts", []))
    max_conflicts = getattr(config, "DATA_CONFIRMATION_MAX_CONFLICTS", 1)

    if conflicts > max_conflicts:
        return f"DATA_CONFLICTS_HIGH {conflicts} > {max_conflicts}"

    conflict_score = context.get("conflict_score", 0)
    max_conflict_score = getattr(config, "DATA_CONFIRMATION_MAX_CONFLICT_SCORE", 1.25)

    if conflict_score > max_conflict_score:
        return f"DATA_CONFLICT_SCORE_HIGH {conflict_score} > {max_conflict_score}"

    return (
        f"DATA_{side}_CONFIRMED confidence={confidence} "
        f"edge={edge} confirmations={confirmations} "
        f"confirmation_score={confirmation_score}"
    )
