import config
from logger import log_info, log_error, log_warning


def score_to_confidence(score, max_score=None):
    if score <= 0:
        return 0

    if max_score is None:
        max_score = get_config_float("LONG_TERM_CONFIDENCE_MAX_SCORE", 42)

    return round(min((score / max(max_score, 1)) * 100, 100), 2)


def get_config_float(name, default):
    try:
        return float(getattr(config, name, default))
    except (TypeError, ValueError):
        return default


def get_config_int(name, default):
    try:
        return int(getattr(config, name, default))
    except (TypeError, ValueError):
        return default


def latest_closed(df):
    return df.iloc[-2] if len(df) > 1 else df.iloc[-1]


def previous_closed(df):
    return df.iloc[-3] if len(df) > 2 else df.iloc[-2]


def pct_distance(a, b):
    if not b:
        return 0

    return abs(a - b) / b * 100


def add_score(score, condition, points):
    return score + points if condition else score


def _safe_float(value, default=0):
    try:
        result = float(value)

        if result != result:
            return default

        return result
    except (TypeError, ValueError):
        return default


def get_structure_stop_loss(df, side):
    try:
        candle = latest_closed(df)
        atr = candle["atr"]

        if atr <= 0:
            return None

        if side == "BUY":
            swing_low = df["low"].iloc[-20:-1].min()
            return swing_low - (atr * 1.5)

        swing_high = df["high"].iloc[-20:-1].max()
        return swing_high + (atr * 1.5)

    except Exception as e:
        log_error(f"STRUCTURE REFERENCE ERROR: {e}")
        return None


def detect_market_structure(df):
    try:
        recent_high = df["high"].iloc[-30:-5].max()
        recent_low = df["low"].iloc[-30:-5].min()
        prev_high = df["high"].iloc[-60:-30].max()
        prev_low = df["low"].iloc[-60:-30].min()
        close = latest_closed(df)["close"]

        return {
            "bullish_structure": recent_high > prev_high and recent_low > prev_low,
            "bearish_structure": recent_high < prev_high and recent_low < prev_low,
            "bullish_breakout": close > recent_high,
            "bearish_breakdown": close < recent_low,
        }

    except Exception:
        return {
            "bullish_structure": False,
            "bearish_structure": False,
            "bullish_breakout": False,
            "bearish_breakdown": False,
        }


def _level_tolerance(df):
    latest = latest_closed(df)
    price = latest["close"]
    atr = latest["atr"] if "atr" in latest.index else 0
    pct_tolerance = price * (
        get_config_float("LONG_TERM_SR_TOLERANCE_PCT", 1.0) / 100
    )
    atr_tolerance = atr * get_config_float("LONG_TERM_SR_ATR_TOLERANCE", 0.75)

    return max(pct_tolerance, atr_tolerance, price * 0.002)


def _collect_pivot_levels(df, side, label, timeframe_weight):
    lookback = get_config_int("LONG_TERM_SR_LOOKBACK", 160)
    min_touches = get_config_int("LONG_TERM_SR_MIN_TOUCHES", 2)
    data = df.tail(lookback).copy()

    if len(data) < 20:
        return []

    tolerance = _level_tolerance(data)
    swing = 3
    column = "low" if side == "BUY" else "high"
    levels = []

    for pos in range(swing, len(data) - swing):
        candle = data.iloc[pos]
        window = data.iloc[pos - swing:pos + swing + 1]
        level = candle[column]

        if side == "BUY" and level > window["low"].min():
            continue

        if side == "SELL" and level < window["high"].max():
            continue

        touches = int((abs(data[column] - level) <= tolerance).sum())

        if touches < min_touches:
            continue

        recency_score = pos / len(data)
        volume_score = 0

        if (
            "volume_sma" in data.columns
            and candle["volume"] > candle["volume_sma"]
        ):
            volume_score = 0.5

        levels.append({
            "level": float(level),
            "score": touches * timeframe_weight + recency_score + volume_score,
            "touches": touches,
            "source": f"{label}_pivot",
        })

    return levels


def _collect_ema_levels(df, side, label, timeframe_weight):
    latest = latest_closed(df)
    levels = []

    for ema_name, bonus in (("ema50", 0.75), ("ema200", 1.25)):
        if ema_name not in latest.index:
            continue

        level = float(latest[ema_name])
        close = float(latest["close"])

        if side == "BUY" and level >= close:
            continue

        if side == "SELL" and level <= close:
            continue

        levels.append({
            "level": level,
            "score": timeframe_weight + bonus,
            "touches": 0,
            "source": f"{label}_{ema_name}",
        })

    return levels


def _collect_range_levels(df, side, label, timeframe_weight):
    lookback = get_config_int("LONG_TERM_SR_LOOKBACK", 160)
    data = df.iloc[:-1].tail(lookback).copy() if len(df) > 1 else df.tail(lookback)

    if len(data) < 20:
        return []

    tolerance = _level_tolerance(data)
    column = "low" if side == "BUY" else "high"
    levels = []

    for window in (20, 50, 100):
        if len(data) < window:
            continue

        recent = data.tail(window)
        level = recent[column].min() if side == "BUY" else recent[column].max()
        touches = int((abs(data[column] - level) <= tolerance).sum())
        score = timeframe_weight + min(window / 100, 1) + (touches * 0.25)

        levels.append({
            "level": float(level),
            "score": round(score, 2),
            "touches": touches,
            "source": f"{label}_{window}_range",
        })

    return levels


def _dedupe_levels(levels, tolerance):
    deduped = []

    for level in sorted(levels, key=lambda item: item["score"], reverse=True):
        if any(abs(level["level"] - item["level"]) <= tolerance for item in deduped):
            continue

        deduped.append(level)

    return deduped


def _take_profit_buffer(entry_price, trend_df, confirm_df):
    pct_buffer = entry_price * (
        get_config_float("STRUCTURE_TP_BUFFER_PCT", 0.15) / 100
    )
    atr_values = []

    for df in (confirm_df, trend_df):
        try:
            atr = latest_closed(df)["atr"]

            if atr > 0:
                atr_values.append(float(atr))
        except Exception:
            continue

    atr_buffer = 0

    if atr_values:
        atr_buffer = min(atr_values) * get_config_float(
            "STRUCTURE_TP_ATR_BUFFER_MULT",
            0.25
        )

    return max(pct_buffer, atr_buffer, entry_price * 0.0005)


def find_structure_take_profit(side, entry_price, trend_df, confirm_df, leverage=None):
    leverage_to_use = leverage or config.LEVERAGE
    target_side = "SELL" if side == "BUY" else "BUY"
    min_roi = get_config_float("STRUCTURE_TP_MIN_ROI", 8)
    max_roi = get_config_float("STRUCTURE_TP_MAX_ROI", 120)
    min_score = get_config_float("STRUCTURE_TP_MIN_SCORE", 2.0)
    buffer = _take_profit_buffer(entry_price, trend_df, confirm_df)

    candidates = []
    candidates.extend(_collect_pivot_levels(trend_df, target_side, "1d", 2.0))
    candidates.extend(_collect_pivot_levels(confirm_df, target_side, "4h", 1.25))
    candidates.extend(_collect_range_levels(trend_df, target_side, "1d", 2.0))
    candidates.extend(_collect_range_levels(confirm_df, target_side, "4h", 1.25))
    candidates.extend(_collect_ema_levels(trend_df, target_side, "1d", 2.0))
    candidates.extend(_collect_ema_levels(confirm_df, target_side, "4h", 1.25))

    tolerance = max(_level_tolerance(trend_df), _level_tolerance(confirm_df))
    candidates = _dedupe_levels(candidates, tolerance)
    valid = []

    for candidate in candidates:
        level = candidate["level"]

        if candidate["score"] < min_score:
            continue

        if side == "BUY":
            if level <= entry_price:
                continue

            target_price = level - buffer

            if target_price <= entry_price:
                continue

            roi = ((target_price - entry_price) / entry_price) * leverage_to_use * 100
        else:
            if level >= entry_price:
                continue

            target_price = level + buffer

            if target_price >= entry_price:
                continue

            roi = ((entry_price - target_price) / entry_price) * leverage_to_use * 100

        if roi < min_roi or roi > max_roi:
            continue

        item = candidate.copy()
        item["target_price"] = float(target_price)
        item["raw_level"] = float(level)
        item["target_roi"] = round(float(roi), 2)
        item["buffer"] = round(float(buffer), 8)
        valid.append(item)

    if not valid:
        return None

    valid.sort(key=lambda item: (item["target_roi"], -item["score"]))
    return valid[0]


def validate_structure_take_profit(side, entry_price, trend_df, confirm_df, leverage=None):
    target = find_structure_take_profit(
        side,
        entry_price,
        trend_df,
        confirm_df,
        leverage=leverage
    )

    if target:
        return True, target

    return False, {
        "reason": "NO VALID STRUCTURE TP LEVEL FOUND"
    }


def find_nearest_profit_room_level(side, entry_price, trend_df, confirm_df, leverage=None):
    if entry_price <= 0:
        return None

    leverage_to_use = leverage or config.LEVERAGE
    target_side = "SELL" if side == "BUY" else "BUY"
    min_score = get_config_float(
        "ENTRY_TP_ROOM_MIN_LEVEL_SCORE",
        get_config_float("STRUCTURE_TP_MIN_SCORE", 2.0)
    )
    buffer = _take_profit_buffer(entry_price, trend_df, confirm_df)

    candidates = []
    candidates.extend(_collect_pivot_levels(trend_df, target_side, "1d", 2.0))
    candidates.extend(_collect_pivot_levels(confirm_df, target_side, "4h", 1.25))
    candidates.extend(_collect_range_levels(trend_df, target_side, "1d", 2.0))
    candidates.extend(_collect_range_levels(confirm_df, target_side, "4h", 1.25))
    candidates.extend(_collect_ema_levels(trend_df, target_side, "1d", 2.0))
    candidates.extend(_collect_ema_levels(confirm_df, target_side, "4h", 1.25))

    tolerance = max(_level_tolerance(trend_df), _level_tolerance(confirm_df))
    candidates = _dedupe_levels(candidates, tolerance)
    valid = []

    for candidate in candidates:
        level = float(candidate["level"])

        if candidate["score"] < min_score:
            continue

        if side == "BUY":
            if level <= entry_price:
                continue

            target_price = level - buffer
            raw_roi = ((level - entry_price) / entry_price) * leverage_to_use * 100
            target_roi = (
                ((target_price - entry_price) / entry_price) *
                leverage_to_use *
                100
            )
        else:
            if level >= entry_price:
                continue

            target_price = level + buffer
            raw_roi = ((entry_price - level) / entry_price) * leverage_to_use * 100
            target_roi = (
                ((entry_price - target_price) / entry_price) *
                leverage_to_use *
                100
            )

        item = candidate.copy()
        item["raw_level"] = level
        item["target_price"] = float(target_price)
        item["raw_level_roi"] = round(float(raw_roi), 2)
        item["target_roi"] = round(float(target_roi), 2)
        item["buffer"] = round(float(buffer), 8)
        valid.append(item)

    if not valid:
        return None

    valid.sort(key=lambda item: (item["raw_level_roi"], -item["score"]))
    return valid[0]


def validate_entry_profit_room(
    side,
    entry_price,
    trend_df,
    confirm_df,
    leverage=None,
    min_roi_override=None
):
    if not getattr(config, "ENTRY_TP_ROOM_CHECK_ENABLED", True):
        return True, {"reason": "ENTRY_TP_ROOM_CHECK_DISABLED"}

    min_roi = (
        min_roi_override
        if min_roi_override is not None
        else get_config_float(
            "ENTRY_MIN_TP_ROOM_ROI",
            get_config_float("STRUCTURE_TP_MIN_ROI", 8)
        )
    )
    level = find_nearest_profit_room_level(
        side,
        entry_price,
        trend_df,
        confirm_df,
        leverage=leverage
    )
    label = "RESISTANCE" if side == "BUY" else "SUPPORT"

    if not level:
        if getattr(config, "ENTRY_TP_ROOM_BLOCK_IF_NO_LEVEL", False):
            return False, {
                "reason": f"NO {label} PROFIT ROOM LEVEL FOUND"
            }

        return True, {
            "reason": f"NO CLEAR {label} FOUND; PROFIT ROOM ALLOWED"
        }

    if level["target_roi"] < min_roi:
        return False, {
            "reason": (
                f"TOO CLOSE TO {label} | "
                f"ROOM={level['target_roi']}% < MIN={min_roi}%"
            ),
            **level
        }

    return True, {
        "reason": "ENTRY_PROFIT_ROOM_OK",
        **level
    }


def _closed_data(df, lookback=None):
    data = df.iloc[:-1].copy() if len(df) > 1 else df.copy()

    if lookback:
        data = data.tail(lookback)

    return data


def _body(candle):
    return abs(float(candle["close"]) - float(candle["open"]))


def _is_bullish(candle):
    return candle["close"] > candle["open"]


def _is_bearish(candle):
    return candle["close"] < candle["open"]


def _candle_atr(candle):
    try:
        return max(float(candle["atr"]), 1e-10)
    except Exception:
        return max(float(candle["high"] - candle["low"]), 1e-10)


def _average_range(df, period=14):
    data = _closed_data(df, period)

    if len(data) == 0:
        return 0

    ranges = data["high"] - data["low"]
    value = ranges.mean()

    return max(float(value), 1e-10)


def _close_position(candle):
    high = float(candle["high"])
    low = float(candle["low"])
    close = float(candle["close"])
    candle_range = high - low

    if candle_range <= 0:
        return 0.5

    return (close - low) / candle_range


def _adverse_zone(side, entry_price, leverage=None):
    leverage_to_use = leverage or config.LEVERAGE
    max_adverse_roi = abs(get_config_float("LONG_TERM_MAX_ADVERSE_ROI", 50))
    max_price_move = (max_adverse_roi / max(leverage_to_use, 1)) / 100

    if side == "BUY":
        return entry_price * (1 - max_price_move), entry_price

    return entry_price, entry_price * (1 + max_price_move)


def _live_entry_timeframe_check(side, df, mark_price, label):
    data = _closed_data(df)
    lookback = get_config_int("LIVE_ENTRY_STRUCTURE_LOOKBACK", 12)

    if len(data) < lookback + 2:
        return {
            "label": label,
            "block": False,
            "structure_break": False,
            "opposite_reversal": False,
            "ema_wrong_side": False,
            "ema_chase": False,
            "close_chase": False,
            "ema20": None,
            "ema_distance_pct": 0,
            "ema_chase_atr": 0,
            "close_position": 0,
            "body_atr": 0,
            "mark_price": round(float(mark_price), 8) if mark_price else None,
            "latest_close": None,
            "reason": "INSUFFICIENT_DATA",
        }

    latest = data.iloc[-1]
    previous = data.iloc[-lookback - 1:-1]
    atr = max(_safe_float(_average_range(df, 14)), 1e-10)
    structure_buffer = atr * get_config_float(
        "LIVE_ENTRY_STRUCTURE_BUFFER_ATR",
        0.08
    )
    retrace_atr = get_config_float("MAX_LIVE_ENTRY_RETRACE_ATR", 0.20)
    min_body_atr = get_config_float("LIVE_ENTRY_MIN_REVERSAL_BODY_ATR", 0.35)
    close_pos_limit = get_config_float("LIVE_ENTRY_REVERSAL_CLOSE_POSITION", 0.30)
    max_chase_atr = get_config_float("MAX_LIVE_ENTRY_CHASE_ATR", 0.25)
    ema_tolerance_pct = get_config_float("LIVE_ENTRY_EMA_TOLERANCE_PCT", 0.08)
    max_close_position = get_config_float("MAX_LIVE_ENTRY_CLOSE_POSITION", 0.88)
    close_position = _close_position(latest)
    body_atr = _body(latest) / atr
    ema20 = _safe_float(latest.get("ema20"))
    ema_distance_pct = pct_distance(mark_price, ema20) if ema20 else 0
    ema_chase_atr = abs(mark_price - ema20) / atr if ema20 else 0
    ema_tolerance = ema20 * (ema_tolerance_pct / 100) if ema20 else 0
    ema_wrong_side = False
    ema_chase = False
    close_chase = False

    if side == "BUY":
        recent_low = float(previous["low"].min())
        structure_break = mark_price < recent_low - structure_buffer
        opposite_reversal = (
            _is_bearish(latest)
            and body_atr >= min_body_atr
            and close_position <= close_pos_limit
            and mark_price <= float(latest["close"]) - (atr * retrace_atr)
        )
        if ema20:
            ema_wrong_side = mark_price < ema20 - ema_tolerance
            ema_chase = (
                max_chase_atr > 0 and
                mark_price > ema20 and
                ema_chase_atr > max_chase_atr
            )
        close_chase = (
            max_close_position > 0 and
            close_position > max_close_position
        )
        reason = "BUY_GUARD"
    else:
        recent_high = float(previous["high"].max())
        structure_break = mark_price > recent_high + structure_buffer
        opposite_reversal = (
            _is_bullish(latest)
            and body_atr >= min_body_atr
            and close_position >= 1 - close_pos_limit
            and mark_price >= float(latest["close"]) + (atr * retrace_atr)
        )
        if ema20:
            ema_wrong_side = mark_price > ema20 + ema_tolerance
            ema_chase = (
                max_chase_atr > 0 and
                mark_price < ema20 and
                ema_chase_atr > max_chase_atr
            )
        close_chase = (
            max_close_position > 0 and
            close_position < 1 - max_close_position
        )
        reason = "SELL_GUARD"

    return {
        "label": label,
        "block": structure_break or opposite_reversal or ema_wrong_side or ema_chase,
        "structure_break": structure_break,
        "opposite_reversal": opposite_reversal,
        "ema_wrong_side": ema_wrong_side,
        "ema_chase": ema_chase,
        "close_chase": close_chase,
        "ema20": round(float(ema20), 8) if ema20 else None,
        "ema_distance_pct": round(float(ema_distance_pct), 3),
        "ema_chase_atr": round(float(ema_chase_atr), 2),
        "close_position": round(float(close_position), 2),
        "body_atr": round(float(body_atr), 2),
        "mark_price": round(float(mark_price), 8),
        "latest_close": round(float(latest["close"]), 8),
        "reason": reason,
    }


def validate_live_entry_guard(side, fast_df, slow_df, mark_price):
    if not config.LIVE_ENTRY_CONFIRMATION_ENABLED:
        return True, {"reason": "LIVE_ENTRY_GUARD_DISABLED"}

    if fast_df is None or slow_df is None or mark_price is None:
        if config.LIVE_ENTRY_REQUIRE_DATA:
            return False, {"reason": "LIVE_ENTRY_GUARD_DATA_UNAVAILABLE"}

        return True, {"reason": "LIVE_ENTRY_GUARD_DATA_UNAVAILABLE_ALLOWED"}

    fast = _live_entry_timeframe_check(
        side,
        fast_df,
        mark_price,
        config.LIVE_ENTRY_FAST_TIMEFRAME
    )
    slow = _live_entry_timeframe_check(
        side,
        slow_df,
        mark_price,
        config.LIVE_ENTRY_SLOW_TIMEFRAME
    )
    structure_break = fast["structure_break"] or slow["structure_break"]
    dual_reversal = fast["opposite_reversal"] and slow["opposite_reversal"]
    dual_ema_wrong_side = fast["ema_wrong_side"] and slow["ema_wrong_side"]
    dual_ema_chase = fast["ema_chase"] and slow["ema_chase"]
    dual_close_chase = fast["close_chase"] and slow["close_chase"]
    live_ema_block = (
        dual_ema_wrong_side or
        dual_ema_chase or
        (dual_close_chase and (fast["ema_chase"] or slow["ema_chase"]))
    )

    if structure_break or dual_reversal or live_ema_block:
        if structure_break:
            reason = "OPPOSITE_STRUCTURE_BREAK"
        elif dual_reversal:
            reason = "DUAL_OPPOSITE_REVERSAL"
        elif dual_ema_wrong_side:
            reason = "DUAL_LIVE_EMA_WRONG_SIDE"
        elif dual_ema_chase:
            reason = "DUAL_LIVE_EMA_CHASE"
        else:
            reason = "DUAL_LIVE_CLOSE_CHASE"
        return False, {
            "reason": reason,
            "fast": fast,
            "slow": slow,
            "mark_price": mark_price,
        }

    return True, {
        "reason": "LIVE_ENTRY_GUARD_OK",
        "fast": fast,
        "slow": slow,
        "mark_price": mark_price,
    }


def detect_liquidity_sweep(side, df, label):
    if not config.SMC_ENABLED or not config.SMC_SWEEP_ENABLED:
        return None

    lookback = get_config_int("SMC_SWEEP_LOOKBACK", 24)
    max_age = get_config_int("SMC_SWEEP_MAX_AGE", 5)
    data = _closed_data(df)

    if len(data) < lookback + 2:
        return None

    start = max(lookback, len(data) - max_age)
    best = None

    for pos in range(start, len(data)):
        prior = data.iloc[pos - lookback:pos]
        candle = data.iloc[pos]
        atr = _candle_atr(candle)

        if side == "BUY":
            swept_level = prior["low"].min()
            swept = candle["low"] < swept_level and candle["close"] > swept_level
            direction_ok = _is_bullish(candle)
            depth = (swept_level - candle["low"]) / atr
        else:
            swept_level = prior["high"].max()
            swept = candle["high"] > swept_level and candle["close"] < swept_level
            direction_ok = _is_bearish(candle)
            depth = (candle["high"] - swept_level) / atr

        if not swept or not direction_ok:
            continue

        recency = 1 - ((len(data) - 1 - pos) / max(max_age, 1))
        volume_bonus = 0.25 if candle.get("volume", 0) > candle.get("volume_sma", 0) else 0
        score = round(1 + max(depth, 0) + recency + volume_bonus, 2)
        item = {
            "type": "liquidity_sweep",
            "source": label,
            "level": float(swept_level),
            "score": score,
            "age": len(data) - 1 - pos,
        }

        if not best or item["score"] > best["score"]:
            best = item

    return best


def _collect_order_blocks(df, side, label, timeframe_weight):
    if not config.SMC_ENABLED or not config.SMC_OB_ENABLED:
        return []

    lookback = get_config_int("SMC_OB_LOOKBACK", 120)
    min_displacement = get_config_float("SMC_OB_DISPLACEMENT_ATR", 0.8)
    max_zone_pct = get_config_float("SMC_OB_MAX_ZONE_PCT", 4.0)
    data = _closed_data(df, lookback)
    blocks = []

    if len(data) < 10:
        return blocks

    for pos in range(2, len(data) - 3):
        candle = data.iloc[pos]
        next_window = data.iloc[pos + 1:pos + 4]
        atr = _candle_atr(candle)
        zone_low = float(candle["low"])
        zone_high = float(candle["high"])
        zone_width_pct = ((zone_high - zone_low) / max(zone_high, 1e-10)) * 100

        if zone_width_pct > max_zone_pct:
            continue

        if side == "BUY":
            if not _is_bearish(candle):
                continue

            displacement = (next_window["close"].max() - candle["high"]) / atr
            broke_structure = next_window["high"].max() > candle["high"]

            if displacement < min_displacement or not broke_structure:
                continue

            anchor = zone_high
        else:
            if not _is_bullish(candle):
                continue

            displacement = (candle["low"] - next_window["close"].min()) / atr
            broke_structure = next_window["low"].min() < candle["low"]

            if displacement < min_displacement or not broke_structure:
                continue

            anchor = zone_low

        recency_score = pos / len(data)
        volume_bonus = 0.25 if candle.get("volume", 0) > candle.get("volume_sma", 0) else 0
        score = timeframe_weight + min(max(displacement, 0), 2.5) + recency_score + volume_bonus
        blocks.append({
            "type": "order_block",
            "source": f"{label}_ob",
            "zone_low": zone_low,
            "zone_high": zone_high,
            "level": float(anchor),
            "score": round(score, 2),
            "displacement": round(float(displacement), 2),
        })

    return blocks


def find_order_block_confirmation(side, entry_price, trend_df, confirm_df, leverage=None):
    zone_min, zone_max = _adverse_zone(side, entry_price, leverage)
    candidates = []
    candidates.extend(_collect_order_blocks(trend_df, side, "1d", 2.0))
    candidates.extend(_collect_order_blocks(confirm_df, side, "4h", 1.25))
    valid = []

    for candidate in candidates:
        if side == "BUY":
            if candidate["level"] >= entry_price:
                continue

            if not (zone_min <= candidate["level"] <= zone_max):
                continue
        else:
            if candidate["level"] <= entry_price:
                continue

            if not (zone_min <= candidate["level"] <= zone_max):
                continue

        distance_pct = pct_distance(entry_price, candidate["level"])
        item = candidate.copy()
        item["distance_pct"] = round(distance_pct, 2)
        valid.append(item)

    if not valid:
        return None

    valid.sort(key=lambda item: (item["score"], -item["distance_pct"]), reverse=True)
    return valid[0]


def _collect_fvgs(df, label):
    if not config.SMC_ENABLED or not config.SMC_FVG_ENABLED:
        return []

    lookback = get_config_int("SMC_FVG_LOOKBACK", 120)
    min_gap_atr = get_config_float("SMC_FVG_MIN_GAP_ATR", 0.12)
    data = _closed_data(df, lookback)
    fvgs = []

    if len(data) < 5:
        return fvgs

    for pos in range(2, len(data)):
        left = data.iloc[pos - 2]
        right = data.iloc[pos]
        atr = _candle_atr(right)

        if left["high"] < right["low"]:
            gap_low = float(left["high"])
            gap_high = float(right["low"])
            gap_atr = (gap_high - gap_low) / atr
            after = data.iloc[pos + 1:]

            if gap_atr >= min_gap_atr and not (
                len(after) and after["low"].min() <= gap_low
            ):
                fvgs.append({
                    "type": "bullish_fvg",
                    "source": f"{label}_bullish_fvg",
                    "zone_low": gap_low,
                    "zone_high": gap_high,
                    "level": (gap_low + gap_high) / 2,
                    "score": round(1 + min(gap_atr, 2), 2),
                })

        if left["low"] > right["high"]:
            gap_low = float(right["high"])
            gap_high = float(left["low"])
            gap_atr = (gap_high - gap_low) / atr
            after = data.iloc[pos + 1:]

            if gap_atr >= min_gap_atr and not (
                len(after) and after["high"].max() >= gap_high
            ):
                fvgs.append({
                    "type": "bearish_fvg",
                    "source": f"{label}_bearish_fvg",
                    "zone_low": gap_low,
                    "zone_high": gap_high,
                    "level": (gap_low + gap_high) / 2,
                    "score": round(1 + min(gap_atr, 2), 2),
                })

    return fvgs


def _estimated_tp_price(side, entry_price, trend_df, confirm_df):
    if config.STATIC_TP_ENABLED:
        roi = config.STATIC_TP_ROI
        if side == "BUY":
            return entry_price * (1 + (roi / config.LEVERAGE) / 100)

        return entry_price * (1 - (roi / config.LEVERAGE) / 100)

    target = find_structure_take_profit(
        side,
        entry_price,
        trend_df,
        confirm_df,
        leverage=config.LEVERAGE
    )

    if target:
        return target["target_price"]

    roi = config.STRUCTURE_TP_FALLBACK_ROI

    if side == "BUY":
        return entry_price * (1 + (roi / config.LEVERAGE) / 100)

    return entry_price * (1 - (roi / config.LEVERAGE) / 100)


def find_fvg_confirmation(side, entry_price, trend_df, confirm_df, leverage=None):
    zone_min, zone_max = _adverse_zone(side, entry_price, leverage)
    target_price = _estimated_tp_price(side, entry_price, trend_df, confirm_df)
    fvgs = []
    fvgs.extend(_collect_fvgs(trend_df, "1d"))
    fvgs.extend(_collect_fvgs(confirm_df, "4h"))
    supportive = []
    blocking = []

    for fvg in fvgs:
        if side == "BUY":
            if (
                fvg["type"] == "bullish_fvg"
                and fvg["zone_high"] < entry_price
                and zone_min <= fvg["level"] <= zone_max
            ):
                supportive.append(fvg)

            if (
                fvg["type"] == "bearish_fvg"
                and entry_price < fvg["zone_low"] < target_price
            ):
                blocking.append(fvg)
        else:
            if (
                fvg["type"] == "bearish_fvg"
                and fvg["zone_low"] > entry_price
                and zone_min <= fvg["level"] <= zone_max
            ):
                supportive.append(fvg)

            if (
                fvg["type"] == "bullish_fvg"
                and target_price < fvg["zone_high"] < entry_price
            ):
                blocking.append(fvg)

    supportive.sort(key=lambda item: item["score"], reverse=True)
    blocking.sort(key=lambda item: item["score"], reverse=True)

    return {
        "supportive": supportive[0] if supportive else None,
        "blocking": blocking[0] if blocking else None,
        "target_price": target_price,
    }


def _smc_context_score(side, trend_df, confirm_df, entry_df):
    if not config.SMC_ENABLED:
        return 0, {"enabled": False}

    entry_price = latest_closed(entry_df)["close"]
    score = 0
    context = {
        "enabled": True,
        "liquidity_sweep": None,
        "order_block": None,
        "fvg_support": None,
        "fvg_block": None,
    }

    entry_sweep = detect_liquidity_sweep(side, entry_df, "1h")
    confirm_sweep = detect_liquidity_sweep(side, confirm_df, "4h")
    opposite_side = "SELL" if side == "BUY" else "BUY"
    opposite_sweep = detect_liquidity_sweep(opposite_side, entry_df, "1h")

    if entry_sweep:
        context["liquidity_sweep"] = entry_sweep
        score += min(entry_sweep["score"], 2.0)
    elif confirm_sweep:
        context["liquidity_sweep"] = confirm_sweep
        score += min(confirm_sweep["score"], 1.25)

    if opposite_sweep:
        score -= min(opposite_sweep["score"], 1.25)

    order_block = find_order_block_confirmation(
        side,
        entry_price,
        trend_df,
        confirm_df,
        leverage=config.LEVERAGE
    )

    if order_block:
        context["order_block"] = order_block
        score += min(order_block["score"] / 2, 2.0)

    fvg = find_fvg_confirmation(
        side,
        entry_price,
        trend_df,
        confirm_df,
        leverage=config.LEVERAGE
    )
    context["fvg_support"] = fvg["supportive"]
    context["fvg_block"] = fvg["blocking"]

    if fvg["supportive"]:
        score += min(fvg["supportive"]["score"] / 2, 1.0)

    if fvg["blocking"]:
        score -= get_config_float("SMC_TP_PATH_BLOCK_PENALTY", 1.5)

    return round(score, 2), context


def find_adverse_zone_level(side, entry_price, trend_df, confirm_df, leverage=None):
    leverage_to_use = leverage or config.LEVERAGE
    max_adverse_roi = abs(get_config_float("LONG_TERM_MAX_ADVERSE_ROI", 50))
    max_price_move = (max_adverse_roi / max(leverage_to_use, 1)) / 100

    if side == "BUY":
        zone_min = entry_price * (1 - max_price_move)
        zone_max = entry_price
    else:
        zone_min = entry_price
        zone_max = entry_price * (1 + max_price_move)

    candidates = []
    candidates.extend(_collect_pivot_levels(trend_df, side, "1d", 2.0))
    candidates.extend(_collect_pivot_levels(confirm_df, side, "4h", 1.25))
    candidates.extend(_collect_ema_levels(trend_df, side, "1d", 2.0))
    candidates.extend(_collect_ema_levels(confirm_df, side, "4h", 1.25))

    tolerance = max(_level_tolerance(trend_df), _level_tolerance(confirm_df))
    candidates = _dedupe_levels(candidates, tolerance)
    valid = []

    for candidate in candidates:
        level = candidate["level"]

        if not (zone_min <= level <= zone_max):
            continue

        if side == "BUY" and level >= entry_price:
            continue

        if side == "SELL" and level <= entry_price:
            continue

        if side == "BUY":
            adverse_roi = ((level - entry_price) / entry_price) * leverage_to_use * 100
        else:
            adverse_roi = ((entry_price - level) / entry_price) * leverage_to_use * 100

        proximity_score = 1 - min(abs(adverse_roi) / max_adverse_roi, 1)
        item = candidate.copy()
        item["adverse_roi"] = round(adverse_roi, 2)
        item["score"] = round(candidate["score"] + proximity_score, 2)
        item["zone_min"] = zone_min
        item["zone_max"] = zone_max
        valid.append(item)

    if not valid:
        return None

    valid.sort(key=lambda item: (item["score"], -abs(item["adverse_roi"])), reverse=True)
    best = valid[0]

    if best["score"] < get_config_float("LONG_TERM_SR_MIN_SCORE", 2.5):
        return None

    return best


def validate_adverse_zone_level(side, entry_price, trend_df, confirm_df, leverage=None):
    if not getattr(config, "LONG_TERM_ADVERSE_ZONE_CHECK_ENABLED", True):
        label = "support" if side == "BUY" else "resistance"
        return True, {
            "reason": f"{label.upper()} ADVERSE-ZONE CHECK DISABLED",
            "level": float(entry_price),
            "adverse_roi": 0,
            "source": "disabled",
            "score": 0,
            "level_check_disabled": True,
        }

    level = find_adverse_zone_level(
        side,
        entry_price,
        trend_df,
        confirm_df,
        leverage=leverage
    )

    if level:
        return True, level

    zone_roi = get_config_float("LONG_TERM_MAX_ADVERSE_ROI", 50)
    label = "support" if side == "BUY" else "resistance"
    return False, {
        "reason": f"NO STRONG {label.upper()} WITHIN -{zone_roi:.0f}% ROI ZONE"
    }


def _dca_structure_tolerance(current_price, entry_df, leverage=None):
    leverage_to_use = leverage or config.LEVERAGE
    atr_tolerance = _average_range(entry_df, 14) * get_config_float(
        "DCA_STRUCTURE_MAX_DISTANCE_ATR",
        0.6
    )
    roi_tolerance = current_price * (
        get_config_float("DCA_STRUCTURE_MAX_DISTANCE_ROI", 6) /
        max(leverage_to_use, 1) /
        100
    )

    return max(atr_tolerance, roi_tolerance, current_price * 0.001)


def _normalise_dca_level(side, current_price, candidate, tolerance, leverage=None):
    leverage_to_use = leverage or config.LEVERAGE
    level = float(candidate.get("level", 0) or 0)

    if level <= 0:
        return None

    zone_low = float(candidate.get("zone_low", level) or level)
    zone_high = float(candidate.get("zone_high", level) or level)

    if zone_low > zone_high:
        zone_low, zone_high = zone_high, zone_low

    if side == "BUY":
        if current_price < zone_low - tolerance:
            return None

        distance = max(0, current_price - zone_high)
    else:
        if current_price > zone_high + tolerance:
            return None

        distance = max(0, zone_low - current_price)

    if distance > tolerance:
        return None

    proximity_score = 1 - min(distance / max(tolerance, 1e-10), 1)
    distance_roi = (distance / current_price) * leverage_to_use * 100
    item = candidate.copy()
    item["level"] = level
    item["zone_low"] = zone_low
    item["zone_high"] = zone_high
    item["distance"] = round(float(distance), 8)
    item["distance_roi"] = round(float(distance_roi), 2)
    item["score"] = round(float(candidate.get("score", 0)) + proximity_score, 2)
    item["max_distance"] = round(float(tolerance), 8)

    return item


def _collect_dca_structure_candidates(side, trend_df, confirm_df):
    candidates = []

    for item in _collect_pivot_levels(trend_df, side, "1d", 2.0):
        candidate = item.copy()
        candidate["kind"] = "pivot"
        candidates.append(candidate)

    for item in _collect_pivot_levels(confirm_df, side, "4h", 1.25):
        candidate = item.copy()
        candidate["kind"] = "pivot"
        candidates.append(candidate)

    for item in _collect_range_levels(trend_df, side, "1d", 2.0):
        candidate = item.copy()
        candidate["kind"] = "range"
        candidates.append(candidate)

    for item in _collect_range_levels(confirm_df, side, "4h", 1.25):
        candidate = item.copy()
        candidate["kind"] = "range"
        candidates.append(candidate)

    for item in _collect_ema_levels(trend_df, side, "1d", 2.0):
        candidate = item.copy()
        candidate["kind"] = "ema"
        candidates.append(candidate)

    for item in _collect_ema_levels(confirm_df, side, "4h", 1.25):
        candidate = item.copy()
        candidate["kind"] = "ema"
        candidates.append(candidate)

    for item in _collect_order_blocks(trend_df, side, "1d", 2.0):
        candidate = item.copy()
        candidate["kind"] = "order_block"
        candidate["score"] = float(candidate.get("score", 0)) + 0.5
        candidates.append(candidate)

    for item in _collect_order_blocks(confirm_df, side, "4h", 1.25):
        candidate = item.copy()
        candidate["kind"] = "order_block"
        candidate["score"] = float(candidate.get("score", 0)) + 0.5
        candidates.append(candidate)

    for fvg in _collect_fvgs(trend_df, "1d"):
        if side == "BUY" and fvg["type"] != "bullish_fvg":
            continue

        if side == "SELL" and fvg["type"] != "bearish_fvg":
            continue

        candidate = fvg.copy()
        candidate["kind"] = "fvg"
        candidate["score"] = float(candidate.get("score", 0)) + 0.25
        candidates.append(candidate)

    for fvg in _collect_fvgs(confirm_df, "4h"):
        if side == "BUY" and fvg["type"] != "bullish_fvg":
            continue

        if side == "SELL" and fvg["type"] != "bearish_fvg":
            continue

        candidate = fvg.copy()
        candidate["kind"] = "fvg"
        candidate["score"] = float(candidate.get("score", 0)) + 0.25
        candidates.append(candidate)

    return candidates


def _level_touched_by_candle(side, candle, level, tolerance):
    zone_low = float(level.get("zone_low", level.get("level")))
    zone_high = float(level.get("zone_high", level.get("level")))

    if side == "BUY":
        return float(candle["low"]) <= zone_high + tolerance

    return float(candle["high"]) >= zone_low - tolerance


def _dca_reaction_confirmation(side, level, entry_df, tolerance):
    if not getattr(config, "DCA_STRUCTURE_REQUIRE_REACTION", True):
        return True, {
            "reaction": "REACTION_NOT_REQUIRED"
        }

    lookback = get_config_int("DCA_STRUCTURE_REACTION_LOOKBACK", 3)
    min_body_atr = get_config_float("DCA_STRUCTURE_REACTION_MIN_BODY_ATR", 0.05)
    close_position_threshold = get_config_float(
        "DCA_STRUCTURE_REACTION_CLOSE_POSITION",
        0.55
    )
    data = _closed_data(entry_df, lookback)

    if len(data) == 0:
        return False, {
            "reason": "DCA_STRUCTURE_REACTION_DATA_UNAVAILABLE"
        }

    best = None

    for _, candle in data.iterrows():
        if not _level_touched_by_candle(side, candle, level, tolerance):
            continue

        atr = _candle_atr(candle)
        body_atr = _body(candle) / atr
        close_position = _close_position(candle)
        close = float(candle["close"])
        zone_low = float(level.get("zone_low", level.get("level")))
        zone_high = float(level.get("zone_high", level.get("level")))

        if side == "BUY":
            reclaimed_level = close >= zone_low
            reaction_ok = (
                reclaimed_level
                and
                close_position >= close_position_threshold
                and (
                    _is_bullish(candle)
                    or body_atr >= min_body_atr
                )
            )
        else:
            reclaimed_level = close <= zone_high
            reaction_ok = (
                reclaimed_level
                and
                close_position <= 1 - close_position_threshold
                and (
                    _is_bearish(candle)
                    or body_atr >= min_body_atr
                )
            )

        item = {
            "close_position": round(float(close_position), 2),
            "body_atr": round(float(body_atr), 2),
            "candle_close": close,
            "reclaimed_level": reclaimed_level,
            "reaction": "OK" if reaction_ok else "WEAK",
        }

        if reaction_ok:
            return True, item

        best = item

    if best:
        best["reason"] = "DCA_STRUCTURE_REACTION_WEAK"
        return False, best

    return False, {
        "reason": "DCA_STRUCTURE_LEVEL_NOT_TOUCHED"
    }


def validate_dca_structure_level(side, current_price, trend_df, confirm_df, entry_df, leverage=None):
    if not getattr(config, "DCA_STRUCTURE_LEVEL_ENABLED", True):
        return True, {
            "reason": "DCA_STRUCTURE_LEVEL_CHECK_DISABLED",
            "level": current_price,
            "source": "disabled",
            "score": 0,
        }

    if current_price <= 0:
        return False, {
            "reason": "DCA_STRUCTURE_INVALID_PRICE"
        }

    min_score = get_config_float("DCA_STRUCTURE_MIN_SCORE", 2.0)
    tolerance = _dca_structure_tolerance(
        current_price,
        entry_df,
        leverage=leverage
    )
    levels = []

    for candidate in _collect_dca_structure_candidates(side, trend_df, confirm_df):
        level = _normalise_dca_level(
            side,
            current_price,
            candidate,
            tolerance,
            leverage=leverage
        )

        if not level:
            continue

        if level["score"] < min_score:
            continue

        levels.append(level)

    if not levels:
        label = "SUPPORT" if side == "BUY" else "RESISTANCE"
        return False, {
            "reason": f"NO DCA {label} LEVEL NEAR CURRENT PRICE"
        }

    levels.sort(key=lambda item: (item["score"], -item["distance"]), reverse=True)
    best = levels[0]
    reaction_ok, reaction = _dca_reaction_confirmation(
        side,
        best,
        entry_df,
        tolerance
    )

    if not reaction_ok:
        return False, {
            "reason": reaction.get("reason", "DCA_STRUCTURE_REACTION_FAILED"),
            **best,
            **reaction,
        }

    return True, {
        "reason": "DCA_STRUCTURE_LEVEL_OK",
        **best,
        **reaction,
    }


def _candle_quality_score(side, candle, max_ema_distance):
    open_price = _safe_float(candle.get("open"))
    high = _safe_float(candle.get("high"))
    low = _safe_float(candle.get("low"))
    close = _safe_float(candle.get("close"))
    atr = max(_safe_float(candle.get("atr")), 1e-10)
    ema20 = _safe_float(candle.get("ema20"))
    volume = _safe_float(candle.get("volume"))
    volume_sma = _safe_float(candle.get("volume_sma"))
    rsi = _safe_float(candle.get("rsi"), 50)
    candle_range = max(high - low, 1e-10)
    body = abs(close - open_price)
    close_position = (close - low) / candle_range
    directional_close = close_position if side == "BUY" else 1 - close_position
    upper_wick = max(high - max(open_price, close), 0)
    lower_wick = max(min(open_price, close) - low, 0)
    rejection_wick = upper_wick if side == "BUY" else lower_wick
    rejection_wick_ratio = rejection_wick / candle_range
    body_ratio = body / candle_range
    directional_body = close - open_price if side == "BUY" else open_price - close
    momentum_atr = directional_body / atr
    candle_atr = candle_range / atr
    volume_mult = volume / volume_sma if volume_sma > 0 else 0
    ema_distance = pct_distance(close, ema20) if ema20 else 0
    chase_atr = abs(close - ema20) / atr if ema20 else 0
    min_body = get_config_float("MIN_SIGNAL_BODY_RATIO", 0.16)
    min_close = get_config_float("MIN_SIGNAL_CLOSE_POSITION", 0.50)
    max_close = get_config_float("MAX_SIGNAL_CLOSE_POSITION", 0.88)
    max_candle_atr = get_config_float("MAX_SIGNAL_CANDLE_ATR", 2.0)
    min_volume_mult = get_config_float("MIN_VOLUME_SMA_MULT", 1.05)
    max_wick_ratio = get_config_float("MAX_SIGNAL_REJECTION_WICK_RATIO", 0.45)
    min_momentum_atr = get_config_float("MIN_SIGNAL_MOMENTUM_ATR", 0.03)
    max_chase_pct = get_config_float("MAX_CHASE_DISTANCE_PCT", max_ema_distance)
    max_late_entry_atr = get_config_float("MAX_LATE_ENTRY_ATR", 2.2)
    late_penalty = get_config_float("LATE_ENTRY_SCORE_PENALTY", 2.0)
    wick_filter = bool(getattr(config, "SIGNAL_WICK_FILTER_ENABLED", True))
    score = 0

    score += 0.5 if body_ratio >= min_body else -0.5
    score += 0.5 if directional_close >= min_close else -0.75
    score += 0.25 if directional_close <= max_close else -0.25

    if wick_filter:
        score += 0.5 if rejection_wick_ratio <= max_wick_ratio else -1.0

    score += 0.25 if candle_atr <= max_candle_atr else -1.0
    score += 0.5 if momentum_atr >= min_momentum_atr else -0.25
    score += 0.5 if volume_mult >= min_volume_mult else -0.25
    score += 0.25 if ema_distance <= max_ema_distance else -0.75

    if max_chase_pct > 0:
        score += 0.25 if ema_distance <= max_chase_pct else -0.5

    if max_late_entry_atr > 0:
        score += 0.25 if chase_atr <= max_late_entry_atr else -late_penalty

    overheat = (
        (side == "BUY" and rsi > get_config_float("BUY_RSI_OVERHEAT", 72)) or
        (side == "SELL" and rsi < get_config_float("SELL_RSI_OVERHEAT", 28))
    )

    if overheat:
        score -= 1.0

    quality_ok = (
        not overheat and
        candle_atr <= max(max_candle_atr * 1.75, max_candle_atr + 1) and
        (
            not wick_filter or
            rejection_wick_ratio <= max(max_wick_ratio * 1.75, 0.80)
        )
    )

    return round(score, 2), {
        "score": round(score, 2),
        "body_ratio": round(float(body_ratio), 3),
        "directional_close": round(float(directional_close), 3),
        "rejection_wick_ratio": round(float(rejection_wick_ratio), 3),
        "volume_mult": round(float(volume_mult), 2),
        "candle_atr": round(float(candle_atr), 2),
        "momentum_atr": round(float(momentum_atr), 3),
        "ema_distance": round(float(ema_distance), 3),
        "chase_atr": round(float(chase_atr), 2),
        "rsi": round(float(rsi), 2),
        "overheat": overheat,
        "quality_ok": quality_ok,
    }


def _market_regime_score(side, trend_df, confirm_df, entry_df):
    trend = latest_closed(trend_df)
    confirm = latest_closed(confirm_df)
    entry = latest_closed(entry_df)
    confirm_structure = detect_market_structure(confirm_df)
    trend_adx = _safe_float(trend.get("adx"))
    confirm_adx = _safe_float(confirm.get("adx"))
    sideways_adx = get_config_float("SIDEWAYS_ADX", 15)
    trending_adx = get_config_float("TRENDING_ADX", 25)
    atr = max(_safe_float(entry.get("atr")), 1e-10)
    entry_close = _safe_float(entry.get("close"))
    entry_ema20 = _safe_float(entry.get("ema20"))
    chase_atr = abs(entry_close - entry_ema20) / atr if entry_ema20 else 0
    max_late_entry_atr = get_config_float("MAX_LATE_ENTRY_ATR", 2.2)

    if side == "BUY":
        trend_aligned = (
            trend["close"] > trend["ema50"] and trend["ema50"] >= trend["ema200"]
        )
        confirm_aligned = (
            confirm["close"] > confirm["ema50"] and confirm["ema20"] >= confirm["ema50"]
        )
        breakout = confirm_structure["bullish_breakout"]
    else:
        trend_aligned = (
            trend["close"] < trend["ema50"] and trend["ema50"] <= trend["ema200"]
        )
        confirm_aligned = (
            confirm["close"] < confirm["ema50"] and confirm["ema20"] <= confirm["ema50"]
        )
        breakout = confirm_structure["bearish_breakdown"]

    if chase_atr > max_late_entry_atr:
        regime = "late_entry"
    elif trend_adx < sideways_adx and confirm_adx < sideways_adx:
        regime = "sideways"
    elif trend_adx >= trending_adx or confirm_adx >= trending_adx:
        regime = "trending"
    elif breakout:
        regime = "breakout"
    else:
        regime = "transition"

    score = 0

    if regime == "late_entry":
        score -= get_config_float("LATE_ENTRY_SCORE_PENALTY", 2.0)
    elif regime == "sideways":
        score += 0.5 if breakout else -1.0
    elif regime == "trending":
        score += 1.0 if trend_aligned and confirm_aligned else -1.25
    elif regime == "breakout":
        score += 0.75 if confirm_aligned else 0
    elif trend_aligned and confirm_aligned:
        score += 0.25

    return round(score, 2), {
        "regime": regime,
        "trend_adx": round(float(trend_adx), 2),
        "confirm_adx": round(float(confirm_adx), 2),
        "trend_aligned": trend_aligned,
        "confirm_aligned": confirm_aligned,
        "breakout": breakout,
        "chase_atr": round(float(chase_atr), 2),
    }


def _ema_gap_score(side, candle):
    enabled = bool(getattr(config, "EMA_GAP_FILTER_ENABLED", True))
    context = {
        "enabled": enabled,
        "score": 0,
    }

    if not enabled:
        context["reason"] = "EMA_GAP_DISABLED"
        return 0, context

    ema20 = _safe_float(candle.get("ema20"))
    ema50 = _safe_float(candle.get("ema50"))
    ema200 = _safe_float(candle.get("ema200"))

    if ema20 <= 0 or ema50 <= 0 or ema200 <= 0:
        context["reason"] = "EMA_GAP_DATA_UNAVAILABLE"
        return 0, context

    gap20_50 = pct_distance(ema20, ema50)
    gap50_200 = pct_distance(ema50, ema200)
    min20_50 = get_config_float("MIN_EMA20_EMA50_GAP_PCT", 0.03)
    min50_200 = get_config_float("MIN_EMA50_EMA200_GAP_PCT", 0.05)
    max20_50 = get_config_float("MAX_EMA20_EMA50_GAP_PCT", 0)
    max50_200 = get_config_float("MAX_EMA50_EMA200_GAP_PCT", 0)
    bonus = get_config_float("EMA_GAP_SCORE_BONUS", 0.75)
    penalty = get_config_float("EMA_GAP_SCORE_PENALTY", 0.5)

    if side == "BUY":
        order_ok = ema20 >= ema50 >= ema200
    else:
        order_ok = ema20 <= ema50 <= ema200

    min_ok = gap20_50 >= min20_50 and gap50_200 >= min50_200
    max20_ok = max20_50 <= 0 or gap20_50 <= max20_50
    max50_ok = max50_200 <= 0 or gap50_200 <= max50_200
    max_ok = max20_ok and max50_ok

    if order_ok and min_ok and max_ok:
        score = bonus
        reason = "EMA_GAP_HEALTHY"
    elif not order_ok:
        score = -penalty
        reason = "EMA_ORDER_NOT_ALIGNED"
    elif not min_ok:
        score = -penalty
        reason = "EMA_GAP_TOO_TIGHT"
    else:
        score = -(penalty / 2)
        reason = "EMA_GAP_TOO_WIDE"

    score = round(float(score), 2)
    context.update({
        "reason": reason,
        "score": score,
        "order_ok": order_ok,
        "min_ok": min_ok,
        "max_ok": max_ok,
        "gap20_50": round(float(gap20_50), 3),
        "gap50_200": round(float(gap50_200), 3),
    })
    return score, context


def _trend_bias_score(side, trend_df):
    trend = latest_closed(trend_df)
    prev = previous_closed(trend_df)
    structure = detect_market_structure(trend_df)
    min_adx = get_config_float("LONG_TERM_MIN_ADX", 14)
    score = 0
    hard_ok = False

    if side == "BUY":
        score = add_score(score, trend["close"] > trend["ema200"], 3)
        score = add_score(score, trend["ema50"] > trend["ema200"], 3)
        score = add_score(score, trend["ema20"] > trend["ema50"], 2)
        score = add_score(score, trend["close"] > trend["ema50"], 2)
        score = add_score(score, trend["ema20"] > prev["ema20"], 1)
        score = add_score(score, structure["bullish_structure"], 2)
        score = add_score(score, structure["bullish_breakout"], 2)
        score = add_score(score, trend["adx"] >= min_adx, 1)
        hard_ok = (
            trend["close"] > trend["ema50"]
            and (trend["ema20"] > trend["ema50"] or trend["ema50"] > trend["ema200"])
        )
    else:
        score = add_score(score, trend["close"] < trend["ema200"], 3)
        score = add_score(score, trend["ema50"] < trend["ema200"], 3)
        score = add_score(score, trend["ema20"] < trend["ema50"], 2)
        score = add_score(score, trend["close"] < trend["ema50"], 2)
        score = add_score(score, trend["ema20"] < prev["ema20"], 1)
        score = add_score(score, structure["bearish_structure"], 2)
        score = add_score(score, structure["bearish_breakdown"], 2)
        score = add_score(score, trend["adx"] >= min_adx, 1)
        hard_ok = (
            trend["close"] < trend["ema50"]
            and (trend["ema20"] < trend["ema50"] or trend["ema50"] < trend["ema200"])
        )

    ema_gap_score, _ = _ema_gap_score(side, trend)
    score += ema_gap_score

    return score, hard_ok


def _confirmation_score(side, confirm_df):
    confirm = latest_closed(confirm_df)
    prev = previous_closed(confirm_df)
    structure = detect_market_structure(confirm_df)
    min_adx = get_config_float("LONG_TERM_MIN_ADX", 14)
    max_ema_distance = get_config_float("MAX_SIGNAL_EMA20_DISTANCE_PCT", 1.2)
    score = 0
    hard_ok = False

    if side == "BUY":
        score = add_score(score, confirm["close"] > confirm["ema50"], 3)
        score = add_score(score, confirm["ema20"] > confirm["ema50"], 2)
        score = add_score(score, confirm["macd"] > confirm["macd_signal"], 2)
        score = add_score(score, 45 <= confirm["rsi"] <= 72, 2)
        score = add_score(score, confirm["adx"] >= min_adx, 1)
        score = add_score(score, structure["bullish_breakout"], 2)
        score = add_score(score, confirm["close"] > prev["high"], 1)
        score = add_score(score, confirm["volume"] > confirm["volume_sma"], 1)
        hard_ok = (
            confirm["close"] > confirm["ema20"]
            and (confirm["macd"] > confirm["macd_signal"] or confirm["rsi"] > 50)
        )
    else:
        score = add_score(score, confirm["close"] < confirm["ema50"], 3)
        score = add_score(score, confirm["ema20"] < confirm["ema50"], 2)
        score = add_score(score, confirm["macd"] < confirm["macd_signal"], 2)
        score = add_score(score, 28 <= confirm["rsi"] <= 55, 2)
        score = add_score(score, confirm["adx"] >= min_adx, 1)
        score = add_score(score, structure["bearish_breakdown"], 2)
        score = add_score(score, confirm["close"] < prev["low"], 1)
        score = add_score(score, confirm["volume"] > confirm["volume_sma"], 1)
        hard_ok = (
            confirm["close"] < confirm["ema20"]
            and (confirm["macd"] < confirm["macd_signal"] or confirm["rsi"] < 50)
        )

    quality_score, quality = _candle_quality_score(
        side,
        confirm,
        max_ema_distance
    )
    ema_gap_score, ema_gap = _ema_gap_score(side, confirm)
    score += quality_score
    score += ema_gap_score
    quality["ema_gap"] = ema_gap
    quality["ema_gap_score"] = ema_gap_score
    quality["score"] = round(float(quality.get("score", 0)) + ema_gap_score, 2)
    hard_ok = hard_ok and quality["quality_ok"]

    return round(score, 2), hard_ok, quality


def _entry_score(side, entry_df):
    entry = latest_closed(entry_df)
    prev = previous_closed(entry_df)
    ema_distance = pct_distance(entry["close"], entry["ema20"])
    max_ema_distance = get_config_float(
        "MAX_ENTRY_EMA20_DISTANCE_PCT",
        get_config_float("LONG_TERM_ENTRY_MAX_EMA_DISTANCE_PCT", 6)
    )
    score = 0
    hard_ok = False

    if side == "BUY":
        bullish_candle = entry["close"] > entry["open"]
        score = add_score(score, entry["close"] > entry["ema20"], 2)
        score = add_score(score, entry["macd"] > entry["macd_signal"], 1)
        score = add_score(score, entry["rsi"] > 50, 1)
        score = add_score(score, bullish_candle, 1)
        score = add_score(score, entry["close"] > prev["high"], 1)
        score = add_score(score, ema_distance <= max_ema_distance, 1)
        score = add_score(score, entry["volume"] > entry["volume_sma"], 1)
        hard_ok = entry["close"] > entry["ema20"] and ema_distance <= max_ema_distance
    else:
        bearish_candle = entry["close"] < entry["open"]
        score = add_score(score, entry["close"] < entry["ema20"], 2)
        score = add_score(score, entry["macd"] < entry["macd_signal"], 1)
        score = add_score(score, entry["rsi"] < 50, 1)
        score = add_score(score, bearish_candle, 1)
        score = add_score(score, entry["close"] < prev["low"], 1)
        score = add_score(score, ema_distance <= max_ema_distance, 1)
        score = add_score(score, entry["volume"] > entry["volume_sma"], 1)
        hard_ok = entry["close"] < entry["ema20"] and ema_distance <= max_ema_distance

    quality_score, quality = _candle_quality_score(
        side,
        entry,
        max_ema_distance
    )
    score += quality_score
    hard_ok = hard_ok and quality["quality_ok"]

    return round(score, 2), hard_ok, ema_distance, quality


def _btc_context_score(side, btc_trend, btc_corr, rs):
    score = 0
    corr_threshold = get_config_float("LONG_TERM_BTC_CORR_THRESHOLD", 0.65)

    if btc_corr is not None and btc_corr >= corr_threshold:
        if side == "BUY":
            score = add_score(score, btc_trend == "BULLISH", 2)
            score -= 2 if btc_trend == "BEARISH" else 0
        else:
            score = add_score(score, btc_trend == "BEARISH", 2)
            score -= 2 if btc_trend == "BULLISH" else 0

    if rs is not None:
        if side == "BUY":
            score = add_score(score, rs > 1, 1)
            score -= 1 if rs < -2 else 0
        else:
            score = add_score(score, rs < -1, 1)
            score -= 1 if rs > 2 else 0

    return score


def _futures_participation_score(side, participation):
    if not participation or not participation.get("available"):
        return 0

    score = 0
    oi_change = participation.get("oi_change_pct")
    taker_ratio = participation.get("taker_buy_sell_ratio")
    global_ratio = participation.get("global_long_short_ratio")
    top_ratio = participation.get("top_long_short_ratio")
    funding_rate = participation.get("funding_rate")
    oi_min = get_config_float("FUTURES_CONTEXT_OI_MIN_CHANGE_PCT", 1.0)
    taker_buy_min = get_config_float("FUTURES_CONTEXT_TAKER_BUY_MIN", 1.05)
    taker_sell_max = get_config_float("FUTURES_CONTEXT_TAKER_SELL_MAX", 0.95)
    crowd_long_max = get_config_float("FUTURES_CONTEXT_CROWD_LONG_MAX", 2.2)
    crowd_short_min = get_config_float("FUTURES_CONTEXT_CROWD_SHORT_MIN", 0.45)
    funding_abs_max = get_config_float("FUTURES_CONTEXT_FUNDING_ABS_MAX", 0.001)

    if oi_change is not None:
        score = add_score(score, oi_change >= oi_min, 1.5)
        score -= 1 if oi_change <= -oi_min else 0

    if taker_ratio is not None:
        if side == "BUY":
            score = add_score(score, taker_ratio >= taker_buy_min, 2)
            score -= 2 if taker_ratio <= taker_sell_max else 0
        else:
            score = add_score(score, taker_ratio <= taker_sell_max, 2)
            score -= 2 if taker_ratio >= taker_buy_min else 0

    crowd_ratio = top_ratio if top_ratio is not None else global_ratio

    if crowd_ratio is not None:
        if side == "BUY":
            score -= 1.5 if crowd_ratio >= crowd_long_max else 0
            score = add_score(score, crowd_ratio <= 1, 0.5)
        else:
            score -= 1.5 if crowd_ratio <= crowd_short_min else 0
            score = add_score(score, crowd_ratio >= 1, 0.5)

    if funding_rate is not None:
        if side == "BUY":
            score -= 1 if funding_rate >= funding_abs_max else 0
            score = add_score(score, funding_rate <= -funding_abs_max, 0.5)
        else:
            score -= 1 if funding_rate <= -funding_abs_max else 0
            score = add_score(score, funding_rate >= funding_abs_max, 0.5)

    return round(score, 2)


def _module_gates_check(
    trend_score,
    confirm_score,
    entry_score,
    quality_score,
    regime_score
):
    if not getattr(config, "SIGNAL_MODULE_GATES_ENABLED", True):
        return True, []

    checks = (
        ("TREND", trend_score, get_config_float("SIGNAL_MIN_TREND_SCORE", 7)),
        ("CONFIRM", confirm_score, get_config_float("SIGNAL_MIN_CONFIRM_SCORE", 7)),
        ("ENTRY", entry_score, get_config_float("SIGNAL_MIN_ENTRY_SCORE", 4)),
        ("QUALITY", quality_score, get_config_float("SIGNAL_MIN_QUALITY_SCORE", 0)),
        ("REGIME", regime_score, get_config_float("SIGNAL_MIN_REGIME_SCORE", -1.5)),
    )
    failures = []

    for label, value, minimum in checks:
        value = _safe_float(value)
        minimum = _safe_float(minimum)

        if value < minimum:
            failures.append(f"{label}={round(value, 2)} < {minimum}")

    return not failures, failures


def _counter_trend_context(side, trend_df, confirm_df):
    trend = latest_closed(trend_df)
    confirm = latest_closed(confirm_df)

    if side == "BUY":
        checks = {
            "trend_close_below_ema50": trend["close"] < trend["ema50"],
            "trend_ema20_below_ema50": trend["ema20"] < trend["ema50"],
            "trend_ema50_below_ema200": trend["ema50"] < trend["ema200"],
            "confirm_close_below_ema50": confirm["close"] < confirm["ema50"],
        }
    else:
        checks = {
            "trend_close_above_ema50": trend["close"] > trend["ema50"],
            "trend_ema20_above_ema50": trend["ema20"] > trend["ema50"],
            "trend_ema50_above_ema200": trend["ema50"] > trend["ema200"],
            "confirm_close_above_ema50": confirm["close"] > confirm["ema50"],
        }

    count = sum(1 for value in checks.values() if value)
    return count >= 2, {"count": count, "checks": checks}


def _reversal_smc_check(smc_score, smc_context):
    if not getattr(config, "SMC_ENABLED", True):
        return False, {"reason": "SMC_DISABLED"}

    smc_context = smc_context or {}
    supportive = {
        "liquidity_sweep": bool(smc_context.get("liquidity_sweep")),
        "order_block": bool(smc_context.get("order_block")),
        "fvg_support": bool(smc_context.get("fvg_support")),
    }
    has_support = any(supportive.values())
    min_smc = get_config_float("REVERSAL_MIN_SMC_SCORE", 1.0)

    return (
        has_support and _safe_float(smc_score) >= min_smc,
        {
            "supportive": supportive,
            "has_block": bool(smc_context.get("fvg_block")),
            "min_smc": min_smc,
        }
    )


def _directional_candle_context(side, candle):
    open_price = _safe_float(candle.get("open"))
    high = _safe_float(candle.get("high"))
    low = _safe_float(candle.get("low"))
    close = _safe_float(candle.get("close"))
    atr = _candle_atr(candle)
    ema20 = _safe_float(candle.get("ema20"))
    macd = _safe_float(candle.get("macd"))
    macd_signal = _safe_float(candle.get("macd_signal"))
    rsi = _safe_float(candle.get("rsi"), 50)
    candle_range = max(high - low, 1e-10)
    close_position = (close - low) / candle_range
    directional_close = close_position if side == "BUY" else 1 - close_position
    directional_body = close - open_price if side == "BUY" else open_price - close
    body_atr = directional_body / atr
    tolerance = get_config_float("REVERSAL_MOMENTUM_EMA_TOLERANCE_PCT", 0.25) / 100

    if side == "BUY":
        direction_ok = close > open_price
        ema_reclaimed = ema20 > 0 and close >= ema20
        ema_near = ema20 > 0 and close >= ema20 * (1 - tolerance)
        oscillator_ok = macd > macd_signal or rsi > 50
    else:
        direction_ok = close < open_price
        ema_reclaimed = ema20 > 0 and close <= ema20
        ema_near = ema20 > 0 and close <= ema20 * (1 + tolerance)
        oscillator_ok = macd < macd_signal or rsi < 50

    return {
        "direction_ok": direction_ok,
        "directional_close": round(float(directional_close), 3),
        "body_atr": round(float(body_atr), 3),
        "ema_reclaimed": ema_reclaimed,
        "ema_near": ema_near,
        "oscillator_ok": oscillator_ok,
        "close": close,
        "rsi": round(float(rsi), 2),
    }


def _reversal_momentum_context(side, confirm_df, entry_df, smc_ok):
    enabled = bool(getattr(config, "REVERSAL_MOMENTUM_SCORE_ENABLED", True))
    context = {"enabled": enabled, "ok": False, "score": 0}

    if not enabled:
        context["reason"] = "REVERSAL_MOMENTUM_DISABLED"
        return 0, context

    lookback = max(get_config_int("REVERSAL_MOMENTUM_LOOKBACK", 3), 2)
    min_candles = max(get_config_int("REVERSAL_MOMENTUM_MIN_CANDLES", 2), 1)
    min_body_atr = get_config_float("REVERSAL_MOMENTUM_MIN_BODY_ATR", 0.20)
    min_close_position = get_config_float(
        "REVERSAL_MOMENTUM_MIN_CLOSE_POSITION",
        0.58
    )
    entry = latest_closed(entry_df)
    confirm = latest_closed(confirm_df)
    prev_confirm = previous_closed(confirm_df)
    entry_context = _directional_candle_context(side, entry)
    confirm_context = _directional_candle_context(side, confirm)
    entry_data = _closed_data(entry_df, lookback + 1)
    recent = entry_data.tail(lookback)

    if side == "BUY":
        direction_count = sum(1 for _, candle in recent.iterrows() if _is_bullish(candle))
        structure_break = (
            len(entry_data) > 1 and
            entry_context["close"] > entry_data.iloc[:-1]["high"].tail(lookback).max()
        )
        confirm_shift = (
            confirm["close"] > prev_confirm["close"] or
            confirm["rsi"] > prev_confirm["rsi"] or
            confirm["macd"] > prev_confirm["macd"]
        )
    else:
        direction_count = sum(1 for _, candle in recent.iterrows() if _is_bearish(candle))
        structure_break = (
            len(entry_data) > 1 and
            entry_context["close"] < entry_data.iloc[:-1]["low"].tail(lookback).min()
        )
        confirm_shift = (
            confirm["close"] < prev_confirm["close"] or
            confirm["rsi"] < prev_confirm["rsi"] or
            confirm["macd"] < prev_confirm["macd"]
        )

    score = 0
    score += 1.0 if entry_context["direction_ok"] else 0
    score += 1.0 if entry_context["body_atr"] >= min_body_atr else 0
    score += 0.75 if entry_context["directional_close"] >= min_close_position else 0
    score += 1.0 if direction_count >= min_candles else 0
    score += 1.25 if entry_context["ema_reclaimed"] else 0.5 if entry_context["ema_near"] else 0
    score += 1.0 if structure_break else 0
    score += 0.75 if entry_context["oscillator_ok"] else 0
    score += 0.75 if confirm_context["direction_ok"] else 0
    score += 0.75 if confirm_shift else 0
    score += 0.75 if confirm_context["ema_reclaimed"] else 0.35 if confirm_context["ema_near"] else 0
    score += 0.5 if smc_ok else 0
    score = round(float(score), 2)

    ok = (
        score >= get_config_float("REVERSAL_MOMENTUM_MIN_SCORE", 4.0)
        and entry_context["direction_ok"]
        and (
            entry_context["ema_near"] or
            structure_break or
            smc_ok
        )
        and (
            confirm_context["direction_ok"] or
            confirm_shift
        )
    )
    context.update({
        "ok": ok,
        "score": score,
        "entry": entry_context,
        "confirm": confirm_context,
        "direction_count": direction_count,
        "min_candles": min_candles,
        "structure_break": structure_break,
        "confirm_shift": bool(confirm_shift),
        "smc_ok": bool(smc_ok),
    })

    if not ok:
        context["reason"] = "REVERSAL_MOMENTUM_WEAK"

    return score, context


def _reversal_signal_check(
    side,
    trend_df,
    confirm_df,
    trend_score,
    confirm_score,
    entry_score,
    quality_score,
    smc_score,
    smc_context,
    momentum_context,
    regime_score,
    confidence,
    confirm_ok,
    entry_ok,
    level_ok
):
    context = {"enabled": bool(getattr(config, "REVERSAL_MODE_ENABLED", True))}
    if not context["enabled"]:
        return False, ["REVERSAL_DISABLED"], context

    failures = []
    counter_ok, counter_context = _counter_trend_context(side, trend_df, confirm_df)
    smc_ok, smc_details = _reversal_smc_check(smc_score, smc_context)
    momentum_context = momentum_context or {}
    momentum_ok = bool(momentum_context.get("ok"))
    context.update({
        "counter_trend": counter_context,
        "smc": smc_details,
        "momentum": momentum_context,
    })

    if getattr(config, "REVERSAL_REQUIRE_COUNTER_TREND", True) and not counter_ok:
        failures.append("COUNTER_TREND_CONTEXT_MISSING")

    if (
        getattr(config, "REVERSAL_REQUIRE_SMC", True)
        and not smc_ok
        and not momentum_ok
    ):
        failures.append("SMC_REVERSAL_EVIDENCE_MISSING")

    confidence_threshold = get_config_float("REVERSAL_SIGNAL_THRESHOLD", 78)
    confirm_min = get_config_float("REVERSAL_MIN_CONFIRM_SCORE", 8)
    entry_min = get_config_float("REVERSAL_MIN_ENTRY_SCORE", 5)
    quality_min = get_config_float("REVERSAL_MIN_QUALITY_SCORE", 0.75)
    regime_min = get_config_float("REVERSAL_MIN_REGIME_SCORE", -1.0)

    if momentum_ok:
        confidence_threshold = get_config_float(
            "REVERSAL_MOMENTUM_SIGNAL_THRESHOLD",
            confidence_threshold
        )
        confirm_min = get_config_float(
            "REVERSAL_MOMENTUM_MIN_CONFIRM_SCORE",
            confirm_min
        )
        entry_min = get_config_float(
            "REVERSAL_MOMENTUM_MIN_ENTRY_SCORE",
            entry_min
        )
        quality_min = get_config_float(
            "REVERSAL_MOMENTUM_MIN_QUALITY_SCORE",
            quality_min
        )
        regime_min = get_config_float(
            "REVERSAL_MOMENTUM_MIN_REGIME_SCORE",
            regime_min
        )

    checks = (
        (
            "CONFIDENCE",
            confidence,
            confidence_threshold,
            ">="
        ),
        (
            "CONFIRM",
            confirm_score,
            confirm_min,
            ">="
        ),
        (
            "ENTRY",
            entry_score,
            entry_min,
            ">="
        ),
        (
            "QUALITY",
            quality_score,
            quality_min,
            ">="
        ),
        (
            "REGIME",
            regime_score,
            regime_min,
            ">="
        ),
        (
            "TREND",
            trend_score,
            get_config_float("REVERSAL_MAX_TREND_SCORE", 8),
            "<="
        ),
    )

    for label, value, limit, operator in checks:
        value = _safe_float(value)
        limit = _safe_float(limit)

        if operator == ">=" and value < limit:
            failures.append(f"{label}={round(value, 2)} < {limit}")
        elif operator == "<=" and value > limit:
            failures.append(f"{label}={round(value, 2)} > {limit}")

    if not confirm_ok and not momentum_ok:
        failures.append("CONFIRMATION_HARD_CHECK_FAILED")

    if not entry_ok and not momentum_ok:
        failures.append("ENTRY_HARD_CHECK_FAILED")

    if not level_ok:
        failures.append("LEVEL_CHECK_FAILED")

    return not failures, failures, context


def _side_signal_score(
    side,
    trend_df,
    confirm_df,
    entry_df,
    btc_trend,
    btc_corr,
    rs,
    participation=None
):
    entry_price = latest_closed(entry_df)["close"]
    level_ok, level = validate_adverse_zone_level(
        side,
        entry_price,
        trend_df,
        confirm_df
    )
    trend_score, trend_ok = _trend_bias_score(side, trend_df)
    confirm_score, confirm_ok, confirm_quality = _confirmation_score(side, confirm_df)
    entry_score, entry_ok, ema_distance, entry_quality = _entry_score(side, entry_df)
    btc_score = _btc_context_score(side, btc_trend, btc_corr, rs)
    participation_score = _futures_participation_score(side, participation)
    smc_score, smc_context = _smc_context_score(side, trend_df, confirm_df, entry_df)
    smc_ok, _ = _reversal_smc_check(smc_score, smc_context)
    momentum_score, momentum_context = _reversal_momentum_context(
        side,
        confirm_df,
        entry_df,
        smc_ok
    )
    regime_score, regime_context = _market_regime_score(
        side,
        trend_df,
        confirm_df,
        entry_df
    )
    quality_score = round(
        float(confirm_quality.get("score", 0)) +
        float(entry_quality.get("score", 0)),
        2
    )
    module_gates_ok, module_gate_reasons = _module_gates_check(
        trend_score,
        confirm_score,
        entry_score,
        quality_score,
        regime_score
    )
    level_check_disabled = bool(level.get("level_check_disabled")) if level else False
    level_score = 4 if level_ok and not level_check_disabled else 0

    total = (
        trend_score +
        confirm_score +
        entry_score +
        btc_score +
        level_score +
        smc_score +
        momentum_score +
        participation_score +
        regime_score
    )
    score = max(0, total)
    trend_confidence = score_to_confidence(score)
    reversal_confidence = score_to_confidence(
        score,
        get_config_float("REVERSAL_CONFIDENCE_MAX_SCORE", 34)
    )
    trend_following_ok = (
        trend_ok and
        confirm_ok and
        entry_ok and
        level_ok and
        module_gates_ok
    )
    reversal_ok, reversal_reasons, reversal_context = _reversal_signal_check(
        side,
        trend_df,
        confirm_df,
        trend_score,
        confirm_score,
        entry_score,
        quality_score,
        smc_score,
        smc_context,
        momentum_context,
        regime_score,
        reversal_confidence,
        confirm_ok,
        entry_ok,
        level_ok
    )
    hard_ok = trend_following_ok or reversal_ok
    confirmation_type = (
        "TREND" if trend_following_ok
        else "REVERSAL" if reversal_ok
        else "NONE"
    )
    confidence = (
        reversal_confidence if reversal_ok and not trend_following_ok
        else trend_confidence
    )

    return {
        "side": side,
        "score": score,
        "confidence": confidence,
        "trend_confidence": trend_confidence,
        "reversal_confidence": reversal_confidence,
        "base_score": trend_score + confirm_score + entry_score + btc_score + level_score,
        "trend_score": trend_score,
        "confirm_score": confirm_score,
        "entry_score": entry_score,
        "btc_score": btc_score,
        "level_score": level_score,
        "smc_score": smc_score,
        "smc_context": smc_context,
        "momentum_score": momentum_score,
        "momentum_context": momentum_context,
        "quality_score": quality_score,
        "confirm_quality": confirm_quality,
        "entry_quality": entry_quality,
        "regime_score": regime_score,
        "regime_context": regime_context,
        "participation_score": participation_score,
        "hard_ok": hard_ok,
        "trend_following_ok": trend_following_ok,
        "reversal_ok": reversal_ok,
        "reversal_reasons": reversal_reasons,
        "reversal_context": reversal_context,
        "confirmation_type": confirmation_type,
        "trend_ok": trend_ok,
        "confirm_ok": confirm_ok,
        "entry_ok": entry_ok,
        "level_ok": level_ok,
        "module_gates_ok": module_gates_ok,
        "module_gate_reasons": module_gate_reasons,
        "level": level,
        "ema_distance": ema_distance,
    }


def _signal_threshold(side_data):
    if side_data.get("confirmation_type") != "REVERSAL":
        return config.LONG_TERM_SIGNAL_THRESHOLD

    momentum = side_data.get("reversal_context", {}).get("momentum", {})

    if momentum.get("ok"):
        return get_config_float(
            "REVERSAL_MOMENTUM_SIGNAL_THRESHOLD",
            get_config_float("REVERSAL_SIGNAL_THRESHOLD", 78)
        )

    return get_config_float(
        "REVERSAL_SIGNAL_THRESHOLD",
        config.LONG_TERM_SIGNAL_THRESHOLD
    )


def _signal_edge(side_data):
    if side_data.get("confirmation_type") == "REVERSAL":
        return get_config_float("REVERSAL_MIN_SIGNAL_EDGE", 0)

    return config.LONG_TERM_MIN_SIGNAL_EDGE


def _signal_candidate(side_data):
    return (
        side_data.get("hard_ok")
        and side_data.get("confidence", 0) >= _signal_threshold(side_data)
    )


def _select_signal(buy, sell):
    buy_ok = _signal_candidate(buy)
    sell_ok = _signal_candidate(sell)

    if buy_ok and not sell_ok:
        return "BUY"

    if sell_ok and not buy_ok:
        return "SELL"

    if not buy_ok and not sell_ok:
        return None

    ignore_trend_confidence = bool(
        getattr(config, "REVERSAL_IGNORE_OPPOSITE_TREND_CONFIDENCE", True)
    )

    if ignore_trend_confidence:
        if (
            buy.get("confirmation_type") == "REVERSAL"
            and sell.get("confirmation_type") == "TREND"
        ):
            return "BUY"

        if (
            sell.get("confirmation_type") == "REVERSAL"
            and buy.get("confirmation_type") == "TREND"
        ):
            return "SELL"

    if buy["confidence"] >= sell["confidence"] + _signal_edge(buy):
        return "BUY"

    if sell["confidence"] >= buy["confidence"] + _signal_edge(sell):
        return "SELL"

    return None


def log_signal_analysis(analysis):
    buy = analysis["buy"]
    sell = analysis["sell"]

    log_info(
        f"BUY conf={buy.get('confidence', 0)}% hard={buy.get('hard_ok', False)} "
        f"type={buy.get('confirmation_type', 'NONE')} "
        f"level={buy.get('level_ok', False)} "
        f"gates={buy.get('module_gates_ok', True)} "
        f"quality={buy.get('quality_score', 0)} "
        f"regime={buy.get('regime_context', {}).get('regime', '')}:"
        f"{buy.get('regime_score', 0)} "
        f"smc={buy.get('smc_score', 0)} "
        f"momentum={buy.get('momentum_score', 0)} "
        f"futures={buy.get('participation_score', 0)} | "
        f"SELL conf={sell.get('confidence', 0)}% "
        f"hard={sell.get('hard_ok', False)} "
        f"type={sell.get('confirmation_type', 'NONE')} "
        f"level={sell.get('level_ok', False)} "
        f"gates={sell.get('module_gates_ok', True)} "
        f"quality={sell.get('quality_score', 0)} "
        f"regime={sell.get('regime_context', {}).get('regime', '')}:"
        f"{sell.get('regime_score', 0)} "
        f"smc={sell.get('smc_score', 0)} "
        f"momentum={sell.get('momentum_score', 0)} "
        f"futures={sell.get('participation_score', 0)}"
    )

    if buy.get("reversal_ok"):
        log_info(
            "BUY REVERSAL CONFIRMED | " +
            f"counter={buy.get('reversal_context', {}).get('counter_trend', {}).get('count')} "
            f"smc={buy.get('smc_score', 0)} "
            f"momentum={buy.get('momentum_score', 0)} "
            f"conf={buy.get('confidence', 0)}"
        )

    if sell.get("reversal_ok"):
        log_info(
            "SELL REVERSAL CONFIRMED | " +
            f"counter={sell.get('reversal_context', {}).get('counter_trend', {}).get('count')} "
            f"smc={sell.get('smc_score', 0)} "
            f"momentum={sell.get('momentum_score', 0)} "
            f"conf={sell.get('confidence', 0)}"
        )

    min_blocked_conf = get_config_float("REVERSAL_LOG_BLOCKED_MIN_CONFIDENCE", 55)

    if (
        getattr(config, "REVERSAL_MODE_ENABLED", True)
        and not buy.get("reversal_ok")
        and buy.get("reversal_confidence", 0) >= min_blocked_conf
    ):
        momentum = buy.get("momentum_context", {})
        log_info(
            "BUY REVERSAL BLOCKED | " +
            "; ".join(buy.get("reversal_reasons", [])) +
            f" | momentum={momentum.get('score', 0)} ok={momentum.get('ok')}"
        )

    if (
        getattr(config, "REVERSAL_MODE_ENABLED", True)
        and not sell.get("reversal_ok")
        and sell.get("reversal_confidence", 0) >= min_blocked_conf
    ):
        momentum = sell.get("momentum_context", {})
        log_info(
            "SELL REVERSAL BLOCKED | " +
            "; ".join(sell.get("reversal_reasons", [])) +
            f" | momentum={momentum.get('score', 0)} ok={momentum.get('ok')}"
        )

    if not buy.get("module_gates_ok", True) and not buy.get("reversal_ok"):
        log_warning(
            "BUY MODULE GATE BLOCKED | " +
            "; ".join(buy.get("module_gate_reasons", []))
        )

    if not sell.get("module_gates_ok", True) and not sell.get("reversal_ok"):
        log_warning(
            "SELL MODULE GATE BLOCKED | " +
            "; ".join(sell.get("module_gate_reasons", []))
        )

    if buy.get("level_ok"):
        log_info(
            f"BUY support {buy.get('level', {}).get('level')} "
            f"ROI={buy.get('level', {}).get('adverse_roi')}% "
            f"SRC={buy.get('level', {}).get('source')}"
        )
    else:
        log_warning(
            f"BUY BLOCKED | {buy.get('level', {}).get('reason', 'NO DETAILS')}"
        )

    if sell.get("level_ok"):
        log_info(
            f"SELL resistance {sell.get('level', {}).get('level')} "
            f"ROI={sell.get('level', {}).get('adverse_roi')}% "
            f"SRC={sell.get('level', {}).get('source')}"
        )
    else:
        log_warning(
            f"SELL BLOCKED | {sell.get('level', {}).get('reason', 'NO DETAILS')}"
        )

    if analysis["signal"]:
        signal_details = analysis[analysis["signal"].lower()]
        log_info(
            f"FINAL LONG-TERM {analysis['signal']} "
            f"TYPE={signal_details.get('confirmation_type', 'NONE')} "
            f"CONFIDENCE: "
            f"{signal_details.get('confidence', 0)}"
        )


def analyze_signal(
    trend_df,
    confirm_df,
    entry_df,
    btc_trend,
    btc_corr,
    rs,
    participation=None,
    log_details=True
):
    try:
        buy = _side_signal_score(
            "BUY",
            trend_df,
            confirm_df,
            entry_df,
            btc_trend,
            btc_corr,
            rs,
            participation=participation
        )
        sell = _side_signal_score(
            "SELL",
            trend_df,
            confirm_df,
            entry_df,
            btc_trend,
            btc_corr,
            rs,
            participation=participation
        )
        signal = _select_signal(buy, sell)
        best = buy if buy["confidence"] >= sell["confidence"] else sell
        analysis = {
            "buy": buy,
            "sell": sell,
            "signal": signal,
            "best_side": best["side"],
            "best_confidence": best["confidence"],
            "threshold": config.LONG_TERM_SIGNAL_THRESHOLD,
            "min_edge": config.LONG_TERM_MIN_SIGNAL_EDGE,
            "participation_available": bool(
                participation and participation.get("available")
            ),
        }

        if log_details:
            log_signal_analysis(analysis)

        return analysis

    except Exception as e:
        log_error(f"STRATEGY ERROR: {e}")
        return {
            "buy": {},
            "sell": {},
            "signal": None,
            "best_side": None,
            "best_confidence": 0,
            "threshold": config.LONG_TERM_SIGNAL_THRESHOLD,
            "min_edge": config.LONG_TERM_MIN_SIGNAL_EDGE,
            "participation_available": False,
            "error": str(e),
        }


def should_fetch_futures_context(analysis):
    if not config.FUTURES_CONTEXT_ENABLED:
        return False

    if analysis.get("best_confidence", 0) < config.FUTURES_CONTEXT_MIN_CONFIDENCE:
        return False

    buy = analysis.get("buy", {})
    sell = analysis.get("sell", {})
    best_key = (analysis.get("best_side") or "").lower()
    best = analysis.get(best_key, {}) if best_key in ("buy", "sell") else {}

    if best:
        momentum = best.get("momentum_context", {})
        return bool(
            best.get("level_ok") and
            (
                best.get("hard_ok") or
                (
                    best.get("reversal_confidence", 0) >=
                    get_config_float("FUTURES_CONTEXT_MIN_CONFIDENCE", 60)
                    and (
                        best.get("reversal_ok") or
                        momentum.get("ok")
                    )
                )
            ) and
            (
                best.get("trend_ok") or
                best.get("reversal_context", {}).get("counter_trend", {}).get("count", 0) >= 2
            ) and
            (
                best.get("confirm_ok") or
                momentum.get("ok")
            ) and
            (
                best.get("entry_ok") or
                momentum.get("ok")
            )
        )

    return bool(
        (buy.get("level_ok") and buy.get("hard_ok")) or
        (sell.get("level_ok") and sell.get("hard_ok"))
    )


def check_signal(trend_df, confirm_df, entry_df, btc_trend, btc_corr, rs):
    return analyze_signal(
        trend_df,
        confirm_df,
        entry_df,
        btc_trend,
        btc_corr,
        rs,
        log_details=True
    )["signal"]
