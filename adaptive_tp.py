import config


def _as_float(value, default=0.0):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default=0):
    try:
        if value in ("", None):
            return default
        if isinstance(value, (list, tuple, set, dict)):
            return len(value)
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def _side_key(side):
    return str(side or "").lower()


def _chart_confidence(side, analysis):
    if not isinstance(analysis, dict):
        return 0.0

    side_data = analysis.get(_side_key(side))

    if isinstance(side_data, dict):
        confidence = _as_float(side_data.get("confidence"), 0.0)

        if confidence > 0:
            return confidence

    return _as_float(analysis.get("best_confidence"), 0.0)


def _signal_type(side, analysis, explicit_signal_type):
    if explicit_signal_type:
        return str(explicit_signal_type).upper()

    if not isinstance(analysis, dict):
        return "UNKNOWN"

    side_data = analysis.get(_side_key(side))

    if isinstance(side_data, dict):
        value = side_data.get("confirmation_type") or side_data.get("signal_type")

        if value:
            return str(value).upper()

    return str(
        analysis.get("confirmation_type") or
        analysis.get("signal_type") or
        "UNKNOWN"
    ).upper()


def calculate_adaptive_fallback_tp(
    side,
    analysis=None,
    data_context=None,
    signal_type=None,
    dca_count=0,
):
    if not config.ADAPTIVE_FALLBACK_TP_ENABLED:
        return {
            "ok": False,
            "reason": "ADAPTIVE_FALLBACK_TP_DISABLED",
        }

    data_context = data_context or {}
    min_roi = config.ADAPTIVE_FALLBACK_TP_MIN_ROI
    max_roi = config.ADAPTIVE_FALLBACK_TP_MAX_ROI
    base_roi = _clamp(config.ADAPTIVE_FALLBACK_TP_BASE_ROI, min_roi, max_roi)

    chart_confidence = _chart_confidence(side, analysis)
    data_confidence = _as_float(data_context.get("confidence"), 0.0)
    data_edge = _as_float(data_context.get("edge"), 0.0)
    confirmations = _as_int(data_context.get("confirmations"), 0)
    conflicts = _as_int(data_context.get("conflicts"), 0)
    resolved_signal_type = _signal_type(side, analysis, signal_type)

    if chart_confidence <= 0 and data_confidence <= 0:
        return {
            "ok": False,
            "reason": "NO_CHART_OR_DATA_CONFIDENCE",
        }

    chart_quality = _clamp((chart_confidence - 70.0) / 30.0, 0.0, 1.0)
    data_floor = max(config.DATA_CONFIRMATION_MIN_CONFIDENCE, 1.0)
    data_quality = _clamp(
        (data_confidence - data_floor) / max(100.0 - data_floor, 1.0),
        0.0,
        1.0,
    )
    edge_quality = _clamp(
        data_edge / max(config.DATA_CONFIRMATION_MIN_EDGE * 2.0, 1.0),
        0.0,
        1.0,
    )

    roi = base_roi
    chart_bonus = chart_quality * config.ADAPTIVE_FALLBACK_TP_CHART_BONUS_ROI
    data_bonus = data_quality * config.ADAPTIVE_FALLBACK_TP_DATA_BONUS_ROI
    edge_bonus = edge_quality * config.ADAPTIVE_FALLBACK_TP_EDGE_BONUS_ROI
    conflict_penalty = conflicts * config.ADAPTIVE_FALLBACK_TP_CONFLICT_PENALTY_ROI
    dca_penalty = max(dca_count, 0) * config.ADAPTIVE_FALLBACK_TP_DCA_REDUCTION_ROI

    roi += chart_bonus + data_bonus + edge_bonus

    if confirmations >= config.DATA_CONFIRMATION_MIN_CONFIRMATIONS:
        roi += config.ADAPTIVE_FALLBACK_TP_CONFIRMATION_BONUS_ROI

    if resolved_signal_type == "TREND":
        roi += config.ADAPTIVE_FALLBACK_TP_TREND_BONUS_ROI
    elif resolved_signal_type == "REVERSAL":
        roi -= config.ADAPTIVE_FALLBACK_TP_REVERSAL_PENALTY_ROI

    roi -= conflict_penalty + dca_penalty
    roi = round(_clamp(roi, min_roi, max_roi), 2)

    return {
        "ok": True,
        "roi": roi,
        "mode": f"ADAPTIVE_FALLBACK_{resolved_signal_type}_ROI_{roi}%",
        "reason": (
            f"chart={round(chart_confidence, 2)} "
            f"data={round(data_confidence, 2)} "
            f"edge={round(data_edge, 2)} "
            f"confirmations={confirmations} conflicts={conflicts}"
        ),
        "components": {
            "chart_bonus": round(chart_bonus, 2),
            "data_bonus": round(data_bonus, 2),
            "edge_bonus": round(edge_bonus, 2),
            "conflict_penalty": round(conflict_penalty, 2),
            "dca_penalty": round(dca_penalty, 2),
        },
    }
