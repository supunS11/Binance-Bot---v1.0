import json
import time
from copy import deepcopy
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

import config
from logger import log_error, log_info, log_warning


_scan_request_count = 0


VALID_ACTIONS = {"ALLOW", "BOOST", "PENALTY", "BLOCK"}
VALID_RISK_LABELS = {"low", "medium", "high"}


def _cache_path():
    path = Path(config.LLM_CACHE_PATH)

    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    return path


def _load_cache():
    path = _cache_path()

    if not path.exists():
        return {
            "items": {},
            "latest_by_symbol_side": {},
            "provider_backoff_until": 0
        }

    try:
        with path.open("r", encoding="utf-8") as file:
            cache = json.load(file)

        if "items" not in cache:
            cache["items"] = {}

        if "provider_backoff_until" not in cache:
            cache["provider_backoff_until"] = 0

        if "latest_by_symbol_side" not in cache:
            cache["latest_by_symbol_side"] = {}

        return cache

    except Exception as e:
        log_error(f"llm cache load error: {e}")
        return {
            "items": {},
            "latest_by_symbol_side": {},
            "provider_backoff_until": 0
        }


def _save_cache(cache):
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(cache, file, indent=2, sort_keys=True, default=str)

    except Exception as e:
        log_error(f"llm cache save error: {e}")


def _empty_context(symbol, reason, enabled=None):
    return {
        "enabled": config.LLM_FILTER_ENABLED if enabled is None else enabled,
        "available": False,
        "symbol": symbol,
        "provider": config.LLM_PROVIDER,
        "model": config.LLM_MODEL,
        "action": "DISABLED" if enabled is False else "ALLOW",
        "raw_action": "",
        "confidence_adjustment": 0,
        "risk_label": "",
        "reason": reason,
    }


def _symbol_side_key(payload):
    return json.dumps(
        {
            "symbol": payload.get("symbol"),
            "side": payload.get("proposed_signal"),
            "model": config.LLM_MODEL,
        },
        sort_keys=True
    )


def _cached_review(cache, key, now, max_age):
    cached = cache.get("items", {}).get(key)

    if not cached:
        return None, ""

    fetched_at = float(cached.get("fetched_at", 0) or 0)
    age = max(now - fetched_at, 0)

    if max_age is not None and age > max_age:
        return None, ""

    source = "cache" if age <= config.LLM_CACHE_SECONDS else "stale_cache"
    return cached.get("review") or {}, source


def _latest_symbol_side_review(cache, payload, now):
    key = _symbol_side_key(payload)
    cached = cache.get("latest_by_symbol_side", {}).get(key)

    if not cached:
        return None, ""

    fetched_at = float(cached.get("fetched_at", 0) or 0)
    age = max(now - fetched_at, 0)

    if age > config.LLM_STALE_CACHE_SECONDS:
        return None, ""

    return cached.get("review") or {}, "symbol_side_stale_cache"


def begin_llm_scan_budget():
    global _scan_request_count

    _scan_request_count = 0


def _round_value(value, digits=4):
    try:
        return round(float(value), digits)
    except Exception:
        return value if value not in (None, "") else ""


def _limit_text(value, max_len=180):
    text = str(value or "").strip()

    if len(text) <= max_len:
        return text

    return text[: max_len - 3] + "..."


def _safe_bool(value):
    if value in ("", None):
        return False

    return bool(value)


def _is_rate_limit_reason(reason):
    text = str(reason or "").lower()
    return (
        "429" in text
        or "too many requests" in text
        or "rate limit" in text
        or "rate_limit" in text
        or "rate-limit" in text
        or "quota" in text
    )


def _compact_level(side_data):
    level = side_data.get("level") if side_data else None

    if not isinstance(level, dict):
        return {}

    return {
        "ok": _safe_bool(side_data.get("level_ok")),
        "price": _round_value(level.get("level")),
        "adverse_roi": _round_value(level.get("adverse_roi"), 2),
        "score": _round_value(level.get("score"), 2),
        "source": level.get("source") or "",
        "reason": level.get("reason") or "",
    }


def _compact_quality(side_data):
    entry_quality = side_data.get("entry_quality") or {}
    confirm_quality = side_data.get("confirm_quality") or {}
    regime_context = side_data.get("regime_context") or {}

    return {
        "quality_score": _round_value(side_data.get("quality_score"), 2),
        "regime": regime_context.get("regime") or "",
        "regime_score": _round_value(side_data.get("regime_score"), 2),
        "entry_quality_ok": _safe_bool(entry_quality.get("quality_ok")),
        "entry_chase_atr": _round_value(entry_quality.get("chase_atr"), 3),
        "entry_volume_mult": _round_value(entry_quality.get("volume_mult"), 2),
        "entry_rejection_wick": _round_value(
            entry_quality.get("rejection_wick_ratio"),
            3
        ),
        "confirm_quality_ok": _safe_bool(confirm_quality.get("quality_ok")),
        "confirm_volume_mult": _round_value(
            confirm_quality.get("volume_mult"),
            2
        ),
    }


def _compact_side(side_data):
    side_data = side_data or {}

    return {
        "side": side_data.get("side") or "",
        "confirmation_type": side_data.get("confirmation_type") or "NONE",
        "confidence": _round_value(side_data.get("confidence"), 2),
        "score": _round_value(side_data.get("score"), 2),
        "hard_ok": _safe_bool(side_data.get("hard_ok")),
        "trend_following_ok": _safe_bool(side_data.get("trend_following_ok")),
        "reversal_ok": _safe_bool(side_data.get("reversal_ok")),
        "trend_ok": _safe_bool(side_data.get("trend_ok")),
        "confirm_ok": _safe_bool(side_data.get("confirm_ok")),
        "entry_ok": _safe_bool(side_data.get("entry_ok")),
        "level": _compact_level(side_data),
        "quality": _compact_quality(side_data),
        "scores": {
            "trend": _round_value(side_data.get("trend_score"), 2),
            "confirm": _round_value(side_data.get("confirm_score"), 2),
            "entry": _round_value(side_data.get("entry_score"), 2),
            "btc": _round_value(side_data.get("btc_score"), 2),
            "smc": _round_value(side_data.get("smc_score"), 2),
            "futures": _round_value(side_data.get("participation_score"), 2),
            "regime": _round_value(side_data.get("regime_score"), 2),
        },
    }


def _compact_news(news_context):
    if not news_context:
        return {}

    return {
        "available": _safe_bool(news_context.get("available")),
        "label": news_context.get("label") or "",
        "score": _round_value(news_context.get("score"), 3),
        "action": news_context.get("action") or "",
        "reason": news_context.get("reason") or "",
        "headline": _limit_text(news_context.get("headline"), 220),
        "source": news_context.get("source") or "",
        "high_impact": _safe_bool(news_context.get("high_impact")),
    }


def _compact_participation(participation):
    if not participation:
        return {}

    return {
        "available": _safe_bool(participation.get("available")),
        "oi_change_pct": _round_value(participation.get("oi_change_pct"), 3),
        "taker_buy_sell_ratio": _round_value(
            participation.get("taker_buy_sell_ratio"),
            3
        ),
        "global_long_short_ratio": _round_value(
            participation.get("global_long_short_ratio"),
            3
        ),
        "top_long_short_ratio": _round_value(
            participation.get("top_long_short_ratio"),
            3
        ),
        "funding_rate": _round_value(participation.get("funding_rate"), 6),
    }


def _build_payload(
    symbol,
    side,
    analysis,
    participation,
    btc_trend,
    btc_corr,
    rs,
    news_context
):
    side_key = (side or "").lower()

    return {
        "symbol": symbol,
        "proposed_signal": side,
        "selected_side": _compact_side(analysis.get(side_key, {})),
        "opposite_side": _compact_side(
            analysis.get("sell" if side_key == "buy" else "buy", {})
        ),
        "threshold": _round_value(analysis.get("threshold"), 2),
        "min_edge": _round_value(analysis.get("min_edge"), 2),
        "best_confidence": _round_value(analysis.get("best_confidence"), 2),
        "btc": {
            "trend": btc_trend,
            "correlation": _round_value(btc_corr, 3),
            "relative_strength_pct": _round_value(rs, 2),
        },
        "futures_context": _compact_participation(participation),
        "news_context": _compact_news(news_context),
    }


def _cache_key(payload):
    selected = payload.get("selected_side", {})
    quality = selected.get("quality", {})
    level = selected.get("level", {})
    news = payload.get("news_context", {})
    futures = payload.get("futures_context", {})
    key_payload = {
        "symbol": payload.get("symbol"),
        "side": payload.get("proposed_signal"),
        "confidence": selected.get("confidence"),
        "opposite_confidence": payload.get("opposite_side", {}).get("confidence"),
        "quality_score": quality.get("quality_score"),
        "regime": quality.get("regime"),
        "level_source": level.get("source"),
        "news_score": news.get("score"),
        "news_action": news.get("action"),
        "oi": futures.get("oi_change_pct"),
        "taker": futures.get("taker_buy_sell_ratio"),
        "model": config.LLM_MODEL,
    }
    return json.dumps(key_payload, sort_keys=True)


def _system_prompt():
    return (
        "You are a risk review layer for an automated Binance futures bot. "
        "The deterministic strategy already selected a signal. You cannot "
        "create a new trade or flip the side. Review only whether the proposed "
        "signal has clear conflict, late-entry risk, weak confirmation, news "
        "risk, or futures-flow risk. Return only valid JSON with keys: "
        "action, confidence_adjustment, risk_label, reason. action must be "
        "ALLOW, BOOST, PENALTY, or BLOCK. risk_label must be low, medium, or "
        "high. Keep reason under 160 characters."
    )


def _user_prompt(payload):
    text = json.dumps(payload, sort_keys=True, default=str)

    if len(text) > config.LLM_MAX_PROMPT_CHARS:
        text = text[: config.LLM_MAX_PROMPT_CHARS]

    return (
        "Review this proposed futures signal. Be conservative with BLOCK; use "
        "BLOCK only for obvious contradictions or high-risk context. JSON "
        f"payload:\n{text}"
    )


def _parse_json_content(content):
    text = str(content or "").strip()

    if text.startswith("```"):
        text = text.strip("`").strip()

        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


def _request_openai_compatible(payload):
    if requests is None:
        return None, "LLM_REQUESTS_PACKAGE_MISSING"

    if not config.LLM_API_KEY:
        return None, "LLM_API_KEY_MISSING"

    if not config.LLM_MODEL:
        return None, "LLM_MODEL_MISSING"

    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _user_prompt(payload)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    last_error = None

    for attempt in range(max(config.LLM_MAX_RETRIES, 1)):
        try:
            response = requests.post(
                config.LLM_BASE_URL,
                headers=headers,
                json=body,
                timeout=config.LLM_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return _parse_json_content(content), ""

        except Exception as e:
            last_error = e

            if attempt + 1 < max(config.LLM_MAX_RETRIES, 1):
                time.sleep(0.5)

    return None, f"LLM_REQUEST_ERROR:{last_error}"


def _normalise_review(review):
    if not isinstance(review, dict):
        review = {}

    action = str(review.get("action") or "ALLOW").strip().upper()

    if action not in VALID_ACTIONS:
        action = "ALLOW"

    risk_label = str(review.get("risk_label") or "medium").strip().lower()

    if risk_label not in VALID_RISK_LABELS:
        risk_label = "medium"

    try:
        raw_adjustment = float(review.get("confidence_adjustment") or 0)
    except Exception:
        raw_adjustment = 0

    if action == "BOOST":
        delta = min(abs(raw_adjustment) or config.LLM_CONFIDENCE_BOOST,
                    config.LLM_CONFIDENCE_BOOST)
    elif action == "PENALTY":
        delta = -min(abs(raw_adjustment) or config.LLM_CONFIDENCE_PENALTY,
                     config.LLM_CONFIDENCE_PENALTY)
    else:
        delta = 0

    return {
        "action": action,
        "confidence_adjustment": round(delta, 2),
        "risk_label": risk_label,
        "reason": _limit_text(review.get("reason") or "LLM_REVIEW_READY", 180),
    }


def _get_review(payload):
    global _scan_request_count

    cache = _load_cache()
    now = time.time()
    key = _cache_key(payload)
    review, source = _cached_review(cache, key, now, config.LLM_CACHE_SECONDS)

    if review is not None:
        return review, source, ""

    provider = config.LLM_PROVIDER
    backoff_until = float(cache.get("provider_backoff_until", 0) or 0)

    if backoff_until > now:
        review, source = _cached_review(
            cache,
            key,
            now,
            config.LLM_STALE_CACHE_SECONDS
        )

        if review is not None:
            return review, source, ""

        review, source = _latest_symbol_side_review(cache, payload, now)

        if review is not None:
            return review, source, ""

        return None, provider, "LLM_PROVIDER_RATE_LIMIT_BACKOFF"

    if provider in ("openai", "openai-compatible", "compatible"):
        if (
            config.LLM_MAX_REQUESTS_PER_SCAN > 0 and
            _scan_request_count >= config.LLM_MAX_REQUESTS_PER_SCAN
        ):
            review, source = _cached_review(
                cache,
                key,
                now,
                config.LLM_STALE_CACHE_SECONDS
            )

            if review is not None:
                return review, source, ""

            review, source = _latest_symbol_side_review(cache, payload, now)

            if review is not None:
                return review, source, ""

            return None, provider, "LLM_SCAN_REQUEST_LIMIT_REACHED"

        _scan_request_count += 1
        review, reason = _request_openai_compatible(payload)
    else:
        review, reason = None, f"LLM_PROVIDER_UNSUPPORTED:{provider}"

    if review is not None:
        cache["provider_backoff_until"] = 0
        cache["items"][key] = {
            "fetched_at": now,
            "review": review,
        }
        cache.setdefault("latest_by_symbol_side", {})[
            _symbol_side_key(payload)
        ] = {
            "fetched_at": now,
            "review": review,
        }
        _save_cache(cache)
    elif _is_rate_limit_reason(reason):
        cache["provider_backoff_until"] = (
            now + config.LLM_RATE_LIMIT_BACKOFF_SECONDS
        )
        _save_cache(cache)
        log_warning(
            f"LLM provider rate limited | "
            f"backoff={config.LLM_RATE_LIMIT_BACKOFF_SECONDS}s"
        )

    return review, provider, reason


def _adjust_side(side_data, delta):
    if not side_data:
        return

    confidence = float(side_data.get("confidence", 0) or 0)
    side_data["confidence"] = round(max(min(confidence + delta, 100), 0), 2)
    side_data["llm_adjustment"] = delta


def apply_llm_filter(
    symbol,
    side,
    analysis,
    participation=None,
    btc_trend=None,
    btc_corr=None,
    rs=None,
    news_context=None
):
    if not config.LLM_FILTER_ENABLED:
        return True, analysis, _empty_context(symbol, "LLM_FILTER_DISABLED", False)

    if not side:
        return True, analysis, _empty_context(symbol, "LLM_NO_SIGNAL")

    payload = _build_payload(
        symbol,
        side,
        analysis,
        participation,
        btc_trend,
        btc_corr,
        rs,
        news_context
    )

    review, source, reason = _get_review(payload)

    if review is None:
        context = _empty_context(symbol, reason or "LLM_UNAVAILABLE")
        context["source"] = source

        if context.get("reason") == "LLM_PROVIDER_RATE_LIMIT_BACKOFF":
            log_info(f"{symbol} LLM skipped | RATE_LIMIT_BACKOFF")
        elif context.get("reason") == "LLM_SCAN_REQUEST_LIMIT_REACHED":
            log_info(f"{symbol} LLM skipped | SCAN_REQUEST_LIMIT")
        else:
            log_warning(f"{symbol} LLM unavailable | {context.get('reason')}")

        if config.LLM_FAIL_OPEN:
            return True, analysis, context

        return False, analysis, context

    normalised = _normalise_review(review)
    adjusted = deepcopy(analysis)
    side_key = side.lower()
    delta = normalised["confidence_adjustment"]
    context = {
        "enabled": True,
        "available": True,
        "symbol": symbol,
        "provider": config.LLM_PROVIDER,
        "model": config.LLM_MODEL,
        "source": source,
        "action": normalised["action"],
        "raw_action": str(review.get("action") or ""),
        "confidence_adjustment": delta,
        "risk_label": normalised["risk_label"],
        "reason": normalised["reason"],
    }

    log_info(
        f"{symbol} LLM | ACTION={context['action']} | "
        f"RISK={context['risk_label']} | ADJ={delta} | "
        f"SOURCE={context['source']} | REASON={context['reason']}"
    )

    if context["action"] == "BLOCK":
        context["reason"] = f"LLM_BLOCK:{context['reason']}"

        if config.LLM_BLOCK_HIGH_RISK:
            return False, adjusted, context

        return True, adjusted, context

    if delta:
        _adjust_side(adjusted.get(side_key, {}), delta)
        current = adjusted.get(side_key, {}).get("confidence", 0)
        adjusted["best_confidence"] = current

        if current < config.LONG_TERM_SIGNAL_THRESHOLD:
            context["action"] = "BLOCK"
            context["reason"] = "LLM_ADJUSTED_CONFIDENCE_BELOW_THRESHOLD"
            return False, adjusted, context

    return True, adjusted, context
