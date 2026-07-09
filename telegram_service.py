try:
    import requests
except ImportError:
    requests = None

import config
from logger import log_error, log_warning


def _enabled():
    return (
        config.TELEGRAM_ENABLED
        and bool(config.TELEGRAM_BOT_TOKEN)
        and bool(config.TELEGRAM_CHAT_ID)
    )


def _fmt(value, empty="-"):
    if value in (None, ""):
        return empty

    if isinstance(value, float):
        return f"{value:.8f}".rstrip("0").rstrip(".")

    return str(value)


def _fmt_pct(value):
    if value in (None, ""):
        return "-"

    try:
        return f"{float(value):.2f}%"
    except Exception:
        return str(value)


def _tp_status(tp_info, price):
    if "ok" in tp_info:
        return "CREATED" if tp_info.get("ok") else "FAILED"

    return "FOUND" if price not in (None, "") else "NOT_FOUND"


def _tp_line(tp_info, label="TP"):
    tp_info = tp_info or {}
    price = tp_info.get("tp_price") or tp_info.get("price")
    mode = tp_info.get("tp_mode") or tp_info.get("mode") or tp_info.get("type")
    status = _tp_status(tp_info, price)

    if price in (None, ""):
        return f"{label}: - | Status: {status}"

    if mode:
        return f"{label}: {_fmt(price)} ({mode}) | Status: {status}"

    return f"{label}: {_fmt(price)} | Status: {status}"


def send_telegram_message(message):
    if not _enabled():
        return False

    if requests is None:
        log_warning("Telegram unavailable | requests package missing")
        return False

    try:
        url = (
            f"https://api.telegram.org/bot"
            f"{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        )
        response = requests.post(
            url,
            json={
                "chat_id": config.TELEGRAM_CHAT_ID,
                "text": message[:4096],
                "disable_web_page_preview": True,
            },
            timeout=config.TELEGRAM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return True

    except Exception as e:
        log_error(f"Telegram send error: {type(e).__name__}")
        return False


def send_order_opened_message(
    symbol,
    side,
    entry_price,
    quantity,
    margin,
    tp_info,
    analysis,
    news_context,
    llm_context
):
    analysis = analysis or {}
    side_data = analysis.get((side or "").lower(), {}) or {}
    data_context = analysis.get("data_confirmation") or {}
    news_context = news_context or {}
    llm_context = llm_context or {}
    signal_mode = side_data.get("confirmation_type") or "LONG_TERM"
    confirmation_type = (
        f"{signal_mode} {config.TREND_TIMEFRAME}/"
        f"{config.CONFIRMATION_TIMEFRAME}/{config.ENTRY_TIMEFRAME}"
    )
    message = "\n".join([
        f"{config.TELEGRAM_MESSAGE_PREFIX} | NEW ORDER",
        f"Symbol: {symbol}",
        f"Side: {side}",
        f"Entry: {_fmt(entry_price)}",
        f"Quantity: {_fmt(quantity)}",
        f"Margin: {_fmt(margin)}",
        _tp_line(tp_info),
        f"Confirmation: {confirmation_type}",
        f"Confidence: {_fmt(side_data.get('confidence'))}",
        (
            f"Technical: trend={side_data.get('trend_ok')} "
            f"confirm={side_data.get('confirm_ok')} "
            f"entry={side_data.get('entry_ok')}"
        ),
        (
            f"Data: {data_context.get('side', '-')} "
            f"ok={data_context.get('ok', '-')} "
            f"conf={_fmt(data_context.get('confidence'))} "
            f"edge={_fmt(data_context.get('edge'))} "
            f"reason={data_context.get('reason', '-')}"
        ),
        (
            f"News: {news_context.get('action', '-')} "
            f"{news_context.get('label', '')} "
            f"score={_fmt(news_context.get('score'))} "
            f"reason={news_context.get('reason', '-')}"
        ),
        (
            f"LLM: {llm_context.get('action', '-')} "
            f"risk={llm_context.get('risk_label', '-')} "
            f"adj={_fmt(llm_context.get('confidence_adjustment'))} "
            f"reason={llm_context.get('reason', '-')}"
        ),
    ])
    return send_telegram_message(message)


def send_dca_filled_message(
    symbol,
    side,
    level,
    max_levels,
    adverse_roi,
    trigger_roi,
    fill_price,
    avg_entry,
    total_quantity,
    dca_margin,
    old_tp_info,
    new_tp_info,
    price_source
):
    message = "\n".join([
        f"{config.TELEGRAM_MESSAGE_PREFIX} | DCA FILLED",
        f"Symbol: {symbol}",
        f"Side: {side}",
        f"Level: {level}/{max_levels}",
        f"DCA ROI: {_fmt_pct(adverse_roi)}",
        f"Trigger ROI: {_fmt_pct(trigger_roi)}",
        f"Fill: {_fmt(fill_price)}",
        f"New Avg Entry: {_fmt(avg_entry)}",
        f"Total Qty: {_fmt(total_quantity)}",
        f"DCA Margin: {_fmt(dca_margin)}",
        _tp_line(old_tp_info, "Old TP"),
        _tp_line(new_tp_info, "New TP"),
        f"Source: {price_source or '-'}",
    ])
    return send_telegram_message(message)


def send_tp_failure_message(
    symbol,
    side,
    context_label,
    entry_price,
    quantity,
    tp_info
):
    message = "\n".join([
        f"{config.TELEGRAM_MESSAGE_PREFIX} | TP FAILED",
        f"Symbol: {symbol}",
        f"Side: {side}",
        f"Context: {context_label or '-'}",
        f"Entry: {_fmt(entry_price)}",
        f"Quantity: {_fmt(quantity)}",
        _tp_line(tp_info),
        "Action: Check Binance open orders and place TP manually if needed.",
    ])
    return send_telegram_message(message)
