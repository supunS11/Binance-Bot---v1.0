import csv
import shutil
from datetime import datetime
from pathlib import Path

import config
from logger import log_error, log_info


FIELDNAMES = [
    "timestamp",
    "symbol",
    "action",
    "decision",
    "best_side",
    "best_confidence",
    "buy_confidence",
    "sell_confidence",
    "buy_score",
    "sell_score",
    "buy_base_score",
    "sell_base_score",
    "buy_smc_score",
    "sell_smc_score",
    "buy_participation_score",
    "sell_participation_score",
    "buy_quality_score",
    "sell_quality_score",
    "buy_regime_score",
    "sell_regime_score",
    "buy_regime",
    "sell_regime",
    "buy_entry_quality_ok",
    "sell_entry_quality_ok",
    "buy_entry_chase_atr",
    "sell_entry_chase_atr",
    "buy_entry_rejection_wick",
    "sell_entry_rejection_wick",
    "buy_entry_volume_mult",
    "sell_entry_volume_mult",
    "buy_hard_ok",
    "sell_hard_ok",
    "buy_level_ok",
    "sell_level_ok",
    "buy_level",
    "sell_level",
    "buy_level_source",
    "sell_level_source",
    "buy_smc_sweep",
    "sell_smc_sweep",
    "buy_smc_order_block",
    "sell_smc_order_block",
    "buy_smc_fvg_support",
    "sell_smc_fvg_support",
    "buy_smc_fvg_block",
    "sell_smc_fvg_block",
    "btc_trend",
    "btc_corr",
    "relative_strength",
    "entry_close",
    "trend_close",
    "confirm_close",
    "futures_context_available",
    "oi_change_pct",
    "taker_buy_sell_ratio",
    "global_long_short_ratio",
    "top_long_short_ratio",
    "funding_rate",
    "news_available",
    "news_score",
    "news_label",
    "news_action",
    "news_reason",
    "news_headline",
    "news_source",
    "llm_enabled",
    "llm_available",
    "llm_action",
    "llm_confidence_adjustment",
    "llm_risk_label",
    "llm_reason",
    "llm_model",
    "data_confirmation_ok",
    "data_confirmation_side",
    "data_confirmation_confidence",
    "data_confirmation_edge",
    "data_confirmation_buy_score",
    "data_confirmation_sell_score",
    "data_confirmation_reason",
    "data_confirmation_confirmations",
    "data_confirmation_conflicts",
    "skip_reason",
]


def _journal_path():
    path = Path(config.SIGNAL_JOURNAL_PATH)

    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    return path


def _ensure_journal_header(path):
    if not path.exists() or path.stat().st_size == 0:
        return True

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = path.with_name(f"{path.stem}.bak_{stamp}{path.suffix}")
    temp_path = path.with_name(f"{path.stem}.tmp_{stamp}{path.suffix}")

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        existing_fields = reader.fieldnames or []

        if existing_fields == FIELDNAMES:
            return False

        with temp_path.open("w", newline="", encoding="utf-8") as temp_file:
            writer = csv.DictWriter(temp_file, fieldnames=FIELDNAMES)
            writer.writeheader()

            for row in reader:
                cleaned = {
                    field: row.get(field, "")
                    for field in FIELDNAMES
                }
                writer.writerow(cleaned)

    shutil.copy2(path, backup)
    temp_path.replace(path)

    log_info(f"signal journal header migrated; backup={backup.name}")
    return False


def _latest_close(df):
    try:
        candle = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        return float(candle["close"])
    except Exception:
        return ""


def _side_value(side, key, default=""):
    if not side:
        return default

    value = side.get(key, default)
    return default if value is None else value


def _level_value(side, key):
    level = side.get("level") if side else None

    if not level or "reason" in level:
        return ""

    return level.get(key, "")


def _smc_source(side, key):
    smc = side.get("smc_context") if side else None

    if not smc:
        return ""

    item = smc.get(key)

    if not item:
        return ""

    return item.get("source", "")


def _nested_value(side, parent, key, default=""):
    if not side:
        return default

    item = side.get(parent) or {}

    if not item:
        return default

    value = item.get(key, default)
    return default if value is None else value


def append_signal_journal(
    symbol,
    analysis,
    participation,
    trend_df,
    confirm_df,
    entry_df,
    btc_trend,
    btc_corr,
    rs,
    action,
    skip_reason="",
    news_context=None,
    llm_context=None,
    data_context=None
):
    if not config.SIGNAL_JOURNAL_ENABLED:
        return

    try:
        path = _journal_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = _ensure_journal_header(path)
        buy = analysis.get("buy", {})
        sell = analysis.get("sell", {})
        participation = participation or {}
        news_context = news_context or {}
        llm_context = llm_context or {}
        data_context = data_context or analysis.get("data_confirmation") or {}
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "symbol": symbol,
            "action": action,
            "decision": analysis.get("signal") or "NONE",
            "best_side": analysis.get("best_side") or "",
            "best_confidence": analysis.get("best_confidence", ""),
            "buy_confidence": _side_value(buy, "confidence"),
            "sell_confidence": _side_value(sell, "confidence"),
            "buy_score": _side_value(buy, "score"),
            "sell_score": _side_value(sell, "score"),
            "buy_base_score": _side_value(buy, "base_score"),
            "sell_base_score": _side_value(sell, "base_score"),
            "buy_smc_score": _side_value(buy, "smc_score"),
            "sell_smc_score": _side_value(sell, "smc_score"),
            "buy_participation_score": _side_value(buy, "participation_score"),
            "sell_participation_score": _side_value(sell, "participation_score"),
            "buy_quality_score": _side_value(buy, "quality_score"),
            "sell_quality_score": _side_value(sell, "quality_score"),
            "buy_regime_score": _side_value(buy, "regime_score"),
            "sell_regime_score": _side_value(sell, "regime_score"),
            "buy_regime": _nested_value(buy, "regime_context", "regime"),
            "sell_regime": _nested_value(sell, "regime_context", "regime"),
            "buy_entry_quality_ok": _nested_value(
                buy,
                "entry_quality",
                "quality_ok"
            ),
            "sell_entry_quality_ok": _nested_value(
                sell,
                "entry_quality",
                "quality_ok"
            ),
            "buy_entry_chase_atr": _nested_value(
                buy,
                "entry_quality",
                "chase_atr"
            ),
            "sell_entry_chase_atr": _nested_value(
                sell,
                "entry_quality",
                "chase_atr"
            ),
            "buy_entry_rejection_wick": _nested_value(
                buy,
                "entry_quality",
                "rejection_wick_ratio"
            ),
            "sell_entry_rejection_wick": _nested_value(
                sell,
                "entry_quality",
                "rejection_wick_ratio"
            ),
            "buy_entry_volume_mult": _nested_value(
                buy,
                "entry_quality",
                "volume_mult"
            ),
            "sell_entry_volume_mult": _nested_value(
                sell,
                "entry_quality",
                "volume_mult"
            ),
            "buy_hard_ok": _side_value(buy, "hard_ok"),
            "sell_hard_ok": _side_value(sell, "hard_ok"),
            "buy_level_ok": _side_value(buy, "level_ok"),
            "sell_level_ok": _side_value(sell, "level_ok"),
            "buy_level": _level_value(buy, "level"),
            "sell_level": _level_value(sell, "level"),
            "buy_level_source": _level_value(buy, "source"),
            "sell_level_source": _level_value(sell, "source"),
            "buy_smc_sweep": _smc_source(buy, "liquidity_sweep"),
            "sell_smc_sweep": _smc_source(sell, "liquidity_sweep"),
            "buy_smc_order_block": _smc_source(buy, "order_block"),
            "sell_smc_order_block": _smc_source(sell, "order_block"),
            "buy_smc_fvg_support": _smc_source(buy, "fvg_support"),
            "sell_smc_fvg_support": _smc_source(sell, "fvg_support"),
            "buy_smc_fvg_block": _smc_source(buy, "fvg_block"),
            "sell_smc_fvg_block": _smc_source(sell, "fvg_block"),
            "btc_trend": btc_trend,
            "btc_corr": btc_corr,
            "relative_strength": rs,
            "entry_close": _latest_close(entry_df),
            "trend_close": _latest_close(trend_df),
            "confirm_close": _latest_close(confirm_df),
            "futures_context_available": participation.get("available", False),
            "oi_change_pct": participation.get("oi_change_pct", ""),
            "taker_buy_sell_ratio": participation.get("taker_buy_sell_ratio", ""),
            "global_long_short_ratio": participation.get("global_long_short_ratio", ""),
            "top_long_short_ratio": participation.get("top_long_short_ratio", ""),
            "funding_rate": participation.get("funding_rate", ""),
            "news_available": news_context.get("available", ""),
            "news_score": news_context.get("score", ""),
            "news_label": news_context.get("label", ""),
            "news_action": news_context.get("action", ""),
            "news_reason": news_context.get("reason", ""),
            "news_headline": news_context.get("headline", ""),
            "news_source": news_context.get("source", ""),
            "llm_enabled": llm_context.get("enabled", ""),
            "llm_available": llm_context.get("available", ""),
            "llm_action": llm_context.get("action", ""),
            "llm_confidence_adjustment": llm_context.get(
                "confidence_adjustment",
                ""
            ),
            "llm_risk_label": llm_context.get("risk_label", ""),
            "llm_reason": llm_context.get("reason", ""),
            "llm_model": llm_context.get("model", ""),
            "data_confirmation_ok": data_context.get("ok", ""),
            "data_confirmation_side": data_context.get("side", ""),
            "data_confirmation_confidence": data_context.get("confidence", ""),
            "data_confirmation_edge": data_context.get("edge", ""),
            "data_confirmation_buy_score": data_context.get("buy_score", ""),
            "data_confirmation_sell_score": data_context.get("sell_score", ""),
            "data_confirmation_reason": data_context.get("reason", ""),
            "data_confirmation_confirmations": "|".join(
                data_context.get("confirmations", []) or []
            ),
            "data_confirmation_conflicts": "|".join(
                data_context.get("conflicts", []) or []
            ),
            "skip_reason": skip_reason,
        }

        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)

            if write_header:
                writer.writeheader()

            writer.writerow(row)

    except Exception as e:
        log_error(f"{symbol} signal journal error: {e}")
