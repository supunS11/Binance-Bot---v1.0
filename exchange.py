from binance.client import Client
from binance.enums import *

import pandas as pd
import time
import numpy as np

import config
from indicators import apply_indicators
from logger import log_info, log_warning, log_error


client = Client(config.API_KEY, config.SECRET_KEY)
_exchange_info_cache = None
_last_kline_request_at = 0.0
_futures_context_cache = {}

# =========================
# SYNC TIME
# =========================
server_time = client.get_server_time()
client.timestamp_offset = server_time['serverTime'] - int(time.time() * 1000)


def _throttle_kline_request():
    global _last_kline_request_at

    delay = getattr(config, "REQUEST_THROTTLE_SECONDS", 0)

    if delay <= 0:
        return

    elapsed = time.time() - _last_kline_request_at

    if elapsed < delay:
        time.sleep(delay - elapsed)

    _last_kline_request_at = time.time()


def get_exchange_info():
    global _exchange_info_cache

    if _exchange_info_cache is None:
        _exchange_info_cache = client.futures_exchange_info()

    return _exchange_info_cache


def get_supported_symbols():

    try:
        symbols = set()

        for item in get_exchange_info().get("symbols", []):
            if item.get("status") != "TRADING":
                continue

            if item.get("contractType") != "PERPETUAL":
                continue

            symbols.add(item["symbol"])

        return symbols

    except Exception as e:
        log_error(f"supported symbols error: {e}")
        return set()


def _to_float(value, default=None):
    try:
        if value in (None, ""):
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def _latest_item(items):
    if not items:
        return None

    return items[-1]


def _change_pct(items, field):
    if not items or len(items) < 2:
        return None

    first = _to_float(items[0].get(field))
    last = _to_float(items[-1].get(field))

    if not first:
        return None

    return round(((last - first) / first) * 100, 2)


def _get_taker_longshort_ratio(params):
    method = getattr(client, "futures_taker_longshort_ratio", None)

    if method:
        return method(**params)

    return client._request_futures_data_api(
        "get",
        "takerlongshortRatio",
        data=params
    )


def get_futures_participation(symbol):
    if not config.FUTURES_CONTEXT_ENABLED:
        return {"available": False, "reason": "DISABLED"}

    key = (
        symbol,
        config.FUTURES_CONTEXT_PERIOD,
        config.FUTURES_CONTEXT_LIMIT
    )
    cached = _futures_context_cache.get(key)

    if cached and time.time() - cached["time"] <= config.FUTURES_CONTEXT_CACHE_SECONDS:
        return cached["data"]

    period = config.FUTURES_CONTEXT_PERIOD
    limit = config.FUTURES_CONTEXT_LIMIT
    params = {"symbol": symbol, "period": period, "limit": limit}
    data = {
        "available": True,
        "symbol": symbol,
        "period": period,
        "limit": limit,
        "oi_change_pct": None,
        "taker_buy_sell_ratio": None,
        "global_long_short_ratio": None,
        "top_long_short_ratio": None,
        "funding_rate": None,
        "errors": [],
    }

    try:
        oi_hist = client.futures_open_interest_hist(**params)
        data["oi_change_pct"] = _change_pct(oi_hist, "sumOpenInterest")
    except Exception as e:
        data["errors"].append(f"OI:{e}")

    try:
        taker = _latest_item(_get_taker_longshort_ratio(params))
        data["taker_buy_sell_ratio"] = _to_float(
            taker.get("buySellRatio") if taker else None
        )
    except Exception as e:
        data["errors"].append(f"TAKER:{e}")

    try:
        global_ratio = _latest_item(client.futures_global_longshort_ratio(**params))
        data["global_long_short_ratio"] = _to_float(
            global_ratio.get("longShortRatio") if global_ratio else None
        )
    except Exception as e:
        data["errors"].append(f"GLOBAL_LS:{e}")

    try:
        top_ratio = _latest_item(client.futures_top_longshort_position_ratio(**params))
        data["top_long_short_ratio"] = _to_float(
            top_ratio.get("longShortRatio") if top_ratio else None
        )
    except Exception as e:
        data["errors"].append(f"TOP_LS:{e}")

    try:
        premium = client.futures_mark_price(symbol=symbol)
        data["funding_rate"] = _to_float(premium.get("lastFundingRate"))
    except Exception as e:
        data["errors"].append(f"FUNDING:{e}")

    usable_values = [
        data["oi_change_pct"],
        data["taker_buy_sell_ratio"],
        data["global_long_short_ratio"],
        data["top_long_short_ratio"],
        data["funding_rate"],
    ]

    data["available"] = any(value is not None for value in usable_values)

    if data["errors"]:
        log_warning(f"{symbol} futures context partial: {' | '.join(data['errors'])}")

    _futures_context_cache[key] = {
        "time": time.time(),
        "data": data
    }

    return data


# =========================
# MARGIN TYPE
# =========================
def set_margin_type(symbol, allow_open_order_block=False):

    try:
        client.futures_change_margin_type(
            symbol=symbol,
            marginType=config.MARGIN_TYPE
        )

        log_info(f"{symbol} Margin: {config.MARGIN_TYPE}")
        return True

    except Exception as e:
        message = str(e)

        if (
            allow_open_order_block
            and (
                "code=-4047" in message
                or "Margin type cannot be changed if there exists open orders" in message
                or "code=-4067" in message
                or "Position side cannot be changed if there exists open orders" in message
            )
        ):
            log_warning(
                f"{symbol} margin setup unchanged | open orders exist; "
                "continuing with current margin type"
            )
            return True

        if "No need to change margin type" not in message:
            log_warning(message)
            return False

        log_info(f"{symbol} Margin already {config.MARGIN_TYPE}")
        return True


# =========================
# LEVERAGE
# =========================
def setup_leverage(symbol):

    try:

        response = client.futures_change_leverage(
            symbol=symbol,
            leverage=config.LEVERAGE
        )

        actual = int(response['leverage'])

        if actual != config.LEVERAGE:
            log_warning(f"{symbol} leverage mismatch")
            return False

        log_info(f"{symbol} leverage set: {actual}x")
        return True

    except Exception as e:
        log_error(f"{symbol} leverage error: {e}")
        return False


# =========================
# BALANCE
# =========================
def get_balance():

    balances = client.futures_account_balance()

    for b in balances:
        if b['asset'] == 'USDT':
            return float(b['balance'])

    return 0


def get_margin_balance():
    return float(client.futures_account()['totalMarginBalance'])


def get_unrealized_pnl():
    return float(client.futures_account()['totalUnrealizedProfit'])


def get_mark_price(symbol):

    try:
        return float(client.futures_mark_price(symbol=symbol)['markPrice'])

    except Exception as e:
        log_error(f"{symbol} mark price error: {e}")
        return None


# =========================
# KLINES
# =========================
def get_klines(symbol, interval, limit=None):

    try:
        limit = limit if limit is not None else config.KLINE_LIMIT
        _throttle_kline_request()

        klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )

        df = pd.DataFrame(klines, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'trades', 'tbbav', 'tbqav', 'ignore'
        ])

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df

    except Exception as e:
        log_error(f"{symbol} klines error: {e}")
        return None


# =========================
# POSITION CHECKS
# =========================
def has_open_position(symbol):

    try:
        positions = client.futures_position_information(symbol=symbol)

        for p in positions:
            if float(p['positionAmt']) != 0:
                return True

        return False

    except Exception as e:
        log_error(str(e))
        return False


def is_position_closed(symbol):

    try:
        positions = client.futures_position_information(symbol=symbol)

        for p in positions:
            if abs(float(p['positionAmt'])) > 0:
                return False

        return True

    except Exception as e:
        log_error(f"{symbol} position check error: {e}")
        return False


def get_open_positions():

    try:
        positions = client.futures_position_information()
        open_positions = {}

        for p in positions:
            amount = float(p["positionAmt"])

            if amount != 0:
                open_positions[p["symbol"]] = amount

        return open_positions

    except Exception as e:
        log_error(f"open positions error: {e}")
        return None


def _build_open_position_detail(position):
    amount = float(position["positionAmt"])

    if amount == 0:
        return None

    position_symbol = position["symbol"]
    position_side = (position.get("positionSide") or "BOTH").upper()

    return {
        "symbol": position_symbol,
        "amount": amount,
        "side": "BUY" if amount > 0 else "SELL",
        "position_side": position_side,
        "quantity": abs(amount),
        "entry_price": abs(_to_float(position.get("entryPrice"), 0) or 0),
        "mark_price": abs(_to_float(position.get("markPrice"), 0) or 0),
        "liquidation_price": abs(
            _to_float(position.get("liquidationPrice"), 0) or 0
        ),
        "unrealized_pnl": _to_float(position.get("unRealizedProfit"), 0) or 0,
    }


def get_open_position_detail_rows(symbol=None):

    try:
        if symbol:
            positions = client.futures_position_information(symbol=symbol)
        else:
            positions = client.futures_position_information()

        open_positions = []

        for position in positions:
            detail = _build_open_position_detail(position)

            if detail:
                open_positions.append(detail)

        return open_positions

    except Exception as e:
        label = symbol if symbol else "all"
        log_error(f"{label} open position row detail error: {e}")
        return None


def get_open_position_details(symbol=None):

    rows = get_open_position_detail_rows(symbol)

    if rows is None:
        return None

    return {
        detail["symbol"]: detail
        for detail in rows
    }


def get_open_position_counts(open_positions=None):

    try:

        if open_positions is None:
            open_positions = get_open_positions()

        if open_positions is None:
            return {"total": 0, "buy": 0, "sell": 0}

        total = buy = sell = 0

        for amt in open_positions.values():

            if amt == 0:
                continue

            total += 1

            if amt > 0:
                buy += 1
            else:
                sell += 1

        return {
            "total": total,
            "buy": buy,
            "sell": sell
        }

    except Exception as e:
        log_error(f"position count error: {e}")
        return {"total": 0, "buy": 0, "sell": 0}


def _get_open_algo_orders(symbol):
    method = getattr(client, "futures_get_open_algo_orders", None)

    if method:
        return method(symbol=symbol)

    return client._request_futures_api(
        "get",
        "openAlgoOrders",
        True,
        data={"symbol": symbol}
    )


def _cancel_algo_order(symbol, algo_id):
    method = getattr(client, "futures_cancel_algo_order", None)

    if method:
        return method(symbol=symbol, algoId=algo_id)

    return client._request_futures_api(
        "delete",
        "algoOrder",
        True,
        data={
            "symbol": symbol,
            "algoId": algo_id,
        }
    )


def _normalise_algo_orders(response):
    if isinstance(response, dict):
        return response.get("orders") or response.get("data") or []

    return response or []


def _order_trigger_price(order):
    for key in ("triggerPrice", "stopPrice", "activatePrice"):
        value = order.get(key)

        if value not in (None, ""):
            return value

    return ""


def get_open_take_profit_info(symbol):
    try:
        tp_types = {"TAKE_PROFIT", "TAKE_PROFIT_MARKET"}

        for order in client.futures_get_open_orders(symbol=symbol):
            order_type = order.get("type")

            if order_type not in tp_types:
                continue

            return {
                "price": _order_trigger_price(order),
                "tp_price": _order_trigger_price(order),
                "type": order_type,
                "source": "order",
                "order_id": order.get("orderId", ""),
            }

        for order in _normalise_algo_orders(_get_open_algo_orders(symbol)):
            order_type = order.get("orderType") or order.get("type")

            if order_type not in tp_types:
                continue

            return {
                "price": _order_trigger_price(order),
                "tp_price": _order_trigger_price(order),
                "type": order_type,
                "source": "algo",
                "order_id": order.get("algoId", ""),
            }

        return {}

    except Exception as e:
        log_warning(f"{symbol} open TP lookup error: {e}")
        return {}


def cancel_open_protection_orders(symbol):

    try:
        orders = client.futures_get_open_orders(symbol=symbol)
        protection_types = {
            "TAKE_PROFIT",
            "TAKE_PROFIT_MARKET",
            "STOP",
            "STOP_MARKET",
            "TRAILING_STOP_MARKET",
        }
        cancelled = 0

        for order in orders:
            order_type = order.get("type")
            close_position = str(order.get("closePosition", "")).lower() == "true"
            reduce_only = str(order.get("reduceOnly", "")).lower() == "true"

            if order_type not in protection_types and not (close_position or reduce_only):
                continue

            client.futures_cancel_order(
                symbol=symbol,
                orderId=order["orderId"]
            )
            cancelled += 1

        if cancelled:
            log_info(f"{symbol} cancelled {cancelled} protection order(s)")

        algo_orders = _normalise_algo_orders(_get_open_algo_orders(symbol))
        algo_cancelled = 0

        for order in algo_orders:
            order_type = order.get("orderType") or order.get("type")

            if order_type not in protection_types:
                continue

            algo_id = order.get("algoId")

            if not algo_id:
                log_warning(f"{symbol} algo protection missing algoId; cancel unsafe")
                return False

            _cancel_algo_order(symbol, algo_id)
            algo_cancelled += 1

        if algo_cancelled:
            log_info(f"{symbol} cancelled {algo_cancelled} algo protection order(s)")

        return True

    except Exception as e:
        log_error(f"{symbol} protection cancel error: {e}")
        return False


# =========================
# PRECISION
# =========================
def get_symbol_precision(symbol):

    info = get_exchange_info()

    for s in info['symbols']:
        if s['symbol'] == symbol:
            return s['quantityPrecision']

    return 3


def get_price_precision(symbol):

    info = get_exchange_info()

    for s in info['symbols']:
        if s['symbol'] == symbol:
            return int(s['pricePrecision'])

    return 4


# =========================
# ENTRY PRICE
# =========================
def get_entry_price(symbol, order=None):

    if order:
        avg_price = float(order.get("avgPrice", 0) or 0)

        if avg_price <= 0:
            executed_qty = float(order.get("executedQty", 0) or 0)
            cum_quote = float(order.get("cumQuote", 0) or 0)

            if executed_qty > 0 and cum_quote > 0:
                avg_price = cum_quote / executed_qty

        if avg_price > 0:
            return avg_price

    last_error = None

    for attempt in range(config.ENTRY_PRICE_RETRIES):
        try:
            positions = client.futures_position_information(symbol=symbol)
            entry_price = abs(float(positions[0]["entryPrice"]))

            if entry_price > 0:
                return entry_price

        except Exception as e:
            last_error = e

        if attempt < config.ENTRY_PRICE_RETRIES - 1:
            time.sleep(config.ENTRY_PRICE_RETRY_DELAY_SECONDS)

    if last_error:
        log_warning(f"{symbol} entry price polling error: {last_error}")

    return 0


# =========================
# MARKET ORDER
# =========================
def place_market_order(symbol, side, quantity):

    try:

        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type=FUTURE_ORDER_TYPE_MARKET,
            quantity=quantity,
            newOrderRespType="RESULT"
        )

        log_info(f"{symbol} MARKET ORDER: {side}")
        return order

    except Exception as e:
        log_error(f"{symbol} order error: {e}")
        return None


def _is_position_side_error(error):
    message = str(error).lower()
    return (
        "position side" in message
        or "hedge" in message
        or "reduceonly" in message
        or "reduce only" in message
    )


def _submit_close_order(symbol, side, quantity, position_side=None, reduce_only=True):
    params = {
        "symbol": symbol,
        "side": side,
        "type": FUTURE_ORDER_TYPE_MARKET,
        "quantity": quantity,
        "newOrderRespType": "RESULT",
    }

    if position_side and position_side != "BOTH":
        params["positionSide"] = position_side
    elif reduce_only:
        params["reduceOnly"] = True

    return client.futures_create_order(**params)


def close_position_market(symbol, amount, position_side=None):

    try:
        amount = float(amount)
        quantity = abs(amount)

        if quantity <= 0:
            return None

        position_side = (position_side or "").upper()

        if position_side in ("LONG", "SHORT"):
            side = SIDE_SELL if position_side == "LONG" else SIDE_BUY
            order = _submit_close_order(
                symbol,
                side,
                quantity,
                position_side=position_side,
                reduce_only=False
            )
            log_warning(
                f"{symbol} HEDGE CLOSE ORDER: {side} | POSITION_SIDE={position_side}"
            )
            return order

        side = SIDE_SELL if amount > 0 else SIDE_BUY

        try:
            order = _submit_close_order(
                symbol,
                side,
                quantity,
                reduce_only=True
            )
            log_warning(f"{symbol} REDUCE-ONLY CLOSE ORDER: {side}")
            return order

        except Exception as e:
            if not _is_position_side_error(e):
                raise

            inferred_position_side = "LONG" if amount > 0 else "SHORT"
            log_warning(
                f"{symbol} reduce-only close failed; retrying hedge close | "
                f"POSITION_SIDE={inferred_position_side} | ERROR={e}"
            )
            order = _submit_close_order(
                symbol,
                side,
                quantity,
                position_side=inferred_position_side,
                reduce_only=False
            )
            log_warning(
                f"{symbol} HEDGE CLOSE ORDER: {side} | "
                f"POSITION_SIDE={inferred_position_side}"
            )
            return order

    except Exception as e:
        log_error(f"{symbol} close position error: {e}")
        return None


# =========================
# STRUCTURE SL (REQUIRED BY MAIN + STRATEGY)
# =========================
def get_structure_stop_loss(df, side):

    try:

        atr = df['atr'].iloc[-1]

        if side == SIDE_BUY:

            swing_low = df['low'].iloc[-10:-1].min()
            return swing_low - (atr * 0.5)

        else:

            swing_high = df['high'].iloc[-10:-1].max()
            return swing_high + (atr * 0.5)

    except Exception as e:
        log_error(f"SL error: {e}")
        return None


def get_roi_take_profit(side, entry_price, roi, precision):
    if side == SIDE_BUY:
        return round(
            entry_price * (1 + (roi / config.LEVERAGE) / 100),
            precision
        )

    return round(
        entry_price * (1 - (roi / config.LEVERAGE) / 100),
        precision
    )


def is_valid_take_profit(side, tp_price, market_price):
    if side == SIDE_BUY:
        return tp_price > market_price

    return tp_price < market_price


def place_algo_order(**params):
    method = getattr(client, "futures_create_algo_order", None)

    if method:
        return method(**params)

    return client._request_futures_api(
        "post",
        "algoOrder",
        True,
        data=params
    )


# =========================
# TP/SL EXECUTION (CLEAN VERSION)
# =========================
def place_tp_sl(
    symbol,
    side,
    entry_price,
    quantity,
    confirm_df,
    structure_tp=None,
    roi_override=None,
    roi_mode_label=None,
    return_details=False
):
    details = {
        "ok": False,
        "symbol": symbol,
        "side": side,
        "entry_price": entry_price,
        "quantity": quantity,
        "tp_price": None,
        "tp_mode": "",
        "sl_price": None,
        "tp_order": None,
    }

    try:
        precision = get_price_precision(symbol)

        market_price = get_mark_price(symbol)

        if market_price is None:
            return details if return_details else False

        if side == SIDE_BUY:
            sl_price = None

            if config.SL_ENABLED:
                sl_price = round(
                    get_structure_stop_loss(confirm_df, SIDE_BUY),
                    precision
                )

            close_side = SIDE_SELL

        # ================= SELL =================
        else:
            sl_price = None

            if config.SL_ENABLED:
                sl_price = round(
                    get_structure_stop_loss(confirm_df, SIDE_SELL),
                    precision
                )

            close_side = SIDE_BUY

        if roi_override is not None:
            tp_mode = roi_mode_label or f"ROI_{roi_override}%"
            tp_price = get_roi_take_profit(
                side,
                entry_price,
                roi_override,
                precision
            )
        elif config.STATIC_TP_ENABLED:
            tp_mode = f"STATIC_ROI_{config.STATIC_TP_ROI}%"
            tp_price = get_roi_take_profit(
                side,
                entry_price,
                config.STATIC_TP_ROI,
                precision
            )
        elif structure_tp and structure_tp.get("target_price"):
            tp_mode = (
                f"STRUCTURE_{structure_tp['source']} "
                f"ROI={structure_tp['target_roi']}%"
            )
            tp_price = round(structure_tp["target_price"], precision)
        else:
            tp_mode = f"FALLBACK_ROI_{config.STRUCTURE_TP_FALLBACK_ROI}%"
            tp_price = get_roi_take_profit(
                side,
                entry_price,
                config.STRUCTURE_TP_FALLBACK_ROI,
                precision
            )

        if (
            not config.STATIC_TP_ENABLED
            and structure_tp
            and structure_tp.get("target_price")
            and not is_valid_take_profit(side, tp_price, market_price)
        ):
            log_warning(f"{symbol} STRUCTURE TP INVALID | USING FALLBACK ROI")
            tp_mode = f"FALLBACK_ROI_{config.STRUCTURE_TP_FALLBACK_ROI}%"
            tp_price = get_roi_take_profit(
                side,
                entry_price,
                config.STRUCTURE_TP_FALLBACK_ROI,
                precision
            )

        details.update({
            "tp_price": tp_price,
            "tp_mode": tp_mode,
            "sl_price": sl_price if config.SL_ENABLED else None,
        })

        # ================= VALIDATION ONLY =================
        if not is_valid_take_profit(side, tp_price, market_price):
            log_warning(
                f"{symbol} TP invalid | SIDE={side} | "
                f"TP={tp_price} | MARKET={market_price} | MODE={tp_mode}"
            )
            return details if return_details else False

        if side == SIDE_BUY and config.SL_ENABLED and sl_price >= market_price:
            log_warning(
                f"{symbol} SL invalid for BUY | "
                f"SL={sl_price} | MARKET={market_price}"
            )
            return details if return_details else False

        if side == SIDE_SELL and config.SL_ENABLED and sl_price <= market_price:
            log_warning(
                f"{symbol} SL invalid for SELL | "
                f"SL={sl_price} | MARKET={market_price}"
            )
            return details if return_details else False

        log_info(
            f"{symbol}\nENTRY: {entry_price}\nTP: {tp_price}\n"
            f"TP_MODE: {tp_mode}\n"
            f"SL: {sl_price if config.SL_ENABLED else 'DISABLED'}"
        )

        # TAKE PROFIT
        tp_order = place_algo_order(
            algoType="CONDITIONAL",
            symbol=symbol,
            side=close_side,
            type="TAKE_PROFIT_MARKET",
            triggerPrice=tp_price,
            closePosition="true",
            workingType="MARK_PRICE",
            priceProtect="TRUE"
        )
        log_info(
            f"{symbol} TP order response | "
            f"ALGO_ID={tp_order.get('algoId')} | "
            f"STATUS={tp_order.get('algoStatus')} | "
            f"TRIGGER={tp_order.get('triggerPrice')} | "
            f"TYPE={tp_order.get('orderType')}"
        )
        details["tp_order"] = tp_order

        if config.SL_ENABLED:
            time.sleep(config.PROTECTION_ORDER_DELAY_SECONDS)

            # STOP LOSS
            sl_order = place_algo_order(
                algoType="CONDITIONAL",
                symbol=symbol,
                side=close_side,
                type="STOP_MARKET",
                triggerPrice=sl_price,
                closePosition="true",
                workingType="MARK_PRICE",
                priceProtect="TRUE"
            )
            log_info(
                f"{symbol} SL order response | "
                f"ALGO_ID={sl_order.get('algoId')} | "
                f"STATUS={sl_order.get('algoStatus')} | "
                f"TRIGGER={sl_order.get('triggerPrice')} | "
                f"TYPE={sl_order.get('orderType')}"
            )
        else:
            log_warning(f"{symbol} SL DISABLED | CROSS-MARGIN LONG-TERM MODE")

        log_info(f"{symbol} TP CREATED")
        details["ok"] = True
        return details if return_details else True

    except Exception as e:
        log_error(f"{symbol} TP/SL error: {e}")
        return details if return_details else False


# =========================
# BTC CORRELATION
# =========================
def get_btc_correlation(symbol):

    try:

        if symbol == "BTCUSDT":
            return 1.0

        coin_df = get_klines(symbol, config.TREND_TIMEFRAME, 100)
        btc_df = get_klines("BTCUSDT", config.TREND_TIMEFRAME, 100)

        if coin_df is None or btc_df is None:
            return 0

        coin_ret = coin_df['close'].pct_change().dropna()
        btc_ret = btc_df['close'].pct_change().dropna()

        return round(float(np.corrcoef(coin_ret, btc_ret)[0, 1]), 2)

    except Exception as e:
        log_error(f"{symbol} corr error: {e}")
        return 0


# =========================
# BTC TREND
# =========================
def get_btc_trend():

    try:

        btc_df = get_klines("BTCUSDT", config.TREND_TIMEFRAME)
        btc_df = apply_indicators(btc_df)

        btc = btc_df.iloc[-2]

        if btc['ema50'] > btc['ema200']:
            return "BULLISH"
        elif btc['ema50'] < btc['ema200']:
            return "BEARISH"

        return "NEUTRAL"

    except Exception as e:
        log_error(f"BTC trend error: {e}")
        return None


# =========================
# RELATIVE STRENGTH
# =========================
def get_relative_strength(symbol):

    try:

        if symbol == "BTCUSDT":
            return 0

        coin = get_klines(symbol, config.TREND_TIMEFRAME, 50)
        btc = get_klines("BTCUSDT", config.TREND_TIMEFRAME, 50)

        if coin is None or btc is None:
            return 0

        coin_r = (coin['close'].iloc[-1] - coin['close'].iloc[-10]) / coin['close'].iloc[-10] * 100
        btc_r = (btc['close'].iloc[-1] - btc['close'].iloc[-10]) / btc['close'].iloc[-10] * 100

        return round(coin_r - btc_r, 2)

    except Exception as e:
        log_error(f"{symbol} RS error: {e}")
        return 0
    
def validate_min_notional(symbol, quantity, price):

    try:

        notional = quantity * price

        # Binance futures minimum notional (safe default buffer)
        MIN_NOTIONAL = 5.0

        if notional < MIN_NOTIONAL:
            return False, notional

        return True, notional

    except Exception:
        return False, 0
