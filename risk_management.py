import config
from exchange import get_symbol_precision


MIN_NOTIONAL = 5.0


def calculate_position_size(balance, entry_price, sl_price, symbol, margin_override=None):

    try:

        margin = margin_override if margin_override is not None else config.MARGIN_PER_TRADE

        # =========================
        # BASE NOTIONAL FROM MARGIN
        # =========================
        base_notional = margin * config.LEVERAGE

        # convert to quantity
        qty_from_margin = base_notional / entry_price

        # =========================
        # ENSURE BINANCE MINIMUM
        # =========================
        qty_min_notional = MIN_NOTIONAL / entry_price

        # FINAL QTY = EXECUTION FIRST
        quantity = max(qty_from_margin, qty_min_notional)

        # =========================
        # OPTIONAL SAFETY (avoid huge spikes)
        # =========================
        max_notional = base_notional * 1.5
        max_qty = max_notional / entry_price

        quantity = min(quantity, max_qty)

        precision = get_symbol_precision(symbol)

        quantity = round(quantity, precision)

        if quantity <= 0:
            return 0

        return quantity

    except Exception:
        return 0