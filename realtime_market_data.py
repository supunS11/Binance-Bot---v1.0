import threading
import time
from collections import defaultdict, deque

import config
from logger import log_error, log_info, log_warning


def _safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _clamp(value, lower=-1.0, upper=1.0):
    return max(lower, min(upper, value))


def _normalise(value, scale):
    scale = abs(scale) or 1.0
    return _clamp(value / scale)


def _sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


class RealtimeMarketDataMonitor:
    def __init__(self, symbols, shutdown_event=None):
        self.enabled = bool(getattr(config, "DATA_CONFIRMATION_REALTIME_ENABLED", True))
        max_symbols = int(getattr(config, "DATA_CONFIRMATION_REALTIME_MAX_SYMBOLS", 120))
        unique_symbols = []

        for symbol in symbols or []:
            symbol = str(symbol).upper().strip()
            if symbol and symbol not in unique_symbols:
                unique_symbols.append(symbol)

        self.symbols = unique_symbols[:max_symbols] if max_symbols > 0 else unique_symbols
        self.shutdown_event = shutdown_event
        self.lock = threading.Lock()
        self.twm = None
        self.socket_key = None
        self.running = False
        self.resetting = False
        self.started_at = 0.0
        self.last_message_at = 0.0
        self.last_restart_at = 0.0
        self.streams = ()
        self.watchdog_thread = None
        self.watchdog_stop_event = threading.Event()
        self.state = defaultdict(self._new_symbol_state)

    @staticmethod
    def _new_symbol_state():
        return {
            "trades": deque(),
            "books": deque(),
            "liquidations": deque(),
            "last_price": 0.0,
            "first_update_at": 0.0,
            "last_update_at": 0.0,
        }

    def start(self):
        if not self.enabled:
            log_info("Realtime data confirmation websocket disabled")
            return

        if not self.symbols:
            log_warning("Realtime data confirmation websocket skipped | no symbols")
            return

        self.streams = tuple(self._build_streams())

        if not self.streams:
            log_warning("Realtime data confirmation websocket skipped | no streams")
            return

        self._start_watchdog()

        try:
            from binance import ThreadedWebsocketManager

            self.twm = ThreadedWebsocketManager(
                api_key=config.API_KEY,
                api_secret=config.SECRET_KEY
            )
            self.twm.start()
            self.socket_key = self.twm.start_futures_multiplex_socket(
                callback=self.handle_message,
                streams=list(self.streams)
            )
            self.running = True
            self.started_at = time.time()
            self.last_message_at = self.started_at
            log_info(
                "Realtime data confirmation websocket started | "
                f"SYMBOLS={len(self.symbols)} | STREAMS={len(self.streams)}"
            )

        except Exception as exc:
            self.running = False
            self.socket_key = None
            log_error(f"Realtime data confirmation websocket start error: {exc}")

    def stop(self):
        self.watchdog_stop_event.set()

        with self.lock:
            self.running = False
            self._stop_socket_locked()

            if self.twm:
                try:
                    self.twm.stop()
                except Exception as exc:
                    log_warning(
                        "Realtime data confirmation websocket stop warning: "
                        f"{exc}"
                    )

            self.twm = None

    def snapshot(self, symbol):
        if not self.enabled:
            return {"available": False, "reason": "REALTIME_DISABLED"}

        symbol = str(symbol).upper().strip()
        now = time.time()
        max_window = max(
            float(getattr(config, "DATA_CONFIRMATION_REALTIME_WINDOW_SECONDS", 300)),
            30.0
        )
        stale_seconds = max(
            float(getattr(config, "DATA_CONFIRMATION_REALTIME_STALE_SECONDS", 45)),
            5.0
        )
        warmup_seconds = max(
            float(getattr(config, "DATA_CONFIRMATION_REALTIME_WARMUP_SECONDS", 15)),
            0.0
        )

        with self.lock:
            state = self.state.get(symbol)

            if not state:
                return {
                    "available": False,
                    "reason": "REALTIME_SYMBOL_NOT_WATCHED",
                }

            self._prune_symbol_locked(symbol, now, max_window)
            trades = list(state["trades"])
            books = list(state["books"])
            liquidations = list(state["liquidations"])
            last_update_at = state["last_update_at"]
            first_update_at = state["first_update_at"]
            last_price = state["last_price"]

        if not last_update_at:
            return {"available": False, "reason": "REALTIME_NO_DATA"}

        age = now - last_update_at

        if age > stale_seconds:
            return {
                "available": False,
                "reason": "REALTIME_STALE",
                "age_seconds": round(age, 2),
            }

        warming = (
            warmup_seconds > 0 and
            first_update_at > 0 and
            now - first_update_at < warmup_seconds
        )

        metrics = self._calculate_snapshot_metrics(
            now,
            trades,
            books,
            liquidations,
            last_price,
        )
        metrics.update({
            "available": not warming,
            "warming": warming,
            "reason": "REALTIME_WARMING" if warming else "REALTIME_OK",
            "age_seconds": round(age, 2),
            "trade_count": len(trades),
            "book_samples": len(books),
            "liquidation_count": len(liquidations),
        })
        return metrics

    def _build_streams(self):
        enabled_streams = {
            item.strip().lower()
            for item in getattr(config, "DATA_CONFIRMATION_REALTIME_STREAMS", [])
        }
        depth_levels = int(
            getattr(config, "DATA_CONFIRMATION_REALTIME_DEPTH_LEVELS", 10)
        )
        depth_max_symbols = int(
            getattr(config, "DATA_CONFIRMATION_REALTIME_DEPTH_MAX_SYMBOLS", 60)
        )
        depth_speed = str(
            getattr(config, "DATA_CONFIRMATION_REALTIME_DEPTH_SPEED", "500ms")
        )

        for index, symbol in enumerate(self.symbols):
            stream_symbol = symbol.lower()

            if "aggtrade" in enabled_streams or "agg_trade" in enabled_streams:
                yield f"{stream_symbol}@aggTrade"

            depth_enabled = (
                "depth" in enabled_streams or
                "book" in enabled_streams
            )
            depth_allowed = depth_max_symbols <= 0 or index < depth_max_symbols

            if depth_enabled and depth_allowed:
                yield f"{stream_symbol}@depth{depth_levels}@{depth_speed}"

            if "forceorder" in enabled_streams or "liquidation" in enabled_streams:
                yield f"{stream_symbol}@forceOrder"

    def _start_watchdog(self):
        if not getattr(config, "DATA_CONFIRMATION_REALTIME_WATCHDOG_ENABLED", True):
            log_info("Realtime data confirmation websocket watchdog disabled")
            return

        if self.watchdog_thread and self.watchdog_thread.is_alive():
            return

        self.watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="data-confirmation-realtime-watchdog",
            daemon=True
        )
        self.watchdog_thread.start()

    def _watchdog_loop(self):
        interval = max(
            float(
                getattr(
                    config,
                    "DATA_CONFIRMATION_REALTIME_WATCHDOG_INTERVAL_SECONDS",
                    15,
                )
            ),
            2.0
        )

        while not self._shutdown_requested():
            if self.watchdog_stop_event.wait(interval):
                return

            self._watchdog_check()

    def _watchdog_check(self):
        if not self.enabled or not self.streams or self.resetting:
            return

        now = time.time()
        stale_seconds = max(
            float(getattr(config, "DATA_CONFIRMATION_REALTIME_STALE_SECONDS", 45)),
            5.0
        )
        cooldown = max(
            float(
                getattr(
                    config,
                    "DATA_CONFIRMATION_REALTIME_RESTART_COOLDOWN_SECONDS",
                    30,
                )
            ),
            5.0
        )

        with self.lock:
            last_message_at = self.last_message_at
            running = self.running
            socket_key = self.socket_key
            last_restart_at = self.last_restart_at

        if not running or not socket_key:
            reason = "socket not running"
        else:
            age = now - last_message_at if last_message_at else stale_seconds + 1

            if age < stale_seconds:
                return

            reason = f"stale {round(age, 1)}s"

        if now - last_restart_at < cooldown:
            return

        log_warning(
            "Realtime data confirmation websocket restart | "
            f"REASON={reason}"
        )
        self.reset_connection(reason)

    def reset_connection(self, reason):
        if self._shutdown_requested() or self.resetting:
            return

        thread = threading.Thread(
            target=self._reset_connection,
            args=(reason,),
            daemon=True
        )
        thread.start()

    def _reset_connection(self, reason):
        with self.lock:
            if self.resetting:
                return

            self.resetting = True
            self.last_restart_at = time.time()

        try:
            log_warning(
                "Realtime data confirmation websocket resetting | "
                f"REASON={reason}"
            )

            with self.lock:
                self._stop_socket_locked()

            if self.twm:
                try:
                    self.twm.stop()
                except Exception as exc:
                    log_warning(
                        "Realtime data confirmation websocket manager stop warning: "
                        f"{exc}"
                    )

            from binance import ThreadedWebsocketManager

            self.twm = ThreadedWebsocketManager(
                api_key=config.API_KEY,
                api_secret=config.SECRET_KEY
            )
            self.twm.start()
            self.socket_key = self.twm.start_futures_multiplex_socket(
                callback=self.handle_message,
                streams=list(self.streams)
            )

            with self.lock:
                self.running = True
                self.last_message_at = time.time()

        except Exception as exc:
            with self.lock:
                self.running = False
                self.socket_key = None

            log_error(
                "Realtime data confirmation websocket reset error: "
                f"{exc}"
            )

        finally:
            with self.lock:
                self.resetting = False

    def _stop_socket_locked(self):
        if not self.socket_key or not self.twm:
            self.socket_key = None
            return

        try:
            self.twm.stop_socket(self.socket_key)
        except Exception as exc:
            log_warning(
                "Realtime data confirmation websocket socket stop warning: "
                f"{exc}"
            )

        self.socket_key = None

    def handle_message(self, message):
        if self._shutdown_requested():
            return

        if isinstance(message, dict) and message.get("e") == "error":
            log_warning(f"Realtime data confirmation websocket error: {message}")
            self.reset_connection(message.get("type") or "websocket error")
            return

        data = message.get("data") if isinstance(message, dict) else None
        if not isinstance(data, dict):
            data = message if isinstance(message, dict) else {}

        event_type = data.get("e")
        symbol = str(data.get("s") or "").upper()

        if event_type == "aggTrade":
            self._handle_trade(symbol, data)
        elif event_type == "depthUpdate":
            self._handle_depth(symbol, data)
        elif event_type == "forceOrder":
            order = data.get("o") or {}
            self._handle_liquidation(str(order.get("s") or symbol).upper(), order)

    def _handle_trade(self, symbol, data):
        if not symbol:
            return

        event_time = (_safe_float(data.get("T") or data.get("E")) / 1000) or time.time()
        price = _safe_float(data.get("p"))
        quantity = _safe_float(data.get("q"))

        if price <= 0 or quantity <= 0:
            return

        notional = price * quantity
        buyer_is_maker = bool(data.get("m"))
        signed_notional = -notional if buyer_is_maker else notional

        with self.lock:
            state = self.state[symbol]
            self._touch_state_locked(state, event_time, price)
            state["trades"].append((event_time, signed_notional, price))
            self.last_message_at = time.time()
            self._prune_symbol_locked(symbol, time.time())

    def _handle_depth(self, symbol, data):
        if not symbol:
            return

        bids = data.get("b") or []
        asks = data.get("a") or []
        bid_notional = sum(
            _safe_float(price) * _safe_float(quantity)
            for price, quantity in bids
        )
        ask_notional = sum(
            _safe_float(price) * _safe_float(quantity)
            for price, quantity in asks
        )
        total = bid_notional + ask_notional

        if total <= 0:
            return

        best_bid = _safe_float(bids[0][0]) if bids else 0
        best_ask = _safe_float(asks[0][0]) if asks else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        imbalance = (bid_notional - ask_notional) / total
        event_time = (_safe_float(data.get("T") or data.get("E")) / 1000) or time.time()

        with self.lock:
            state = self.state[symbol]
            self._touch_state_locked(state, event_time, mid_price)
            state["books"].append((event_time, _clamp(imbalance), mid_price))
            self.last_message_at = time.time()
            self._prune_symbol_locked(symbol, time.time())

    def _handle_liquidation(self, symbol, order):
        if not symbol:
            return

        price = _safe_float(order.get("ap") or order.get("p"))
        quantity = _safe_float(order.get("z") or order.get("q"))

        if price <= 0 or quantity <= 0:
            return

        side = str(order.get("S") or "").upper()
        notional = price * quantity
        signed_notional = notional if side == "BUY" else -notional
        event_time = (_safe_float(order.get("T")) / 1000) or time.time()

        with self.lock:
            state = self.state[symbol]
            self._touch_state_locked(state, event_time, price)
            state["liquidations"].append((event_time, signed_notional, price))
            self.last_message_at = time.time()
            self._prune_symbol_locked(symbol, time.time())

    def _touch_state_locked(self, state, event_time, price):
        if not state["first_update_at"]:
            state["first_update_at"] = event_time

        state["last_update_at"] = event_time

        if price > 0:
            state["last_price"] = price

    def _prune_symbol_locked(self, symbol, now, window_seconds=None):
        state = self.state[symbol]
        window = max(
            float(
                window_seconds if window_seconds is not None
                else getattr(config, "DATA_CONFIRMATION_REALTIME_WINDOW_SECONDS", 300)
            ),
            30.0
        )
        cutoff = now - window

        for key in ("trades", "books", "liquidations"):
            rows = state[key]

            while rows and rows[0][0] < cutoff:
                rows.popleft()

    def _calculate_snapshot_metrics(
        self,
        now,
        trades,
        books,
        liquidations,
        last_price,
    ):
        windows = getattr(
            config,
            "DATA_CONFIRMATION_REALTIME_CVD_WINDOWS",
            [30, 60, 180, 300],
        )
        cvd_by_window = {
            int(window): self._cvd_for_window(now, trades, float(window))
            for window in windows
        }
        cvd_30 = cvd_by_window.get(30, self._cvd_for_window(now, trades, 30))
        cvd_60 = cvd_by_window.get(60, self._cvd_for_window(now, trades, 60))
        cvd_180 = cvd_by_window.get(180, self._cvd_for_window(now, trades, 180))
        cvd_300 = cvd_by_window.get(300, self._cvd_for_window(now, trades, 300))
        book_bias, book_persistence = self._book_pressure(now, books)
        price_move_pct = self._price_move_pct(now, trades, last_price, 60)
        absorption_bias = self._absorption_bias(
            cvd_60,
            price_move_pct,
            book_bias,
        )
        liquidation_bias, liquidation_reaction = self._liquidation_bias(
            now,
            liquidations,
            cvd_30,
            price_move_pct,
        )

        return {
            "live_cvd_30s": round(cvd_30, 5),
            "live_cvd_1m": round(cvd_60, 5),
            "live_cvd_3m": round(cvd_180, 5),
            "live_cvd_5m": round(cvd_300, 5),
            "book_pressure": round(book_bias, 5),
            "book_persistence": round(book_persistence, 3),
            "absorption_bias": round(absorption_bias, 5),
            "liquidation_bias": round(liquidation_bias, 5),
            "liquidation_reaction_bias": round(liquidation_reaction, 5),
            "price_momentum_1m": round(_normalise(price_move_pct, 0.5), 5),
            "price_move_pct_1m": round(price_move_pct, 5),
        }

    @staticmethod
    def _cvd_for_window(now, trades, seconds):
        rows = [row for row in trades if now - row[0] <= seconds]
        total = sum(abs(row[1]) for row in rows)

        if total <= 0:
            return 0.0

        return _clamp(sum(row[1] for row in rows) / total)

    @staticmethod
    def _price_move_pct(now, trades, last_price, seconds):
        rows = [row for row in trades if now - row[0] <= seconds and row[2] > 0]

        if len(rows) < 2:
            return 0.0

        first_price = rows[0][2]
        final_price = last_price or rows[-1][2]

        if first_price <= 0:
            return 0.0

        return ((final_price - first_price) / first_price) * 100

    @staticmethod
    def _book_pressure(now, books):
        rows = [row for row in books if now - row[0] <= 20]
        min_samples = max(
            int(getattr(config, "DATA_CONFIRMATION_REALTIME_BOOK_MIN_SAMPLES", 3)),
            1
        )

        if len(rows) < min_samples:
            return 0.0, 0.0

        values = [row[1] for row in rows]
        average = sum(values) / len(values)
        direction = _sign(average)

        if not direction:
            return 0.0, 0.0

        threshold = abs(
            float(getattr(config, "DATA_CONFIRMATION_REALTIME_BOOK_CONFLICT_AT", 0.12))
        ) / 2
        same_direction = sum(
            1 for value in values
            if _sign(value) == direction and abs(value) >= threshold
        )
        persistence = same_direction / len(values)

        if persistence < 0.55:
            return 0.0, persistence

        return _clamp(average), persistence

    @staticmethod
    def _absorption_bias(cvd_60, price_move_pct, book_bias):
        flow_at = abs(
            float(
                getattr(
                    config,
                    "DATA_CONFIRMATION_REALTIME_ABSORPTION_FLOW_AT",
                    0.16,
                )
            )
        )
        max_move = abs(
            float(
                getattr(
                    config,
                    "DATA_CONFIRMATION_REALTIME_ABSORPTION_MAX_PRICE_MOVE_PCT",
                    0.08,
                )
            )
        )
        book_at = abs(
            float(getattr(config, "DATA_CONFIRMATION_REALTIME_BOOK_CONFLICT_AT", 0.12))
        )

        if cvd_60 <= -flow_at and price_move_pct >= -max_move and book_bias >= book_at / 2:
            return _clamp(abs(cvd_60))

        if cvd_60 >= flow_at and price_move_pct <= max_move and book_bias <= -book_at / 2:
            return -_clamp(abs(cvd_60))

        return 0.0

    @staticmethod
    def _liquidation_bias(now, liquidations, cvd_30, price_move_pct):
        rows = [row for row in liquidations if now - row[0] <= 180]
        total = sum(abs(row[1]) for row in rows)
        min_notional = max(
            float(
                getattr(
                    config,
                    "DATA_CONFIRMATION_REALTIME_LIQUIDATION_MIN_NOTIONAL",
                    10000,
                )
            ),
            0.0
        )

        if total < min_notional:
            return 0.0, 0.0

        bias = _clamp(sum(row[1] for row in rows) / total)
        max_move = abs(
            float(
                getattr(
                    config,
                    "DATA_CONFIRMATION_REALTIME_ABSORPTION_MAX_PRICE_MOVE_PCT",
                    0.08,
                )
            )
        )

        if bias < 0:
            if price_move_pct >= -max_move or cvd_30 > 0.05:
                return bias, _clamp(abs(bias) * 0.75)

            return bias, bias

        if bias > 0:
            if price_move_pct <= max_move or cvd_30 < -0.05:
                return bias, -_clamp(abs(bias) * 0.75)

            return bias, bias

        return 0.0, 0.0

    def _shutdown_requested(self):
        return bool(
            self.shutdown_event is not None and self.shutdown_event.is_set()
        )


_monitor = None
_monitor_lock = threading.Lock()


def start_realtime_market_data(symbols, shutdown_event=None):
    global _monitor

    with _monitor_lock:
        if _monitor is not None:
            return _monitor

        _monitor = RealtimeMarketDataMonitor(symbols, shutdown_event=shutdown_event)
        _monitor.start()
        return _monitor


def stop_realtime_market_data():
    global _monitor

    with _monitor_lock:
        monitor = _monitor
        _monitor = None

    if monitor is not None:
        monitor.stop()


def get_realtime_market_snapshot(symbol):
    with _monitor_lock:
        monitor = _monitor

    if monitor is None:
        return {"available": False, "reason": "REALTIME_NOT_STARTED"}

    return monitor.snapshot(symbol)
