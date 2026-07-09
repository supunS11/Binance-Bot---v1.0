import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import config
from logger import log_error, log_info, log_warning


def _state_path():
    path = Path(config.DCA_STATE_PATH)

    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    return path


def _lock_path():
    path = _state_path()
    return path.with_name(f"{path.name}.lock")


@contextmanager
def _state_file_lock():
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    timeout = max(float(getattr(config, "DCA_STATE_LOCK_TIMEOUT_SECONDS", 10)), 0.1)
    stale_seconds = max(timeout * 3, 30)
    start = time.time()
    fd = None

    while fd is None:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time()}".encode("utf-8"))
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime

                if age > stale_seconds:
                    path.unlink()
                    continue

            except FileNotFoundError:
                continue
            except Exception as e:
                log_warning(f"trade state lock inspect warning: {e}")

            if time.time() - start >= timeout:
                raise TimeoutError(f"timed out waiting for trade state lock: {path}")

            time.sleep(0.05)

    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)

        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            log_warning(f"trade state lock cleanup warning: {e}")


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _load_trade_state_unlocked():
    path = _state_path()

    if not path.exists():
        return {"positions": {}}

    try:
        with path.open("r", encoding="utf-8") as file:
            state = json.load(file)

        if "positions" not in state:
            state["positions"] = {}

        return state

    except Exception as e:
        log_error(f"trade state load error: {e}")
        return {"positions": {}}


def load_trade_state():
    return _load_trade_state_unlocked()


def _save_trade_state_unlocked(state):
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, sort_keys=True, default=str)

    os.replace(temp_path, path)


def save_trade_state(state):
    try:
        with _state_file_lock():
            _save_trade_state_unlocked(state)

    except Exception as e:
        log_error(f"trade state save error: {e}")


def get_position_state(state, symbol):
    return state.get("positions", {}).get(symbol)


def upsert_position_state(state, symbol, data):
    try:
        with _state_file_lock():
            latest_state = _load_trade_state_unlocked()
            latest_state.setdefault("positions", {})[symbol] = data
            _save_trade_state_unlocked(latest_state)
            state["positions"] = latest_state.get("positions", {})

    except Exception as e:
        log_error(f"{symbol} trade state upsert error: {e}")


def remove_position_state(state, symbol):
    try:
        with _state_file_lock():
            latest_state = _load_trade_state_unlocked()
            positions = latest_state.setdefault("positions", {})

            if symbol in positions:
                del positions[symbol]
                _save_trade_state_unlocked(latest_state)

            state["positions"] = latest_state.get("positions", {})

    except Exception as e:
        log_error(f"{symbol} trade state remove error: {e}")


def update_position_tp_status(state, symbol, tp_info, context=""):
    try:
        with _state_file_lock():
            latest_state = _load_trade_state_unlocked()
            item = get_position_state(latest_state, symbol)

            if not item:
                return

            tp_info = tp_info or {}
            ok = bool(tp_info.get("ok"))
            item["tp_status"] = "CREATED" if ok else "FAILED"
            item["tp_price"] = tp_info.get("tp_price")
            item["tp_mode"] = tp_info.get("tp_mode")
            item["tp_context"] = context
            item["tp_updated_at"] = now_iso()
            latest_state.setdefault("positions", {})[symbol] = item
            _save_trade_state_unlocked(latest_state)
            state["positions"] = latest_state.get("positions", {})

    except Exception as e:
        log_error(f"{symbol} TP status update error: {e}")


def prune_closed_positions(state, open_positions):
    try:
        with _state_file_lock():
            latest_state = _load_trade_state_unlocked()
            positions = latest_state.setdefault("positions", {})
            closed_symbols = [
                symbol
                for symbol in positions
                if symbol not in open_positions
            ]

            for symbol in closed_symbols:
                log_info(f"{symbol} removed from trade state; position is closed")
                del positions[symbol]

            if closed_symbols:
                _save_trade_state_unlocked(latest_state)

            state["positions"] = latest_state.get("positions", {})
            return closed_symbols

    except Exception as e:
        log_error(f"trade state prune error: {e}")
        return []


def create_position_state(
    symbol,
    side,
    entry_price,
    quantity,
    planned_margin,
    used_margin,
    reference_price,
    level_info=None
):
    return {
        "symbol": symbol,
        "side": side,
        "managed_by_bot": True,
        "opened_at": now_iso(),
        "updated_at": now_iso(),
        "initial_entry": entry_price,
        "avg_entry": entry_price,
        "quantity": quantity,
        "planned_margin": planned_margin,
        "used_margin": used_margin,
        "dca_count": 0,
        "last_dca_price": None,
        "last_dca_at": None,
        "tp_status": "PENDING",
        "tp_price": None,
        "tp_mode": "",
        "reference_price": reference_price,
        "level_info": level_info or {},
    }


def record_dca_fill(
    state,
    symbol,
    avg_entry,
    quantity,
    used_margin,
    dca_price,
    level_info=None,
    dca_count_increment=1
):
    try:
        with _state_file_lock():
            latest_state = _load_trade_state_unlocked()
            item = get_position_state(latest_state, symbol)

            if not item:
                log_error(f"{symbol} DCA fill not recorded | state missing")
                return False

            current_count = int(item.get("dca_count", 0) or 0)
            completed_level = None

            if level_info:
                try:
                    completed_level = int(level_info.get("dca_level") or 0)
                except Exception:
                    completed_level = None

            item["avg_entry"] = avg_entry
            item["quantity"] = quantity
            item["used_margin"] = round(
                float(item.get("used_margin", 0)) + used_margin,
                8
            )
            item["dca_count"] = current_count + dca_count_increment

            if completed_level:
                item["dca_count"] = max(item["dca_count"], completed_level)

            item["last_dca_price"] = dca_price
            item["last_dca_at"] = now_iso()
            item["updated_at"] = now_iso()
            item.pop("pending_dca", None)

            if level_info:
                item["last_dca_level_info"] = level_info
                completed_levels = item.setdefault("completed_dca_levels", [])

                if completed_level and completed_level not in completed_levels:
                    completed_levels.append(completed_level)
                    completed_levels.sort()

            latest_state.setdefault("positions", {})[symbol] = item
            _save_trade_state_unlocked(latest_state)
            state["positions"] = latest_state.get("positions", {})
            return True

    except Exception as e:
        log_error(f"{symbol} DCA fill record error: {e}")
        return False


def _pending_dca_is_active(pending):
    if not pending:
        return False

    timeout = max(float(getattr(config, "DCA_PENDING_TIMEOUT_SECONDS", 300)), 1)

    try:
        started_at = datetime.fromisoformat(pending.get("started_at", ""))
        return (datetime.now() - started_at).total_seconds() < timeout
    except Exception:
        return False


def has_active_dca_reservation(state, symbol):
    item = get_position_state(state, symbol)

    if not item:
        return False

    return _pending_dca_is_active(item.get("pending_dca"))


def reserve_dca_level(state, symbol, expected_dca_count, level_info=None):
    level = int(expected_dca_count) + 1

    try:
        with _state_file_lock():
            latest_state = _load_trade_state_unlocked()
            item = get_position_state(latest_state, symbol)

            if not item:
                return False, "state missing"

            current_count = int(item.get("dca_count", 0) or 0)

            if current_count != int(expected_dca_count):
                state["positions"] = latest_state.get("positions", {})
                return False, f"state count changed to {current_count}"

            pending = item.get("pending_dca")

            if _pending_dca_is_active(pending):
                return False, f"level {pending.get('level')} already pending"

            item["pending_dca"] = {
                "level": level,
                "started_at": now_iso(),
                "trigger_roi": (level_info or {}).get("trigger_roi"),
                "adverse_roi": (level_info or {}).get("adverse_roi"),
                "price_source": (level_info or {}).get("price_source"),
            }
            item["updated_at"] = now_iso()
            latest_state.setdefault("positions", {})[symbol] = item
            _save_trade_state_unlocked(latest_state)
            state["positions"] = latest_state.get("positions", {})
            return True, "reserved"

    except Exception as e:
        log_error(f"{symbol} DCA reservation error: {e}")
        return False, str(e)


def clear_dca_reservation(state, symbol, level=None):
    try:
        with _state_file_lock():
            latest_state = _load_trade_state_unlocked()
            item = get_position_state(latest_state, symbol)

            if not item:
                return

            pending = item.get("pending_dca")

            if not pending:
                return

            if level is not None and int(pending.get("level", 0) or 0) != int(level):
                return

            item.pop("pending_dca", None)
            item["updated_at"] = now_iso()
            latest_state.setdefault("positions", {})[symbol] = item
            _save_trade_state_unlocked(latest_state)
            state["positions"] = latest_state.get("positions", {})

    except Exception as e:
        log_error(f"{symbol} DCA reservation clear error: {e}")
