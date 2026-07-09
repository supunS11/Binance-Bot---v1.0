import re
import threading
import time


_BAN_UNTIL_RE = re.compile(r"banned until\s+(\d{12,})", re.IGNORECASE)
_API_CODE_RE = re.compile(r"code=(-?\d+)")
_PRIVATE_BACKOFF_UNTIL = 0.0
_PRIVATE_BACKOFF_LOCK = threading.Lock()


def get_api_error_code(error):
    code = getattr(error, "code", None)

    if code is not None:
        try:
            return int(code)
        except (TypeError, ValueError):
            pass

    match = _API_CODE_RE.search(str(error))

    if not match:
        return None

    try:
        return int(match.group(1))
    except ValueError:
        return None


def is_binance_rate_limit_error(error):
    text = str(error).lower()
    code = get_api_error_code(error)

    return (
        code == -1003
        or "too many requests" in text
        or "way too many requests" in text
        or "banned until" in text
    )


def get_rate_limit_backoff_seconds(
    error,
    default_seconds=300,
    buffer_seconds=10,
):
    default_seconds = max(float(default_seconds), 1)
    buffer_seconds = max(float(buffer_seconds), 0)
    text = str(error)
    match = _BAN_UNTIL_RE.search(text)

    if not match:
        return default_seconds

    try:
        banned_until_ms = int(match.group(1))
    except ValueError:
        return default_seconds

    banned_until_seconds = banned_until_ms / 1000
    wait_seconds = (banned_until_seconds - time.time()) + buffer_seconds

    return max(wait_seconds, default_seconds)


def set_private_api_backoff(seconds):
    global _PRIVATE_BACKOFF_UNTIL

    seconds = max(float(seconds), 0)

    if seconds <= 0:
        return 0

    with _PRIVATE_BACKOFF_LOCK:
        backoff_until = time.time() + seconds
        _PRIVATE_BACKOFF_UNTIL = max(_PRIVATE_BACKOFF_UNTIL, backoff_until)
        return max(_PRIVATE_BACKOFF_UNTIL - time.time(), 0)


def register_private_rate_limit(
    error,
    default_seconds=300,
    buffer_seconds=10,
):
    backoff_seconds = get_rate_limit_backoff_seconds(
        error,
        default_seconds=default_seconds,
        buffer_seconds=buffer_seconds,
    )
    return set_private_api_backoff(backoff_seconds)


def get_private_api_backoff_seconds():
    with _PRIVATE_BACKOFF_LOCK:
        return max(_PRIVATE_BACKOFF_UNTIL - time.time(), 0)
