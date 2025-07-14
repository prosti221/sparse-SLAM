import logging
import inspect
import threading

# === Config ===
# TODO: Make this configurable via environment variables or a config file
DEBUG = True

DEBUG_TAG_FILTER = ["SuperPoint", "Parser", "Renderer", "Map"]

SEVERITY_FILTER = ["info", "error", "warning"]

MSG_TYPE_TO_COLOR = {
    "debug": "\x1b[90m",   # Bright Black / Gray
    "info": "\x1b[32m",    # Green
    "error": "\x1b[31m",   # Red
    "warning": "\x1b[33m",  # Yellow
    "tag": "\x1b[2m"
}
RESET_COLOR = "\x1b[0m"

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(message)s"
)

# === Thread-local storage for scope context ===
_log_context = threading.local()
_log_context.active_debug_scope = None


def colorize(msg_type: str, content: str) -> str:
    color = MSG_TYPE_TO_COLOR.get(msg_type.lower(), "")
    return f"{color}{content}{RESET_COLOR}"


def get_calling_function_name(skip=2):
    # skip=2: 0=get_calling_function_name, 1=debug_log, 2=caller of debug_log
    stack = inspect.stack()
    if len(stack) > skip:
        return stack[skip].function
    return None


def set_debug_scope(func_name: str):
    """Set the active debug scope function name."""
    _log_context.active_debug_scope = func_name


def clear_debug_scope():
    """Clear the active debug scope."""
    _log_context.active_debug_scope = None


def debug_log(log_tag, message):
    if not DEBUG or log_tag in DEBUG_TAG_FILTER:
        return
    active_scope = getattr(_log_context, "active_debug_scope", None)
    calling_func = get_calling_function_name()
    if active_scope is None or active_scope == calling_func:
        logging.debug(
            colorize("debug", f"[DEBUG]") +
            colorize("tag", f" [{log_tag}]") +
            colorize("tag", f"[{calling_func}] ") +
            message
        )


def info_log(log_tag, message):
    if "info" not in SEVERITY_FILTER:
        return
    logging.info(colorize("info", f"[INFO]") +
                 colorize("tag", f" [{log_tag}] ") + message)


def error_log(log_tag, message):
    if "error" not in SEVERITY_FILTER:
        return
    logging.error(colorize("error", f"[ERROR]") +
                  colorize("tag", f" [{log_tag}] ") + message)


def warning_log(log_tag, message):
    if "warning" not in SEVERITY_FILTER:
        return
    logging.warning(
        colorize("warning", f"[WARNING]") + colorize("tag", f" [{log_tag}] ") + message)
