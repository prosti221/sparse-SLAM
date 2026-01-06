import logging
import inspect
import threading
import colorama

# Initialize colorama (for Windows compatibility)
colorama.init(autoreset=True)

# === Config ===
# TODO: Make this configurable via environment variables or a config file
DEBUG = True
DEBUG_TAG_FILTER = ["Map", "LoopClosure", "Tracker", "g2oBA"]
SEVERITY_FILTER = ["info", "error", "warning"]

MSG_TYPE_TO_COLOR = {
    "debug": colorama.Fore.LIGHTBLACK_EX,
    "info": colorama.Fore.GREEN,
    "error": colorama.Fore.RED,
    "warning": colorama.Fore.YELLOW,
    "tag": colorama.Fore.LIGHTBLACK_EX
}
RESET_COLOR = colorama.Style.RESET_ALL

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(message)s"
)

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
    _log_context.active_debug_scope = func_name


def clear_debug_scope():
    _log_context.active_debug_scope = None


def debug_log(log_tag, message):
    if not DEBUG or log_tag not in DEBUG_TAG_FILTER:
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
