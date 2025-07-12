import logging
import os

# TODO: Set DEBUG flag and filters based on environment variable or config
DEBUG = True
# Filter specific debug tags
DEBUG_TAG_FILTER = []

MSG_TYPE_TO_COLOR = {
    "debug": "\x1b[90m",   # Bright Black / Gray
    "info": "\x1b[32m",    # Green
    "error": "\x1b[31m",   # Red
    "warning": "\x1b[33m",  # Yellow
}
RESET_COLOR = "\x1b[0m"

# Set up the base logger
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(message)s"
)


def colorize(msg_type: str, content: str) -> str:
    color = MSG_TYPE_TO_COLOR.get(msg_type.lower(), "")
    return f"{color}{content}{RESET_COLOR}"


def debug_log(log_tag, message):
    if DEBUG and log_tag not in DEBUG_TAG_FILTER:
        logging.debug(
            colorize("debug", f"[DEBUG]") + f" [{log_tag}] {message}")


def info_log(log_tag, message):
    logging.info(colorize("info", f"[INFO]") + f" [{log_tag}] {message}")


def error_log(log_tag, message):
    logging.error(colorize("error", f"[ERROR]") + f" [{log_tag}] {message}")


def warning_log(log_tag, message):
    logging.warning(
        colorize("warning", f"[WARNING]") + f" [{log_tag}] {message}")
