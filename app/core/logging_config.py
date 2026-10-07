"""Server logs go to the console and to logs/server.log (rotated).

Call setup_logging() from the app lifespan, not at import time: uvicorn applies its own
logging config when it boots, and the lifespan runs after it however the server is started.
"""

import logging
import logging.handlers
from pathlib import Path

from app.config import settings

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_FILE = LOG_DIR / "server.log"
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5

LOG_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-40s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Libraries that log per-request detail nobody reads.
NOISY_LOGGERS = ("httpx", "httpcore", "pymongo", "watchfiles", "LiteLLM", "openhands")

_FILE_HANDLER_NAME = "ng_automate_server_file"


def setup_logging() -> None:
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    if not any(handler.get_name() == _FILE_HANDLER_NAME for handler in root.handlers):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
        )
        file_handler.set_name(_FILE_HANDLER_NAME)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        root.addHandler(file_handler)

    if not any(isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler) for handler in root.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        root.addHandler(console_handler)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
