from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def setup_logging(config: dict[str, Any], app_root: Path) -> logging.Logger:
    settings = config["logging"]
    log_directory = _resolve(app_root, str(settings["directory"]))
    log_directory.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("miscord")
    logger.setLevel(getattr(logging, str(settings["level"]).upper(), logging.INFO))
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(threadName)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        log_directory / "miscord.log",
        maxBytes=int(settings["max_bytes"]),
        backupCount=int(settings["backup_count"]),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
