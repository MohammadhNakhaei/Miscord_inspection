# -*- coding: utf-8 -*-
"""Basler tire-cord missing-thread inspection application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


APP_ROOT = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Basler real-time missing tire-cord detector"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=APP_ROOT / "config.yaml",
        help="YAML configuration file",
    )
    parser.add_argument(
        "--offline",
        type=Path,
        help="Use an image, video, or image directory instead of a camera",
    )
    parser.add_argument(
        "--headless-check",
        type=Path,
        help="Process one image, print the result, and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from miscord.config import ConfigManager
    from miscord.logging_utils import setup_logging

    manager = ConfigManager(args.config, APP_ROOT)
    config = manager.load()

    if args.offline:
        config["camera"]["mode"] = "offline"
        config["camera"]["offline_path"] = str(args.offline.resolve())

    logger = setup_logging(config, APP_ROOT)
    logger.info("Application starting; config=%s", args.config.resolve())

    if args.headless_check:
        from miscord.detector import CordDetector

        import cv2 as cv

        image = cv.imread(str(args.headless_check), cv.IMREAD_UNCHANGED)
        if image is None:
            logger.error("Cannot read image: %s", args.headless_check)
            return 2
        result = CordDetector(config["detection"]).process(image)
        print(result.summary())
        return 1 if result.status == "DEFECT" else 0

    try:
        from PySide6.QtWidgets import QApplication
        from miscord.ui import MainWindow
    except ImportError as exc:
        logger.exception("GUI dependency is missing")
        print(
            "PySide6 is not installed. Run setup_env.ps1 first.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        return 3

    app = QApplication(sys.argv)
    app.setApplicationName("Miscord Inspector")
    app.setOrganizationName("Kurdistan Tire")
    window = MainWindow(manager, config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
