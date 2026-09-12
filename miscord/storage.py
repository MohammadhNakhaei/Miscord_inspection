from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2 as cv

from miscord.detector import DetectionResult


LOGGER = logging.getLogger("miscord.storage")


@dataclass
class StorageItem:
    base_path: Path
    result: DetectionResult


class DefectStorage:
    """Write defect images away from the acquisition/processing thread."""

    def __init__(self, settings: dict[str, Any], app_root: Path):
        self.settings = settings
        directory = Path(str(settings["directory"]))
        self.root = directory if directory.is_absolute() else app_root / directory
        self.enabled = bool(settings["save_defects"])
        self.queue: queue.Queue[StorageItem | None] = queue.Queue(
            maxsize=int(settings["queue_size"])
        )
        self.thread: threading.Thread | None = None
        self.dropped_count = 0
        self.sequence = 0
        self.lock = threading.Lock()

    def start(self) -> None:
        if not self.enabled or self.thread is not None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(
            target=self._run, name="DefectWriter", daemon=True
        )
        self.thread.start()
        LOGGER.info("Defect storage started: %s", self.root.resolve())

    def enqueue(self, result: DetectionResult) -> Path | None:
        if not self.enabled:
            return None
        timestamp = result.timestamp.astimezone()
        day_directory = self.root / timestamp.strftime("%Y-%m-%d")
        with self.lock:
            self.sequence += 1
            sequence = self.sequence
        stem = timestamp.strftime("%H%M%S_%f") + f"_{sequence:06d}"
        base_path = day_directory / stem
        try:
            self.queue.put_nowait(StorageItem(base_path, result))
            return base_path
        except queue.Full:
            self.dropped_count += 1
            LOGGER.error(
                "Defect save queue is full; dropped=%d", self.dropped_count
            )
            return None

    @staticmethod
    def _write_image(path: Path, image: Any, jpeg_quality: int) -> None:
        ok, encoded = cv.imencode(
            ".jpg", image, [cv.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
        )
        if not ok:
            raise RuntimeError(f"OpenCV failed to encode {path.name}")
        encoded.tofile(str(path))

    def _write(self, item: StorageItem) -> None:
        item.base_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path = item.base_path.with_name(item.base_path.name + "_raw.jpg")
        annotated_path = item.base_path.with_name(
            item.base_path.name + "_annotated.jpg"
        )
        metadata_path = item.base_path.with_suffix(".json")
        self._write_image(
            raw_path, item.result.raw_frame, int(self.settings["jpeg_quality"])
        )
        self._write_image(
            annotated_path,
            item.result.annotated_frame,
            int(self.settings["jpeg_quality"]),
        )
        metadata = item.result.metadata()
        metadata["raw_image"] = raw_path.name
        metadata["annotated_image"] = annotated_path.name
        temporary = metadata_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
        temporary.replace(metadata_path)
        LOGGER.info("Defect saved: %s", annotated_path.resolve())

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                self._write(item)
            except Exception:
                LOGGER.exception("Failed to save defect")
            finally:
                self.queue.task_done()

    def close(self, timeout: float = 5.0) -> None:
        if self.thread is None:
            return
        try:
            self.queue.put(None, timeout=1.0)
        except queue.Full:
            LOGGER.error("Could not enqueue storage shutdown marker")
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            LOGGER.warning("Defect storage did not stop before timeout")
        self.thread = None
