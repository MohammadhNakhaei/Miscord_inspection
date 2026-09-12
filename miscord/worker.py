from __future__ import annotations

import copy
import logging
import threading
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal

from miscord.camera import FrameSource, create_source
from miscord.detector import CordDetector, DetectionResult, STATUS_DEFECT
from miscord.storage import DefectStorage


LOGGER = logging.getLogger("miscord.worker")


class InspectionThread(QThread):
    frame_ready = Signal(object)
    defect_ready = Signal(object)
    stats_ready = Signal(dict)
    source_status = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, config: dict[str, Any], app_root: Path):
        super().__init__()
        self.setObjectName("InspectionThread")
        self.app_root = app_root
        self._lock = threading.Lock()
        self._config = copy.deepcopy(config)
        self._revision = 0
        self._stop_event = threading.Event()

    def update_config(self, config: dict[str, Any]) -> None:
        with self._lock:
            self._config = copy.deepcopy(config)
            self._revision += 1
        LOGGER.info("Runtime configuration update requested")

    def request_stop(self) -> None:
        self._stop_event.set()

    def _snapshot(self) -> tuple[dict[str, Any], int]:
        with self._lock:
            return copy.deepcopy(self._config), self._revision

    def _wait_interruptibly(self, seconds: float) -> bool:
        return self._stop_event.wait(timeout=max(0.0, seconds))

    def _open_source(
        self, config: dict[str, Any]
    ) -> tuple[FrameSource | None, bool]:
        camera = config["camera"]
        source = create_source(camera, self.app_root)
        try:
            source.open()
            return source, False
        except Exception as primary_error:
            source.close()
            LOGGER.exception("Primary frame source failed")
            self.error_occurred.emit(str(primary_error))
            if camera["mode"] != "basler" or not camera.get(
                "fallback_to_offline", False
            ):
                return None, False
            try:
                fallback = create_source(camera, self.app_root, force_offline=True)
                fallback.open()
                LOGGER.warning("Using offline fallback while Basler is unavailable")
                self.source_status.emit(
                    f"OFFLINE FALLBACK | camera error: {primary_error}"
                )
                return fallback, True
            except Exception as fallback_error:
                LOGGER.exception("Offline fallback failed")
                self.error_occurred.emit(
                    f"Camera: {primary_error}; offline fallback: {fallback_error}"
                )
                return None, False

    def run(self) -> None:
        import cv2 as cv

        threading.current_thread().name = "InspectionWorker"
        cv.setUseOptimized(True)
        config, revision = self._snapshot()
        storage = DefectStorage(config["storage"], self.app_root)
        storage.start()
        source: FrameSource | None = None
        consecutive_defects = 0
        last_saved_at = -1e9
        total_frames = 0
        window_frames = 0
        window_started = time.perf_counter()
        last_display_at = -1e9
        last_stats_at = window_started

        try:
            while not self._stop_event.is_set():
                config, revision = self._snapshot()
                if config["storage"] != storage.settings:
                    storage.close()
                    storage = DefectStorage(config["storage"], self.app_root)
                    storage.start()
                detector = CordDetector(config["detection"])
                source, using_fallback = self._open_source(config)
                if source is None:
                    self.source_status.emit("DISCONNECTED | reconnecting...")
                    if self._wait_interruptibly(
                        float(config["camera"]["reconnect_delay_seconds"])
                    ):
                        break
                    continue

                if not using_fallback:
                    self.source_status.emit(f"CONNECTED | {source.description}")
                opened_at = time.perf_counter()
                try:
                    while not self._stop_event.is_set():
                        _, current_revision = self._snapshot()
                        if current_revision != revision:
                            LOGGER.info("Restarting source to apply new settings")
                            break
                        if using_fallback and time.perf_counter() - opened_at >= max(
                            5.0, float(config["camera"]["reconnect_delay_seconds"])
                        ):
                            LOGGER.info("Retrying Basler connection from offline fallback")
                            break

                        frame = source.read()
                        if frame is None:
                            if config["camera"]["mode"] == "offline" and not config[
                                "camera"
                            ].get("offline_loop", True):
                                self.source_status.emit("OFFLINE SOURCE FINISHED")
                                self._stop_event.set()
                                break
                            continue

                        try:
                            result = detector.process(frame)
                        except Exception as exc:
                            LOGGER.exception("Frame processing failed")
                            self.error_occurred.emit(f"Processing error: {exc}")
                            continue

                        now = time.perf_counter()
                        total_frames += 1
                        window_frames += 1
                        if result.status == STATUS_DEFECT:
                            consecutive_defects += 1
                        else:
                            consecutive_defects = 0

                        confirmed = consecutive_defects >= int(
                            config["runtime"]["confirmation_frames"]
                        )
                        if confirmed:
                            cooldown = float(
                                config["runtime"]["duplicate_suppression_seconds"]
                            )
                            if now - last_saved_at >= cooldown:
                                saved_base = storage.enqueue(result)
                                last_saved_at = now
                                LOGGER.warning(
                                    "%s; saved=%s", result.summary(), saved_base
                                )
                                self.defect_ready.emit(result)

                        display_period = 1.0 / float(
                            config["runtime"]["display_max_fps"]
                        )
                        if now - last_display_at >= display_period:
                            self.frame_ready.emit(result)
                            last_display_at = now

                        stats_period = float(
                            config["runtime"]["stats_interval_seconds"]
                        )
                        if now - last_stats_at >= stats_period:
                            elapsed = max(now - window_started, 1e-6)
                            self.stats_ready.emit(
                                {
                                    "fps": window_frames / elapsed,
                                    "total_frames": total_frames,
                                    "processing_ms": result.processing_ms,
                                    "status": result.status,
                                    "thread_count": len(result.centers),
                                    "pitch_px": result.pitch_px,
                                    "quality": result.quality_score,
                                    "save_queue": storage.queue.qsize(),
                                    "dropped_saves": storage.dropped_count,
                                }
                            )
                            window_frames = 0
                            window_started = now
                            last_stats_at = now
                except Exception as exc:
                    LOGGER.exception("Frame source failed during acquisition")
                    self.error_occurred.emit(f"Acquisition error: {exc}")
                finally:
                    source.close()
                    source = None

                if not self._stop_event.is_set():
                    self.source_status.emit("RECONNECTING...")
                    if self._wait_interruptibly(
                        float(config["camera"]["reconnect_delay_seconds"])
                    ):
                        break
        finally:
            if source is not None:
                source.close()
            storage.close()
            self.source_status.emit("STOPPED")
            LOGGER.info("Inspection thread stopped; total frames=%d", total_frames)
