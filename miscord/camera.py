from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np


LOGGER = logging.getLogger("miscord.camera")
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _device_property(device: Any, method_name: str, default: str = "") -> str:
    try:
        method = getattr(device, method_name)
        return str(method())
    except Exception:
        return default


def enumerate_basler_devices() -> list[dict[str, str]]:
    """Return display-safe information for every currently connected Basler camera."""
    from pypylon import pylon

    factory = pylon.TlFactory.GetInstance()
    devices: list[dict[str, str]] = []
    for device in factory.EnumerateDevices():
        model = _device_property(device, "GetModelName", "Unknown model")
        serial = _device_property(device, "GetSerialNumber", "")
        transport = _device_property(device, "GetDeviceClass", "Unknown transport")
        friendly_name = _device_property(device, "GetFriendlyName", model)
        user_name = _device_property(device, "GetUserDefinedName", "")
        ip_address = _device_property(device, "GetIpAddress", "")
        parts = [model, f"S/N {serial or '?'}", transport]
        if ip_address:
            parts.append(f"IP {ip_address}")
        if user_name:
            parts.append(user_name)
        devices.append(
            {
                "model": model,
                "serial_number": serial,
                "transport": transport,
                "friendly_name": friendly_name,
                "user_name": user_name,
                "ip_address": ip_address,
                "label": " | ".join(parts),
            }
        )
    devices.sort(key=lambda item: (item["model"], item["serial_number"]))
    return devices


class FrameSource(ABC):
    description = "Unknown source"

    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self) -> np.ndarray | None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class BaslerCameraSource(FrameSource):
    def __init__(self, settings: dict[str, Any]):
        self.settings = settings
        self.camera: Any = None
        self.description = "Basler camera"

    @staticmethod
    def _available(node: Any) -> bool:
        try:
            from pypylon import genicam

            return bool(genicam.IsAvailable(node))
        except Exception:
            return node is not None

    @staticmethod
    def _writable(node: Any) -> bool:
        try:
            from pypylon import genicam

            return bool(genicam.IsWritable(node))
        except Exception:
            return node is not None

    def _node(self, name: str) -> Any | None:
        try:
            node = getattr(self.camera, name)
            return node if self._available(node) else None
        except Exception:
            return None

    def _set_enum(self, name: str, value: str, required: bool = False) -> bool:
        node = self._node(name)
        if node is None or not self._writable(node):
            if required:
                raise RuntimeError(f"Required camera feature is not writable: {name}")
            LOGGER.warning("Camera feature is unavailable/read-only: %s", name)
            return False
        try:
            node.SetValue(str(value))
            return True
        except Exception as exc:
            if required:
                raise RuntimeError(f"Cannot set {name}={value}: {exc}") from exc
            LOGGER.warning("Cannot set camera feature %s=%s: %s", name, value, exc)
            return False

    def _set_bool(self, name: str, value: bool) -> bool:
        node = self._node(name)
        if node is None or not self._writable(node):
            return False
        try:
            node.SetValue(bool(value))
            return True
        except Exception as exc:
            LOGGER.warning("Cannot set camera feature %s=%s: %s", name, value, exc)
            return False

    def _set_number(self, name: str, value: float) -> float | None:
        node = self._node(name)
        if node is None or not self._writable(node):
            LOGGER.warning("Camera feature is unavailable/read-only: %s", name)
            return None
        try:
            minimum = float(node.Min)
            maximum = float(node.Max)
            target = min(max(float(value), minimum), maximum)
            node.SetValue(target)
            return float(node.GetValue())
        except Exception as exc:
            LOGGER.warning("Cannot set camera feature %s=%s: %s", name, value, exc)
            return None

    def _set_integer(self, name: str, value: int, use_max_for_zero: bool = False) -> int:
        node = self._node(name)
        if node is None or not self._writable(node):
            raise RuntimeError(f"Camera ROI feature is not writable: {name}")
        minimum = int(node.Min)
        maximum = int(node.Max)
        increment = max(1, int(getattr(node, "Inc", 1)))
        requested = maximum if use_max_for_zero and int(value) == 0 else int(value)
        requested = min(max(requested, minimum), maximum)
        aligned = minimum + ((requested - minimum) // increment) * increment
        node.SetValue(aligned)
        return int(node.GetValue())

    def _configure_roi(self) -> dict[str, int]:
        roi = self.settings["roi"]
        # Reset offsets first so the requested width/height maxima are valid.
        self._set_integer("OffsetX", 0)
        self._set_integer("OffsetY", 0)
        width = self._set_integer("Width", int(roi["width"]), use_max_for_zero=True)
        height = self._set_integer("Height", int(roi["height"]), use_max_for_zero=True)
        x = self._set_integer("OffsetX", int(roi["x"]))
        y = self._set_integer("OffsetY", int(roi["y"]))
        return {"x": x, "y": y, "width": width, "height": height}

    def open(self) -> None:
        from pypylon import pylon

        factory = pylon.TlFactory.GetInstance()
        devices = list(factory.EnumerateDevices())
        if not devices:
            raise RuntimeError("No Basler camera was found")

        requested_serial = str(self.settings.get("serial_number", "")).strip()
        selected = None
        for device in devices:
            if not requested_serial or device.GetSerialNumber() == requested_serial:
                selected = device
                break
        if selected is None:
            available = ", ".join(device.GetSerialNumber() for device in devices)
            raise RuntimeError(
                f"Basler serial {requested_serial!r} was not found; available: {available}"
            )

        self.camera = pylon.InstantCamera(factory.CreateDevice(selected))
        self.camera.Open()
        try:
            self._set_enum("AcquisitionMode", "Continuous")
            self._set_enum("TriggerMode", "Off")
            self._set_enum("ExposureAuto", "Off")
            self._set_enum("GainAuto", "Off")
            self._set_enum(
                "PixelFormat", str(self.settings.get("pixel_format", "Mono8")), required=True
            )
            actual_roi = self._configure_roi()
            exposure = self._set_number("ExposureTime", self.settings["exposure_us"])
            if exposure is None:
                exposure = self._set_number("ExposureTimeAbs", self.settings["exposure_us"])
            gain = self._set_number("Gain", self.settings["gain_db"])
            if gain is None:
                gain = self._set_number("GainRaw", self.settings["gain_db"])

            requested_fps = float(self.settings.get("acquisition_frame_rate", 0.0))
            if requested_fps > 0:
                self._set_bool("AcquisitionFrameRateEnable", True)
                actual_fps = self._set_number("AcquisitionFrameRate", requested_fps)
                if actual_fps is None:
                    actual_fps = self._set_number("AcquisitionFrameRateAbs", requested_fps)
            else:
                self._set_bool("AcquisitionFrameRateEnable", False)
                actual_fps = None

            try:
                self.camera.MaxNumBuffer = int(self.settings["buffer_count"])
            except Exception as exc:
                LOGGER.warning("Cannot set camera buffer count: %s", exc)

            self.camera.StartGrabbing(
                pylon.GrabStrategy_LatestImageOnly,
                pylon.GrabLoop_ProvidedByUser,
            )
            self.description = (
                f"Basler {selected.GetModelName()} | S/N {selected.GetSerialNumber()} | "
                f"ROI {actual_roi['width']}x{actual_roi['height']}+"
                f"{actual_roi['x']}+{actual_roi['y']} | exposure={exposure}us | "
                f"gain={gain} | fps={'max' if actual_fps is None else f'{actual_fps:.1f}'}"
            )
            LOGGER.info("Camera opened: %s", self.description)
        except Exception:
            self.close()
            raise

    def read(self) -> np.ndarray | None:
        from pypylon import pylon

        if self.camera is None or not self.camera.IsGrabbing():
            raise RuntimeError("Camera is not grabbing")
        result = self.camera.RetrieveResult(
            int(self.settings["grab_timeout_ms"]), pylon.TimeoutHandling_Return
        )
        if result is None:
            return None
        try:
            if not result.GrabSucceeded():
                raise RuntimeError(
                    f"Basler grab failed {result.GetErrorCode()}: "
                    f"{result.GetErrorDescription()}"
                )
            array = result.Array
            return np.asarray(array).copy()
        finally:
            result.Release()

    def close(self) -> None:
        camera, self.camera = self.camera, None
        if camera is None:
            return
        try:
            if camera.IsGrabbing():
                camera.StopGrabbing()
        except Exception:
            LOGGER.exception("Error while stopping Basler acquisition")
        try:
            if camera.IsOpen():
                camera.Close()
        except Exception:
            LOGGER.exception("Error while closing Basler camera")


class OfflineFrameSource(FrameSource):
    def __init__(self, settings: dict[str, Any], path: Path):
        self.settings = settings
        self.path = path
        self.images: list[Path] = []
        self.index = 0
        self.capture: cv.VideoCapture | None = None
        self.next_frame_time = 0.0
        self.description = f"Offline: {path}"

    def open(self) -> None:
        if not self.path.exists():
            raise RuntimeError(f"Offline source does not exist: {self.path}")
        if self.path.is_dir():
            self.images = sorted(
                item
                for item in self.path.iterdir()
                if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not self.images:
                raise RuntimeError(f"No supported images found in {self.path}")
        elif self.path.suffix.lower() in IMAGE_EXTENSIONS:
            self.images = [self.path]
        else:
            self.capture = cv.VideoCapture(str(self.path))
            if not self.capture.isOpened():
                raise RuntimeError(f"Cannot open offline video: {self.path}")
        self.next_frame_time = time.perf_counter()
        LOGGER.info("Offline source opened: %s", self.path)

    def _throttle(self) -> None:
        period = 1.0 / float(self.settings["offline_fps"])
        now = time.perf_counter()
        remaining = self.next_frame_time - now
        if remaining > 0:
            time.sleep(remaining)
        self.next_frame_time = max(self.next_frame_time + period, time.perf_counter())

    def read(self) -> np.ndarray | None:
        self._throttle()
        if self.images:
            if self.index >= len(self.images):
                if not self.settings.get("offline_loop", True):
                    return None
                self.index = 0
            path = self.images[self.index]
            self.index += 1
            image = cv.imread(str(path), cv.IMREAD_UNCHANGED)
            if image is None:
                raise RuntimeError(f"Cannot read offline image: {path}")
            return image

        if self.capture is None:
            return None
        ok, frame = self.capture.read()
        if ok:
            return frame
        if not self.settings.get("offline_loop", True):
            return None
        self.capture.set(cv.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.capture.read()
        return frame if ok else None

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def create_source(
    settings: dict[str, Any], app_root: Path, force_offline: bool = False
) -> FrameSource:
    if settings["mode"] == "basler" and not force_offline:
        return BaslerCameraSource(settings)
    path = Path(str(settings["offline_path"]))
    if not path.is_absolute():
        path = (app_root / path).resolve()
    return OfflineFrameSource(settings, path)
