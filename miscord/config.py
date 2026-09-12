from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "camera": {
        "mode": "basler",
        "serial_number": "",
        "exposure_us": 800.0,
        "gain_db": 0.0,
        "pixel_format": "Mono8",
        "acquisition_frame_rate": 0.0,
        "grab_timeout_ms": 1000,
        "buffer_count": 12,
        "reconnect_delay_seconds": 2.0,
        "roi": {"x": 0, "y": 0, "width": 0, "height": 0},
        "offline_path": "5M25mmUnknowExp1.bmp",
        "offline_fps": 20.0,
        "offline_loop": True,
        "fallback_to_offline": True,
    },
    "detection": {
        "algorithm": "auto",
        "software_roi": {"x": 0, "y": 20, "width": 0, "height": 0},
        "profile_auto_aspect_ratio": 3.0,
        "profile_tophat_px": 9,
        "profile_path_x_step": 4,
        "profile_path_max_step_px": 2,
        "profile_path_smoothness": 2.0,
        "profile_band_half_height_px": 3,
        "profile_border_margin_px": 8,
        "profile_min_path_coverage": 0.90,
        "profile_presence_min_contrast": 10.0,
        "profile_presence_max_gap_px": 60,
        "profile_presence_min_run_peaks": 6,
        "profile_presence_max_pitch_px": 16.0,
        "profile_presence_max_pitch_mad_ratio": 0.45,
        "profile_recovery_min_contrast": 2.0,
        "profile_recovery_min_neighbor_ratio": 0.10,
        "profile_recovery_max_gap_ratio": 1.90,
        "profile_recovery_search_px": 3,
        "profile_complete_min_coverage": 0.90,
        "profile_complete_edge_margin_ratio": 0.04,
        "profile_background_px": 21,
        "profile_peak_window_px": 7,
        "profile_peak_percentile": 25.0,
        "profile_min_contrast": 1.0,
        "profile_min_peak_support_ratio": 0.75,
        "profile_maximum_pitch_mad_ratio": 0.35,
        "edge_exclusion_pitch_count": 5.0,
        "adaptive_block_size": 7,
        "adaptive_c": 9.0,
        "mask_erode_px": 10,
        "mask_open_px": 24,
        "intensity_threshold": 113,
        "object_open_px": 6,
        "min_area_px": 18.0,
        "max_area_px": 220.0,
        "min_radius_px": 2.5,
        "max_radius_px": 12.0,
        "min_circularity": 0.55,
        "row_tolerance_px": 12.0,
        "minimum_threads": 6,
        "nominal_pitch_px": 0.0,
        "missing_gap_ratio": 1.60,
        "maximum_pitch_mad_ratio": 0.22,
        "minimum_defect_quality_score": 0.55,
        "maximum_missing_fraction": 0.20,
        "expected_thread_count": 0,
        "draw_all_centers": True,
    },
    "runtime": {
        "display_max_fps": 25.0,
        "stats_interval_seconds": 0.5,
        "confirmation_frames": 2,
        "alarm_blink_seconds": 4.0,
        "duplicate_suppression_seconds": 1.0,
    },
    "storage": {
        "save_defects": True,
        "directory": "defects",
        "jpeg_quality": 95,
        "queue_size": 100,
    },
    "logging": {
        "directory": "logs",
        "level": "INFO",
        "max_bytes": 10_485_760,
        "backup_count": 10,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _non_negative_int(value: Any, name: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be zero or positive")
    return result


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    config = _deep_merge(DEFAULT_CONFIG, config)
    camera = config["camera"]
    detection = config["detection"]
    runtime = config["runtime"]

    camera["mode"] = str(camera["mode"]).lower()
    if camera["mode"] not in {"basler", "offline"}:
        raise ValueError("camera.mode must be 'basler' or 'offline'")
    camera["exposure_us"] = max(1.0, float(camera["exposure_us"]))
    camera["gain_db"] = max(0.0, float(camera["gain_db"]))
    camera["acquisition_frame_rate"] = max(
        0.0, float(camera["acquisition_frame_rate"])
    )
    camera["grab_timeout_ms"] = max(50, int(camera["grab_timeout_ms"]))
    camera["buffer_count"] = max(3, int(camera["buffer_count"]))
    camera["offline_fps"] = max(0.1, float(camera["offline_fps"]))
    camera["reconnect_delay_seconds"] = max(
        0.1, float(camera["reconnect_delay_seconds"])
    )
    for key in ("x", "y", "width", "height"):
        camera["roi"][key] = _non_negative_int(
            camera["roi"][key], f"camera.roi.{key}"
        )

    roi = detection["software_roi"]
    for key in ("x", "y", "width", "height"):
        roi[key] = _non_negative_int(roi[key], f"detection.software_roi.{key}")

    detection["algorithm"] = str(detection["algorithm"]).lower()
    if detection["algorithm"] not in {"auto", "profile", "contour"}:
        raise ValueError("detection.algorithm must be auto, profile, or contour")
    detection["profile_auto_aspect_ratio"] = max(
        1.0, float(detection["profile_auto_aspect_ratio"])
    )
    for key in (
        "profile_tophat_px",
        "profile_background_px",
        "profile_peak_window_px",
    ):
        value = max(3, int(detection[key]))
        detection[key] = value if value % 2 else value + 1
    detection["profile_path_x_step"] = max(
        1, int(detection["profile_path_x_step"])
    )
    detection["profile_path_max_step_px"] = min(
        8, max(1, int(detection["profile_path_max_step_px"]))
    )
    detection["profile_path_smoothness"] = max(
        0.0, float(detection["profile_path_smoothness"])
    )
    detection["profile_band_half_height_px"] = max(
        1, int(detection["profile_band_half_height_px"])
    )
    detection["profile_border_margin_px"] = max(
        0, int(detection["profile_border_margin_px"])
    )
    detection["profile_min_path_coverage"] = min(
        1.0, max(0.1, float(detection["profile_min_path_coverage"]))
    )
    detection["profile_presence_min_contrast"] = max(
        0.0, float(detection["profile_presence_min_contrast"])
    )
    detection["profile_presence_max_gap_px"] = max(
        1, int(detection["profile_presence_max_gap_px"])
    )
    detection["profile_presence_min_run_peaks"] = max(
        3, int(detection["profile_presence_min_run_peaks"])
    )
    detection["profile_presence_max_pitch_px"] = max(
        1.0, float(detection["profile_presence_max_pitch_px"])
    )
    detection["profile_presence_max_pitch_mad_ratio"] = max(
        0.05, float(detection["profile_presence_max_pitch_mad_ratio"])
    )
    detection["profile_recovery_min_contrast"] = max(
        0.0, float(detection["profile_recovery_min_contrast"])
    )
    detection["profile_recovery_min_neighbor_ratio"] = min(
        1.0, max(0.0, float(detection["profile_recovery_min_neighbor_ratio"]))
    )
    detection["profile_recovery_max_gap_ratio"] = max(
        1.1, float(detection["profile_recovery_max_gap_ratio"])
    )
    detection["profile_recovery_search_px"] = max(
        1, int(detection["profile_recovery_search_px"])
    )
    detection["profile_complete_min_coverage"] = min(
        1.0, max(0.1, float(detection["profile_complete_min_coverage"]))
    )
    detection["profile_complete_edge_margin_ratio"] = min(
        0.25,
        max(0.0, float(detection["profile_complete_edge_margin_ratio"])),
    )
    detection["profile_peak_percentile"] = min(
        95.0, max(0.0, float(detection["profile_peak_percentile"]))
    )
    detection["profile_min_contrast"] = max(
        0.0, float(detection["profile_min_contrast"])
    )
    detection["profile_min_peak_support_ratio"] = min(
        1.0,
        max(0.0, float(detection["profile_min_peak_support_ratio"])),
    )
    detection["profile_maximum_pitch_mad_ratio"] = max(
        0.05, float(detection["profile_maximum_pitch_mad_ratio"])
    )
    detection["edge_exclusion_pitch_count"] = max(
        0.0, float(detection["edge_exclusion_pitch_count"])
    )

    block_size = max(3, int(detection["adaptive_block_size"]))
    detection["adaptive_block_size"] = block_size if block_size % 2 else block_size + 1
    for key in ("mask_erode_px", "mask_open_px", "object_open_px"):
        detection[key] = max(1, int(detection[key]))
    detection["intensity_threshold"] = min(
        254, max(1, int(detection["intensity_threshold"]))
    )
    detection["min_area_px"] = max(1.0, float(detection["min_area_px"]))
    detection["max_area_px"] = max(
        detection["min_area_px"], float(detection["max_area_px"])
    )
    detection["min_radius_px"] = max(0.5, float(detection["min_radius_px"]))
    detection["max_radius_px"] = max(
        detection["min_radius_px"], float(detection["max_radius_px"])
    )
    detection["min_circularity"] = min(
        1.0, max(0.0, float(detection["min_circularity"]))
    )
    detection["row_tolerance_px"] = max(1.0, float(detection["row_tolerance_px"]))
    detection["minimum_threads"] = max(3, int(detection["minimum_threads"]))
    detection["nominal_pitch_px"] = max(0.0, float(detection["nominal_pitch_px"]))
    detection["missing_gap_ratio"] = max(
        1.1, float(detection["missing_gap_ratio"])
    )
    detection["maximum_pitch_mad_ratio"] = max(
        0.02, float(detection["maximum_pitch_mad_ratio"])
    )
    detection["minimum_defect_quality_score"] = min(
        1.0, max(0.0, float(detection["minimum_defect_quality_score"]))
    )
    detection["maximum_missing_fraction"] = min(
        1.0, max(0.01, float(detection["maximum_missing_fraction"]))
    )
    detection["expected_thread_count"] = max(
        0, int(detection["expected_thread_count"])
    )

    runtime["display_max_fps"] = max(1.0, float(runtime["display_max_fps"]))
    runtime["stats_interval_seconds"] = max(
        0.1, float(runtime["stats_interval_seconds"])
    )
    runtime["confirmation_frames"] = max(1, int(runtime["confirmation_frames"]))
    runtime["alarm_blink_seconds"] = max(
        0.5, float(runtime["alarm_blink_seconds"])
    )
    runtime["duplicate_suppression_seconds"] = max(
        0.0, float(runtime["duplicate_suppression_seconds"])
    )

    config["storage"]["jpeg_quality"] = min(
        100, max(50, int(config["storage"]["jpeg_quality"]))
    )
    config["storage"]["queue_size"] = max(
        10, int(config["storage"]["queue_size"])
    )
    return config


class ConfigManager:
    def __init__(self, path: Path, app_root: Path):
        self.path = Path(path).resolve()
        self.app_root = Path(app_root).resolve()

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            config = validate_config({})
            self.save(config)
            return config
        with self.path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
        if not isinstance(loaded, dict):
            raise ValueError("The root of config.yaml must be a mapping")
        return validate_config(loaded)

    def save(self, config: dict[str, Any]) -> dict[str, Any]:
        validated = validate_config(config)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                yaml.safe_dump(
                    validated,
                    stream,
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False,
                )
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return validated

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.app_root / path).resolve()
