from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import cv2 as cv
import numpy as np


STATUS_OK = "OK"
STATUS_DEFECT = "DEFECT"
STATUS_LOW_QUALITY = "LOW_QUALITY"
STATUS_WAITING = "WAITING"


@dataclass(frozen=True)
class CordCandidate:
    x: float
    y: float
    radius: float
    area: float
    circularity: float


@dataclass(frozen=True)
class DefectGap:
    start: tuple[float, float]
    end: tuple[float, float]
    midpoint: tuple[float, float]
    distance_px: float
    ratio: float
    estimated_missing_count: int


@dataclass(frozen=True)
class ProfileAnalysis:
    candidates: list[CordCandidate]
    path_coverage: float
    material_coverage: float
    material_complete: bool
    material_bounds: tuple[float, float] | None


@dataclass
class DetectionResult:
    status: str
    timestamp: datetime
    annotated_frame: np.ndarray
    raw_frame: np.ndarray
    centers: list[tuple[float, float]] = field(default_factory=list)
    distances_px: list[float] = field(default_factory=list)
    defects: list[DefectGap] = field(default_factory=list)
    pitch_px: float = 0.0
    pitch_mad_ratio: float = 0.0
    quality_score: float = 0.0
    path_coverage: float = 1.0
    material_coverage: float = 1.0
    message: str = ""
    processing_ms: float = 0.0

    def summary(self) -> str:
        return (
            f"status={self.status}, threads={len(self.centers)}, "
            f"pitch={self.pitch_px:.2f}px, defects={len(self.defects)}, "
            f"quality={self.quality_score:.2f}, "
            f"material={self.material_coverage:.0%}, message={self.message}"
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "message": self.message,
            "thread_count": len(self.centers),
            "pitch_px": round(self.pitch_px, 4),
            "pitch_mad_ratio": round(self.pitch_mad_ratio, 4),
            "quality_score": round(self.quality_score, 4),
            "path_coverage": round(self.path_coverage, 4),
            "material_coverage": round(self.material_coverage, 4),
            "processing_ms": round(self.processing_ms, 3),
            "centers": [[round(x, 2), round(y, 2)] for x, y in self.centers],
            "distances_px": [round(value, 3) for value in self.distances_px],
            "defects": [
                {
                    "start": [round(v, 2) for v in defect.start],
                    "end": [round(v, 2) for v in defect.end],
                    "midpoint": [round(v, 2) for v in defect.midpoint],
                    "distance_px": round(defect.distance_px, 3),
                    "gap_ratio": round(defect.ratio, 3),
                    "estimated_missing_count": defect.estimated_missing_count,
                }
                for defect in self.defects
            ],
        }


class CordDetector:
    """Detect bright tire-cord cross sections and abnormal gaps between them."""

    def __init__(self, settings: dict[str, Any]):
        self.settings = dict(settings)

    def update_settings(self, settings: dict[str, Any]) -> None:
        self.settings = dict(settings)

    @staticmethod
    def _odd(value: int) -> int:
        value = max(3, int(value))
        return value if value % 2 else value + 1

    @staticmethod
    def _kernel(size: int) -> np.ndarray:
        size = max(1, int(size))
        # Rectangular kernels intentionally match the validated optical setup.
        # Kernel sizes remain configurable because they scale with magnification.
        return np.ones((size, size), dtype=np.uint8)

    def _crop(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        height, width = frame.shape[:2]
        roi = self.settings["software_roi"]
        x = min(max(0, int(roi["x"])), max(0, width - 1))
        y = min(max(0, int(roi["y"])), max(0, height - 1))
        roi_width = int(roi["width"]) or (width - x)
        roi_height = int(roi["height"]) or (height - y)
        roi_width = min(max(1, roi_width), width - x)
        roi_height = min(max(1, roi_height), height - y)
        return frame[y : y + roi_height, x : x + roi_width], (x, y, roi_width, roi_height)

    def _segment(self, image: np.ndarray) -> np.ndarray:
        gray = self._gray(image)

        mask_seed = cv.adaptiveThreshold(
            gray,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            self._odd(self.settings["adaptive_block_size"]),
            float(self.settings["adaptive_c"]),
        )
        mask_seed = cv.erode(
            mask_seed,
            self._kernel(self.settings["mask_erode_px"]),
            iterations=1,
        )
        mask_seed = cv.bitwise_not(mask_seed)
        material_mask = cv.morphologyEx(
            mask_seed,
            cv.MORPH_OPEN,
            self._kernel(self.settings["mask_open_px"]),
        )
        masked_gray = cv.bitwise_and(gray, gray, mask=material_mask)
        _, objects = cv.threshold(
            masked_gray,
            int(self.settings["intensity_threshold"]),
            255,
            cv.THRESH_BINARY,
        )
        return cv.morphologyEx(
            objects,
            cv.MORPH_OPEN,
            self._kernel(self.settings["object_open_px"]),
        )

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.shape[2] == 4:
            return cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def _use_profile_algorithm(self, image: np.ndarray) -> bool:
        algorithm = str(self.settings.get("algorithm", "auto")).lower()
        if algorithm == "profile":
            return True
        if algorithm == "contour":
            return False
        height, width = image.shape[:2]
        return width / max(height, 1) >= float(
            self.settings["profile_auto_aspect_ratio"]
        )

    def _track_cord_path(self, gray: np.ndarray) -> np.ndarray:
        height, width = gray.shape
        top_hat_size = self._odd(self.settings["profile_tophat_px"])
        top_hat = cv.morphologyEx(
            gray,
            cv.MORPH_TOPHAT,
            cv.getStructuringElement(
                cv.MORPH_ELLIPSE, (top_hat_size, top_hat_size)
            ),
        )
        score_full = cv.GaussianBlur(
            top_hat.astype(np.float32),
            (0, 0),
            sigmaX=2.0,
            sigmaY=1.2,
        )
        x_step = int(self.settings["profile_path_x_step"])
        sampled_score = score_full[:, ::x_step]
        sampled_width = sampled_score.shape[1]
        if sampled_width < 2:
            return np.full(width, height // 2, dtype=np.int32)

        max_step = int(self.settings["profile_path_max_step_px"])
        smoothness = float(self.settings["profile_path_smoothness"])
        offsets = list(range(-max_step, max_step + 1))
        dynamic = sampled_score[:, 0].copy()
        backtrack = np.zeros((height, sampled_width), dtype=np.int8)
        rows = np.arange(height)

        for column in range(1, sampled_width):
            options = np.full(
                (len(offsets), height), -1e9, dtype=np.float32
            )
            for option_index, offset in enumerate(offsets):
                if offset < 0:
                    options[option_index, :offset] = (
                        dynamic[-offset:] - smoothness * abs(offset)
                    )
                elif offset > 0:
                    options[option_index, offset:] = (
                        dynamic[:-offset] - smoothness * abs(offset)
                    )
                else:
                    options[option_index] = dynamic
            best = np.argmax(options, axis=0)
            dynamic = sampled_score[:, column] + options[best, rows]
            backtrack[:, column] = best.astype(np.int16) - max_step

        sampled_path = np.empty(sampled_width, dtype=np.int32)
        sampled_path[-1] = int(np.argmax(dynamic))
        for column in range(sampled_width - 1, 0, -1):
            offset = int(backtrack[sampled_path[column], column])
            sampled_path[column - 1] = sampled_path[column] - offset

        sampled_x = np.arange(0, width, x_step)
        path = np.rint(
            np.interp(np.arange(width), sampled_x, sampled_path)
        ).astype(np.int32)
        return np.clip(path, 0, height - 1)

    def _profile_candidates(
        self, image: np.ndarray
    ) -> ProfileAnalysis:
        gray = self._gray(image)
        height, width = gray.shape
        if height < 12 or width < 20:
            return ProfileAnalysis([], 0.0, 0.0, False, None)
        path = self._track_cord_path(gray)
        configured_margin = int(self.settings["profile_border_margin_px"])
        border_margin = min(configured_margin, max(0, (height - 1) // 2))
        if border_margin:
            valid_path = (path >= border_margin) & (
                path < height - border_margin
            )
        else:
            valid_path = np.ones(width, dtype=bool)
        path_coverage = float(np.mean(valid_path))
        x = np.arange(width)
        half_height = int(self.settings["profile_band_half_height_px"])
        samples = np.stack(
            [
                gray[np.clip(path + offset, 0, height - 1), x]
                for offset in range(-half_height, half_height + 1)
            ]
        )
        brightest = np.max(samples, axis=0).astype(np.float32)
        second_brightest = np.partition(samples, -2, axis=0)[-2].astype(
            np.float32
        )
        vertical_support = second_brightest / np.maximum(brightest, 1.0)
        line = brightest
        horizontal_samples = np.stack(
            (
                np.concatenate((line[:1], line[:-1])),
                line,
                np.concatenate((line[1:], line[-1:])),
            )
        )
        horizontal_support = np.partition(
            horizontal_samples, -2, axis=0
        )[-2] / np.maximum(np.max(horizontal_samples, axis=0), 1.0)
        peak_support = np.minimum(vertical_support, horizontal_support)
        smoothed = cv.GaussianBlur(line.reshape(1, -1), (0, 0), 1.0).ravel()
        background_size = self._odd(self.settings["profile_background_px"])
        background = cv.morphologyEx(
            smoothed.reshape(1, -1),
            cv.MORPH_OPEN,
            np.ones((1, background_size), dtype=np.float32),
        ).ravel()
        response = smoothed - background
        peak_window = self._odd(self.settings["profile_peak_window_px"])
        local_maximum = cv.dilate(
            response.reshape(1, -1),
            np.ones((1, peak_window), dtype=np.float32),
        ).ravel()
        threshold = max(
            float(self.settings["profile_min_contrast"]),
            float(
                np.percentile(
                    response, float(self.settings["profile_peak_percentile"])
                )
            ),
        )
        peak_x = np.flatnonzero(
            (response >= local_maximum - 1e-6)
            & (response >= threshold)
            & valid_path
            & (
                peak_support
                >= float(self.settings["profile_min_peak_support_ratio"])
            )
        )
        radius = max(2.0, peak_window / 2.0)
        all_candidates = [
            CordCandidate(
                x=float(peak),
                y=float(path[peak]),
                radius=radius,
                area=float(response[peak]),
                circularity=1.0,
            )
            for peak in peak_x
        ]

        # Presence is derived from the already-computed profile peaks, so it
        # adds no second image-processing pass. Weak background peaks cannot
        # form the dense, high-contrast chain produced by real tire cords.
        presence_contrast = float(
            self.settings["profile_presence_min_contrast"]
        )
        maximum_gap = float(self.settings["profile_presence_max_gap_px"])
        strong = [item for item in all_candidates if item.area >= presence_contrast]
        runs: list[list[CordCandidate]] = []
        run: list[CordCandidate] = []
        for item in strong:
            if run and item.x - run[-1].x > maximum_gap:
                runs.append(run)
                run = []
            run.append(item)
        if run:
            runs.append(run)
        minimum_run = int(self.settings["profile_presence_min_run_peaks"])
        strongest_run = self._select_presence_run(runs, minimum_run)
        if not strongest_run:
            return ProfileAnalysis([], path_coverage, 0.0, False, None)

        material_left = strongest_run[0].x
        material_right = strongest_run[-1].x
        material_candidates = [
            item
            for item in all_candidates
            if material_left <= item.x <= material_right
        ]
        material_candidates = self._recover_weak_profile_peaks(
            material_candidates,
            response,
            path,
            valid_path,
            peak_support,
        )
        material_coverage = max(
            0.0,
            min(1.0, (material_right - material_left) / max(width - 1, 1)),
        )
        edge_margin = (
            float(self.settings["profile_complete_edge_margin_ratio"]) * width
        )
        material_complete = (
            material_coverage
            >= float(self.settings["profile_complete_min_coverage"])
            and material_left <= edge_margin
            and (width - 1 - material_right) <= edge_margin
        )
        return ProfileAnalysis(
            material_candidates,
            path_coverage,
            material_coverage,
            material_complete,
            (material_left, material_right),
        )

    def _select_presence_run(
        self,
        runs: list[list[CordCandidate]],
        minimum_run: int,
    ) -> list[CordCandidate]:
        valid_runs = [
            candidate_run
            for candidate_run in runs
            if len(candidate_run) >= minimum_run
            and self._is_regular_presence_run(candidate_run)
        ]
        # A long irregular conveyor reflection must not hide a shorter real
        # layer in the middle of the image.
        return max(valid_runs, key=len, default=[])

    def _is_regular_presence_run(
        self, candidates: list[CordCandidate]
    ) -> bool:
        points = np.asarray(
            [(item.x, item.y) for item in candidates], dtype=np.float64
        )
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if distances.size == 0:
            return False
        pitch = float(np.median(distances))
        mad_ratio = float(
            1.4826
            * np.median(np.abs(distances - pitch))
            / max(pitch, 1e-6)
        )
        return (
            pitch <= float(self.settings["profile_presence_max_pitch_px"])
            and mad_ratio
            <= float(self.settings["profile_presence_max_pitch_mad_ratio"])
        )

    def _recover_weak_profile_peaks(
        self,
        candidates: list[CordCandidate],
        response: np.ndarray,
        path: np.ndarray,
        valid_path: np.ndarray,
        peak_support: np.ndarray,
    ) -> list[CordCandidate]:
        """Recover a dim cord only when it explains one suspicious gap."""
        if len(candidates) < 4:
            return candidates

        points = np.asarray(
            [(item.x, item.y) for item in candidates], dtype=np.float64
        )
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        pitch = float(np.median(distances))
        if pitch <= 0:
            return candidates

        minimum_gap_ratio = float(self.settings["missing_gap_ratio"])
        maximum_gap_ratio = float(
            self.settings["profile_recovery_max_gap_ratio"]
        )
        minimum_contrast = float(
            self.settings["profile_recovery_min_contrast"]
        )
        minimum_neighbor_ratio = float(
            self.settings["profile_recovery_min_neighbor_ratio"]
        )
        search_px = int(self.settings["profile_recovery_search_px"])
        weak_local_maximum = cv.dilate(
            response.reshape(1, -1), np.ones((1, 3), dtype=np.float32)
        ).ravel()
        recovered: list[CordCandidate] = []

        for index, distance in enumerate(distances):
            gap_ratio = float(distance / pitch)
            if not minimum_gap_ratio <= gap_ratio <= maximum_gap_ratio:
                continue

            start = candidates[index]
            end = candidates[index + 1]
            expected_x = 0.5 * (start.x + end.x)
            search_left = max(
                int(np.ceil(start.x)) + 1,
                int(round(expected_x)) - search_px,
            )
            search_right = min(
                int(np.floor(end.x)) - 1,
                int(round(expected_x)) + search_px,
            )
            if search_right < search_left:
                continue

            positions = np.arange(search_left, search_right + 1)
            weak_maxima = positions[
                valid_path[positions]
                & (
                    peak_support[positions]
                    >= float(
                        self.settings["profile_min_peak_support_ratio"]
                    )
                )
                & (
                    response[positions]
                    >= weak_local_maximum[positions] - 1e-6
                )
            ]
            if weak_maxima.size == 0:
                continue
            peak = int(weak_maxima[np.argmax(response[weak_maxima])])
            contrast = float(response[peak])
            neighbor_contrast = min(start.area, end.area)
            if (
                contrast < minimum_contrast
                or contrast < minimum_neighbor_ratio * neighbor_contrast
            ):
                continue
            recovered.append(
                CordCandidate(
                    x=float(peak),
                    y=float(path[peak]),
                    radius=max(
                        2.0,
                        self._odd(self.settings["profile_peak_window_px"]) / 2.0,
                    ),
                    area=contrast,
                    circularity=1.0,
                )
            )

        if not recovered:
            return candidates
        return sorted([*candidates, *recovered], key=lambda item: item.x)

    def _candidates(self, binary: np.ndarray) -> list[CordCandidate]:
        contours, _ = cv.findContours(
            binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        candidates: list[CordCandidate] = []
        for contour in contours:
            area = float(cv.contourArea(contour))
            if not self.settings["min_area_px"] <= area <= self.settings["max_area_px"]:
                continue
            perimeter = float(cv.arcLength(contour, True))
            if perimeter <= 0:
                continue
            circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
            if circularity < self.settings["min_circularity"]:
                continue
            (x, y), radius = cv.minEnclosingCircle(contour)
            if not self.settings["min_radius_px"] <= radius <= self.settings["max_radius_px"]:
                continue
            candidates.append(CordCandidate(x, y, float(radius), area, circularity))
        return candidates

    def _select_row(self, candidates: list[CordCandidate]) -> list[CordCandidate]:
        if len(candidates) < 3:
            return sorted(candidates, key=lambda item: item.x)
        x = np.asarray([item.x for item in candidates], dtype=np.float64)
        y = np.asarray([item.y for item in candidates], dtype=np.float64)
        active = np.ones(len(candidates), dtype=bool)
        tolerance = float(self.settings["row_tolerance_px"])
        for _ in range(4):
            count = int(active.sum())
            if count < 3:
                break
            degree = 2 if count >= 5 else 1
            coefficients = np.polyfit(x[active], y[active], degree)
            residuals = np.abs(y - np.polyval(coefficients, x))
            updated = residuals <= tolerance
            if np.array_equal(updated, active) or int(updated.sum()) < 3:
                break
            active = updated
        row = [item for item, keep in zip(candidates, active, strict=True) if keep]
        return sorted(row, key=lambda item: item.x)

    def _low_quality_result(
        self,
        frame: np.ndarray,
        centers: list[tuple[float, float]],
        message: str,
        roi: tuple[int, int, int, int],
    ) -> DetectionResult:
        annotated = self._as_bgr(frame)
        self._draw_header(annotated, STATUS_LOW_QUALITY, message)
        self._draw_roi(annotated, roi)
        return DetectionResult(
            status=STATUS_LOW_QUALITY,
            timestamp=datetime.now(timezone.utc),
            annotated_frame=annotated,
            raw_frame=frame,
            centers=centers,
            quality_score=0.0,
            message=message,
        )

    def _waiting_result(
        self,
        frame: np.ndarray,
        centers: list[tuple[float, float]],
        material_coverage: float,
        path_coverage: float,
        roi: tuple[int, int, int, int],
    ) -> DetectionResult:
        if centers:
            message = (
                "Waiting for complete tire layer: "
                f"detected cords={len(centers)}, "
                f"coverage={material_coverage:.0%}"
            )
        else:
            message = "Waiting for tire layer"
        annotated = self._as_bgr(frame)
        self._draw_roi(annotated, roi)
        self._draw_centers(annotated, centers)
        self._draw_header(annotated, STATUS_WAITING, message)
        return DetectionResult(
            status=STATUS_WAITING,
            timestamp=datetime.now(timezone.utc),
            annotated_frame=annotated,
            raw_frame=frame,
            centers=centers,
            path_coverage=path_coverage,
            material_coverage=material_coverage,
            message=message,
        )

    def process(self, frame: np.ndarray) -> DetectionResult:
        start_tick = cv.getTickCount()
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame received")

        image, roi = self._crop(frame)
        profile_mode = self._use_profile_algorithm(image)
        path_coverage = 1.0
        material_coverage = 1.0
        material_complete = True
        material_bounds: tuple[float, float] | None = None
        if profile_mode:
            profile = self._profile_candidates(image)
            row = profile.candidates
            path_coverage = profile.path_coverage
            material_coverage = profile.material_coverage
            material_complete = profile.material_complete
            material_bounds = profile.material_bounds
        else:
            binary = self._segment(image)
            row = self._select_row(self._candidates(binary))
        offset_x, offset_y, _, _ = roi
        centers = [(item.x + offset_x, item.y + offset_y) for item in row]

        minimum_path_coverage = float(
            self.settings["profile_min_path_coverage"]
        )
        if profile_mode and material_bounds is None:
            result = self._waiting_result(
                frame,
                centers,
                material_coverage,
                path_coverage,
                roi,
            )
            result.processing_ms = self._elapsed_ms(start_tick)
            return result

        if profile_mode and path_coverage < minimum_path_coverage:
            result = self._low_quality_result(
                frame,
                centers,
                (
                    "Cord path reaches software ROI border: "
                    f"coverage={path_coverage:.0%}"
                ),
                roi,
            )
            result.path_coverage = path_coverage
            result.material_coverage = material_coverage
            result.processing_ms = self._elapsed_ms(start_tick)
            return result

        if len(row) < int(self.settings["minimum_threads"]):
            result = self._low_quality_result(
                frame,
                centers,
                f"Too few valid cords: {len(row)}",
                roi,
            )
            result.processing_ms = self._elapsed_ms(start_tick)
            result.path_coverage = path_coverage
            result.material_coverage = material_coverage
            return result

        local_points = np.asarray([(item.x, item.y) for item in row], dtype=np.float64)
        vectors = np.diff(local_points, axis=0)
        distances = np.linalg.norm(vectors, axis=1)
        configured_pitch = float(self.settings["nominal_pitch_px"])
        pitch = configured_pitch if configured_pitch > 0 else float(np.median(distances))
        if pitch <= 0:
            result = self._low_quality_result(
                frame, centers, "Invalid pitch estimate", roi
            )
            result.processing_ms = self._elapsed_ms(start_tick)
            result.path_coverage = path_coverage
            result.material_coverage = material_coverage
            return result

        gap_limit = float(self.settings["missing_gap_ratio"]) * pitch
        regular = distances[distances < gap_limit]
        if regular.size < max(3, len(distances) // 2):
            result = self._low_quality_result(
                frame, centers, "Unstable spacing: insufficient regular gaps", roi
            )
            result.processing_ms = self._elapsed_ms(start_tick)
            result.path_coverage = path_coverage
            result.material_coverage = material_coverage
            return result

        median_regular = float(np.median(regular))
        mad = float(np.median(np.abs(regular - median_regular)))
        mad_ratio = 1.4826 * mad / max(pitch, 1e-6)
        close_gap_count = int(np.sum(distances < 0.55 * pitch))
        allowed_close_gaps = max(1, len(distances) // 20)
        maximum_mad_ratio = float(
            self.settings["profile_maximum_pitch_mad_ratio"]
            if profile_mode
            else self.settings["maximum_pitch_mad_ratio"]
        )
        if (
            mad_ratio > maximum_mad_ratio
            or close_gap_count > allowed_close_gaps
        ):
            message = (
                f"Unstable segmentation: MAD={mad_ratio:.2f}, "
                f"close gaps={close_gap_count}"
            )
            result = self._low_quality_result(frame, centers, message, roi)
            result.pitch_px = pitch
            result.pitch_mad_ratio = mad_ratio
            result.distances_px = distances.tolist()
            result.processing_ms = self._elapsed_ms(start_tick)
            result.path_coverage = path_coverage
            result.material_coverage = material_coverage
            return result

        defects: list[DefectGap] = []
        edge_margin = float(self.settings["edge_exclusion_pitch_count"]) * pitch
        if profile_mode and material_bounds is not None:
            inspection_left = float(offset_x) + material_bounds[0]
            inspection_right = float(offset_x) + material_bounds[1]
        else:
            inspection_left = float(offset_x)
            inspection_right = float(offset_x + roi[2])
        for index, distance in enumerate(distances):
            ratio = float(distance / pitch)
            if ratio < float(self.settings["missing_gap_ratio"]):
                continue
            start = centers[index]
            end = centers[index + 1]
            midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
            if (
                midpoint[0] - inspection_left < edge_margin
                or inspection_right - midpoint[0] < edge_margin
            ):
                continue
            missing_count = max(1, int(round(ratio)) - 1)
            defects.append(
                DefectGap(
                    start=start,
                    end=end,
                    midpoint=midpoint,
                    distance_px=float(distance),
                    ratio=ratio,
                    estimated_missing_count=missing_count,
                )
            )

        expected_count = int(self.settings["expected_thread_count"])
        count_deficit = (
            max(0, expected_count - len(centers))
            if expected_count and material_complete
            else 0
        )
        status = STATUS_DEFECT if defects or count_deficit else STATUS_OK
        if defects:
            missing_total = sum(item.estimated_missing_count for item in defects)
            message = f"Missing cord detected: estimated count={missing_total}"
        elif count_deficit:
            message = f"Cord count below expected value by {count_deficit}"
        elif profile_mode and not material_complete:
            message = (
                "Visible layer spacing is normal: "
                f"coverage={material_coverage:.0%}"
            )
        else:
            message = "Spacing is normal"

        quality_score = max(
            0.0,
            min(
                1.0,
                1.0
                - mad_ratio / max(maximum_mad_ratio, 1e-6),
            ),
        )
        estimated_missing = sum(item.estimated_missing_count for item in defects)
        estimated_total = len(centers) + estimated_missing
        missing_fraction = estimated_missing / max(estimated_total, 1)
        if defects and (
            quality_score < float(self.settings["minimum_defect_quality_score"])
            or missing_fraction > float(self.settings["maximum_missing_fraction"])
        ):
            quality_message = (
                "Possible gaps rejected by quality gate: "
                f"quality={quality_score:.2f}, missing fraction={missing_fraction:.2f}"
            )
            annotated = self._annotate(
                frame,
                centers,
                defects,
                STATUS_LOW_QUALITY,
                quality_message,
                roi,
                pitch,
            )
            return DetectionResult(
                status=STATUS_LOW_QUALITY,
                timestamp=datetime.now(timezone.utc),
                annotated_frame=annotated,
                raw_frame=frame,
                centers=centers,
                distances_px=distances.tolist(),
                defects=defects,
                pitch_px=pitch,
                pitch_mad_ratio=mad_ratio,
                quality_score=quality_score,
                path_coverage=path_coverage,
                material_coverage=material_coverage,
                message=quality_message,
                processing_ms=self._elapsed_ms(start_tick),
            )
        annotated = self._annotate(frame, centers, defects, status, message, roi, pitch)
        return DetectionResult(
            status=status,
            timestamp=datetime.now(timezone.utc),
            annotated_frame=annotated,
            raw_frame=frame,
            centers=centers,
            distances_px=distances.tolist(),
            defects=defects,
            pitch_px=pitch,
            pitch_mad_ratio=mad_ratio,
            quality_score=quality_score,
            path_coverage=path_coverage,
            material_coverage=material_coverage,
            message=message,
            processing_ms=self._elapsed_ms(start_tick),
        )

    @staticmethod
    def _elapsed_ms(start_tick: int) -> float:
        return (cv.getTickCount() - start_tick) * 1000.0 / cv.getTickFrequency()

    @staticmethod
    def _as_bgr(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        if frame.shape[2] == 4:
            return cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
        return frame.copy()

    @staticmethod
    def _draw_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> None:
        x, y, width, height = roi
        if x or y or width != frame.shape[1] or height != frame.shape[0]:
            cv.rectangle(frame, (x, y), (x + width - 1, y + height - 1), (255, 255, 0), 1)

    @staticmethod
    def _draw_header(frame: np.ndarray, status: str, message: str) -> None:
        colors = {
            STATUS_OK: (20, 130, 20),
            STATUS_DEFECT: (0, 0, 220),
            STATUS_LOW_QUALITY: (0, 170, 230),
            STATUS_WAITING: (150, 95, 25),
        }
        color = colors[status]
        cv.rectangle(frame, (0, 0), (frame.shape[1], 34), color, -1)
        cv.putText(
            frame,
            f"{status} | {message}",
            (8, 23),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )

    def _annotate(
        self,
        frame: np.ndarray,
        centers: list[tuple[float, float]],
        defects: list[DefectGap],
        status: str,
        message: str,
        roi: tuple[int, int, int, int],
        pitch: float,
    ) -> np.ndarray:
        annotated = self._as_bgr(frame)
        self._draw_roi(annotated, roi)
        self._draw_centers(annotated, centers)

        for defect in defects:
            start = tuple(int(round(value)) for value in defect.start)
            end = tuple(int(round(value)) for value in defect.end)
            midpoint = tuple(int(round(value)) for value in defect.midpoint)
            radius = max(12, int(round(defect.distance_px * 0.55)))
            cv.line(annotated, start, end, (0, 0, 255), 3, cv.LINE_AA)
            cv.circle(annotated, midpoint, radius, (0, 0, 255), 3, cv.LINE_AA)
            cv.putText(
                annotated,
                f"GAP {defect.distance_px:.1f}px ({defect.ratio:.2f}x)",
                (max(2, midpoint[0] - 80), max(52, midpoint[1] - radius - 8)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
        self._draw_header(annotated, status, f"{message}; pitch={pitch:.1f}px")
        return annotated

    def _draw_centers(
        self,
        annotated: np.ndarray,
        centers: list[tuple[float, float]],
    ) -> None:
        if self.settings.get("draw_all_centers", True):
            for center in centers:
                point = (int(round(center[0])), int(round(center[1])))
                cv.circle(annotated, point, 3, (0, 255, 0), -1, cv.LINE_AA)
            if len(centers) > 1:
                points = np.asarray(centers, dtype=np.int32).reshape((-1, 1, 2))
                cv.polylines(annotated, [points], False, (0, 180, 0), 1, cv.LINE_AA)
