from __future__ import annotations

import copy
import tempfile
import time
import unittest
from pathlib import Path

import cv2 as cv
import numpy as np

from miscord.camera import OfflineFrameSource
from miscord.config import ConfigManager, DEFAULT_CONFIG, validate_config
from miscord.detector import (
    CordCandidate,
    CordDetector,
    STATUS_DEFECT,
    STATUS_LOW_QUALITY,
    STATUS_OK,
    STATUS_WAITING,
)
from miscord.storage import DefectStorage


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "5M25mmUnknowExp1.bmp"
if not SAMPLE.exists():
    SAMPLE = ROOT / "dist" / "MiscordInspector" / SAMPLE.name
LEGACY_IMAGES = ROOT / "images"
RECENT_IMAGES = ROOT / "new"
FULL_LAYER_IMAGES = (
    "Image__2026-08-25__10-06-48.bmp",
    "Image__2026-08-25__10-06-50.bmp",
    "Image__2026-08-25__10-06-52.bmp",
)
PARTIAL_LAYER_IMAGES = (
    "Image__2026-08-25__11-02-53.bmp",
    "Image__2026-08-25__11-03-07.bmp",
)
EMPTY_LAYER_IMAGES = (
    "Image__2026-08-25__11-05-30.bmp",
    "Image__2026-08-25__11-05-33.bmp",
)


def remove_profile_cord(image: np.ndarray, x: int) -> np.ndarray:
    modified = image.copy()
    half_width = 5
    left = image[:, x - half_width - 1].astype(float)
    right = image[:, x + half_width + 1].astype(float)
    for column in range(x - half_width, x + half_width + 1):
        weight = (column - (x - half_width) + 1) / (2 * half_width + 2)
        modified[:, column] = np.rint(left + (right - left) * weight).astype(
            image.dtype
        )
    return modified


class ConfigTests(unittest.TestCase):
    def test_defaults_merge_and_even_block_is_corrected(self) -> None:
        config = validate_config({"detection": {"adaptive_block_size": 8}})
        self.assertEqual(config["detection"]["adaptive_block_size"], 9)
        self.assertEqual(
            config["detection"]["profile_presence_min_run_peaks"], 6
        )
        self.assertEqual(
            config["detection"]["profile_min_peak_support_ratio"], 0.75
        )
        self.assertEqual(config["camera"]["mode"], "basler")

    def test_atomic_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = ConfigManager(root / "settings.yaml", root)
            config = manager.save(copy.deepcopy(DEFAULT_CONFIG))
            self.assertEqual(manager.load(), config)


@unittest.skipUnless(SAMPLE.exists(), "sample image is not present")
class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ConfigManager(ROOT / "config.yaml", ROOT).load()
        cls.detector = CordDetector(cls.config["detection"])
        cls.image = cv.imread(str(SAMPLE), cv.IMREAD_COLOR)

    def test_known_good_sample(self) -> None:
        result = self.detector.process(self.image)
        self.assertEqual(result.status, STATUS_OK, result.summary())
        self.assertGreaterEqual(len(result.centers), 35)
        self.assertAlmostEqual(result.pitch_px, 14.1, delta=1.0)

    def test_synthetic_missing_cord(self) -> None:
        baseline = self.detector.process(self.image)
        target = baseline.centers[len(baseline.centers) // 2]
        modified = self.image.copy()
        cv.circle(
            modified,
            (round(target[0]), round(target[1])),
            9,
            (30, 40, 40),
            -1,
        )
        result = self.detector.process(modified)
        self.assertEqual(result.status, STATUS_DEFECT, result.summary())
        self.assertEqual(len(result.defects), 1)
        self.assertGreater(result.defects[0].ratio, 1.8)

    @unittest.skipUnless(LEGACY_IMAGES.exists(), "camera images are not present")
    def test_panoramic_camera_images_use_tracked_profile(self) -> None:
        images = [LEGACY_IMAGES / name for name in FULL_LAYER_IMAGES]
        images = [path for path in images if path.exists()]
        self.assertGreater(len(images), 0)
        for path in images:
            with self.subTest(path=path.name):
                image = cv.imread(str(path), cv.IMREAD_UNCHANGED)
                result = self.detector.process(image)
                self.assertEqual(result.status, STATUS_OK, result.summary())
                self.assertGreaterEqual(len(result.centers), 225)
                self.assertAlmostEqual(result.pitch_px, 10.3, delta=1.0)

    @unittest.skipUnless(LEGACY_IMAGES.exists(), "camera images are not present")
    def test_partial_layer_is_inspected_without_false_defect(self) -> None:
        images = [LEGACY_IMAGES / name for name in PARTIAL_LAYER_IMAGES]
        images = [path for path in images if path.exists()]
        self.assertGreater(len(images), 0)
        for path in images:
            with self.subTest(path=path.name):
                image = cv.imread(str(path), cv.IMREAD_UNCHANGED)
                result = self.detector.process(image)
                self.assertEqual(result.status, STATUS_OK, result.summary())
                self.assertGreaterEqual(len(result.centers), 80)
                self.assertEqual(len(result.defects), 0)
                self.assertGreater(result.material_coverage, 0.30)
                self.assertLess(result.material_coverage, 0.60)

    @unittest.skipUnless(LEGACY_IMAGES.exists(), "camera images are not present")
    def test_empty_frame_waits_without_false_cords_or_defect(self) -> None:
        images = [LEGACY_IMAGES / name for name in EMPTY_LAYER_IMAGES]
        images = [path for path in images if path.exists()]
        self.assertGreater(len(images), 0)
        for path in images:
            with self.subTest(path=path.name):
                image = cv.imread(str(path), cv.IMREAD_UNCHANGED)
                result = self.detector.process(image)
                self.assertEqual(result.status, STATUS_WAITING, result.summary())
                self.assertEqual(len(result.centers), 0)
                self.assertEqual(len(result.defects), 0)
                self.assertEqual(result.material_coverage, 0.0)

    @unittest.skipUnless(LEGACY_IMAGES.exists(), "camera images are not present")
    def test_panoramic_synthetic_missing_cord(self) -> None:
        images = [LEGACY_IMAGES / FULL_LAYER_IMAGES[0]]
        images = [path for path in images if path.exists()]
        if not images:
            self.skipTest("reference panoramic image is not present")
        path = images[0]
        image = cv.imread(str(path), cv.IMREAD_UNCHANGED)
        baseline = self.detector.process(image)
        x, _ = map(round, baseline.centers[len(baseline.centers) // 2])
        modified = remove_profile_cord(image, x)
        result = self.detector.process(modified)
        self.assertEqual(result.status, STATUS_DEFECT, result.summary())
        self.assertEqual(len(result.defects), 1)
        self.assertGreater(result.defects[0].ratio, 1.8)

    @unittest.skipUnless(LEGACY_IMAGES.exists(), "camera images are not present")
    def test_clipped_profile_roi_is_quality_warning_not_false_defect(self) -> None:
        images = [LEGACY_IMAGES / FULL_LAYER_IMAGES[0]]
        images = [path for path in images if path.exists()]
        if not images:
            self.skipTest("reference panoramic image is not present")
        image = cv.imread(str(images[0]), cv.IMREAD_UNCHANGED)
        settings = copy.deepcopy(self.config["detection"])
        settings["software_roi"]["height"] = 210
        result = CordDetector(settings).process(image)
        self.assertEqual(result.status, STATUS_LOW_QUALITY, result.summary())
        self.assertIn("ROI border", result.message)
        self.assertLess(result.path_coverage, settings["profile_min_path_coverage"])

    @unittest.skipUnless(RECENT_IMAGES.exists(), "recent images are not present")
    def test_conveyor_reflection_is_not_detected_as_cords(self) -> None:
        paths = sorted(RECENT_IMAGES.glob("*_0056.bmp"))
        self.assertGreater(len(paths), 0)
        result = self.detector.process(
            cv.imread(str(paths[0]), cv.IMREAD_UNCHANGED)
        )
        self.assertEqual(result.status, STATUS_WAITING, result.summary())
        self.assertEqual(len(result.centers), 0)
        self.assertEqual(result.material_coverage, 0.0)

    def test_small_regular_run_beats_long_irregular_reflection(self) -> None:
        false_x = [0.0]
        for distance in [8.0, 25.0] * 11:
            false_x.append(false_x[-1] + distance)
        false_run = [
            CordCandidate(x, 80.0, 3.5, 20.0, 1.0) for x in false_x
        ]
        real_run = [
            CordCandidate(1_000.0 + 10.0 * index, 180.0, 3.5, 20.0, 1.0)
            for index in range(6)
        ]
        selected = self.detector._select_presence_run(
            [false_run, real_run], minimum_run=6
        )
        self.assertEqual(selected, real_run)

    @unittest.skipUnless(RECENT_IMAGES.exists(), "recent images are not present")
    def test_recent_partial_layer_is_actively_inspected(self) -> None:
        paths = sorted(RECENT_IMAGES.glob("*_0135.bmp"))
        self.assertGreater(len(paths), 0)
        image = cv.imread(str(paths[0]), cv.IMREAD_UNCHANGED)
        result = self.detector.process(image)
        self.assertEqual(result.status, STATUS_OK, result.summary())
        self.assertGreaterEqual(len(result.centers), 120)
        self.assertGreater(result.material_coverage, 0.45)
        self.assertLess(result.material_coverage, 0.65)

        x, _ = map(round, result.centers[len(result.centers) // 2])
        missing = self.detector.process(remove_profile_cord(image, x))
        self.assertEqual(missing.status, STATUS_DEFECT, missing.summary())
        self.assertEqual(len(missing.defects), 1)

    @unittest.skipUnless(RECENT_IMAGES.exists(), "recent images are not present")
    def test_dim_visible_cords_do_not_raise_false_alarm(self) -> None:
        paths = sorted(RECENT_IMAGES.glob("*_0145.bmp")) + sorted(
            RECENT_IMAGES.glob("*_0168.bmp")
        )
        self.assertEqual(len(paths), 2)
        for path in paths:
            with self.subTest(path=path.name):
                image = cv.imread(str(path), cv.IMREAD_UNCHANGED)
                result = self.detector.process(image)
                self.assertEqual(result.status, STATUS_OK, result.summary())
                self.assertEqual(len(result.defects), 0)

                x, _ = map(round, result.centers[len(result.centers) // 2])
                missing = self.detector.process(remove_profile_cord(image, x))
                self.assertEqual(
                    missing.status, STATUS_DEFECT, missing.summary()
                )

    @unittest.skipUnless(RECENT_IMAGES.exists(), "recent images are not present")
    def test_black_rubber_occlusion_rejects_thin_false_peak(self) -> None:
        paths = sorted(RECENT_IMAGES.glob("*_0145.bmp"))
        self.assertGreater(len(paths), 0)
        image = cv.imread(str(paths[0]), cv.IMREAD_UNCHANGED)
        baseline = self.detector.process(image)
        x, y = map(round, baseline.centers[len(baseline.centers) // 2])

        for reflection in ("point", "horizontal", "vertical"):
            with self.subTest(reflection=reflection):
                covered = image.copy()
                cv.rectangle(covered, (x - 7, y - 10), (x + 7, y + 10), 0, -1)
                if reflection == "point":
                    covered[y, x] = 80
                elif reflection == "horizontal":
                    covered[y, x - 3 : x + 4] = 80
                else:
                    covered[y - 3 : y + 4, x] = 80

                result = self.detector.process(covered)
                self.assertEqual(result.status, STATUS_DEFECT, result.summary())
                self.assertEqual(len(result.defects), 1)
                self.assertFalse(
                    any(abs(center_x - x) <= 4 for center_x, _ in result.centers)
                )

    def test_blank_frame_is_not_reported_as_defect(self) -> None:
        blank = np.zeros_like(self.image)
        result = self.detector.process(blank)
        self.assertEqual(result.status, STATUS_LOW_QUALITY)

    def test_blurred_frame_is_quality_alarm_not_product_defect(self) -> None:
        blurred = cv.GaussianBlur(self.image, (5, 5), 0)
        result = self.detector.process(blurred)
        self.assertEqual(result.status, STATUS_LOW_QUALITY, result.summary())

    def test_offline_source_reads_sample(self) -> None:
        settings = copy.deepcopy(self.config["camera"])
        settings["offline_fps"] = 1000.0
        source = OfflineFrameSource(settings, SAMPLE)
        source.open()
        try:
            frame = source.read()
        finally:
            source.close()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape[:2], self.image.shape[:2])

    def test_defect_storage_writes_three_artifacts(self) -> None:
        baseline = self.detector.process(self.image)
        target = baseline.centers[len(baseline.centers) // 2]
        modified = self.image.copy()
        cv.circle(modified, tuple(map(round, target)), 9, (30, 40, 40), -1)
        result = self.detector.process(modified)
        with tempfile.TemporaryDirectory() as directory:
            settings = copy.deepcopy(self.config["storage"])
            settings["directory"] = "saved"
            storage = DefectStorage(settings, Path(directory))
            storage.start()
            base = storage.enqueue(result)
            deadline = time.monotonic() + 5.0
            while storage.queue.unfinished_tasks and time.monotonic() < deadline:
                time.sleep(0.02)
            storage.close()
            self.assertIsNotNone(base)
            self.assertTrue(base.with_name(base.name + "_raw.jpg").is_file())
            self.assertTrue(base.with_name(base.name + "_annotated.jpg").is_file())
            self.assertTrue(base.with_suffix(".json").is_file())


if __name__ == "__main__":
    unittest.main()
