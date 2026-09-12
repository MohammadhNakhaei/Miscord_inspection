from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from miscord.config import ConfigManager
from miscord.camera import enumerate_basler_devices
from miscord.detector import (
    DetectionResult,
    STATUS_DEFECT,
    STATUS_LOW_QUALITY,
    STATUS_OK,
    STATUS_WAITING,
)
from miscord.worker import InspectionThread


LOGGER = logging.getLogger("miscord.ui")


APP_STYLE = """
QMainWindow, QDialog { background: #15191e; color: #edf1f5; }
QWidget { font-family: "Segoe UI"; font-size: 10pt; }
QGroupBox { border: 1px solid #3d4650; border-radius: 5px; margin-top: 12px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #dce6ee; }
QLabel#imageView { background: #050607; border: 1px solid #3d4650; color: #78838d; }
QPushButton { background: #2c353e; border: 1px solid #55616c; padding: 7px 14px; border-radius: 4px; }
QPushButton:hover { background: #394651; }
QPushButton:disabled { color: #68727b; background: #20262c; }
QLineEdit, QComboBox { background: #20262c; border: 1px solid #52606b; padding: 5px 8px; border-radius: 3px; }
QSpinBox, QDoubleSpinBox { background: #20262c; border: 1px solid #52606b; padding: 5px 30px 5px 8px; min-height: 26px; border-radius: 3px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 24px; }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 24px; }
QTabWidget::pane { border: 1px solid #3d4650; }
QTabBar::tab { background: #242c33; padding: 8px 16px; }
QTabBar::tab:selected { background: #3a4650; }
"""


class ImageView(QLabel):
    def __init__(self, placeholder: str):
        super().__init__(placeholder)
        self.setObjectName("imageView")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(420, 300)
        self._image: QImage | None = None

    def set_array(self, frame: np.ndarray) -> None:
        if frame.ndim == 2:
            height, width = frame.shape
            image = QImage(
                frame.data,
                width,
                height,
                int(frame.strides[0]),
                QImage.Format.Format_Grayscale8,
            )
        else:
            if frame.shape[2] == 4:
                frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
            height, width = frame.shape[:2]
            image = QImage(
                frame.data,
                width,
                height,
                int(frame.strides[0]),
                QImage.Format.Format_BGR888,
            )
        self._image = image.copy()
        self._refresh()

    def _refresh(self) -> None:
        if self._image is None:
            return
        pixmap = QPixmap.fromImage(self._image).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._refresh()


def _spin(minimum: int = 0, maximum: int = 100_000) -> QSpinBox:
    widget = QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
    widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
    widget.setMinimumWidth(210)
    widget.setKeyboardTracking(False)
    widget.setAccelerated(True)
    return widget


def _double_spin(
    minimum: float, maximum: float, decimals: int = 2, step: float = 0.1
) -> QDoubleSpinBox:
    widget = QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setSingleStep(step)
    widget.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
    widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
    widget.setMinimumWidth(210)
    widget.setKeyboardTracking(False)
    widget.setAccelerated(True)
    return widget


class CameraDiscoveryThread(QThread):
    discovered = Signal(list)
    failed = Signal(str)

    def run(self) -> None:
        try:
            self.discovered.emit(enumerate_basler_devices())
        except Exception as exc:
            LOGGER.exception("Basler camera discovery failed")
            self.failed.emit(str(exc))


class SettingsDialog(QDialog):
    def __init__(self, config: dict[str, Any], app_root: Path, parent: QWidget | None = None):
        super().__init__(parent)
        self.original = copy.deepcopy(config)
        self.app_root = app_root
        self.discovery_thread: CameraDiscoveryThread | None = None
        self.setWindowTitle("تنظیمات دوربین و تشخیص")
        self.setMinimumSize(650, 620)
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)

        tabs = QTabWidget()
        tabs.addTab(self._camera_tab(config), "دوربین")
        tabs.addTab(self._detection_tab(config), "تشخیص")
        tabs.addTab(self._runtime_tab(config), "اجرا و ذخیره")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("ذخیره و اعمال")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("انصراف")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)
        if config["camera"]["mode"] == "basler":
            QTimer.singleShot(0, self._refresh_cameras)

    @staticmethod
    def _scroll_form(form: QFormLayout) -> QScrollArea:
        container = QWidget()
        container.setLayout(form)
        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        return scroll

    def _camera_tab(self, config: dict[str, Any]) -> QWidget:
        values = config["camera"]
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.mode = QComboBox()
        self.mode.addItem("Basler", "basler")
        self.mode.addItem("Offline / simulation", "offline")
        self.mode.setCurrentIndex(max(0, self.mode.findData(values["mode"])))
        self.serial = QLineEdit(str(values["serial_number"]))
        self.camera_list = QComboBox()
        self.camera_list.setMinimumContentsLength(45)
        self.camera_list.addItem("برای نمایش دوربین‌ها، جست‌وجو را بزنید", None)
        self.camera_list.currentIndexChanged.connect(self._camera_selected)
        self.refresh_cameras_button = QPushButton("جست‌وجوی مجدد")
        self.refresh_cameras_button.clicked.connect(self._refresh_cameras)
        camera_list_row = QWidget()
        camera_list_layout = QHBoxLayout(camera_list_row)
        camera_list_layout.setContentsMargins(0, 0, 0, 0)
        camera_list_layout.addWidget(self.camera_list, 1)
        camera_list_layout.addWidget(self.refresh_cameras_button)
        self.exposure = _double_spin(1.0, 10_000_000.0, 1, 10.0)
        self.exposure.setValue(float(values["exposure_us"]))
        self.gain = _double_spin(0.0, 100.0, 2, 0.1)
        self.gain.setValue(float(values["gain_db"]))
        self.camera_fps = _double_spin(0.0, 10_000.0, 2, 1.0)
        self.camera_fps.setSpecialValueText("بیشترین سرعت")
        self.camera_fps.setValue(float(values["acquisition_frame_rate"]))

        roi = values["roi"]
        self.camera_roi = {key: _spin() for key in ("x", "y", "width", "height")}
        for key, widget in self.camera_roi.items():
            widget.setValue(int(roi[key]))
        self.camera_roi["width"].setSpecialValueText("حداکثر")
        self.camera_roi["height"].setSpecialValueText("حداکثر")

        self.offline_path = QLineEdit(str(values["offline_path"]))
        browse = QPushButton("انتخاب...")
        browse.clicked.connect(self._browse_offline)
        offline_row = QWidget()
        offline_layout = QHBoxLayout(offline_row)
        offline_layout.setContentsMargins(0, 0, 0, 0)
        offline_layout.addWidget(self.offline_path)
        offline_layout.addWidget(browse)
        self.fallback = QCheckBox("در نبود دوربین، تصویر آفلاین نمایش داده شود")
        self.fallback.setChecked(bool(values["fallback_to_offline"]))

        form.addRow("حالت منبع:", self.mode)
        form.addRow("دوربین‌های متصل:", camera_list_row)
        form.addRow("سریال دوربین (خالی = اولین دوربین):", self.serial)
        form.addRow("Exposure (µs):", self.exposure)
        form.addRow("Gain:", self.gain)
        form.addRow("Frame rate (0 = max):", self.camera_fps)
        form.addRow("ROI X:", self.camera_roi["x"])
        form.addRow("ROI Y:", self.camera_roi["y"])
        form.addRow("ROI Width:", self.camera_roi["width"])
        form.addRow("ROI Height:", self.camera_roi["height"])
        form.addRow("مسیر تصویر/ویدئو آفلاین:", offline_row)
        form.addRow("", self.fallback)
        return self._scroll_form(form)

    def _refresh_cameras(self) -> None:
        if self.discovery_thread is not None and self.discovery_thread.isRunning():
            return
        self.refresh_cameras_button.setEnabled(False)
        self.camera_list.blockSignals(True)
        self.camera_list.clear()
        self.camera_list.addItem("در حال جست‌وجوی دوربین‌های Basler...", None)
        self.camera_list.blockSignals(False)
        thread = CameraDiscoveryThread(self)
        thread.discovered.connect(self._show_discovered_cameras)
        thread.failed.connect(self._camera_discovery_failed)
        thread.finished.connect(self._camera_discovery_finished)
        self.discovery_thread = thread
        thread.start()

    def _show_discovered_cameras(self, devices: list[dict[str, str]]) -> None:
        configured_serial = self.serial.text().strip()
        self.camera_list.blockSignals(True)
        self.camera_list.clear()
        self.camera_list.addItem("خودکار — اولین دوربین موجود", "")
        selected_index = 0
        for device in devices:
            self.camera_list.addItem(device["label"], device["serial_number"])
            self.camera_list.setItemData(
                self.camera_list.count() - 1,
                device["friendly_name"],
                Qt.ItemDataRole.ToolTipRole,
            )
            if device["serial_number"] == configured_serial:
                selected_index = self.camera_list.count() - 1
        if configured_serial and selected_index == 0:
            self.camera_list.addItem(
                f"تنظیم‌شده ولی اکنون متصل نیست | S/N {configured_serial}",
                configured_serial,
            )
            selected_index = self.camera_list.count() - 1
        self.camera_list.setCurrentIndex(selected_index)
        self.camera_list.blockSignals(False)
        if not devices:
            self.camera_list.setToolTip("هیچ دوربین Basler متصلی پیدا نشد")
        else:
            self.camera_list.setToolTip(f"{len(devices)} دوربین پیدا شد")
        self._camera_selected(selected_index)

    def _camera_selected(self, index: int) -> None:
        serial = self.camera_list.itemData(index)
        if isinstance(serial, str):
            self.serial.setText(serial)

    def _camera_discovery_failed(self, message: str) -> None:
        configured_serial = self.serial.text().strip()
        self.camera_list.blockSignals(True)
        self.camera_list.clear()
        self.camera_list.addItem(f"خطا در جست‌وجو: {message}", configured_serial)
        self.camera_list.blockSignals(False)
        self.camera_list.setToolTip(message)

    def _camera_discovery_finished(self) -> None:
        self.refresh_cameras_button.setEnabled(True)
        thread = self.discovery_thread
        self.discovery_thread = None
        if thread is not None:
            thread.deleteLater()

    def done(self, result: int) -> None:
        thread = self.discovery_thread
        if thread is not None and thread.isRunning():
            thread.wait(5000)
        super().done(result)

    def _browse_offline(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "انتخاب تصویر یا ویدئو",
            str(self.app_root),
            "Images and videos (*.bmp *.png *.jpg *.jpeg *.tif *.tiff *.avi *.mp4 *.mkv);;All files (*.*)",
        )
        if path:
            self.offline_path.setText(path)

    def _detection_tab(self, config: dict[str, Any]) -> QWidget:
        values = config["detection"]
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.detection_algorithm = QComboBox()
        self.detection_algorithm.addItem("خودکار (پیشنهادی)", "auto")
        self.detection_algorithm.addItem("ردیابی پروفایل برای تصویر عریض", "profile")
        self.detection_algorithm.addItem("کانتور برای تصویر قدیمی", "contour")
        self.detection_algorithm.setCurrentIndex(
            max(0, self.detection_algorithm.findData(values["algorithm"]))
        )

        roi = values["software_roi"]
        self.software_roi = {key: _spin() for key in ("x", "y", "width", "height")}
        for key, widget in self.software_roi.items():
            widget.setValue(int(roi[key]))
        self.software_roi["width"].setSpecialValueText("کل عرض")
        self.software_roi["height"].setSpecialValueText("کل ارتفاع")

        self.threshold = _spin(1, 254)
        self.threshold.setValue(int(values["intensity_threshold"]))
        self.min_area = _double_spin(1, 100_000, 1, 1)
        self.min_area.setValue(float(values["min_area_px"]))
        self.max_area = _double_spin(1, 100_000, 1, 1)
        self.max_area.setValue(float(values["max_area_px"]))
        self.min_radius = _double_spin(0.5, 1_000, 1, 0.5)
        self.min_radius.setValue(float(values["min_radius_px"]))
        self.max_radius = _double_spin(0.5, 1_000, 1, 0.5)
        self.max_radius.setValue(float(values["max_radius_px"]))
        self.circularity = _double_spin(0.0, 1.0, 2, 0.05)
        self.circularity.setValue(float(values["min_circularity"]))
        self.row_tolerance = _double_spin(1.0, 1_000, 1, 1)
        self.row_tolerance.setValue(float(values["row_tolerance_px"]))
        self.minimum_threads = _spin(3, 10_000)
        self.minimum_threads.setValue(int(values["minimum_threads"]))
        self.nominal_pitch = _double_spin(0.0, 10_000, 2, 0.5)
        self.nominal_pitch.setSpecialValueText("خودکار")
        self.nominal_pitch.setValue(float(values["nominal_pitch_px"]))
        self.gap_ratio = _double_spin(1.1, 10.0, 2, 0.05)
        self.gap_ratio.setValue(float(values["missing_gap_ratio"]))
        self.profile_peak_percentile = _double_spin(0.0, 95.0, 1, 1.0)
        self.profile_peak_percentile.setValue(
            float(values["profile_peak_percentile"])
        )
        self.profile_border_margin = _spin(0, 1_000)
        self.profile_border_margin.setValue(
            int(values["profile_border_margin_px"])
        )
        self.profile_min_path_coverage = _double_spin(0.1, 1.0, 2, 0.05)
        self.profile_min_path_coverage.setValue(
            float(values["profile_min_path_coverage"])
        )
        self.profile_presence_contrast = _double_spin(0.0, 255.0, 1, 0.5)
        self.profile_presence_contrast.setValue(
            float(values["profile_presence_min_contrast"])
        )
        self.profile_complete_coverage = _double_spin(0.1, 1.0, 2, 0.05)
        self.profile_complete_coverage.setValue(
            float(values["profile_complete_min_coverage"])
        )
        self.edge_exclusion = _double_spin(0.0, 20.0, 1, 0.5)
        self.edge_exclusion.setValue(
            float(values["edge_exclusion_pitch_count"])
        )
        self.minimum_defect_quality = _double_spin(0.0, 1.0, 2, 0.05)
        self.minimum_defect_quality.setValue(
            float(values["minimum_defect_quality_score"])
        )
        self.maximum_missing_fraction = _double_spin(0.01, 1.0, 2, 0.05)
        self.maximum_missing_fraction.setValue(
            float(values["maximum_missing_fraction"])
        )
        self.expected_count = _spin(0, 10_000)
        self.expected_count.setSpecialValueText("غیرفعال")
        self.expected_count.setValue(int(values["expected_thread_count"]))

        form.addRow("روش تشخیص:", self.detection_algorithm)
        form.addRow("Software ROI X:", self.software_roi["x"])
        form.addRow("Software ROI Y:", self.software_roi["y"])
        form.addRow("Software ROI Width:", self.software_roi["width"])
        form.addRow("Software ROI Height:", self.software_roi["height"])
        form.addRow("آستانه روشنایی:", self.threshold)
        form.addRow("حداقل مساحت نخ (px²):", self.min_area)
        form.addRow("حداکثر مساحت نخ (px²):", self.max_area)
        form.addRow("حداقل شعاع نخ (px):", self.min_radius)
        form.addRow("حداکثر شعاع نخ (px):", self.max_radius)
        form.addRow("حداقل گردی کانتور:", self.circularity)
        form.addRow("تلرانس ردیف نخ‌ها (px):", self.row_tolerance)
        form.addRow("حداقل نخ لازم برای پردازش:", self.minimum_threads)
        form.addRow("فاصله اسمی نخ‌ها (0 = auto):", self.nominal_pitch)
        form.addRow("ضریب تشخیص نخ مفقود:", self.gap_ratio)
        form.addRow("حساسیت پروفایل (percentile):", self.profile_peak_percentile)
        form.addRow("حاشیه امن مسیر از لبه ROI (px):", self.profile_border_margin)
        form.addRow("حداقل پوشش معتبر مسیر:", self.profile_min_path_coverage)
        form.addRow("حداقل کنتراست برای حضور لایه:", self.profile_presence_contrast)
        form.addRow("حداقل پوشش برای لایه کامل:", self.profile_complete_coverage)
        form.addRow("حذف لبه بر حسب فاصله نخ:", self.edge_exclusion)
        form.addRow("حداقل کیفیت لازم برای اعلام عیب:", self.minimum_defect_quality)
        form.addRow("حداکثر نسبت نخ مفقود معتبر:", self.maximum_missing_fraction)
        form.addRow("تعداد مورد انتظار (0 = off):", self.expected_count)
        return self._scroll_form(form)

    def _runtime_tab(self, config: dict[str, Any]) -> QWidget:
        runtime = config["runtime"]
        storage = config["storage"]
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.display_fps = _double_spin(1.0, 240.0, 1, 1)
        self.display_fps.setValue(float(runtime["display_max_fps"]))
        self.confirmation = _spin(1, 100)
        self.confirmation.setValue(int(runtime["confirmation_frames"]))
        self.alarm_seconds = _double_spin(0.5, 60.0, 1, 0.5)
        self.alarm_seconds.setValue(float(runtime["alarm_blink_seconds"]))
        self.cooldown = _double_spin(0.0, 60.0, 2, 0.1)
        self.cooldown.setValue(float(runtime["duplicate_suppression_seconds"]))
        self.save_defects = QCheckBox("ذخیره تصویر خام، علامت‌گذاری‌شده و JSON")
        self.save_defects.setChecked(bool(storage["save_defects"]))
        self.storage_directory = QLineEdit(str(storage["directory"]))

        form.addRow("حداکثر FPS نمایش رابط:", self.display_fps)
        form.addRow("فریم متوالی برای تأیید عیب:", self.confirmation)
        form.addRow("مدت چشمک آلارم (ثانیه):", self.alarm_seconds)
        form.addRow("فاصله ذخیره عیب تکراری (ثانیه):", self.cooldown)
        form.addRow("", self.save_defects)
        form.addRow("پوشه ذخیره عیب:", self.storage_directory)
        return self._scroll_form(form)

    def result_config(self) -> dict[str, Any]:
        config = copy.deepcopy(self.original)
        camera = config["camera"]
        camera.update(
            {
                "mode": self.mode.currentData(),
                "serial_number": self.serial.text().strip(),
                "exposure_us": self.exposure.value(),
                "gain_db": self.gain.value(),
                "acquisition_frame_rate": self.camera_fps.value(),
                "offline_path": self.offline_path.text().strip(),
                "fallback_to_offline": self.fallback.isChecked(),
                "roi": {key: widget.value() for key, widget in self.camera_roi.items()},
            }
        )
        detection = config["detection"]
        detection.update(
            {
                "algorithm": self.detection_algorithm.currentData(),
                "software_roi": {
                    key: widget.value() for key, widget in self.software_roi.items()
                },
                "intensity_threshold": self.threshold.value(),
                "min_area_px": self.min_area.value(),
                "max_area_px": self.max_area.value(),
                "min_radius_px": self.min_radius.value(),
                "max_radius_px": self.max_radius.value(),
                "min_circularity": self.circularity.value(),
                "row_tolerance_px": self.row_tolerance.value(),
                "minimum_threads": self.minimum_threads.value(),
                "nominal_pitch_px": self.nominal_pitch.value(),
                "missing_gap_ratio": self.gap_ratio.value(),
                "profile_peak_percentile": self.profile_peak_percentile.value(),
                "profile_border_margin_px": self.profile_border_margin.value(),
                "profile_min_path_coverage": self.profile_min_path_coverage.value(),
                "profile_presence_min_contrast": self.profile_presence_contrast.value(),
                "profile_complete_min_coverage": self.profile_complete_coverage.value(),
                "edge_exclusion_pitch_count": self.edge_exclusion.value(),
                "minimum_defect_quality_score": self.minimum_defect_quality.value(),
                "maximum_missing_fraction": self.maximum_missing_fraction.value(),
                "expected_thread_count": self.expected_count.value(),
            }
        )
        config["runtime"].update(
            {
                "display_max_fps": self.display_fps.value(),
                "confirmation_frames": self.confirmation.value(),
                "alarm_blink_seconds": self.alarm_seconds.value(),
                "duplicate_suppression_seconds": self.cooldown.value(),
            }
        )
        config["storage"].update(
            {
                "save_defects": self.save_defects.isChecked(),
                "directory": self.storage_directory.text().strip() or "defects",
            }
        )
        return config


class MainWindow(QMainWindow):
    def __init__(self, manager: ConfigManager, config: dict[str, Any]):
        super().__init__()
        self.manager = manager
        self.config = config
        self.worker: InspectionThread | None = None
        self.alarm_until = 0.0
        self.alarm_on = False

        self.setWindowTitle("Miscord Inspector | بازرسی نخ لایه لاستیک")
        self.resize(1450, 850)
        self.setMinimumSize(1000, 650)
        self.setStyleSheet(APP_STYLE)
        self._build_ui()

        self.blink_timer = QTimer(self)
        self.blink_timer.setInterval(280)
        self.blink_timer.timeout.connect(self._blink_alarm)
        self.blink_timer.start()
        QTimer.singleShot(0, self.start_inspection)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        top = QHBoxLayout()
        self.start_button = QPushButton("شروع")
        self.stop_button = QPushButton("توقف")
        settings_button = QPushButton("تنظیمات")
        self.start_button.clicked.connect(self.start_inspection)
        self.stop_button.clicked.connect(self.stop_inspection)
        settings_button.clicked.connect(self.open_settings)
        top.addWidget(self.start_button)
        top.addWidget(self.stop_button)
        top.addWidget(settings_button)
        top.addStretch(1)
        self.source_label = QLabel("آماده‌سازی...")
        self.source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        top.addWidget(self.source_label)
        root.addLayout(top)

        self.alarm_label = QLabel("سیستم آماده")
        self.alarm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alarm_label.setMinimumHeight(54)
        self.alarm_label.setStyleSheet(
            "background:#174d2a;color:white;font-size:18pt;font-weight:700;border-radius:5px;"
        )
        root.addWidget(self.alarm_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        live_group = QGroupBox("تصویر زنده دوربین و نتیجه پردازش")
        live_layout = QVBoxLayout(live_group)
        self.live_view = ImageView("در انتظار تصویر دوربین")
        live_layout.addWidget(self.live_view)

        defect_group = QGroupBox("آخرین نخ مفقود شناسایی‌شده")
        defect_layout = QVBoxLayout(defect_group)
        self.defect_view = ImageView("هنوز عیبی ثبت نشده است")
        self.defect_details = QLabel("—")
        self.defect_details.setWordWrap(True)
        self.defect_details.setAlignment(Qt.AlignmentFlag.AlignCenter)
        defect_layout.addWidget(self.defect_view)
        defect_layout.addWidget(self.defect_details)

        splitter.addWidget(live_group)
        splitter.addWidget(defect_group)
        splitter.setSizes([750, 650])
        root.addWidget(splitter, 1)

        stats = QGridLayout()
        self.fps_label = QLabel("FPS: —")
        self.processing_label = QLabel("پردازش: —")
        self.thread_count_label = QLabel("تعداد نخ: —")
        self.pitch_label = QLabel("فاصله: —")
        self.quality_label = QLabel("کیفیت: —")
        self.queue_label = QLabel("صف ذخیره: —")
        widgets = [
            self.fps_label,
            self.processing_label,
            self.thread_count_label,
            self.pitch_label,
            self.quality_label,
            self.queue_label,
        ]
        for column, widget in enumerate(widgets):
            widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            widget.setStyleSheet("background:#20262c;padding:8px;border-radius:3px;")
            stats.addWidget(widget, 0, column)
        root.addLayout(stats)
        self.setCentralWidget(central)
        self.stop_button.setEnabled(False)

    def start_inspection(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.worker = InspectionThread(self.config, self.manager.app_root)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.defect_ready.connect(self._on_defect)
        self.worker.stats_ready.connect(self._on_stats)
        self.worker.source_status.connect(self._on_source_status)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        LOGGER.info("Inspection started from UI")

    def stop_inspection(self) -> None:
        if self.worker is None:
            return
        self.worker.request_stop()
        self.stop_button.setEnabled(False)
        self.source_label.setText("در حال توقف...")

    def _on_worker_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.worker = None

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.config, self.manager.app_root, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.config = self.manager.save(dialog.result_config())
            if self.worker is not None and self.worker.isRunning():
                self.worker.update_config(self.config)
            self.statusBar().showMessage("تنظیمات ذخیره و برای اعمال ارسال شد", 5000)
            LOGGER.info("Settings saved by operator")
        except Exception as exc:
            LOGGER.exception("Could not save settings")
            QMessageBox.critical(self, "خطای تنظیمات", str(exc))

    def _on_frame(self, result: DetectionResult) -> None:
        self.live_view.set_array(result.annotated_frame)
        if result.status == STATUS_WAITING and time.perf_counter() >= self.alarm_until:
            if result.centers:
                text = (
                    "در انتظار ورود کامل لایه | "
                    f"نخ‌های بخش موجود: {len(result.centers)} | "
                    f"پوشش: {result.material_coverage * 100:.0f}%"
                )
            else:
                text = "در انتظار ورود لایه"
            self.alarm_label.setText(text)
            self.alarm_label.setStyleSheet(
                "background:#255b78;color:white;font-size:17pt;font-weight:700;border-radius:5px;"
            )
        elif result.status == STATUS_LOW_QUALITY and time.perf_counter() >= self.alarm_until:
            self.alarm_label.setText("کیفیت تصویر برای تصمیم‌گیری کافی نیست")
            self.alarm_label.setStyleSheet(
                "background:#ad7600;color:white;font-size:18pt;font-weight:700;border-radius:5px;"
            )
        elif result.status == STATUS_OK and time.perf_counter() >= self.alarm_until:
            self.alarm_label.setText("OK | وضعیت نخ‌ها عادی است")
            self.alarm_label.setStyleSheet(
                "background:#174d2a;color:white;font-size:18pt;font-weight:700;border-radius:5px;"
            )

    def _on_defect(self, result: DetectionResult) -> None:
        self.defect_view.set_array(result.annotated_frame)
        missing = sum(item.estimated_missing_count for item in result.defects)
        largest_ratio = max((item.ratio for item in result.defects), default=0.0)
        timestamp = result.timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        if result.defects:
            details = (
                f"زمان: {timestamp} | نخ مفقود تخمینی: {missing} | "
                f"بیشترین ضریب فاصله: {largest_ratio:.2f}x"
            )
        else:
            details = f"زمان: {timestamp} | {result.message}"
        self.defect_details.setText(details)
        self.alarm_until = time.perf_counter() + float(
            self.config["runtime"]["alarm_blink_seconds"]
        )
        self.alarm_on = False
        self._blink_alarm()

    def _blink_alarm(self) -> None:
        if time.perf_counter() >= self.alarm_until:
            return
        self.alarm_on = not self.alarm_on
        color = "#f21b1b" if self.alarm_on else "#5c0707"
        self.alarm_label.setText("آلارم: نخ مفقود شناسایی شد")
        self.alarm_label.setStyleSheet(
            f"background:{color};color:white;font-size:19pt;font-weight:800;border-radius:5px;"
        )

    def _on_stats(self, stats: dict[str, Any]) -> None:
        self.fps_label.setText(f"FPS پردازش: {stats['fps']:.1f}")
        self.processing_label.setText(f"پردازش: {stats['processing_ms']:.2f} ms")
        self.thread_count_label.setText(f"تعداد نخ: {stats['thread_count']}")
        self.pitch_label.setText(f"فاصله: {stats['pitch_px']:.2f} px")
        self.quality_label.setText(f"کیفیت: {stats['quality'] * 100:.0f}%")
        self.queue_label.setText(
            f"صف ذخیره: {stats['save_queue']} | از دست رفته: {stats['dropped_saves']}"
        )

    def _on_source_status(self, message: str) -> None:
        self.source_label.setText(message)

    def _on_error(self, message: str) -> None:
        self.source_label.setText(f"خطا: {message}")
        if time.perf_counter() >= self.alarm_until:
            self.alarm_label.setText("خطای دوربین/پردازش — تلاش برای اتصال مجدد")
            self.alarm_label.setStyleSheet(
                "background:#ad7600;color:white;font-size:17pt;font-weight:700;border-radius:5px;"
            )

    def closeEvent(self, event: QCloseEvent) -> None:
        worker = self.worker
        if worker is not None and worker.isRunning():
            worker.request_stop()
            if not worker.wait(6000):
                LOGGER.error("Inspection worker did not stop before UI shutdown timeout")
        event.accept()
