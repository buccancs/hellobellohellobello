"""Camera Calibration Dialog implementation.

This module provides the GUI interface for camera calibration workflow,
implementing the missing piece from FR9 requirements.
"""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pc_controller.src.tools.camera_calibration import (
    CalibrationResult,
    calibrate_camera,
    save_calibration,
)


class CalibrationWorker(QObject):
    """Worker thread for performing camera calibration without blocking UI."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self, image_paths: list[str], board_size: tuple[int, int], square_size: float
    ):
        super().__init__()
        self.image_paths = image_paths
        self.board_size = board_size
        self.square_size = square_size

    def run(self):
        """Perform calibration in background thread."""
        try:
            self.progress.emit("Starting camera calibration...")
            self.progress.emit(f"Processing {len(self.image_paths)} images...")

            result = calibrate_camera(
                image_paths=self.image_paths,
                board_size=self.board_size,
                square_size=self.square_size,
            )

            self.progress.emit(
                f"Calibration complete! RMS error: {result.rms_error:.4f}"
            )
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(f"Calibration failed: {e!s}")


class CalibrationDialog(QDialog):
    """Camera calibration dialog for configuring and running calibration workflow."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Camera Calibration")
        self.setMinimumSize(500, 400)

        self.calibration_result: CalibrationResult | None = None
        self.worker_thread: QThread | None = None
        self.worker: CalibrationWorker | None = None

        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        image_group = QGroupBox("Calibration Images")
        image_layout = QFormLayout(image_group)

        self.images_path_edit = QLineEdit()
        self.images_path_edit.setPlaceholderText(
            "Select folder containing checkerboard images..."
        )
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_images_folder)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.images_path_edit)
        path_layout.addWidget(self.browse_button)

        image_layout.addRow("Images Folder:", path_layout)
        layout.addWidget(image_group)

        # Checkerboard parameters group
        params_group = QGroupBox("Checkerboard Parameters")
        params_layout = QFormLayout(params_group)

        self.board_width_spin = QSpinBox()
        self.board_width_spin.setRange(3, 20)
        self.board_width_spin.setValue(9)

        self.board_height_spin = QSpinBox()
        self.board_height_spin.setRange(3, 20)
        self.board_height_spin.setValue(6)

        self.square_size_spin = QDoubleSpinBox()
        self.square_size_spin.setRange(0.001, 1.0)
        self.square_size_spin.setValue(0.025)
        self.square_size_spin.setDecimals(3)
        self.square_size_spin.setSuffix(" m")

        params_layout.addRow("Board Width (corners):", self.board_width_spin)
        params_layout.addRow("Board Height (corners):", self.board_height_spin)
        params_layout.addRow("Square Size:", self.square_size_spin)
        layout.addWidget(params_group)

        progress_group = QGroupBox("Calibration Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        self.progress_text = QTextEdit()
        self.progress_text.setMaximumHeight(100)
        self.progress_text.setReadOnly(True)

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_text)
        layout.addWidget(progress_group)

        button_layout = QHBoxLayout()

        self.calibrate_button = QPushButton("Start Calibration")
        self.calibrate_button.clicked.connect(self._start_calibration)

        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self._save_results)
        self.save_button.setEnabled(False)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)

        button_layout.addWidget(self.calibrate_button)
        button_layout.addWidget(self.save_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

    def _browse_images_folder(self):
        """Browse for folder containing calibration images."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Calibration Images Folder",
            self.images_path_edit.text() or os.path.expanduser("~"),
        )
        if folder:
            self.images_path_edit.setText(folder)

    def _start_calibration(self):
        """Start the calibration process."""
        images_folder = self.images_path_edit.text().strip()
        if not images_folder or not Path(images_folder).is_dir():
            QMessageBox.warning(
                self, "Input Error", "Please select a valid images folder."
            )
            return

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = [
            str(p)
            for p in Path(images_folder).iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        ]

        if not image_files:
            QMessageBox.warning(
                self, "Input Error", "No image files found in the selected folder."
            )
            return

        # Get parameters
        board_size = (self.board_width_spin.value(), self.board_height_spin.value())
        square_size = self.square_size_spin.value()

        self._log_progress(f"Found {len(image_files)} images for calibration")
        self._log_progress(
            f"Board size: {board_size[0]}x{board_size[1]}, square size: {square_size}m"
        )

        self.calibrate_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        self.worker = CalibrationWorker(image_files, board_size, square_size)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)

        self.worker.progress.connect(self._log_progress)
        self.worker.finished.connect(self._calibration_finished)
        self.worker.error.connect(self._calibration_error)
        self.worker_thread.started.connect(self.worker.run)

        self.worker_thread.start()

    def _log_progress(self, message: str):
        """Add progress message to the text area."""
        self.progress_text.append(message)

    def _calibration_finished(self, result: CalibrationResult):
        """Handle successful calibration completion."""
        self.calibration_result = result

        self._log_progress("=" * 50)
        self._log_progress("CALIBRATION RESULTS:")
        self._log_progress(f"RMS Reprojection Error: {result.rms_error:.4f} pixels")
        self._log_progress(f"Image Size: {result.image_size[0]}x{result.image_size[1]}")
        self._log_progress("Camera Matrix:")
        for i in range(3):
            row_str = (
                "  ["
                + ", ".join(f"{result.camera_matrix[i, j]:8.3f}" for j in range(3))
                + "]"
            )
            self._log_progress(row_str)
        self._log_progress(f"Distortion Coefficients: {result.dist_coeffs.flatten()}")

        # Enable save button
        self.save_button.setEnabled(True)
        self._cleanup_thread()

    def _calibration_error(self, error_msg: str):
        """Handle calibration error."""
        self._log_progress(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Calibration Error", error_msg)
        self._cleanup_thread()

    def _cleanup_thread(self):
        """Clean up worker thread."""
        self.calibrate_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
        self.worker = None

    def _save_results(self):
        """Save calibration results to file."""
        if not self.calibration_result:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Calibration Results",
            "camera_calibration.json",
            "JSON files (*.json)",
        )

        if filename:
            try:
                save_calibration(filename, self.calibration_result)
                QMessageBox.information(
                    self, "Success", f"Calibration saved to:\n{filename}"
                )
                self._log_progress(f"Results saved to: {filename}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Save Error", f"Failed to save calibration:\n{e!s}"
                )

    def closeEvent(self, event):
        """Handle dialog close event."""
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Calibration Running",
                "Calibration is still running. Do you want to cancel it and close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self._cleanup_thread()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
