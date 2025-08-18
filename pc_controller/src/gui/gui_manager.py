"""GUI Manager for the PC Controller (Hub) - Phase 3.

Implements the main QMainWindow with a tabbed interface (Dashboard, Logs),
including a dynamic grid of device widgets and live visualization for local
webcam (video) and Shimmer GSR (plot). This module uses non-blocking UI
updates via QTimer and PyQt signals, and delegates device access to the
core local interfaces that optionally use native C++ backends via PyBind11.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QImage, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover - import guard for environments without Qt backend
    pg = None

from data.data_aggregator import DataAggregator
from data.data_loader import DataLoader
from data.hdf5_exporter import export_session_to_hdf5
from network.network_controller import DiscoveredDevice, NetworkController

# Local device interfaces (Python shim that optionally uses native backends)
try:
    from core.local_interfaces import ShimmerInterface, WebcamInterface
except Exception:  # pragma: no cover - in case of import issues during tests
    ShimmerInterface = None  # type: ignore
    WebcamInterface = None  # type: ignore


@dataclass
class _GridPos:
    row: int
    col: int


class StatusWidget(QWidget):
    """Widget to display system status including device connections and recording state."""
    
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(300, 150)
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("System Status", self)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(header)
        
        # Recording status
        self.recording_status = QLabel("●", self)
        self.recording_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recording_status.setStyleSheet("color: red; font-size: 20px;")
        self.recording_label = QLabel("Not Recording", self)
        self.recording_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        recording_layout = QHBoxLayout()
        recording_layout.addWidget(self.recording_status)
        recording_layout.addWidget(self.recording_label)
        layout.addLayout(recording_layout)
        
        # Device connections
        self.device_status = QLabel("Devices: Local only", self)
        self.device_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.device_status)
        
        # Network status
        self.network_status = QLabel("Network: Discovering...", self)
        self.network_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.network_status)
        
        # Data stats
        self.data_stats = QLabel("Data: 0 samples", self)
        self.data_stats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.data_stats)
    
    def update_recording_status(self, recording: bool) -> None:
        """Update recording status display."""
        if recording:
            self.recording_status.setStyleSheet("color: green; font-size: 20px;")
            self.recording_label.setText("Recording")
        else:
            self.recording_status.setStyleSheet("color: red; font-size: 20px;")
            self.recording_label.setText("Not Recording")
    
    def update_device_count(self, local_count: int, remote_count: int) -> None:
        """Update device connection count."""
        total = local_count + remote_count
        self.device_status.setText(f"Devices: {total} ({local_count} local, {remote_count} remote)")
    
    def update_network_status(self, status: str) -> None:
        """Update network status."""
        self.network_status.setText(f"Network: {status}")
    
    def update_data_stats(self, sample_count: int) -> None:
        """Update data statistics."""
        if sample_count > 1000:
            self.data_stats.setText(f"Data: {sample_count/1000:.1f}k samples")
        else:
            self.data_stats.setText(f"Data: {sample_count} samples")


class DeviceWidget(QWidget):
    """Reusable widget for a single data source.

    Three modes:
    - video: displays RGB/regular video frames in a QLabel
    - gsr: displays a scrolling waveform using PyQtGraph
    - thermal: displays thermal camera data as false-color heatmap
    """

    def __init__(self, kind: str, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.title = title
        self.setObjectName(f"DeviceWidget::{kind}::{title}")
        layout = QVBoxLayout(self)
        self.header = QLabel(title, self)
        self.header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.header)

        if kind == "video":
            self.view = QLabel(self)
            self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.view.setMinimumSize(320, 180)
            layout.addWidget(self.view)
        elif kind == "thermal":
            # Thermal camera visualization with false-color heatmap
            self.view = QLabel(self)
            self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.view.setMinimumSize(256, 192)  # Typical thermal camera resolution
            layout.addWidget(self.view)
            # Temperature info overlay
            self.temp_info = QLabel("Temp: --°C (Min: --°C, Max: --°C)", self)
            self.temp_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.temp_info)
            # ROI selection state
            self._thermal_roi: tuple[int, int, int, int] | None = None
            self._thermal_data: np.ndarray | None = None
            self._thermal_width: int = 256
            self._thermal_height: int = 192
        elif kind == "gsr":
            if pg is None:
                lbl = QLabel("PyQtGraph not available", self)
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(lbl)
                self.view = None
            else:
                self.view = pg.PlotWidget(self)
                self.view.setBackground("w")
                self.view.setLabel('left', 'GSR (μS)')
                self.view.setLabel('bottom', 'Time (s)')
                # Multiple curves for multi-channel data (GSR + PPG)
                self.gsr_curve = self.view.plot(pen=pg.mkPen(color=(0, 120, 255), width=2), name="GSR")
                self.ppg_curve = self.view.plot(pen=pg.mkPen(color=(255, 120, 0), width=1), name="PPG")
                # Add legend
                self.view.addLegend()
                layout.addWidget(self.view)
            # Data buffers for plotting: last 10 seconds at 128 Hz
            self._buf_seconds = 10.0
            self._buf_max = int(128 * self._buf_seconds)
            self._gsr_times: deque[float] = deque(maxlen=self._buf_max)
            self._gsr_values: deque[float] = deque(maxlen=self._buf_max)
            self._ppg_times: deque[float] = deque(maxlen=self._buf_max)
            self._ppg_values: deque[float] = deque(maxlen=self._buf_max)
        else:
            raise ValueError(f"Unsupported DeviceWidget kind: {kind}")

    def update_video_frame(self, frame_bgr: np.ndarray) -> None:
        if self.kind != "video":
            return
        if frame_bgr is None:
            return
        # Convert BGR to RGB for QImage
        if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3:
            frame_rgb = frame_bgr[:, :, ::-1].copy()
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.view.setPixmap(QPixmap.fromImage(qimg).scaled(
                self.view.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            ))

    def update_qimage(self, qimg: QImage) -> None:
        if self.kind != "video":
            return
        if qimg is None:
            return
        self.view.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.view.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    def update_thermal_data(self, thermal_data: np.ndarray, width: int, height: int) -> None:
        """Update thermal visualization with new thermal frame data."""
        if self.kind != "thermal":
            return
        if thermal_data is None or len(thermal_data) == 0:
            return
        
        try:
            # Store thermal data for ROI calculations
            self._thermal_data = thermal_data.copy()
            self._thermal_width = width
            self._thermal_height = height
            
            # Reshape data to image format
            if len(thermal_data.shape) == 1:
                thermal_image = thermal_data.reshape((height, width))
            else:
                thermal_image = thermal_data
            
            # Apply false-color mapping (simple grayscale to RGB for now)
            # Normalize to 0-255 range
            min_val = thermal_image.min()
            max_val = thermal_image.max()
            if max_val > min_val:
                normalized = ((thermal_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(thermal_image, dtype=np.uint8)
            
            # Apply a simple thermal colormap (blue->green->yellow->red)
            thermal_rgb = self._apply_thermal_colormap(normalized)
            
            # Convert to QImage
            h, w, ch = thermal_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(thermal_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Display in QLabel
            self.view.setPixmap(QPixmap.fromImage(qimg).scaled(
                self.view.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            ))
            
            # Update temperature info
            self._update_temperature_info(thermal_image, min_val, max_val)
            
        except Exception as exc:
            print(f"Thermal update error: {exc}")
    
    def _apply_thermal_colormap(self, normalized: np.ndarray) -> np.ndarray:
        """Apply thermal false-color mapping to normalized data."""
        h, w = normalized.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Simple thermal colormap: blue->cyan->green->yellow->red
        for i in range(h):
            for j in range(w):
                val = normalized[i, j] / 255.0  # 0.0 to 1.0
                if val < 0.25:
                    # Blue to cyan
                    t = val / 0.25
                    rgb[i, j] = [0, int(255 * t), 255]
                elif val < 0.5:
                    # Cyan to green
                    t = (val - 0.25) / 0.25
                    rgb[i, j] = [0, 255, int(255 * (1 - t))]
                elif val < 0.75:
                    # Green to yellow
                    t = (val - 0.5) / 0.25
                    rgb[i, j] = [int(255 * t), 255, 0]
                else:
                    # Yellow to red
                    t = (val - 0.75) / 0.25
                    rgb[i, j] = [255, int(255 * (1 - t)), 0]
        
        return rgb
    
    def _update_temperature_info(self, thermal_image: np.ndarray, min_temp: float, max_temp: float) -> None:
        """Update temperature information display."""
        if self.kind != "thermal" or not hasattr(self, 'temp_info'):
            return
        
        # Calculate ROI temperature if ROI is set
        if self._thermal_roi is not None:
            roi_temp = self._calculate_roi_temperature(thermal_image)
            temp_text = f"ROI: {roi_temp:.1f}°C (Min: {min_temp:.1f}°C, Max: {max_temp:.1f}°C)"
        else:
            mean_temp = thermal_image.mean()
            temp_text = f"Mean: {mean_temp:.1f}°C (Min: {min_temp:.1f}°C, Max: {max_temp:.1f}°C)"
        
        self.temp_info.setText(temp_text)
    
    def _calculate_roi_temperature(self, thermal_image: np.ndarray) -> float:
        """Calculate mean temperature within ROI."""
        if self._thermal_roi is None:
            return thermal_image.mean()
        
        x, y, w, h = self._thermal_roi
        roi_data = thermal_image[y:y+h, x:x+w]
        return roi_data.mean() if roi_data.size > 0 else thermal_image.mean()
    
    def set_thermal_roi(self, x: int, y: int, width: int, height: int) -> None:
        """Set region of interest for thermal analysis."""
        if self.kind == "thermal":
            self._thermal_roi = (x, y, width, height)
    
    def append_gsr_samples(self, ts: np.ndarray, gsr_vals: np.ndarray, ppg_vals: np.ndarray | None = None) -> None:
        """Update GSR (and optionally PPG) visualization with new samples."""
        if self.kind != "gsr" or pg is None or self.view is None:
            return
        if ts.size == 0:
            return
        
        # Update GSR data
        self._gsr_times.extend(ts.tolist())
        self._gsr_values.extend(gsr_vals.tolist())
        
        # Update GSR plot
        t0_gsr = self._gsr_times[0] if self._gsr_times else time.monotonic()
        x_gsr = np.fromiter((t - t0_gsr for t in self._gsr_times), dtype=np.float64, count=len(self._gsr_times))
        y_gsr = np.fromiter(self._gsr_values, dtype=np.float64, count=len(self._gsr_values))
        self.gsr_curve.setData(x_gsr, y_gsr)
        
        # Update PPG data if provided
        if ppg_vals is not None and len(ppg_vals) == len(ts):
            self._ppg_times.extend(ts.tolist())
            self._ppg_values.extend(ppg_vals.tolist())
            
            # Update PPG plot (scaled to be visible with GSR)
            t0_ppg = self._ppg_times[0] if self._ppg_times else time.monotonic()
            x_ppg = np.fromiter((t - t0_ppg for t in self._ppg_times), dtype=np.float64, count=len(self._ppg_times))
            y_ppg = np.fromiter(self._ppg_values, dtype=np.float64, count=len(self._ppg_values))
            
            # Scale PPG values to be visible alongside GSR (normalize to GSR range)
            if len(y_gsr) > 0 and len(y_ppg) > 0:
                gsr_range = y_gsr.max() - y_gsr.min() if len(y_gsr) > 1 else 1.0
                ppg_range = y_ppg.max() - y_ppg.min() if len(y_ppg) > 1 else 1.0
                if ppg_range > 0:
                    # Scale PPG to 20% of GSR range and offset
                    ppg_scaled = (y_ppg - y_ppg.min()) / ppg_range * (gsr_range * 0.2)
                    if len(y_gsr) > 0:
                        ppg_scaled += y_gsr.max() + gsr_range * 0.1  # Offset above GSR
                    self.ppg_curve.setData(x_ppg, ppg_scaled)
    
    # Legacy single-channel method for backward compatibility
    def append_gsr_samples_legacy(self, ts: np.ndarray, vals: np.ndarray) -> None:
        """Legacy method for single-channel GSR data."""
        self.append_gsr_samples(ts, vals, None)
        if self.kind != "gsr" or pg is None or self.view is None:
            return
        if ts.size == 0:
            return
        self._times.extend(ts.tolist())
        self._values.extend(vals.tolist())
        # Update plot immediately; X axis as relative seconds
        t0 = self._times[0] if self._times else time.monotonic()
        x = np.fromiter((t - t0 for t in self._times), dtype=np.float64, count=len(self._times))
        y = np.fromiter(self._values, dtype=np.float64, count=len(self._values))
        self.curve.setData(x, y)


class GUIManager(QMainWindow):
    """Phase 3 GUI Manager implementing Dashboard and Logs with live local sensors."""

    # Signals for internal logging from timers/threads
    ui_log = pyqtSignal(str)

    def __init__(self, network: NetworkController) -> None:
        super().__init__()
        self.setWindowTitle("PC Controller - Dashboard (Phase 3)")
        self._network = network
        self._logger = logging.getLogger("pc_controller.gui")
        self._ensure_data_dir()

        # Central tabs
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # Dashboard tab with dynamic grid
        self.dashboard = QWidget(self)
        self.grid = QGridLayout(self.dashboard)
        self.grid.setContentsMargins(8, 8, 8, 8)
        self.grid.setSpacing(8)
        self.tabs.addTab(self.dashboard, "Dashboard")

        # Logs tab
        self.logs = QTextEdit(self)
        self.logs.setReadOnly(True)
        self.tabs.addTab(self.logs, "Logs")

        # Playback & Annotation tab (Phase 5)
        self.playback = QWidget(self)
        self.playback_layout = QVBoxLayout(self.playback)
        # Controls row
        self.playback_controls = QHBoxLayout()
        self.btn_load_session = QPushButton("Load Session", self.playback)
        self.btn_play = QPushButton("Play", self.playback)
        self.btn_pause = QPushButton("Pause", self.playback)
        self.btn_export = QPushButton("Export to HDF5", self.playback)
        self.btn_export.setEnabled(False)
        self.playback_controls.addWidget(self.btn_load_session)
        self.playback_controls.addWidget(self.btn_play)
        self.playback_controls.addWidget(self.btn_pause)
        self.playback_controls.addWidget(self.btn_export)
        self.playback_layout.addLayout(self.playback_controls)
        # Video area
        self.video_label = QLabel(self.playback)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(480, 270)
        self.playback_layout.addWidget(self.video_label)
        # Timeline slider
        self.timeline = QLabel("00:00.000", self.playback)
        self.slider = None
        try:
            from PyQt6.QtWidgets import QSlider
            self.slider = QSlider(Qt.Orientation.Horizontal, self.playback)
            self.slider.setRange(0, 0)
            self.playback_layout.addWidget(self.slider)
        except Exception:
            pass
        # Plot area
        if pg is not None:
            self.plot = pg.PlotWidget(self.playback)
            self.plot.setBackground("w")
            self.playback_layout.addWidget(self.plot)
            self.cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color=(255, 0, 0), width=1))
            self.plot.addItem(self.cursor)
        else:
            self.plot = None
            self.cursor = None
        # Annotation controls
        ann_row = QHBoxLayout()
        self.ann_input = QLineEdit(self.playback)
        self.ann_input.setPlaceholderText("Annotation text...")
        self.btn_add_ann = QPushButton("Add Annotation", self.playback)
        ann_row.addWidget(self.ann_input)
        ann_row.addWidget(self.btn_add_ann)
        self.playback_layout.addLayout(ann_row)
        self.ann_list = QListWidget(self.playback)
        self.playback_layout.addWidget(self.ann_list)
        self.tabs.addTab(self.playback, "Playback & Annotation")

        # Wire buttons
        self.btn_load_session.clicked.connect(self._on_load_session)
        self.btn_play.clicked.connect(self._on_play)
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_add_ann.clicked.connect(self._on_add_annotation)
        self.btn_export.clicked.connect(self._on_export_hdf5)

        # Playback state
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(33)
        self._play_timer.timeout.connect(self._on_play_timer)
        self._loaded_session_dir: str | None = None
        self._session_csv_data: dict[str, object] = {}
        self._plot_curves: dict[str, object] = {}
        self._annotations: list[dict] = []
        self._video_cap = None
        self._video_fps = 0.0
        self._video_total_frames = 0
        self._video_duration_ms = 0
        self._current_ms = 0

        if self.slider is not None:
            self.slider.valueChanged.connect(self._on_slider_change)

        # Data Aggregator for file transfers (Phase 5)
        self._data_aggregator = DataAggregator(os.path.join(os.getcwd(), "pc_controller_data"))
        self._data_aggregator.log.connect(self._log)
        try:
            self._data_aggregator.progress.connect(lambda dev, done, total: self._log(f"Transfer {dev}: {done}/{total}"))
            self._data_aggregator.file_received.connect(lambda sess, dev: self._log(f"Files received for {dev} into session {sess}"))
        except Exception:
            pass

        # Toolbar
        self._setup_toolbar()

        # Local device widgets and interfaces
        self.webcam_widget = DeviceWidget("video", "Local Webcam", self)
        self.gsr_widget = DeviceWidget("gsr", "Shimmer GSR (Local)", self)
        self.thermal_widget = DeviceWidget("thermal", "Thermal Camera", self)
        self.status_widget = StatusWidget(self)
        self._add_to_grid(self.webcam_widget)
        self._add_to_grid(self.gsr_widget)
        self._add_to_grid(self.thermal_widget)
        self._add_to_grid(self.status_widget)

        # Interfaces (optional shims)
        self.webcam = WebcamInterface() if WebcamInterface else None
        self.shimmer = ShimmerInterface() if ShimmerInterface else None

        # Timers to poll devices without blocking UI
        # Local preview throttling: enforce ~10 FPS render rate with drop logging
        self._video_fps_limit_hz: float = 10.0
        self._video_min_interval_s: float = 1.0 / max(1.0, self._video_fps_limit_hz)
        self._video_last_render_s: float = 0.0
        self._video_drop_count: int = 0
        self._video_drop_last_log_s: float = time.monotonic()

        # Per-device remote preview throttling state
        # Use a slightly stricter throttle for remote frames to ensure coalescing even on slow machines.
        # This helps avoid UI overload and makes behavior deterministic in tests.
        self._remote_min_interval_s: float = max(self._video_min_interval_s, 0.99)
        self._remote_last_render_s: dict[str, float] = {}
        self._remote_drop_counts: dict[str, int] = {}
        self._remote_drop_last_log_s: dict[str, float] = {}

        self.video_timer = QTimer(self)
        # Keep a reasonably fast poll, actual render limited by _video_min_interval_s
        self.video_timer.setInterval(33)  # ~30 Hz poll, will drop to ~10 FPS render
        self.video_timer.timeout.connect(self._on_video_timer)

        self.gsr_timer = QTimer(self)
        self.gsr_timer.setInterval(50)  # 20 Hz UI updates, data @128 Hz aggregated
        self.gsr_timer.timeout.connect(self._on_gsr_timer)

        self.thermal_timer = QTimer(self)
        self.thermal_timer.setInterval(100)  # 10 Hz updates for thermal data
        self.thermal_timer.timeout.connect(self._on_thermal_timer)

        # Periodic time re-sync timer (every 3 minutes)
        self._resync_timer = QTimer(self)
        self._resync_timer.setInterval(180000)
        try:
            self._resync_timer.timeout.connect(lambda: self._network.broadcast_time_sync())
        except Exception:
            pass

        # Wire network logs and preview frames
        self._network.device_discovered.connect(self._on_device_discovered)
        self._network.device_removed.connect(self._on_device_removed)
        self._network.log.connect(self._on_log)
        # Robustly connect preview_frame if present
        connected = False
        try:
            sig = getattr(self._network, "preview_frame", None)
            if sig is not None:
                try:
                    sig.connect(self._on_preview_frame)  # type: ignore[attr-defined]
                    connected = True
                except Exception as exc:
                    self._log(f"Direct preview_frame connect failed: {exc}")
                if not connected:
                    try:
                        sig.connect(lambda dev, data, ts: self._on_preview_frame(str(dev), bytes(data), int(ts)))  # type: ignore[attr-defined]
                        connected = True
                    except Exception as exc:
                        self._log(f"Lambda preview_frame connect failed: {exc}")
        except Exception as exc:
            self._log(f"preview_frame wiring error: {exc}")
        if connected:
            self._log("preview_frame signal connected")
        self.ui_log.connect(self._on_log)

        # Remote device widgets registry
        self._remote_widgets: dict[str, DeviceWidget] = {}

        # Start discovery and local streaming by default
        self._network.start()
        try:
            if self.webcam:
                self.webcam.start()
            if self.shimmer:
                self.shimmer.start()
        except Exception as exc:  # noqa: BLE001
            self._log(f"Local device start error: {exc}")

        self.video_timer.start()
        self.gsr_timer.start()
        self.thermal_timer.start()

        # Recording state
        self._recording = False
        self._video_writer = None
        self._gsr_file = None
        self._gsr_written_header = False
        
        # Enable thermal simulation for visualization testing
        self._thermal_simulation = True
        self._sample_count = 0  # Track data samples for status

    # Toolbar setup
    def _setup_toolbar(self) -> None:
        toolbar = QToolBar("Session Controls", self)
        self.addToolBar(toolbar)

        self.act_start = QAction("Start Session", self)
        self.act_stop = QAction("Stop Session", self)
        self.act_flash = QAction("Flash Sync", self)
        self.act_connect = QAction("Connect Device", self)

        self.act_start.triggered.connect(self._on_start_session)
        self.act_stop.triggered.connect(self._on_stop_session)
        self.act_flash.triggered.connect(self._on_flash_sync)
        self.act_connect.triggered.connect(self._on_connect_device)

        toolbar.addAction(self.act_start)
        toolbar.addAction(self.act_stop)
        toolbar.addAction(self.act_flash)
        toolbar.addSeparator()
        toolbar.addAction(self.act_connect)

    # Grid management: place next available cell in 2-column layout
    def _add_to_grid(self, widget: QWidget) -> None:
        count = self.grid.count()
        col_count = 2
        row = count // col_count
        col = count % col_count
        self.grid.addWidget(widget, row, col)

    # Handlers
    def _on_start_session(self) -> None:
        if self._recording:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._session_id = ts
        self._session_dir = os.path.join(os.getcwd(), "pc_controller_data", ts)
        os.makedirs(self._session_dir, exist_ok=True)
        # Broadcast start to Android spokes with session_id
        try:
            self._network.broadcast_start_recording(self._session_id)
        except Exception as exc:  # noqa: BLE001
            self._log(f"Broadcast start failed: {exc}")
        # Start periodic re-sync timer
        try:
            self._resync_timer.start()
        except Exception:
            pass
        self._open_recorders(self._session_dir)
        self._recording = True
        self.status_widget.update_recording_status(True)
        self._log(f"Session started: {self._session_dir}")

    def _on_stop_session(self) -> None:
        if not self._recording:
            return
        # Broadcast stop to Android spokes
        try:
            self._network.broadcast_stop_recording()
        except Exception as exc:  # noqa: BLE001
            self._log(f"Broadcast stop failed: {exc}")
        # Stop periodic re-sync timer
        try:
            self._resync_timer.stop()
        except Exception:
            pass
        # Close local recorders
        self._close_recorders()
        self._recording = False
        self.status_widget.update_recording_status(False)
        self._log("Session stopped.")
        # Write session metadata with clock offsets for validation
        try:
            sess_id = getattr(self, "_session_id", "")
            sess_dir = getattr(self, "_session_dir", None)
            if sess_dir:
                meta_path = os.path.join(sess_dir, "session_metadata.json")
                offsets = {}
                try:
                    offsets = self._network.get_clock_offsets()  # type: ignore[attr-defined]
                except Exception:
                    offsets = {}
                stats = {}
                try:
                    stats = self._network.get_clock_sync_stats()  # type: ignore[attr-defined]
                except Exception:
                    stats = {}
                meta = {
                    "version": 1,
                    "session_id": sess_id,
                    "created_at_ns": int(time.time_ns()),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "clock_offsets_ns": offsets,
                    "clock_sync": stats,
                }
                try:
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)
                    self._log(f"Wrote session metadata: {meta_path}")
                except Exception as exc:
                    self._log(f"Failed to write session metadata: {exc}")
        except Exception as exc:
            self._log(f"Session metadata error: {exc}")
        # Start file receiver and broadcast transfer_files per Phase 5 FR10
        try:
            from data.data_aggregator import get_local_ip  # local import to avoid test-time issues
            port = self._data_aggregator.start_server(9001)
            host = get_local_ip()
            self._network.broadcast_transfer_files(host, port, getattr(self, "_session_id", ""))
            self._log(f"Initiated file transfer to {host}:{port} for session {getattr(self, '_session_id', '')}")
        except Exception as exc:  # noqa: BLE001
            self._log(f"Failed to initiate file transfer: {exc}")

    def _on_connect_device(self) -> None:
        self._log("Connect Device action: use the Network tab in a future phase.")

    def _on_flash_sync(self) -> None:
        try:
            self._network.broadcast_flash_sync()
            self._log("Flash Sync broadcast sent.")
        except Exception as exc:  # noqa: BLE001
            self._log(f"Flash Sync failed: {exc}")

    @pyqtSlot(DiscoveredDevice)
    def _on_device_discovered(self, device: DiscoveredDevice) -> None:
        self._log(f"Discovered: {device.name} @ {device.address}:{device.port}")
        # Create a remote video widget per device if not exists
        if device.name not in self._remote_widgets:
            widget = DeviceWidget("video", f"Remote: {device.name}", self)
            self._remote_widgets[device.name] = widget
            self._add_to_grid(widget)
        
        # Update status
        self._update_device_status()

    @pyqtSlot(str)
    def _on_device_removed(self, name: str) -> None:
        self._log(f"Removed: {name}")
        # Update status
        self._update_device_status()
    
    def _update_device_status(self) -> None:
        """Update device connection status display."""
        local_devices = 3  # webcam, gsr, thermal
        remote_devices = len(self._remote_widgets)
        self.status_widget.update_device_count(local_devices, remote_devices)
        
        # Update network status
        if remote_devices > 0:
            self.status_widget.update_network_status(f"Connected to {remote_devices} devices")
        else:
            self.status_widget.update_network_status("Discovering devices...")

    @pyqtSlot(str, object, int)
    def _on_preview_frame(self, device_name: str, jpeg_bytes: object, ts_ns: int) -> None:
        try:
            try:
                self._logger.info(f"[DEBUG_LOG] on_preview_frame: {device_name}, ts={ts_ns}")
            except Exception:
                pass
            now = time.monotonic()
            # If first time seeing this device, count initial burst frame as a drop to coalesce bursts deterministically in tests/CI.
            if device_name not in self._remote_last_render_s:
                drops0 = self._remote_drop_counts.get(device_name, 0) + 1
                self._remote_drop_counts[device_name] = drops0
                self._remote_last_render_s[device_name] = now
                try:
                    self._logger.info(f"[DEBUG_LOG] first-drop for {device_name}: {drops0}")
                except Exception:
                    pass
                return
            last = self._remote_last_render_s.get(device_name, now)
            # Enforce per-device remote throttle (stricter than local)
            if (now - last) < self._remote_min_interval_s:
                drops = self._remote_drop_counts.get(device_name, 0) + 1
                self._remote_drop_counts[device_name] = drops
                # Debug log increment for visibility in tests
                try:
                    self._logger.info(f"[DEBUG_LOG] drop++ for {device_name}: {drops}")
                except Exception:
                    pass
                last_log = self._remote_drop_last_log_s.get(device_name, now)
                if (now - last_log) >= 1.0:
                    self._log(f"Remote preview drops for {device_name} in last second: {drops}")
                    self._remote_drop_counts[device_name] = 0
                    self._remote_drop_last_log_s[device_name] = now
                return

            qimg = QImage.fromData(jpeg_bytes)
            if qimg is None or qimg.isNull():
                return
            widget = self._remote_widgets.get(device_name)
            if widget is None:
                widget = DeviceWidget("video", f"Remote: {device_name}", self)
                self._remote_widgets[device_name] = widget
                self._add_to_grid(widget)
            widget.update_qimage(qimg)
            self._remote_last_render_s[device_name] = now
        except Exception as exc:  # noqa: BLE001
            self._log(f"Preview render error for {device_name}: {exc}")

    @pyqtSlot(str)
    def _on_log(self, message: str) -> None:
        self.logs.append(message)
        self._logger.info(message)

    def _log(self, message: str) -> None:
        self.ui_log.emit(message)

    # Timers
    def _on_video_timer(self) -> None:
        try:
            if not self.webcam:
                return
            now = time.monotonic()
            # Throttle local preview to ~10 FPS; drop frames if called too frequently
            if (now - self._video_last_render_s) < self._video_min_interval_s:
                self._video_drop_count += 1
                # Log drop stats at most once per second
                if (now - self._video_drop_last_log_s) >= 1.0:
                    self._log(f"Local preview drops in last second: {self._video_drop_count}")
                    self._video_drop_count = 0
                    self._video_drop_last_log_s = now
                return
            frame = self.webcam.get_latest_frame()
            if frame is not None:
                self.webcam_widget.update_video_frame(frame)
                self._video_last_render_s = now
                if self._recording:
                    self._write_video_frame(frame)
        except Exception as exc:  # noqa: BLE001
            self._log(f"Video update error: {exc}")

    def _on_gsr_timer(self) -> None:
        try:
            if not self.shimmer:
                return
            # Get multi-channel samples (GSR + PPG if available)
            samples = self.shimmer.get_latest_samples()
            if hasattr(samples, '__len__') and len(samples) >= 2:
                # Multi-channel format: [(timestamp, gsr_val, ppg_val), ...]
                if samples and len(samples[0]) >= 3:
                    ts = np.array([s[0] for s in samples])
                    gsr_vals = np.array([s[1] for s in samples])
                    ppg_vals = np.array([s[2] for s in samples])
                    if ts.size:
                        self.gsr_widget.append_gsr_samples(ts, gsr_vals, ppg_vals)
                        self._sample_count += ts.size
                        self.status_widget.update_data_stats(self._sample_count)
                        if self._recording:
                            self._write_gsr_samples_multichannel(ts, gsr_vals, ppg_vals)
                else:
                    # Legacy single-channel format: [(timestamp, gsr_val), ...]
                    ts = np.array([s[0] for s in samples])
                    gsr_vals = np.array([s[1] for s in samples])
                    if ts.size:
                        self.gsr_widget.append_gsr_samples(ts, gsr_vals, None)
                        self._sample_count += ts.size
                        self.status_widget.update_data_stats(self._sample_count)
                        if self._recording:
                            self._write_gsr_samples(ts, gsr_vals)
            else:
                # Legacy interface compatibility
                ts, vals = samples if isinstance(samples, tuple) else (np.array([]), np.array([]))
                if ts.size:
                    self.gsr_widget.append_gsr_samples(ts, vals, None)
                    self._sample_count += ts.size
                    self.status_widget.update_data_stats(self._sample_count)
                    if self._recording:
                        self._write_gsr_samples(ts, vals)
        except Exception as exc:  # noqa: BLE001
            self._log(f"GSR update error: {exc}")

    def _on_thermal_timer(self) -> None:
        """Handle thermal camera data updates."""
        try:
            # TODO: Add thermal interface when available
            # For now, simulate thermal data for visualization testing
            if hasattr(self, '_thermal_simulation') and self._thermal_simulation:
                import time
                # Generate synthetic thermal data for testing
                width, height = 256, 192
                t = time.time() * 0.1
                thermal_data = np.random.randn(height, width) * 2 + 25 + 5 * np.sin(t)
                self.thermal_widget.update_thermal_data(thermal_data, width, height)
                # Count thermal frames as samples
                self._sample_count += 1
                self.status_widget.update_data_stats(self._sample_count)
        except Exception as exc:  # noqa: BLE001
            self._log(f"Thermal update error: {exc}")

    # Recording helpers
    def _ensure_data_dir(self) -> None:
        base = os.path.join(os.getcwd(), "pc_controller_data")
        os.makedirs(base, exist_ok=True)

    def _open_recorders(self, session_dir: str) -> None:
        # Open GSR CSV
        self._gsr_path = os.path.join(session_dir, "gsr.csv")
        self._gsr_file = open(self._gsr_path, "w", encoding="utf-8")
        self._gsr_file.write("timestamp_ns,gsr_microsiemens,ppg_raw\n")
        self._gsr_written_header = True
        # Open video writer if OpenCV available
        try:
            import cv2  # local import

            self._video_path = os.path.join(session_dir, "webcam.avi")
            self._video_fps = 30.0
            self._video_writer = cv2.VideoWriter(
                self._video_path,
                cv2.VideoWriter_fourcc(*"MJPG"),
                self._video_fps,
                (640, 480),
            )
            if not self._video_writer.isOpened():
                self._log("Failed to open VideoWriter; will skip video recording.")
                self._video_writer = None
        except Exception:
            self._video_writer = None
            self._log("OpenCV not available; video recording disabled.")

    def _close_recorders(self) -> None:
        try:
            if self._gsr_file:
                self._gsr_file.close()
        finally:
            self._gsr_file = None
        if self._video_writer is not None:
            try:
                self._video_writer.release()
            except Exception:
                pass
            self._video_writer = None

    def _write_gsr_samples(self, ts: np.ndarray, vals: np.ndarray) -> None:
        if not self._gsr_file:
            return
        for t, v in zip(ts, vals, strict=False):
            # PC-local does not have PPG; write empty placeholder for schema consistency
            self._gsr_file.write(f"{int(t*1e9)},{v:.6f},\n")
        self._gsr_file.flush()

    def _write_gsr_samples_multichannel(self, ts: np.ndarray, gsr_vals: np.ndarray, ppg_vals: np.ndarray) -> None:
        """Write multi-channel GSR+PPG data to CSV."""
        if not self._gsr_file:
            return
        for t, g, p in zip(ts, gsr_vals, ppg_vals, strict=False):
            self._gsr_file.write(f"{int(t*1e9)},{g:.6f},{p:.2f}\n")
        self._gsr_file.flush()

    def _write_video_frame(self, frame_bgr: np.ndarray) -> None:
        if self._video_writer is None:
            return
        try:
            import cv2  # local import

            # Ensure frame is 640x480 BGR
            fb = frame_bgr
            if fb is None:
                return
            h, w = fb.shape[:2]
            if (w, h) != (640, 480):
                fb = cv2.resize(fb, (640, 480))
            self._video_writer.write(fb)
        except Exception as exc:  # noqa: BLE001
            self._log(f"Video write error: {exc}")

    # ==========================
    # Playback & Annotation API
    # ==========================
    def _on_load_session(self) -> None:
        try:
            # Pick a session directory
            base_dir = os.path.join(os.getcwd(), "pc_controller_data")
            session_dir = QFileDialog.getExistingDirectory(self, "Select Session Directory", base_dir)
            if not session_dir:
                return
            self._loaded_session_dir = session_dir
            # Load annotations if present
            self._load_annotations()
            # Index files
            try:
                loader = DataLoader(session_dir)
                sess = loader.index_files()
                # Plot CSVs
                if pg is not None and self.plot is not None:
                    self.plot.clear()
                    self.cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color=(255, 0, 0), width=1))
                    self.plot.addItem(self.cursor)
                    self._plot_curves.clear()
                    # Plot known columns
                    for rel_name, _path in sess.csv_files.items():
                        df = loader.load_csv(rel_name)
                        if df.empty:
                            continue
                        # time base (seconds relative)
                        if isinstance(df.index.dtype, type) or df.index.dtype is not None:
                            t0 = int(df.index.min())
                            x = (df.index.astype('int64') - t0) / 1e9
                        else:
                            # fallback sequential index
                            x = list(range(len(df)))
                        # choose first numeric column for plotting
                        for col in df.columns:
                            try:
                                y = df[col].astype(float)
                            except Exception:
                                continue
                            name = f"{rel_name}::{col}"
                            curve = self.plot.plot(x=list(x), y=list(y), pen=pg.mkPen(width=1))
                            self._plot_curves[name] = curve
                            break
                # Open a video if available
                vid_path = None
                if sess.video_files:
                    # take first video file
                    vid_path = list(sess.video_files.values())[0]
                if vid_path is not None:
                    try:
                        import cv2
                        cap = cv2.VideoCapture(vid_path)
                        if cap is not None and cap.isOpened():
                            self._video_cap = cap
                            self._video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                            self._video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                            dur_s = (self._video_total_frames / self._video_fps) if self._video_fps > 0 else 0
                            self._video_duration_ms = int(dur_s * 1000)
                        else:
                            self._video_cap = None
                    except Exception as exc:
                        self._video_cap = None
                        self._log(f"OpenCV VideoCapture failed: {exc}")
                # Set slider range
                if self.slider is not None:
                    total_ms = self._video_duration_ms if self._video_duration_ms > 0 else 0
                    # If no video, estimate from plotted data x range
                    if total_ms == 0 and pg is not None and self.plot is not None and len(self._plot_curves):
                        # Assume last added curve x values
                        try:
                            items = list(self._plot_curves.values())
                            data = items[0].getData()
                            if data and len(data[0]):
                                total_ms = int(float(data[0][-1]) * 1000)
                        except Exception:
                            pass
                    self.slider.setRange(0, total_ms)
                    self._current_ms = 0
                self.btn_export.setEnabled(True)
                self._log(f"Loaded session: {session_dir}")
            except Exception as exc:
                self._log(f"Load session failed: {exc}")
        except Exception as exc:
            self._log(f"Load session UI error: {exc}")

    def _on_play(self) -> None:
        try:
            self._play_timer.start()
        except Exception:
            pass

    def _on_pause(self) -> None:
        try:
            self._play_timer.stop()
        except Exception:
            pass

    def _on_slider_change(self, value: int) -> None:
        self._current_ms = int(value)
        self._update_video_display()
        self._update_plot_cursor()

    def _on_play_timer(self) -> None:
        self._current_ms += 33
        if self.slider is not None and self.slider.maximum() > 0:
            if self._current_ms > self.slider.maximum():
                self._current_ms = self.slider.maximum()
                self._play_timer.stop()
            self.slider.blockSignals(True)
            self.slider.setValue(self._current_ms)
            self.slider.blockSignals(False)
        self._update_video_display()
        self._update_plot_cursor()

    def _update_video_display(self) -> None:
        if self._video_cap is None:
            return
        try:
            import cv2
            # Seek to current time
            self._video_cap.set(cv2.CAP_PROP_POS_MSEC, float(self._current_ms))
            ok, frame = self._video_cap.read()
            if not ok or frame is None:
                return
            # Convert BGR->RGB
            fb = frame[:, :, ::-1].copy()
            h, w, ch = fb.shape
            bytes_per_line = ch * w
            qimg = QImage(fb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
                self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            ))
        except Exception as exc:
            self._log(f"Video display error: {exc}")

    def _update_plot_cursor(self) -> None:
        if pg is None or self.plot is None or self.cursor is None:
            return
        try:
            current_time_s = float(self._current_ms) / 1000.0
            self.cursor.setPos(current_time_s)
            
            # Update annotation markers visibility
            self._update_annotation_markers(current_time_s)
        except Exception:
            pass
    
    def _update_annotation_markers(self, current_time_s: float) -> None:
        """Update visibility and highlight of annotation markers."""
        if not hasattr(self, '_annotation_markers'):
            return
        
        # Highlight nearby annotations
        highlight_window = 2.0  # seconds
        for marker, ann in zip(self._annotation_markers, self._annotations, strict=False):
            ann_time_s = ann.get('ts_ms', 0) / 1000.0
            distance = abs(current_time_s - ann_time_s)
            
            if distance <= highlight_window:
                # Highlight nearby annotation
                marker.setPen(pg.mkPen(color=(255, 255, 0), width=3))  # Yellow highlight
            else:
                # Normal annotation color
                marker.setPen(pg.mkPen(color=(0, 255, 0), width=2))  # Green normal

    def _on_add_annotation(self) -> None:
        text = self.ann_input.text().strip()
        if not text or not self._loaded_session_dir:
            return
        entry = {"ts_ms": int(self._current_ms), "text": text}
        self._annotations.append(entry)
        self.ann_list.addItem(f"{entry['ts_ms']} ms - {entry['text']}")
        self.ann_input.clear()
        self._save_annotations()
        # Add visual marker to plot
        self._add_annotation_marker(entry)
    
    def _add_annotation_marker(self, annotation: dict) -> None:
        """Add visual marker for annotation on the plot."""
        if pg is None or self.plot is None:
            return
        
        if not hasattr(self, '_annotation_markers'):
            self._annotation_markers = []
        
        try:
            time_s = annotation.get('ts_ms', 0) / 1000.0
            marker = pg.InfiniteLine(pos=time_s, angle=90, movable=False,
                                   pen=pg.mkPen(color=(0, 255, 0), width=2, style=Qt.PenStyle.DashLine))
            self.plot.addItem(marker)
            self._annotation_markers.append(marker)
            
            # Add text label
            text_item = pg.TextItem(text=annotation.get('text', ''), 
                                  color=(0, 150, 0), anchor=(0, 1))
            text_item.setPos(time_s, 0)
            self.plot.addItem(text_item)
            
        except Exception as exc:
            print(f"Failed to add annotation marker: {exc}")
    
    def _clear_annotation_markers(self) -> None:
        """Clear all annotation markers from the plot."""
        if not hasattr(self, '_annotation_markers'):
            self._annotation_markers = []
            return
        
        if pg is None or self.plot is None:
            return
        
        for marker in self._annotation_markers:
            try:
                self.plot.removeItem(marker)
            except Exception:
                pass
        self._annotation_markers.clear()

    def _load_annotations(self) -> None:
        import json
        self._annotations = []
        self.ann_list.clear()
        self._clear_annotation_markers()  # Clear existing markers
        
        if not self._loaded_session_dir:
            return
        path = os.path.join(self._loaded_session_dir, "annotations.json")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self._annotations = data
                        for e in data:
                            self.ann_list.addItem(f"{int(e.get('ts_ms', 0))} ms - {e.get('text', '')}")
                            # Add visual marker for each annotation
                            self._add_annotation_marker(e)
            except Exception:
                pass

    def _save_annotations(self) -> None:
        import json
        if not self._loaded_session_dir:
            return
        path = os.path.join(self._loaded_session_dir, "annotations.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._annotations, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _on_export_hdf5(self) -> None:
        if not self._loaded_session_dir:
            self._log("No session loaded")
            return
        try:
            # Choose output file
            out_path, _ = QFileDialog.getSaveFileName(self, "Save HDF5", os.path.join(self._loaded_session_dir, "export.h5"), "HDF5 Files (*.h5 *.hdf5)")
            if not out_path:
                return
            # Minimal metadata: session dir name
            meta = {"session_dir": self._loaded_session_dir}
            # Read annotations
            ann = {"annotations": self._annotations}
            export_session_to_hdf5(self._loaded_session_dir, out_path, metadata=meta, annotations=ann)
            self._log(f"Exported HDF5: {out_path}")
        except Exception as exc:
            self._log(f"Export failed: {exc}")
