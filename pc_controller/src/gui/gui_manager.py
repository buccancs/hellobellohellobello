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


class DeviceWidget(QWidget):
    """Reusable widget for a single data source.

    Two modes:
    - video: displays frames in a QLabel
    - gsr: displays a scrolling waveform using PyQtGraph
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
            # Thermal camera visualization - similar to video but with temperature colormap
            self.view = QLabel(self)
            self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.view.setMinimumSize(320, 240)  # Slightly taller for thermal resolution
            layout.addWidget(self.view)
            # Add temperature range indicator
            self.temp_range_label = QLabel("Temp Range: --°C to --°C", self)
            self.temp_range_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.temp_range_label.setStyleSheet("color: orange; font-size: 10px;")
            layout.addWidget(self.temp_range_label)
        elif kind == "gsr":
            if pg is None:
                lbl = QLabel("PyQtGraph not available", self)
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(lbl)
                self.view = None
            else:
                self.view = pg.PlotWidget(self)
                self.view.setBackground("w")
                self.curve = self.view.plot(pen=pg.mkPen(color=(0, 120, 255), width=2))
                # Add axis labels and grid for better visualization
                self.view.setLabel('left', 'GSR (μS)', color='black', size='10pt')
                self.view.setLabel('bottom', 'Time (seconds)', color='black', size='10pt')
                self.view.showGrid(True, True, alpha=0.3)
                # Add data quality indicator
                self.quality_label = QLabel("Quality: --", self)
                self.quality_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.quality_label.setStyleSheet("color: green; font-size: 10px;")
                layout.addWidget(self.view)
                layout.addWidget(self.quality_label)
            # Data buffer for plotting: last 10 seconds at 128 Hz
            self._buf_seconds = 10.0
            self._buf_max = int(128 * self._buf_seconds)
            self._times: deque[float] = deque(maxlen=self._buf_max)
            self._values: deque[float] = deque(maxlen=self._buf_max)
            self._sample_count = 0
        else:
            raise ValueError(f"Unsupported DeviceWidget kind: {kind}")

    def update_thermal_frame(self, thermal_data: np.ndarray, temp_min: float = None, temp_max: float = None) -> None:
        """Update thermal camera display with false-color temperature visualization."""
        if self.kind != "thermal":
            return
        if thermal_data is None:
            return
        
        try:
            import cv2
            # Apply false-color mapping for thermal data
            # Normalize thermal data to 0-255 range
            if thermal_data.min() == thermal_data.max():
                # Avoid division by zero
                normalized = np.zeros_like(thermal_data, dtype=np.uint8)
            else:
                normalized = ((thermal_data - thermal_data.min()) / 
                            (thermal_data.max() - thermal_data.min()) * 255).astype(np.uint8)
            
            # Apply colormap (COLORMAP_JET gives blue-to-red temperature visualization)
            colored_thermal = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            
            # Convert BGR to RGB for Qt
            thermal_rgb = colored_thermal[:, :, ::-1]
            h, w, ch = thermal_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(thermal_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            self.view.setPixmap(QPixmap.fromImage(qimg).scaled(
                self.view.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            ))
            
            # Update temperature range display
            if temp_min is not None and temp_max is not None:
                self.temp_range_label.setText(f"Temp Range: {temp_min:.1f}°C to {temp_max:.1f}°C")
            else:
                # Use data min/max if actual temps not provided
                data_min, data_max = thermal_data.min(), thermal_data.max()
                self.temp_range_label.setText(f"Temp Range: {data_min:.1f} to {data_max:.1f}")
                
        except Exception:
            # Fallback to grayscale if OpenCV not available
            if thermal_data.ndim == 2:
                # Convert single channel to RGB
                normalized = ((thermal_data - thermal_data.min()) / 
                            (thermal_data.max() - thermal_data.min()) * 255).astype(np.uint8)
                h, w = normalized.shape
                qimg = QImage(normalized.data, w, h, w, QImage.Format.Format_Grayscale8)
                self.view.setPixmap(QPixmap.fromImage(qimg).scaled(
                    self.view.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                ))

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

    def append_gsr_samples(self, ts: np.ndarray, vals: np.ndarray) -> None:
        if self.kind != "gsr" or pg is None or self.view is None:
            return
        if ts.size == 0:
            return
        self._times.extend(ts.tolist())
        self._values.extend(vals.tolist())
        self._sample_count += ts.size
        
        # Update plot immediately; X axis as relative seconds
        t0 = self._times[0] if self._times else time.monotonic()
        x = np.fromiter((t - t0 for t in self._times), dtype=np.float64, count=len(self._times))
        y = np.fromiter(self._values, dtype=np.float64, count=len(self._values))
        self.curve.setData(x, y)
        
        # Update data quality indicator based on sampling rate and signal stability
        if hasattr(self, 'quality_label') and self._sample_count > 10:
            # Calculate approximate sampling rate
            if len(self._times) > 1:
                time_span = self._times[-1] - self._times[0]
                if time_span > 0:
                    sample_rate = len(self._times) / time_span
                    # Check signal quality based on sampling rate and variance
                    signal_variance = np.var(y[-min(128, len(y)):])  # Last second of data
                    
                    if sample_rate > 120:  # Expected ~128 Hz
                        if signal_variance < 0.01:  # Very stable signal might be disconnected
                            quality = "Poor (No signal?)"
                            color = "red"
                        elif signal_variance > 10.0:  # Very noisy signal
                            quality = "Fair (Noisy)"
                            color = "orange"
                        else:
                            quality = "Good"
                            color = "green"
                    else:
                        quality = f"Poor ({sample_rate:.0f} Hz)"
                        color = "red"
                        
                    self.quality_label.setText(f"Quality: {quality}")
                    self.quality_label.setStyleSheet(f"color: {color}; font-size: 10px;")


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
        
        # Storage for annotation markers on the plot
        self._annotation_markers = []
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
        self._add_to_grid(self.webcam_widget)
        self._add_to_grid(self.gsr_widget)
        self._add_to_grid(self.thermal_widget)

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

        # Recording state
        self._recording = False
        self._video_writer = None
        self._gsr_file = None
        self._gsr_written_header = False

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

    @pyqtSlot(str)
    def _on_device_removed(self, name: str) -> None:
        self._log(f"Removed: {name}")

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
            ts, vals = self.shimmer.get_latest_samples()
            if ts.size:
                self.gsr_widget.append_gsr_samples(ts, vals)
                if self._recording:
                    self._write_gsr_samples(ts, vals)
        except Exception as exc:  # noqa: BLE001
            self._log(f"GSR update error: {exc}")

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
            self.cursor.setPos(float(self._current_ms) / 1000.0)
        except Exception:
            pass

    def _on_add_annotation(self) -> None:
        text = self.ann_input.text().strip()
        if not text or not self._loaded_session_dir:
            return
        entry = {"ts_ms": int(self._current_ms), "text": text}
        self._annotations.append(entry)
        self.ann_list.addItem(f"{entry['ts_ms']} ms - {entry['text']}")
        self.ann_input.clear()
        self._save_annotations()
        # Add visual marker to the plot
        self._add_annotation_marker(entry)

    def _add_annotation_marker(self, entry: dict) -> None:
        """Add a visual marker for an annotation on the plot."""
        if pg is None or self.plot is None:
            return
        try:
            # Convert timestamp to x-axis coordinate (seconds)
            x_pos = entry['ts_ms'] / 1000.0
            
            # Create a vertical line marker
            marker_line = pg.InfiniteLine(
                pos=x_pos,
                angle=90,
                pen=pg.mkPen(color=(255, 165, 0), width=2, style=pg.QtCore.Qt.PenStyle.DashLine),
                movable=False
            )
            
            # Add text label
            text_item = pg.TextItem(
                text=entry['text'][:20] + ('...' if len(entry['text']) > 20 else ''),
                color=(255, 165, 0),
                anchor=(0, 1)  # Top-left anchor
            )
            text_item.setPos(x_pos, self.plot.viewRange()[1][1])  # Position at top of plot
            
            # Store references to remove later if needed
            self._annotation_markers.append((marker_line, text_item))
            
            # Add to plot
            self.plot.addItem(marker_line)
            self.plot.addItem(text_item)
            
        except Exception as exc:
            self._log(f"Error adding annotation marker: {exc}")

    def _clear_annotation_markers(self) -> None:
        """Clear all annotation markers from the plot."""
        if pg is None or self.plot is None:
            return
        try:
            for marker_line, text_item in self._annotation_markers:
                self.plot.removeItem(marker_line)
                self.plot.removeItem(text_item)
            self._annotation_markers.clear()
        except Exception as exc:
            self._log(f"Error clearing annotation markers: {exc}")

    def _load_annotations(self) -> None:
        import json
        self._annotations = []
        self.ann_list.clear()
        self._clear_annotation_markers()  # Clear existing markers first
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
                            # Add visual marker for each loaded annotation
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
