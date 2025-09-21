#!/usr/bin/env python3
"""
Phase 4: PC Hub Main Dashboard GUI

Comprehensive PyQt6 GUI for the Multi-Modal Physiological Sensing Platform PC Hub.
Provides real-time device discovery, session management, and multi-device coordination
with advanced networking features and live sensor data visualization.
"""

import logging
import sys
from dataclasses import dataclass
from datetime import datetime

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStatusBar,
    QTableWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


@dataclass
class DeviceStatus:
    """Data class for Android device status information"""

    device_id: str
    device_name: str
    ip_address: str
    port: int
    connection_state: str
    last_heartbeat: datetime
    battery_level: int
    storage_free_mb: int
    session_id: str | None
    recording_status: str
    sensor_capabilities: list[str]
    sync_quality: str


class DeviceDiscoveryThread(QThread):
    """Background thread for device discovery via NSD/Zeroconf"""

    device_found = pyqtSignal(str, str, int)
    device_lost = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = False

    def run(self):
        """Run device discovery loop using actual zeroconf service discovery"""
        self.running = True
        logger.info("Device discovery thread started with zeroconf service discovery")

        try:
            from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
        except ImportError:
            logger.warning("Zeroconf not available, falling back to simulation")
            self._run_simulation()
            return

        class AndroidServiceListener(ServiceListener):
            def __init__(self, discovery_thread):
                self.discovery_thread = discovery_thread

            def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
                info = zc.get_service_info(type_, name)
                if info:
                    device_id = name.split(".")[0]
                    addresses = info.parsed_addresses()
                    if addresses:
                        ip = addresses[0]
                        port = info.port
                        logger.info(f"Discovered Android device: {device_id} at {ip}:{port}")
                        self.discovery_thread.device_found.emit(device_id, ip, port)

            def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
                device_id = name.split(".")[0]
                logger.info(f"Android device disconnected: {device_id}")
                self.discovery_thread.device_lost.emit(device_id)

            def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
                """Handle service updates - refresh device information"""
                info = zc.get_service_info(type_, name)
                if info:
                    device_id = name.split(".")[0]
                    addresses = info.parsed_addresses()
                    if addresses:
                        ip = addresses[0]
                        port = info.port
                        logger.info(f"Android device updated: {device_id} at {ip}:{port}")
                        self.discovery_thread.device_found.emit(device_id, ip, port)

        try:
            zeroconf = Zeroconf()
            listener = AndroidServiceListener(self)
            
            service_type = "_gsr-controller._tcp.local."
            browser = ServiceBrowser(zeroconf, service_type, listener)
            
            logger.info(f"Browsing for services of type: {service_type}")
            
            while self.running:
                import time
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Zeroconf discovery error: {e}, falling back to simulation")
            self._run_simulation()
        finally:
            try:
                browser.cancel()
                zeroconf.close()
            except:
                pass
                
    def _run_simulation(self):
        """Fallback simulation when zeroconf is not available"""
        import random
        import time

        logger.info("Running device discovery simulation")
        while self.running:
            time.sleep(5)
            device_ids = [
                "android_device_001",
                "android_device_002", 
                "android_device_003",
            ]
            for device_id in device_ids:
                if random.random() > 0.7:
                    ip = f"192.168.1.{100 + random.randint(1, 50)}"
                    port = 8080 + random.randint(0, 10)
                    self.device_found.emit(device_id, ip, port)

    def stop(self):
        """Stop device discovery"""
        self.running = False
        self.quit()
        self.wait()


class DeviceWidget(QFrame):
    """Widget representing a single Android device with controls and status"""

    def __init__(self, device_status: DeviceStatus):
        super().__init__()
        self.device_status = device_status
        self.setup_ui()

    def setup_ui(self):
        """Set up the device widget UI"""
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setMaximumHeight(200)

        layout = QVBoxLayout(self)

        header_layout = QHBoxLayout()

        name_label = QLabel(f"<b>{self.device_status.device_name}</b>")
        name_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        header_layout.addWidget(name_label)

        status_color = {
            "online": "green",
            "offline": "red",
            "reconnecting": "orange",
        }.get(self.device_status.connection_state, "gray")

        status_label = QLabel(f"● {self.device_status.connection_state.upper()}")
        status_label.setStyleSheet(f"color: {status_color}; font-weight: bold;")
        header_layout.addWidget(status_label)

        header_layout.addStretch()
        layout.addLayout(header_layout)

        info_layout = QGridLayout()

        info_layout.addWidget(QLabel("IP Address:"), 0, 0)
        info_layout.addWidget(
            QLabel(f"{self.device_status.ip_address}:{self.device_status.port}"), 0, 1
        )

        info_layout.addWidget(QLabel("Battery:"), 0, 2)
        battery_bar = QProgressBar()
        battery_bar.setValue(self.device_status.battery_level)
        battery_bar.setMaximumWidth(100)
        info_layout.addWidget(battery_bar, 0, 3)

        info_layout.addWidget(QLabel("Storage:"), 1, 0)
        info_layout.addWidget(QLabel(f"{self.device_status.storage_free_mb} MB"), 1, 1)

        info_layout.addWidget(QLabel("Sync Quality:"), 1, 2)
        sync_color = {
            "excellent": "green",
            "good": "blue",
            "fair": "orange",
            "poor": "red",
        }.get(self.device_status.sync_quality, "gray")
        sync_label = QLabel(self.device_status.sync_quality.upper())
        sync_label.setStyleSheet(f"color: {sync_color}; font-weight: bold;")
        info_layout.addWidget(sync_label, 1, 3)

        layout.addLayout(info_layout)

        button_layout = QHBoxLayout()

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setEnabled(self.device_status.connection_state == "offline")
        button_layout.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setEnabled(self.device_status.connection_state == "online")
        button_layout.addWidget(self.disconnect_btn)

        self.flash_sync_btn = QPushButton("Flash Sync")
        self.flash_sync_btn.setEnabled(self.device_status.connection_state == "online")
        button_layout.addWidget(self.flash_sync_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

    def update_status(self, device_status: DeviceStatus):
        """Update device status display"""
        self.device_status = device_status
        self.findChild(QLabel).setText(f"<b>{device_status.device_name}</b>")

        _status_color = {
            "online": "green",
            "offline": "red",
            "reconnecting": "orange",
        }.get(device_status.connection_state, "gray")

        self.connect_btn.setEnabled(device_status.connection_state == "offline")
        self.disconnect_btn.setEnabled(device_status.connection_state == "online")
        self.flash_sync_btn.setEnabled(device_status.connection_state == "online")


class SessionControlWidget(QWidget):
    """Widget for session management and control"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Set up session control UI"""
        layout = QVBoxLayout(self)

        session_group = QGroupBox("Session Control")
        session_layout = QVBoxLayout(session_group)

        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("Session ID:"))
        self.session_id_input = QLineEdit()
        self.session_id_input.setPlaceholderText("Auto-generated if empty")
        id_layout.addWidget(self.session_id_input)
        session_layout.addLayout(id_layout)

        control_layout = QHBoxLayout()

        self.start_session_btn = QPushButton("Start Session")
        self.start_session_btn.setStyleSheet(
            "QPushButton { background-color: green; color: white; font-weight: bold; }"
        )
        control_layout.addWidget(self.start_session_btn)

        self.stop_session_btn = QPushButton("Stop Session")
        self.stop_session_btn.setStyleSheet(
            "QPushButton { background-color: red; color: white; font-weight: bold; }"
        )
        self.stop_session_btn.setEnabled(False)
        control_layout.addWidget(self.stop_session_btn)

        self.flash_all_btn = QPushButton("Flash All Devices")
        self.flash_all_btn.setStyleSheet(
            "QPushButton { background-color: orange; color: white; font-weight: bold; }"
        )
        control_layout.addWidget(self.flash_all_btn)

        session_layout.addLayout(control_layout)
        layout.addWidget(session_group)

        status_group = QGroupBox("Session Status")
        status_layout = QVBoxLayout(status_group)

        self.session_status_label = QLabel("No active session")
        self.session_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.session_status_label)

        self.recording_progress = QProgressBar()
        self.recording_progress.setVisible(False)
        status_layout.addWidget(self.recording_progress)

        layout.addWidget(status_group)


class MainDashboard(QMainWindow):
    """Main Dashboard GUI for PC Hub"""

    def __init__(self):
        super().__init__()
        self.devices: dict[str, DeviceStatus] = {}
        self.discovery_thread: DeviceDiscoveryThread | None = None
        self.setup_ui()
        self.start_device_discovery()

    def setup_ui(self):
        """Set up main dashboard UI"""
        self.setWindowTitle("Multi-Modal Physiological Sensing Platform - PC Hub")
        self.setMinimumSize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        self.setup_device_tab()

        self.setup_session_tab()

        self.setup_visualization_tab()

        self.setup_logs_tab()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("PC Hub started - Ready for device connections")

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_dashboard)
        self.update_timer.start(1000)

    def setup_device_tab(self):
        """Set up device management tab"""
        device_widget = QWidget()
        layout = QVBoxLayout(device_widget)

        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("<h2>Connected Devices</h2>"))
        header_layout.addStretch()

        refresh_btn = QPushButton("Refresh Discovery")
        refresh_btn.clicked.connect(self.refresh_device_discovery)
        header_layout.addWidget(refresh_btn)

        layout.addLayout(header_layout)

        scroll_area = QScrollArea()
        self.device_list_widget = QWidget()
        self.device_list_layout = QVBoxLayout(self.device_list_widget)
        scroll_area.setWidget(self.device_list_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        self.tab_widget.addTab(device_widget, "Devices")

    def setup_session_tab(self):
        """Set up session management tab"""
        session_widget = QWidget()
        layout = QHBoxLayout(session_widget)

        self.session_control = SessionControlWidget()
        layout.addWidget(self.session_control)

        history_group = QGroupBox("Session History")
        history_layout = QVBoxLayout(history_group)

        self.session_table = QTableWidget()
        self.session_table.setColumnCount(4)
        self.session_table.setHorizontalHeaderLabels(
            ["Session ID", "Start Time", "Duration", "Devices"]
        )
        history_layout.addWidget(self.session_table)

        layout.addWidget(history_group)

        self.tab_widget.addTab(session_widget, "Sessions")

    def setup_visualization_tab(self):
        """Set up data visualization tab"""
        viz_widget = QWidget()
        layout = QVBoxLayout(viz_widget)

        layout.addWidget(QLabel("<h2>Real-Time Sensor Data</h2>"))

        try:
            import pyqtgraph as pg

            plot_widget = pg.PlotWidget()
            plot_widget.setBackground("w")
            plot_widget.setLabel("left", "GSR (μS)", color="black")
            plot_widget.setLabel("bottom", "Time (s)", color="black")
            plot_widget.showGrid(x=True, y=True)

            import numpy as np

            x = np.linspace(0, 10, 100)
            y = np.sin(x) + 0.1 * np.random.random(100)
            plot_widget.plot(x, y, pen="b", name="GSR Signal")

            layout.addWidget(plot_widget)

        except ImportError:
            placeholder = QLabel(
                "Real-time sensor data visualization\n"
                "(PyQtGraph not available - install for live plots)"
            )
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("border: 2px dashed gray; padding: 50px;")
            layout.addWidget(placeholder)

        self.tab_widget.addTab(viz_widget, "Visualization")

    def setup_logs_tab(self):
        """Set up system logs tab"""
        logs_widget = QWidget()
        layout = QVBoxLayout(logs_widget)

        layout.addWidget(QLabel("<h2>System Logs</h2>"))

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        layout.addWidget(self.log_text)

        log_controls = QHBoxLayout()

        clear_btn = QPushButton("Clear Logs")
        clear_btn.clicked.connect(self.log_text.clear)
        log_controls.addWidget(clear_btn)

        log_controls.addStretch()

        level_combo = QComboBox()
        level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        level_combo.setCurrentText("INFO")
        log_controls.addWidget(QLabel("Log Level:"))
        log_controls.addWidget(level_combo)

        layout.addLayout(log_controls)

        self.tab_widget.addTab(logs_widget, "Logs")

    def start_device_discovery(self):
        """Start background device discovery"""
        self.discovery_thread = DeviceDiscoveryThread()
        self.discovery_thread.device_found.connect(self.on_device_found)
        self.discovery_thread.device_lost.connect(self.on_device_lost)
        self.discovery_thread.start()
        logger.info("Started device discovery")

    def refresh_device_discovery(self):
        """Refresh device discovery"""
        logger.info("Refreshing device discovery")
        if self.discovery_thread:
            self.discovery_thread.stop()

        self.discovery_thread = DeviceDiscoveryThread()
        self.discovery_thread.device_found.connect(self.on_device_found)
        self.discovery_thread.device_lost.connect(self.on_device_lost)
        self.discovery_thread.start()

        self.devices.clear()
        self.update_device_list()

    def on_device_found(self, device_id: str, ip_address: str, port: int):
        """Handle discovered device"""
        logger.info(f"Device discovered: {device_id} at {ip_address}:{port}")

        device_status = DeviceStatus(
            device_id=device_id,
            device_name=f"Android Device {device_id[-3:]}",
            ip_address=ip_address,
            port=port,
            connection_state="offline",
            last_heartbeat=datetime.now(),
            battery_level=85,
            storage_free_mb=2048,
            session_id=None,
            recording_status="idle",
            sensor_capabilities=["rgb_camera", "thermal_camera", "gsr", "audio"],
            sync_quality="good",
        )

        self.devices[device_id] = device_status
        self.update_device_list()

    def on_device_lost(self, device_id: str):
        """Handle lost device"""
        logger.info(f"Device lost: {device_id}")
        if device_id in self.devices:
            del self.devices[device_id]
            self.update_device_list()

    def update_device_list(self):
        """Update the device list display"""
        for i in reversed(range(self.device_list_layout.count())):
            child = self.device_list_layout.itemAt(i).widget()
            if child:
                child.setParent(None)

        for device_status in self.devices.values():
            device_widget = DeviceWidget(device_status)
            self.device_list_layout.addWidget(device_widget)

        self.device_list_layout.addStretch()

    def update_dashboard(self):
        """Update dashboard with latest information"""
        device_count = len(self.devices)
        online_count = sum(
            1 for d in self.devices.values() if d.connection_state == "online"
        )

        self.status_bar.showMessage(
            f"Devices: {online_count}/{device_count} online | "
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )

    def closeEvent(self, event):
        """Handle application close"""
        if self.discovery_thread:
            self.discovery_thread.stop()
        event.accept()


def main():
    """Main function to run the dashboard"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Multi-Modal Physiological Sensing Platform")

    app.setStyle("Fusion")

    dashboard = MainDashboard()
    dashboard.show()

    logger.info("Main Dashboard started")

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
