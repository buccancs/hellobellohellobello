"""Quick Start Guide and Tutorial System for improved user onboarding."""

from collections.abc import Callable
from dataclasses import dataclass

try:
    from PyQt6.QtCore import QDateTime, Qt, pyqtSignal
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QDialog,
        QFrame,
        QHBoxLayout,
        QLabel,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False

    class QDialog:  # type: ignore[no-redef]
        class DialogCode:
            Accepted = 1
            Rejected = 0

    class pyqtSignal:  # type: ignore[no-redef]
        def __call__(self, *args):
            return self

        def emit(self, *args):
            pass

        def connect(self, slot):
            pass


@dataclass
class TutorialStep:
    """Represents a single step in the tutorial."""

    title: str
    content: str
    image_path: str | None = None
    action_text: str | None = None
    action_callback: Callable | None = None
    skip_allowed: bool = True


class QuickStartGuide(QDialog):
    """Interactive quick start guide dialog for first-time users."""

    tutorial_completed = pyqtSignal()
    tutorial_skipped = pyqtSignal()

    def __init__(self, parent=None):
        if not _QT_AVAILABLE:
            raise ImportError("Qt libraries not available - GUI mode disabled")
        super().__init__(parent)
        self.setWindowTitle(
            "Quick Start Guide - Multi-Modal Physiological Sensing Platform"
        )
        self.setFixedSize(800, 600)
        self.setModal(True)

        self.current_step = 0
        self.steps = self._create_tutorial_steps()

        self._setup_ui()
        self._show_step(0)

    def _create_tutorial_steps(self) -> list[TutorialStep]:
        """Create the tutorial steps."""
        return [
            TutorialStep(
                title="Welcome to the Multi-Modal Physiological Sensing Platform",
                content="""
                <h3>Get started in 5 minutes!</h3>
                <p>This guide will help you:</p>
                <ul>
                <li>Set up your first recording session</li>
                <li>Connect Android devices</li>
                <li>Configure sensors and cameras</li>
                <li>Export your data for analysis</li>
                </ul>
                <p><b>Tip:</b> You can access this guide anytime from the Help menu.</p>
                """,
                skip_allowed=True,
            ),
            TutorialStep(
                title="Step 1: Network Setup",
                content="""
                <h3>Connect Your Devices</h3>
                <p>To start recording, you need to connect your Android devices to the PC:</p>
                <ol>
                <li><b>WiFi Network:</b> Ensure your PC and Android devices are on the same WiFi
                network</li>
                <li><b>Automatic Discovery:</b> Android devices will automatically discover this
                PC Hub</li>
                <li><b>Manual Connection:</b> If automatic discovery fails, use the "Connect Device"
                button</li>
                </ol>

                <div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px;
                margin-top: 10px;'>
                <b>Troubleshooting:</b> If devices don't appear, check your firewall settings and
                ensure both devices have internet access.
                </div>
                """,
                action_text="Test Network Discovery",
                action_callback=self._test_network_discovery,
            ),
            TutorialStep(
                title="Step 2: Device Configuration",
                content="""
                <h3>Configure Your Sensors</h3>
                <p>The system supports multiple sensor types:</p>
                <ul>
                <li><b>RGB Camera:</b> Records high-quality video with timestamps</li>
                <li><b>Thermal Camera:</b> Captures thermal imaging data (if available)</li>
                <li><b>GSR Sensor:</b> Measures galvanic skin response via Shimmer devices</li>
                </ul>

                <p><b>Camera Calibration:</b> For best results, calibrate your cameras before
                recording:</p>
                <ol>
                <li>Click "Calibrate Cameras" in the toolbar</li>
                <li>Use a checkerboard pattern (print from our templates)</li>
                <li>Capture 10+ images from different angles</li>
                </ol>
                """,
                action_text="Open Calibration Dialog",
                action_callback=self._demo_calibration,
            ),
            TutorialStep(
                title="Step 3: Recording Session",
                content="""
                <h3>Start Your First Recording</h3>
                <p>Follow these steps to record data:</p>
                <ol>
                <li><b>Start Session:</b> Click "Start Session" to begin recording</li>
                <li><b>Monitor Devices:</b> Watch the device status indicators for connection
                health</li>
                <li><b>Flash Sync:</b> Use "Flash Sync" to synchronize timestamps across
                devices</li>
                <li><b>Stop Session:</b> Click "Stop Session" when finished</li>
                </ol>

                <div style='background-color: #f0f8e8; padding: 10px; border-radius: 5px;
                margin-top: 10px;'>
                <b>Best Practice:</b> Start with short test sessions (1-2 minutes) to verify
                everything works before longer recordings.
                </div>
                """,
                action_text="Show Session Controls",
                action_callback=self._highlight_session_controls,
            ),
            TutorialStep(
                title="Step 4: Data Export and Analysis",
                content="""
                <h3>Export Your Data</h3>
                <p>After recording, export data in multiple formats:</p>
                <ul>
                <li><b>HDF5:</b> Structured format for MATLAB, Python analysis</li>
                <li><b>CSV:</b> Raw sensor data for spreadsheet analysis</li>
                <li><b>MP4:</b> Video files from cameras</li>
                </ul>

                <p><b>Export Process:</b></p>
                <ol>
                <li>Click "Export Data" in the toolbar</li>
                <li>Select your session directory</li>
                <li>Choose export formats and destination</li>
                <li>Click OK to start export</li>
                </ol>

                <div style='background-color: #f8f0e8; padding: 10px; border-radius: 5px;
                margin-top: 10px;'>
                <b>File Locations:</b> Export locations are clearly shown in the log and status
                messages.
                </div>
                """,
                action_text="Demo Export Dialog",
                action_callback=self._demo_export,
            ),
            TutorialStep(
                title="Quick Reference",
                content="""
                <h3>Quick Reference Card</h3>
                <p>Keep these shortcuts handy:</p>

                <table style='width: 100%; border-collapse: collapse;'>
                <tr style='background-color: #f5f5f5;'>
                    <td style='padding: 8px; border: 1px solid #ddd;'><b>Action</b></td>
                    <td style='padding: 8px; border: 1px solid #ddd;'><b>Button/Location</b></td>
                </tr>
                <tr>
                    <td style='padding: 8px; border: 1px solid #ddd;'>Start Recording</td>
                    <td style='padding: 8px; border: 1px solid #ddd;'>Toolbar → "Start Session"</td>
                </tr>
                <tr style='background-color: #f9f9f9;'>
                    <td style='padding: 8px; border: 1px solid #ddd;'>Connect Device</td>
                    <td style='padding: 8px; border: 1px solid #ddd;'>Toolbar →
                    "Connect Device"</td>
                </tr>
                <tr>
                    <td style='padding: 8px; border: 1px solid #ddd;'>Calibrate Cameras</td>
                    <td style='padding: 8px; border: 1px solid #ddd;'>Toolbar →
                    "Calibrate Cameras"</td>
                </tr>
                <tr style='background-color: #f9f9f9;'>
                    <td style='padding: 8px; border: 1px solid #ddd;'>Export Data</td>
                    <td style='padding: 8px; border: 1px solid #ddd;'>Toolbar → "Export Data"</td>
                </tr>
                <tr>
                    <td style='padding: 8px; border: 1px solid #ddd;'>Flash Sync</td>
                    <td style='padding: 8px; border: 1px solid #ddd;'>Toolbar → "Flash Sync"</td>
                </tr>
                </table>

                <p style='margin-top: 15px;'><b>Need Help?</b> Check the logs tab for detailed
                status messages and troubleshooting information.</p>
                """,
                skip_allowed=False,
            ),
        ]

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)

        header_layout = QHBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(len(self.steps))
        self.progress_bar.setValue(0)

        self.step_label = QLabel()
        step_font = QFont()
        step_font.setBold(True)
        self.step_label.setFont(step_font)

        header_layout.addWidget(self.step_label)
        header_layout.addStretch()
        header_layout.addWidget(self.progress_bar)

        layout.addLayout(header_layout)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)

        self.title_label = QLabel()
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        self.title_label.setFont(title_font)

        self.content_text = QTextEdit()
        self.content_text.setReadOnly(True)
        self.content_text.setMinimumHeight(300)

        self.content_layout.addWidget(self.title_label)
        self.content_layout.addWidget(self.content_text)

        # Action button (optional)
        self.action_button = QPushButton()
        self.action_button.hide()
        self.action_button.clicked.connect(self._on_action_clicked)
        self.content_layout.addWidget(self.action_button)

        self.scroll_area.setWidget(self.content_widget)
        layout.addWidget(self.scroll_area)

        nav_layout = QHBoxLayout()

        self.skip_button = QPushButton("Skip Tutorial")
        self.skip_button.clicked.connect(self._on_skip)

        self.prev_button = QPushButton("← Previous")
        self.prev_button.clicked.connect(self._on_previous)
        self.prev_button.setEnabled(False)

        self.next_button = QPushButton("Next →")
        self.next_button.clicked.connect(self._on_next)

        self.finish_button = QPushButton("Finish")
        self.finish_button.clicked.connect(self._on_finish)
        self.finish_button.hide()

        nav_layout.addWidget(self.skip_button)
        nav_layout.addStretch()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        nav_layout.addWidget(self.finish_button)

        layout.addLayout(nav_layout)

    def _show_step(self, step_index: int):
        """Display the specified tutorial step."""
        if step_index < 0 or step_index >= len(self.steps):
            return

        step = self.steps[step_index]
        self.current_step = step_index

        self.progress_bar.setValue(step_index + 1)
        self.step_label.setText(f"Step {step_index + 1} of {len(self.steps)}")

        self.title_label.setText(step.title)
        self.content_text.setHtml(step.content)

        if step.action_text and step.action_callback:
            self.action_button.setText(step.action_text)
            self.action_button.show()
        else:
            self.action_button.hide()

        self.prev_button.setEnabled(step_index > 0)
        self.skip_button.setVisible(step.skip_allowed)

        if step_index == len(self.steps) - 1:
            self.next_button.hide()
            self.finish_button.show()
        else:
            self.next_button.show()
            self.finish_button.hide()

    def _on_next(self):
        """Move to next step."""
        if self.current_step < len(self.steps) - 1:
            self._show_step(self.current_step + 1)

    def _on_previous(self):
        """Move to previous step."""
        if self.current_step > 0:
            self._show_step(self.current_step - 1)

    def _on_skip(self):
        """Skip the tutorial."""
        self.tutorial_skipped.emit()
        self.accept()

    def _on_finish(self):
        """Finish the tutorial."""
        self.tutorial_completed.emit()
        self.accept()

    def _on_action_clicked(self):
        """Execute the action for the current step."""
        step = self.steps[self.current_step]
        if step.action_callback:
            step.action_callback()

    def _test_network_discovery(self):
        """Demo network discovery testing."""
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.information(
            self,
            "Network Discovery Test",
            "This would test network discovery.\nIn the real app, this would:\n"
            "• Scan for available Android devices\n"
            "• Show connection status\n"
            "• Provide troubleshooting tips if no devices found",
        )

    def _demo_calibration(self):
        """Open the camera calibration dialog."""
        try:
            from pc_controller.src.gui.calibration_dialog import CalibrationDialog

            dialog = CalibrationDialog(self)
            dialog.exec()

        except ImportError as e:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self,
                "Feature Unavailable",
                f"Camera calibration feature requires OpenCV:\n{e}\n\n"
                "Please install OpenCV to use this feature:\n"
                "pip install opencv-python",
            )
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.critical(
                self,
                "Calibration Error",
                f"Failed to open calibration dialog:\n{e}",
            )

    def _highlight_session_controls(self):
        """Demo highlighting session controls."""
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.information(
            self,
            "Session Controls",
            "Session controls are located in the main toolbar:\n"
            "• Start Session: Begin recording\n"
            "• Stop Session: End recording\n"
            "• Flash Sync: Synchronize device timestamps\n"
            "• Connect Device: Manual device connection",
        )

    def _demo_export(self):
        """Demo export dialog."""
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.information(
            self,
            "Export Demo",
            "This would open the export dialog.\n"
            "Features include:\n"
            "• Multiple format selection (HDF5, CSV, MP4)\n"
            "• Session and output directory selection\n"
            "• Progress tracking during export\n"
            "• Clear file location indicators",
        )


class FirstTimeSetupWizard:
    """Manages first-time setup and tutorial flow."""

    def __init__(self, settings_manager):
        self.settings = settings_manager
        self._tutorial_shown = False

    def should_show_tutorial(self) -> bool:
        """Check if tutorial should be shown to the user."""
        return not self.settings.get_boolean("tutorial_completed", False)

    def show_tutorial_if_needed(self, parent_widget) -> bool:
        """Show tutorial if it hasn't been completed yet."""
        if not _QT_AVAILABLE:
            return False  # Skip tutorial in headless mode

        if not self.should_show_tutorial() or self._tutorial_shown:
            return False

        self._tutorial_shown = True
        tutorial = QuickStartGuide(parent_widget)

        tutorial.tutorial_completed.connect(self._on_tutorial_completed)
        tutorial.tutorial_skipped.connect(self._on_tutorial_skipped)

        result = tutorial.exec()
        return result == QDialog.DialogCode.Accepted

    def _on_tutorial_completed(self):
        """Handle tutorial completion."""
        self.settings.set_boolean("tutorial_completed", True)
        if _QT_AVAILABLE:
            self.settings.set_string(
                "tutorial_completion_date", str(QDateTime.currentDateTime().toString())
            )
        else:
            import datetime

            self.settings.set_string(
                "tutorial_completion_date", str(datetime.datetime.now())
            )

    def _on_tutorial_skipped(self):
        """Handle tutorial being skipped."""
        self.settings.set_boolean("tutorial_skipped", True)

    def show_tutorial_on_demand(self, parent_widget):
        """Show tutorial when explicitly requested by user."""
        tutorial = QuickStartGuide(parent_widget)
        tutorial.exec()


def integrate_quick_start_guide(gui_manager, settings_manager):
    """Integrate quick start guide into the main GUI manager."""

    setup_wizard = FirstTimeSetupWizard(settings_manager)

    def show_tutorial_if_first_time():
        setup_wizard.show_tutorial_if_needed(gui_manager)

    def show_tutorial_on_demand():
        setup_wizard.show_tutorial_on_demand(gui_manager)

    return {
        "show_tutorial_if_first_time": show_tutorial_if_first_time,
        "show_tutorial_on_demand": show_tutorial_on_demand,
        "setup_wizard": setup_wizard,
    }
