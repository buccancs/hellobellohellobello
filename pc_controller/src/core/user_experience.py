"""User-friendly error message utility for improved UX."""

import logging
from typing import ClassVar


class ErrorMessageTranslator:
    """Translates technical errors into user-friendly messages with actionable advice."""

    ERROR_TRANSLATIONS: ClassVar[dict[str, str]] = {
        "ConnectionRefusedError": "Unable to connect to device. Please check that the device is "
        "on the same WiFi network and try again.",
        "TimeoutError": "Connection timed out. Please check your network connection and ensure "
        "the device is responding.",
        "ConnectionResetError": "Connection was lost unexpectedly. The device may have restarted "
        "or lost network connectivity.",
        "NetworkUnreachableError": "Network is unreachable. Please check your WiFi connection "
        "and try again.",
        "FileNotFoundError": "Required file could not be found. Please check that all files "
        "are in the correct location.",
        "PermissionError": "Permission denied. Please check file permissions or run as "
        "administrator if needed.",
        "DiskSpaceError": "Insufficient disk space. Please free up storage space and try again.",
        "FileExistsError": "A file with this name already exists. Please choose a different "
        "name or location.",
        "DeviceNotFoundError": "Device not detected. Please ensure the device is connected "
        "and powered on.",
        "DeviceBusyError": "Device is currently busy or being used by another application. "
        "Please close other applications and try again.",
        "DeviceDisconnectedError": "Device was disconnected during operation. Please reconnect "
        "the device and try again.",
        "CalibrationError": (
            "Camera calibration failed. Please ensure:\n"
            "• Checkerboard pattern is clearly visible\n"
            "• Good lighting conditions\n"
            "• Multiple angles captured\n"
            "• Pattern is flat and undamaged"
        ),
        "CalibrationPatternNotFoundError": (
            "Checkerboard pattern not detected in images. "
            "Please ensure the pattern is clearly visible and well-lit."
        ),
        "CalibrationInsufficientDataError": (
            "Not enough calibration images. "
            "Please capture at least 10 clear images from different angles."
        ),
        "RecordingError": (
            "Recording failed to start. "
            "Please check device connections and available storage space."
        ),
        "RecordingInterruptedError": (
            "Recording was interrupted unexpectedly. Data may be incomplete. "
            "Please check device connections."
        ),
        "SynchronizationError": (
            "Device synchronization failed. "
            "Please ensure all devices are connected to the same network."
        ),
        "ExportError": (
            "Data export failed. Please check that the destination folder "
            "has write permissions and sufficient space."
        ),
        "ImportError": (
            "Data import failed. Please verify the file is not corrupted "
            "and is in the correct format."
        ),
        "FormatError": (
            "File format not supported. Please ensure the file is in one of the "
            "supported formats (CSV, HDF5, MP4)."
        ),
        # Configuration errors
        "ConfigurationError": "Configuration is invalid. Please check all settings and try again.",
        "SettingsError": "Unable to save settings. Please check file permissions and try again.",
    }

    CONTEXT_ADVICE: ClassVar[dict[str, dict[str, str]]] = {
        "network": {
            "troubleshooting": (
                "Network Troubleshooting:\n"
                "• Ensure all devices are on the same WiFi network\n"
                "• Check that no firewall is blocking connections\n"
                "• Try restarting your router if problems persist"
            ),
            "prevention": (
                "To prevent network issues:\n"
                "• Use a dedicated research WiFi network\n"
                "• Avoid networks with many connected devices\n"
                "• Keep devices close to the WiFi router"
            ),
        },
        "calibration": {
            "troubleshooting": (
                "Calibration Troubleshooting:\n"
                "• Use a high-quality printed checkerboard pattern\n"
                "• Ensure even lighting without shadows or reflections\n"
                "• Capture images from various angles and distances\n"
                "• Keep the pattern flat against a rigid surface"
            ),
            "prevention": (
                "For best calibration results:\n"
                "• Print the pattern on rigid paper or mount on cardboard\n"
                "• Use a tripod for steady image capture\n"
                "• Take your time to ensure each image is clear"
            ),
        },
        "recording": {
            "troubleshooting": (
                "Recording Troubleshooting:\n"
                "• Check all device connections are secure\n"
                "• Verify sufficient storage space on all devices\n"
                "• Ensure devices are fully charged or plugged in\n"
                "• Close other applications that might use cameras or sensors"
            ),
            "prevention": (
                "For reliable recording:\n"
                "• Always verify connections before starting\n"
                "• Monitor battery levels during long sessions\n"
                "• Keep backup storage available"
            ),
        },
    }

    @classmethod
    def translate_error(cls, error: Exception, context: str | None = None) -> str:
        """
        Translate a technical error into a user-friendly message.

        Args:
            error: The exception that occurred
            context: Optional context for additional advice (network, calibration, recording)

        Returns:
            User-friendly error message with actionable advice
        """
        error_type = type(error).__name__
        base_message = cls.ERROR_TRANSLATIONS.get(
            error_type, f"An unexpected error occurred: {error!s}"
        )

        if context and context in cls.CONTEXT_ADVICE:
            advice = cls.CONTEXT_ADVICE[context].get("troubleshooting", "")
            if advice:
                base_message += f"\n\n{advice}"

        return base_message

    @classmethod
    def get_prevention_advice(cls, context: str) -> str:
        """
        Get prevention advice for a specific context.

        Args:
            context: The context to get advice for

        Returns:
            Prevention advice string
        """
        return cls.CONTEXT_ADVICE.get(context, {}).get("prevention", "")

    @classmethod
    def log_user_friendly_error(
        cls,
        error: Exception,
        context: str | None = None,
        logger: logging.Logger | None = None,
    ) -> str:
        """
        Log an error with both technical details and user-friendly message.

        Args:
            error: The exception that occurred
            context: Optional context for additional advice
            logger: Logger to use (creates default if None)

        Returns:
            User-friendly error message
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        logger.error(
            f"Technical error: {type(error).__name__}: {error!s}", exc_info=True
        )

        user_message = cls.translate_error(error, context)

        logger.info(f"User message: {user_message}")

        return user_message


class StatusIndicator:
    """Provides clear status indicators for file locations and system state."""

    @staticmethod
    def format_file_location(path: str, description: str = "Files") -> str:
        """
        Format file location for clear display in UI.

        Args:
            path: The file or directory path
            description: Description of what files are located there

        Returns:
            Formatted location string
        """
        import os

        abs_path = os.path.abspath(path)

        return f"{description} location: {abs_path}"

    @staticmethod
    def format_export_status(output_dir: str, file_count: int, formats: list) -> str:
        """
        Format export status message.

        Args:
            output_dir: Directory where files were exported
            file_count: Number of files exported
            formats: List of formats exported

        Returns:
            Formatted status message
        """
        format_list = ", ".join(formats)
        return (
            f"Export completed successfully!\n"
            f"• {file_count} files exported\n"
            f"• Formats: {format_list}\n"
            f"• Location: {output_dir}"
        )

    @staticmethod
    def format_device_status(device_name: str, status: str, details: dict) -> str:
        """
        Format device status for clear display.

        Args:
            device_name: Name of the device
            status: Current status (connected, disconnected, error, etc.)
            details: Additional status details

        Returns:
            Formatted status string
        """
        status_symbols = {
            "connected": "🟢",
            "disconnected": "🔴",
            "connecting": "🟡",
            "error": "❌",
            "unknown": "⚪",
        }

        symbol = status_symbols.get(status.lower(), "⚪")
        base_status = f"{symbol} {device_name}: {status.title()}"

        detail_parts = []
        if "battery" in details:
            battery = details["battery"]
            if battery < 20:
                detail_parts.append(f"Battery: {battery}% ⚠️")
            else:
                detail_parts.append(f"Battery: {battery}%")

        if "signal_strength" in details:
            signal = details["signal_strength"]
            if signal < 50:
                detail_parts.append(f"Signal: {signal}% ⚠️")
            else:
                detail_parts.append(f"Signal: {signal}%")

        if detail_parts:
            base_status += f" ({', '.join(detail_parts)})"

        return base_status


def show_user_friendly_error(error: Exception, context: str | None = None) -> str:
    """Quick function to get user-friendly error message."""
    return ErrorMessageTranslator.translate_error(error, context)


def show_file_location(path: str, description: str = "Data files") -> str:
    """Quick function to show file location."""
    return StatusIndicator.format_file_location(path, description)


def show_export_status(output_dir: str, file_count: int, formats: list) -> str:
    """Quick function to show export status."""
    return StatusIndicator.format_export_status(output_dir, file_count, formats)
