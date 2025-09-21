"""Tests for user experience enhancements."""


import pytest
from core.user_experience import ErrorMessageTranslator, StatusIndicator


class TestErrorMessageTranslator:
    """Test error message translation functionality."""

    def test_common_error_translations(self):
        """Test translation of common technical errors."""
        conn_error = ConnectionRefusedError("Connection refused")
        user_msg = ErrorMessageTranslator.translate_error(conn_error)
        assert "unable to connect" in user_msg.lower()
        assert "same WiFi network" in user_msg

        file_error = FileNotFoundError("File not found")
        user_msg = ErrorMessageTranslator.translate_error(file_error)
        assert "could not be found" in user_msg
        assert "correct location" in user_msg

    def test_context_specific_advice(self):
        """Test that context-specific advice is added."""
        error = ConnectionRefusedError("Connection refused")
        user_msg = ErrorMessageTranslator.translate_error(error, "network")
        assert "Network Troubleshooting:" in user_msg
        assert "same WiFi network" in user_msg

    def test_unknown_error_handling(self):
        """Test handling of unknown error types."""
        class CustomError(Exception):
            pass

        error = CustomError("Custom error message")
        user_msg = ErrorMessageTranslator.translate_error(error)
        assert "unexpected error occurred" in user_msg.lower()
        assert "Custom error message" in user_msg

    def test_prevention_advice(self):
        """Test prevention advice retrieval."""
        advice = ErrorMessageTranslator.get_prevention_advice("network")
        assert "dedicated research WiFi" in advice
        assert "prevent network issues" in advice


class TestStatusIndicator:
    """Test status indicator functionality."""

    def test_file_location_formatting(self):
        """Test file location formatting for UI display."""
        test_path = "/test/path/to/files"
        formatted = StatusIndicator.format_file_location(test_path, "Test files")
        assert "Test files location:" in formatted
        assert test_path in formatted

    def test_export_status_formatting(self):
        """Test export status message formatting."""
        output_dir = "/export/directory"
        file_count = 5
        formats = ["HDF5", "CSV"]

        status = StatusIndicator.format_export_status(output_dir, file_count, formats)
        assert "5 files exported" in status
        assert "HDF5, CSV" in status
        assert output_dir in status
        assert "Export completed successfully" in status

    def test_device_status_formatting(self):
        """Test device status formatting."""
        status = StatusIndicator.format_device_status(
            "Device1", "connected", {"battery": 80, "signal_strength": 95}
        )
        assert "🟢" in status
        assert "Device1: Connected" in status
        assert "Battery: 80%" in status
        assert "Signal: 95%" in status

        # Test low battery warning
        status = StatusIndicator.format_device_status(
            "Device2", "connected", {"battery": 15}
        )
        assert "Battery: 15% ⚠️" in status

        status = StatusIndicator.format_device_status(
            "Device3", "disconnected", {}
        )
        assert "🔴" in status
        assert "Device3: Disconnected" in status


class TestUserExperienceIntegration:
    """Test integration of user experience features."""

    def test_error_logging_integration(self):
        """Test error logging with user-friendly messages."""
        import logging

        logger = logging.getLogger("test_ux")

        error = FileNotFoundError("test.txt not found")
        user_msg = ErrorMessageTranslator.log_user_friendly_error(error, "recording", logger)

        assert "could not be found" in user_msg
        assert isinstance(user_msg, str)
        assert len(user_msg) > 20

    def test_quick_access_functions(self):
        """Test quick access convenience functions."""
        from core.user_experience import (
            show_export_status,
            show_file_location,
            show_user_friendly_error,
        )

        error = ConnectionRefusedError("test error")
        msg = show_user_friendly_error(error)
        assert "unable to connect" in msg.lower()

        location = show_file_location("/test/path", "Test data")
        assert "Test data location:" in location

        status = show_export_status("/output", 3, ["HDF5"])
        assert "3 files exported" in status
        assert "HDF5" in status


if __name__ == "__main__":
    pytest.main([__file__])
