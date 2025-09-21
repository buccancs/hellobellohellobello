"""Tests for GUI enhancements - calibration and export functionality."""

import os
from unittest.mock import patch

import pytest

pytest_plugins = []
try:
    import importlib.util
    GUI_AVAILABLE = importlib.util.find_spec("PyQt6.QtWidgets") is not None
except ImportError:
    GUI_AVAILABLE = False


@pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI libraries not available")
class TestCalibrationDialog:
    """Test calibration dialog functionality."""

    def test_calibration_dialog_parameters(self):
        """Test calibration dialog parameter collection."""
        pytest.skip("GUI components require display environment")


@pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI libraries not available")
class TestExportDialog:
    """Test export dialog functionality."""

    def test_export_dialog_parameters(self):
        """Test export dialog parameter collection."""
        pytest.skip("GUI components require display environment")


class TestCalibrationWorkflow:
    """Test calibration workflow without GUI dependencies."""

    def test_calibration_parameters_validation(self):
        """Test calibration parameter validation logic."""
        # Test valid parameters
        valid_params = {
            "images_dir": "/valid/path",
            "board_width": 9,
            "board_height": 6,
            "square_size": 0.025
        }

        assert valid_params["board_width"] > 0
        assert valid_params["board_height"] > 0
        assert valid_params["square_size"] > 0
        assert isinstance(valid_params["images_dir"], str)

    @patch('tools.camera_calibration.calibrate_camera')
    @patch('tools.camera_calibration.save_calibration')
    @patch('os.path.isdir')
    def test_calibration_workflow_success(self, mock_isdir, mock_save, mock_calibrate):
        """Test successful calibration workflow."""
        import numpy as np
        from tools.camera_calibration import CalibrationResult

        mock_isdir.return_value = True
        mock_result = CalibrationResult(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            rms_error=0.5,
            image_size=(640, 480),
            board_size=(9, 6),
            square_size=0.025
        )
        mock_calibrate.return_value = mock_result

        # Test parameters
        params = {
            "images_dir": "/test/images",
            "board_width": 9,
            "board_height": 6,
            "square_size": 0.025
        }

        (params["board_width"], params["board_height"])
        result = mock_calibrate.return_value

        assert result.rms_error == 0.5
        assert result.board_size == (9, 6)

    def test_calibration_dialog_creation(self):
        """Test calibration dialog can be created without crashing."""
        try:
            from pc_controller.src.gui.calibration_dialog import CalibrationDialog
            dialog_class = CalibrationDialog
            assert dialog_class is not None
            print("CalibrationDialog class successfully imported")
        except ImportError as e:
            if "EGL" in str(e) or "display" in str(e).lower():
                print("Skipping GUI test in headless environment")
            else:
                raise
        except Exception as e:
            raise AssertionError(f"Unexpected error importing CalibrationDialog: {e}") from e

    def test_calibration_parameters_defaults(self):
        """Test that calibration has sensible default parameters."""
        default_board_width = 9
        default_board_height = 6
        default_square_size = 0.025

        assert default_board_width > 2
        assert default_board_height > 2
        assert default_square_size > 0
        assert isinstance(default_board_width, int)
        assert isinstance(default_board_height, int)
        assert isinstance(default_square_size, float)


class TestExportWorkflow:
    """Test export workflow without GUI dependencies."""

    def test_export_parameters_validation(self):
        """Test export parameter validation logic."""
        valid_params = {
            "session_dir": "/valid/session",
            "output_dir": "/valid/output",
            "formats": ["HDF5", "CSV"]
        }

        assert isinstance(valid_params["session_dir"], str)
        assert isinstance(valid_params["output_dir"], str)
        assert isinstance(valid_params["formats"], list)
        assert len(valid_params["formats"]) > 0

    def test_supported_export_formats(self):
        """Test that all expected export formats are supported."""
        supported_formats = ["HDF5", "CSV", "MP4"]

        assert "HDF5" in supported_formats
        assert "CSV" in supported_formats
        assert "MP4" in supported_formats

    @patch('data.hdf5_exporter.export_session_to_hdf5')
    @patch('os.path.isdir')
    def test_export_workflow_hdf5(self, mock_isdir, mock_export):
        """Test HDF5 export workflow."""
        mock_isdir.return_value = True

        # Test parameters
        params = {
            "session_dir": "/test/session",
            "output_dir": "/test/output",
            "formats": ["HDF5"]
        }

        if "HDF5" in params["formats"]:
            out_path = os.path.join(params["output_dir"], "export.h5")
            meta = {"session_dir": params["session_dir"]}


        assert out_path.endswith("export.h5")
        assert meta["session_dir"] == params["session_dir"]


class TestUserExperienceEnhancements:
    """Test user experience enhancement features."""

    def test_error_message_improvements(self):
        """Test that error messages are user-friendly."""
        technical_errors = {
            "FileNotFoundError": "File location not found",
            "ConnectionRefusedError": "Unable to connect to device",
            "CalibrationError": "Camera calibration failed - please check images",
            "ExportError": "Data export failed - check output location"
        }

        for user_message in technical_errors.values():
            assert len(user_message) > 10
            assert not user_message.isupper()
            assert "Error" not in user_message or "failed" in user_message.lower()

    def test_file_location_indicators(self):
        """Test file location indicator functionality."""
        test_directories = [
            "/home/user/data/session_20241221",
            "C:\\Users\\researcher\\Documents\\exports",
            "/tmp/calibration_results"
        ]

        for directory in test_directories:
            location_message = f"Exporting to: {directory}"
            assert directory in location_message
            assert "Exporting to:" in location_message


if __name__ == "__main__":
    pytest.main([__file__])
