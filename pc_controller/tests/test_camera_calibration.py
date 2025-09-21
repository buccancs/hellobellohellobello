from pathlib import Path

import numpy as np
import pytest

from pc_controller.src.tools.camera_calibration import (
    CalibrationResult,
    find_checkerboard_corners,
    load_calibration,
    save_calibration,
)


def test_save_load_roundtrip(tmp_path: Path):
    camera_matrix = np.array(
        [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    dist_coeffs = np.array([0.1, -0.05, 0.001, 0.002, 0.0], dtype=np.float64)
    res = CalibrationResult(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms_error=0.42,
        image_size=(1280, 720),
        board_size=(9, 6),
        square_size=0.025,
    )

    out_file = tmp_path / "calibration.json"
    save_calibration(out_file, res)

    loaded = load_calibration(out_file)
    assert loaded.image_size == (1280, 720)
    assert loaded.board_size == (9, 6)
    assert loaded.square_size == pytest.approx(0.025)
    assert loaded.rms_error == pytest.approx(0.42)
    assert np.allclose(loaded.camera_matrix, camera_matrix)
    assert np.allclose(loaded.dist_coeffs, dist_coeffs)


def test_find_checkerboard_corners_invalid_inputs():
    with pytest.raises(ValueError):
        find_checkerboard_corners([], (9, 6), 0.025)
    with pytest.raises(ValueError):
        find_checkerboard_corners(["a.jpg"], (1, 6), 0.025)
    with pytest.raises(ValueError):
        find_checkerboard_corners(["a.jpg"], (9, 6), 0.0)


def test_find_checkerboard_corners_unreadable_images(tmp_path: Path):
    img1 = tmp_path / "does_not_exist1.jpg"
    img2 = tmp_path / "does_not_exist2.jpg"
    with pytest.raises(ValueError, match="No readable images"):
        find_checkerboard_corners([img1, img2], (9, 6), 0.025)
