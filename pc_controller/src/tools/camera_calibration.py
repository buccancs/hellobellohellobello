"""Camera calibration utility functions.

This module provides functions to perform checkerboard-based camera calibration
using OpenCV and to persist calibration parameters. It is designed for FR9.

Note: The actual calibration requires a dataset of checkerboard images. The
unit tests cover parameter I/O and validation paths and do not perform a full
calibration.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2

    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False


@dataclass
class CalibrationResult:
    """Holds camera calibration results.

    Attributes:
        camera_matrix: 3x3 intrinsic matrix.
        dist_coeffs: Distortion coefficients (k1, k2, p1, p2, k3[, k4, k5, k6]).
        rms_error: Root-mean-square reprojection error from calibration.
        image_size: (width, height) of the images used for calibration.
        board_size: (cols, rows) inner-corner count of the checkerboard.
        square_size: Size of a checkerboard square in meters (or any unit).
    """

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rms_error: float
    image_size: tuple[int, int]
    board_size: tuple[int, int]
    square_size: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "rms_error": self.rms_error,
            "image_size": list(self.image_size),
            "board_size": list(self.board_size),
            "square_size": self.square_size,
        }

    @staticmethod
    def from_json_dict(data: dict[str, Any]) -> CalibrationResult:
        try:
            return CalibrationResult(
                camera_matrix=np.array(data["camera_matrix"], dtype=np.float64),
                dist_coeffs=np.array(data["dist_coeffs"], dtype=np.float64),
                rms_error=float(data["rms_error"]),
                image_size=(int(data["image_size"][0]), int(data["image_size"][1])),
                board_size=(int(data["board_size"][0]), int(data["board_size"][1])),
                square_size=float(data["square_size"]),
            )
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"Invalid calibration data format: {e}") from e


def _prepare_object_points(
    board_size: tuple[int, int], square_size: float
) -> np.ndarray:
    cols, rows = board_size
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * square_size
    return objp


def find_checkerboard_corners(
    image_paths: Sequence[Path | str],
    board_size: tuple[int, int],
    square_size: float,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    """Finds checkerboard corners in the given images.

    Returns object_points (list per image), image_points (list per image), and image_size (w, h).

    Raises:
        ValueError: If no image paths are provided or board_size/square_size invalid.
    """
    if not image_paths:
        raise ValueError("No image paths provided for calibration.")
    if len(board_size) != 2 or board_size[0] < 2 or board_size[1] < 2:
        raise ValueError("board_size must be a (cols, rows) with both >= 2.")
    if square_size <= 0:
        raise ValueError("square_size must be > 0.")

    if not CV2_AVAILABLE:
        raise ValueError("No readable images provided for calibration.")

    objp_template = _prepare_object_points(board_size, square_size)
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for p in image_paths:
        pp = Path(p)
        if not pp.exists():
            continue
        img = cv2.imread(str(pp), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if image_size is None:
            image_size = (img.shape[1], img.shape[0])
        ret, corners = cv2.findChessboardCorners(img, board_size, None)
        if ret:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            )
            corners2 = cv2.cornerSubPix(
                img,
                corners,
                (11, 11),
                (-1, -1),
                criteria,
            )
            objpoints.append(objp_template.copy())
            imgpoints.append(corners2)

    if image_size is None:
        raise ValueError("No readable images provided for calibration.")

    if len(objpoints) == 0:
        raise ValueError(
            "No checkerboard corners were detected in the provided images."
        )

    return objpoints, imgpoints, image_size


def calibrate_camera(
    image_paths: Sequence[Path | str],
    board_size: tuple[int, int],
    square_size: float,
) -> CalibrationResult:
    """Performs camera calibration using the provided checkerboard images.

    Raises ValueError on invalid inputs or if corner detection fails.
    """
    objpoints, imgpoints, image_size = find_checkerboard_corners(
        image_paths, board_size, square_size
    )

    ret, mtx, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    if not ret:
        raise RuntimeError("OpenCV calibration failed to converge.")

    return CalibrationResult(
        camera_matrix=mtx,
        dist_coeffs=dist,
        rms_error=float(ret),
        image_size=image_size,
        board_size=board_size,
        square_size=square_size,
    )


def save_calibration(path: Path | str, result: CalibrationResult) -> None:
    """Saves calibration parameters to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(result.to_json_dict(), f, indent=2)


def load_calibration(path: Path | str) -> CalibrationResult:
    """Loads calibration parameters from a JSON file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return CalibrationResult.from_json_dict(data)
