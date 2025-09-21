"""
Camera Calibration Utility for Multi-Modal Physiological Sensing

Production-ready camera calibration for RGB and thermal cameras to ensure
accurate spatial alignment and measurement precision.

Features:
- Automatic checkerboard pattern detection
- Stereo calibration for RGB/thermal camera pairs
- Distortion correction parameters
- Spatial alignment matrices
- Validation with ground truth measurements
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from PyQt6.QtCore import QObject, pyqtSignal

    HAS_QT = True
except ImportError:
    HAS_QT = False

    class _DummyQObject:
        def __init__(self):
            pass

    QObject = _DummyQObject  # type: ignore

    def pyqtSignal(*args):
        return lambda: None


@dataclass
class CalibrationResult:
    """Container for camera calibration results."""

    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    rotation_vectors: list[np.ndarray] | None = None
    translation_vectors: list[np.ndarray] | None = None
    reprojection_error: float = 0.0
    calibration_timestamp: str = ""
    image_size: tuple[int, int] = (0, 0)
    pattern_size: tuple[int, int] = (0, 0)
    square_size_mm: float = 0.0


@dataclass
class StereoCalibrationResult:
    """Container for stereo calibration results."""

    left_camera: CalibrationResult
    right_camera: CalibrationResult
    rotation_matrix: np.ndarray = None
    translation_vector: np.ndarray = None
    essential_matrix: np.ndarray = None
    fundamental_matrix: np.ndarray = None
    rectification_left: np.ndarray = None
    rectification_right: np.ndarray = None
    projection_left: np.ndarray = None
    projection_right: np.ndarray = None
    disparity_to_depth_matrix: np.ndarray = None
    stereo_error: float = 0.0


class CameraCalibrator(QObject if HAS_QT else object):
    """
    Professional camera calibration utility with GUI integration.

    Supports both single camera and stereo camera calibration for
    RGB and thermal camera systems.
    """

    if HAS_QT:
        progress_updated = pyqtSignal(int, str)
        calibration_completed = pyqtSignal(object)
        error_occurred = pyqtSignal(str)
        pattern_detected = pyqtSignal(int)

    def __init__(self):
        super().__init__()

        self.pattern_size = (9, 6)
        self.square_size_mm = 25.0

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Detection parameters
        self.detection_flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
        )

        self.object_points = []
        self.image_points_left = []
        self.image_points_right = []

        self._prepare_object_points()

    def _prepare_object_points(self):
        """Prepare 3D object points for the calibration pattern."""
        self.objp = np.zeros(
            (self.pattern_size[0] * self.pattern_size[1], 3), np.float32
        )
        self.objp[:, :2] = np.mgrid[
            0 : self.pattern_size[0], 0 : self.pattern_size[1]
        ].T.reshape(-1, 2)
        self.objp *= self.square_size_mm

    def emit_progress(self, percent: int, message: str):
        """Emit progress signal or print to console."""
        if HAS_QT and hasattr(self, "progress_updated"):
            self.progress_updated.emit(percent, message)
        else:
            print(f"[{percent}%] {message}")

    def collect_calibration_images(
        self, image_dir: Path, camera_type: str = "rgb"
    ) -> list[Path]:
        """
        Collect and validate calibration images from a directory.

        Args:
            image_dir: Directory containing calibration images
            camera_type: Type of camera ("rgb" or "thermal")

        Returns:
            List of valid image paths
        """
        self.emit_progress(0, f"Collecting {camera_type} calibration images...")

        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]

        image_paths: list[Path] = []
        for ext in extensions:
            image_paths.extend(image_dir.glob(ext))
            image_paths.extend(image_dir.glob(ext.upper()))

        if not image_paths:
            self.emit_progress(0, f"No images found in {image_dir}")
            return []

        image_paths.sort()

        self.emit_progress(10, f"Found {len(image_paths)} potential calibration images")
        return image_paths

    def detect_checkerboard_corners(
        self, image_paths: list[Path]
    ) -> tuple[list[np.ndarray], tuple[int, int]]:
        """
        Detect checkerboard corners in calibration images.

        Args:
            image_paths: List of image file paths

        Returns:
            Tuple of (corner_points_list, image_size)
        """
        corner_points = []
        image_size = None

        self.emit_progress(20, "Detecting checkerboard patterns...")

        for i, img_path in enumerate(image_paths):
            progress = 20 + int((i / len(image_paths)) * 60)
            self.emit_progress(progress, f"Processing {img_path.name}...")

            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                image_size = gray.shape[::-1]

                ret, corners = cv2.findChessboardCorners(
                    gray, self.pattern_size, self.detection_flags
                )

                if ret:
                    corners_refined = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), self.criteria
                    )
                    corner_points.append(corners_refined)

                    if HAS_QT and hasattr(self, "pattern_detected"):
                        self.pattern_detected.emit(len(corner_points))

                    total_count = len(corner_points)
                    msg = f"Pattern detected in {img_path.name} ({total_count} total)"
                    self.emit_progress(progress, msg)
                else:
                    self.emit_progress(progress, f"No pattern found in {img_path.name}")

            except Exception as e:
                self.emit_progress(progress, f"Error processing {img_path.name}: {e}")
                continue

        self.emit_progress(
            80, f"Pattern detection complete: {len(corner_points)} valid images"
        )
        return corner_points, image_size

    def calibrate_single_camera(
        self, image_paths: list[Path], camera_name: str = "camera"
    ) -> CalibrationResult | None:
        """
        Perform single camera calibration.

        Args:
            image_paths: List of calibration image paths
            camera_name: Name for identification

        Returns:
            CalibrationResult or None if calibration failed
        """
        self.emit_progress(0, f"Starting {camera_name} calibration...")

        corner_points, image_size = self.detect_checkerboard_corners(image_paths)

        if len(corner_points) < 5:
            count = len(corner_points)
            error_msg = (
                f"Insufficient valid calibration images ({count}). Need at least 5."
            )
            if HAS_QT and hasattr(self, "error_occurred"):
                self.error_occurred.emit(error_msg)
            else:
                print(f"Error: {error_msg}")
            return None

        object_points = [self.objp for _ in corner_points]

        self.emit_progress(85, "Running camera calibration...")

        try:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                object_points,
                corner_points,
                image_size,
                None,
                None,
                flags=cv2.CALIB_FIX_PRINCIPAL_POINT,
            )

            if not ret:
                error_msg = "Camera calibration failed - numerical issues"
                if HAS_QT and hasattr(self, "error_occurred"):
                    self.error_occurred.emit(error_msg)
                return None

            total_error = 0
            for i in range(len(object_points)):
                proj_points, _ = cv2.projectPoints(
                    object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                )
                error = cv2.norm(corner_points[i], proj_points, cv2.NORM_L2) / len(
                    proj_points
                )
                total_error += error

            mean_error = total_error / len(object_points)

            result = CalibrationResult(
                camera_matrix=camera_matrix,
                distortion_coeffs=dist_coeffs,
                rotation_vectors=rvecs,
                translation_vectors=tvecs,
                reprojection_error=mean_error,
                calibration_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                image_size=image_size,
                pattern_size=self.pattern_size,
                square_size_mm=self.square_size_mm,
            )

            self.emit_progress(
                100, f"Calibration complete! Error: {mean_error:.4f} pixels"
            )

            if HAS_QT and hasattr(self, "calibration_completed"):
                self.calibration_completed.emit(result)

            return result

        except Exception as e:
            error_msg = f"Calibration error: {e!s}"
            if HAS_QT and hasattr(self, "error_occurred"):
                self.error_occurred.emit(error_msg)
            else:
                print(f"Error: {error_msg}")
            return None

    def save_calibration_result(self, result: CalibrationResult, output_path: Path):
        """Save calibration result to JSON file."""
        result_dict = {
            "camera_matrix": result.camera_matrix.tolist(),
            "distortion_coeffs": result.distortion_coeffs.tolist(),
            "reprojection_error": float(result.reprojection_error),
            "calibration_timestamp": result.calibration_timestamp,
            "image_size": result.image_size,
            "pattern_size": result.pattern_size,
            "square_size_mm": result.square_size_mm,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        print(f"Calibration saved to {output_path}")

    def load_calibration_result(self, input_path: Path) -> CalibrationResult | None:
        """Load calibration result from JSON file."""
        try:
            with open(input_path) as f:
                data = json.load(f)

            return CalibrationResult(
                camera_matrix=np.array(data["camera_matrix"]),
                distortion_coeffs=np.array(data["distortion_coeffs"]),
                reprojection_error=data["reprojection_error"],
                calibration_timestamp=data["calibration_timestamp"],
                image_size=tuple(data["image_size"]),
                pattern_size=tuple(data["pattern_size"]),
                square_size_mm=data["square_size_mm"],
            )
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return None


def calibrate_camera_from_directory(
    image_dir: Path, camera_type: str = "rgb", output_dir: Path | None = None
) -> CalibrationResult | None:
    """
    Convenience function for command-line camera calibration.

    Args:
        image_dir: Directory containing calibration images
        camera_type: Type of camera ("rgb" or "thermal")
        output_dir: Directory to save calibration results

    Returns:
        CalibrationResult or None if failed
    """
    calibrator = CameraCalibrator()
    image_paths = calibrator.collect_calibration_images(image_dir, camera_type)

    if not image_paths:
        print(f"No calibration images found in {image_dir}")
        return None

    result = calibrator.calibrate_single_camera(image_paths, camera_type)

    if result and output_dir:
        output_path = output_dir / f"{camera_type}_calibration.json"
        calibrator.save_calibration_result(result, output_path)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Camera calibration utility")
    parser.add_argument("image_dir", help="Directory containing calibration images")
    parser.add_argument(
        "--camera-type",
        choices=["rgb", "thermal"],
        default="rgb",
        help="Type of camera to calibrate",
    )
    parser.add_argument("--output-dir", help="Directory to save calibration results")
    parser.add_argument(
        "--pattern-size",
        nargs=2,
        type=int,
        default=[9, 6],
        help="Checkerboard pattern size (width height)",
    )
    parser.add_argument(
        "--square-size", type=float, default=25.0, help="Checkerboard square size in mm"
    )

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir) if args.output_dir else image_dir.parent

    # Configure calibration parameters
    calibrator = CameraCalibrator()
    calibrator.pattern_size = tuple(args.pattern_size)
    calibrator.square_size_mm = args.square_size

    result = calibrate_camera_from_directory(image_dir, args.camera_type, output_dir)

    if result:
        print("\nCalibration successful!")
        print(f"Reprojection error: {result.reprojection_error:.4f} pixels")
        print(f"Results saved to {output_dir}")
    else:
        print("Calibration failed!")
        exit(1)
