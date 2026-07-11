"""
Phase 1 – Camera Initialisation & Calibration
Phase 2 – Automated Image Acquisition

Handles:
  • Auto-detecting and opening the connected camera
  • Retrieving every available sensor property
  • Checkerboard / known-object calibration (pixels → mm)
  • Automated capture with full metadata logging
"""

import cv2
import numpy as np
import json
import time
import os
import threading
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import config


class CameraManager:
    """Wraps OpenCV VideoCapture with auto-property discovery and calibration."""

    # ------------------------------------------------------------------
    # Map human-readable names → OpenCV CAP_PROP constants
    # ------------------------------------------------------------------
    _PROP_MAP = {
        "resolution_width": cv2.CAP_PROP_FRAME_WIDTH,
        "resolution_height": cv2.CAP_PROP_FRAME_HEIGHT,
        "fps": cv2.CAP_PROP_FPS,
        "brightness": cv2.CAP_PROP_BRIGHTNESS,
        "contrast": cv2.CAP_PROP_CONTRAST,
        "saturation": cv2.CAP_PROP_SATURATION,
        "hue": cv2.CAP_PROP_HUE,
        "gain": cv2.CAP_PROP_GAIN,
        "exposure": cv2.CAP_PROP_EXPOSURE,
        "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
        "white_balance_blue_u": cv2.CAP_PROP_WHITE_BALANCE_BLUE_U,
        "white_balance_red_v": cv2.CAP_PROP_WHITE_BALANCE_RED_V,
        "focus": cv2.CAP_PROP_FOCUS,
        "auto_focus": cv2.CAP_PROP_AUTOFOCUS,
        "zoom": cv2.CAP_PROP_ZOOM,
        "sharpness": cv2.CAP_PROP_SHARPNESS,
        "gamma": cv2.CAP_PROP_GAMMA,
        "temperature": cv2.CAP_PROP_TEMPERATURE,
        "backlight": cv2.CAP_PROP_BACKLIGHT,
        "iso_speed": cv2.CAP_PROP_ISO_SPEED,
        "format": cv2.CAP_PROP_FORMAT,
        "mode": cv2.CAP_PROP_MODE,
        "convert_rgb": cv2.CAP_PROP_CONVERT_RGB,
        "buffer_size": cv2.CAP_PROP_BUFFERSIZE,
    }

    def __init__(self, camera_index: int = config.CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.properties: Dict[str, Any] = {}
        self.calibration_data: Dict[str, Any] = {}
        self.pixels_per_mm: float = config.PIXELS_PER_MM
        self._is_open = False
        self._lock = threading.Lock()
        # Load calibration from disk at startup so it survives camera power cycles
        self._load_calibration()

    # ------------------------------------------------------------------
    # Phase 1 – Initialisation (camera is only powered on when needed)
    # ------------------------------------------------------------------
    def open(self) -> bool:
        """Open the camera without taking permanent ownership (thread-safe)."""
        with self._lock:
            if self.cap is not None:
                self.close()
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                # Try AVFoundation backend explicitly on macOS
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
            if not self.cap.isOpened():
                self.cap = None
                return False
            self._is_open = True
            return True

    def _discover_properties(self):
        """Read every CAP_PROP that the backend exposes."""
        if self.cap is None:
            return
        self.properties = {}

        # Backend & camera identification
        backend = self.cap.getBackendName() if hasattr(self.cap, "getBackendName") else "unknown"
        self.properties["backend"] = backend

        for name, prop_id in self._PROP_MAP.items():
            try:
                val = self.cap.get(prop_id)
                # OpenCV returns -1.0 when the property is unsupported
                if val is not None and val != -1.0:
                    self.properties[name] = float(val)
            except Exception:
                pass

        # Resolution is always useful
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.properties["resolution_width"] = w
        self.properties["resolution_height"] = h
        self.properties["resolution_label"] = f"{w} × {h}"

        # FourCC codec
        try:
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            self.properties["fourcc"] = "".join(
                [chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]
            )
        except Exception:
            pass

    def get_properties(self) -> Dict[str, Any]:
        """Return the full property dict for display."""
        return dict(self.properties)

    def get_optimal_config(self) -> Dict[str, Any]:
        """Suggest the optimal capture configuration for morphometric work."""
        return {
            "resolution": f"{self.properties.get('resolution_width', '?')} × "
                          f"{self.properties.get('resolution_height', '?')}",
            "recommended_format": "JPEG (lossless PNG for calibration)",
            "recommended_exposure": "Lowest value that avoids motion blur",
            "recommended_gain": "0 (minimise noise)",
            "recommended_white_balance": "Fixed (disable auto)",
            "recommended_focus": "Manual, locked on the sample plane",
            "recommended_fps": "1–5 fps (quality over speed)",
            "lighting": "Diffuse, uniform, no reflections on matte black surface",
            "calibration": "Checkerboard with known square size before first run",
        }

    # ------------------------------------------------------------------
    # Phase 1b – Calibration
    # ------------------------------------------------------------------
    def calibrate_checkerboard(
        self,
        image_paths: list,
        square_size_mm: float = config.CHECKERBOARD_SQUARE_MM,
        pattern: Tuple[int, int] = config.CHECKERBOARD_PATTERN,
    ) -> Dict[str, Any]:
        """
        Calibrate pixels-per-mm from a set of checkerboard images.

        Returns a dict with the calibration result and stores it on disk.
        """
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
        objp *= square_size_mm

        objpoints = []
        imgpoints = []

        for fpath in image_paths:
            img = cv2.imread(fpath)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern, None)
            if found:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)

        if len(objpoints) < 1:
            return {"success": False, "error": "No checkerboard corners detected in any image"}

        h = img.shape[0]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (img.shape[1], h), None, None
        )

        # Compute pixels-per-mm from the first image
        first_img = cv2.imread(image_paths[0])
        gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern, None)
        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # Distance between first two adjacent corners in pixels
            d_px = np.linalg.norm(corners2[0].ravel() - corners2[1].ravel())
            pixels_per_mm = d_px / square_size_mm
        else:
            pixels_per_mm = 0.0

        self.pixels_per_mm = pixels_per_mm
        self.calibration_data = {
            "success": True,
            "pixels_per_mm": pixels_per_mm,
            "mm_per_pixel": 1.0 / pixels_per_mm if pixels_per_mm > 0 else 0.0,
            "camera_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist(),
            "num_images_used": len(objpoints),
            "square_size_mm": square_size_mm,
            "pattern": list(pattern),
            "calibrated_at": datetime.now().isoformat(),
        }

        self._save_calibration()
        return self.calibration_data

    def calibrate_known_object(
        self,
        image_path: str,
        object_length_mm: float,
    ) -> Dict[str, Any]:
        """
        Calibrate pixels-per-mm from a single image containing an object
        of known length.  The user clicks two endpoints on the object
        (or the system auto-detects the largest bright contour).
        """
        img = cv2.imread(image_path)
        if img is None:
            return {"success": False, "error": "Cannot read image"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {"success": False, "error": "No contours found in calibration image"}

        # Use the largest contour's bounding rectangle as the known object
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        width_px, height_px = rect[1]
        longest_side_px = max(width_px, height_px)

        if longest_side_px == 0:
            return {"success": False, "error": "Detected object has zero length"}

        pixels_per_mm = longest_side_px / object_length_mm
        self.pixels_per_mm = pixels_per_mm
        self.calibration_data = {
            "success": True,
            "pixels_per_mm": pixels_per_mm,
            "mm_per_pixel": 1.0 / pixels_per_mm,
            "method": "known_object",
            "object_length_mm": object_length_mm,
            "detected_length_px": float(longest_side_px),
            "calibrated_at": datetime.now().isoformat(),
        }
        self._save_calibration()
        return self.calibration_data

    def set_manual_calibration(self, pixels_per_mm: float) -> Dict[str, Any]:
        """Manually set the pixels-per-mm ratio."""
        self.pixels_per_mm = pixels_per_mm
        self.calibration_data = {
            "success": True,
            "pixels_per_mm": pixels_per_mm,
            "mm_per_pixel": 1.0 / pixels_per_mm,
            "method": "manual",
            "calibrated_at": datetime.now().isoformat(),
        }
        self._save_calibration()
        return self.calibration_data

    def _save_calibration(self):
        path = os.path.join(config.CALIBRATION_DIR, "calibration.json")
        with open(path, "w") as f:
            json.dump(self.calibration_data, f, indent=2)

    def _load_calibration(self):
        path = os.path.join(config.CALIBRATION_DIR, "calibration.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self.calibration_data = json.load(f)
                self.pixels_per_mm = self.calibration_data.get("pixels_per_mm", 0.0)
            except Exception:
                pass

    def is_calibrated(self) -> bool:
        return self.pixels_per_mm > 0

    # ------------------------------------------------------------------
    # Phase 2 – Automated Image Acquisition
    # ------------------------------------------------------------------
    def capture(self, save: bool = True) -> Dict[str, Any]:
        """Capture a single frame and release the camera immediately."""
        if not self.open():
            return {"success": False, "error": "Camera not available"}

        try:
            # Set preferred resolution each time the camera is opened
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)

            # Warm up the camera (first few frames are often dark)
            for _ in range(5):
                self.cap.grab()
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return {"success": False, "error": "Failed to capture frame"}

            timestamp = datetime.now()
            filename = f"capture_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}{config.CAPTURE_FORMAT}"
            filepath = os.path.join(config.UPLOAD_DIR, filename)

            if save:
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, config.CAPTURE_JPEG_QUALITY])

            metadata = {
                "timestamp": timestamp.isoformat(),
                "filename": filename,
                "filepath": filepath,
                "resolution": f"{frame.shape[1]} × {frame.shape[0]}",
                "width": frame.shape[1],
                "height": frame.shape[0],
                "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
                "camera_properties": self.get_properties(),
                "calibration_status": "calibrated" if self.is_calibrated() else "uncalibrated",
                "pixels_per_mm": self.pixels_per_mm,
            }

            return {
                "success": True,
                "frame": frame,
                "metadata": metadata,
            }
        finally:
            self.close()

    def capture_multiple(
        self,
        count: int = 5,
        interval_sec: float = 1.0,
    ) -> list:
        """Capture multiple images at fixed intervals, opening/closing per shot."""
        results = []
        for i in range(count):
            result = self.capture()
            results.append(result)
            if i < count - 1:
                time.sleep(interval_sec)
        return results

    def close(self):
        """Release the camera so the LED turns off."""
        with self._lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self._is_open = False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def get_preview(self) -> Optional[np.ndarray]:
        """Grab a single frame for live preview, then release the camera."""
        if not self.open():
            return None
        try:
            # Set preferred resolution each time the camera is opened
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
            # Give the sensor a moment to adjust
            for _ in range(3):
                self.cap.grab()
            ret, frame = self.cap.read()
            return frame if ret else None
        finally:
            self.close()

    def discover_and_close(self) -> bool:
        """Open camera, discover properties, then close it."""
        if not self.open():
            return False
        try:
            self._discover_properties()
            return True
        finally:
            self.close()

    def set_property(self, prop_name: str, value: float) -> bool:
        """Set a camera property by name."""
        if self.cap is None:
            return False
        prop_id = self._PROP_MAP.get(prop_name)
        if prop_id is None:
            return False
        return self.cap.set(prop_id, value)
