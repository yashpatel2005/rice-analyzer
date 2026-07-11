"""
Phase 5 – Grain Detection and Measurement

For every segmented rice grain compute a comprehensive set of
geometrical / morphometric measurements using OpenCV and scikit-image.

All measurements are reported in pixels and, when calibration is
available, in millimetres.
"""

import cv2
import numpy as np
from skimage.measure import regionprops, label as sk_label
from skimage.measure import moments_hu
from typing import Dict, Any, List, Optional

import config


class GrainMeasurer:
    """Compute morphometric features for each individual grain."""

    def __init__(self, pixels_per_mm: float = 0.0):
        self.ppm = pixels_per_mm  # 0 → uncalibrated (px only)

    # ------------------------------------------------------------------
    # Unit helpers
    # ------------------------------------------------------------------
    def _mm(self, px: float) -> Optional[float]:
        """Convert pixels to mm; return None when uncalibrated."""
        if self.ppm and self.ppm > 0:
            return px / self.ppm
        return None

    def _mm2(self, px2: float) -> Optional[float]:
        if self.ppm and self.ppm > 0:
            return px2 / (self.ppm ** 2)
        return None

    @staticmethod
    def _safe_div(a: float, b: float) -> float:
        return a / b if b != 0 else 0.0

    # ------------------------------------------------------------------
    # Zernike moment approximation
    # ------------------------------------------------------------------
    @staticmethod
    def _zernike_moments(mask: np.ndarray, order: int = 4) -> List[float]:
        """
        Compute a small set of Zernike-like moments.
        Uses a simplified radial polynomial on the unit disk.
        """
        # Map mask to unit disk
        h, w = mask.shape
        y, x = np.nonzero(mask)
        if len(x) == 0:
            return [0.0] * (order + 1)
        cx, cy = x.mean(), y.mean()
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r_max = r.max() if r.max() > 0 else 1.0
        rn = r / r_max
        theta = np.arctan2(y - cy, x - cx)

        zernikes = []
        for n in range(order + 1):
            # Z_n^0 = R_n^0  (simplified)
            R_n = np.sum(rn ** n) / max(len(rn), 1)
            zernikes.append(float(R_n))
        return zernikes

    # ------------------------------------------------------------------
    # Single-grain measurement
    # ------------------------------------------------------------------
    def measure_grain(
        self, contour: np.ndarray, mask: np.ndarray, label: int
    ) -> Dict[str, Any]:
        """Compute every available morphometric metric for one grain."""
        M = cv2.moments(contour)
        area_px = M["m00"]
        area_mm = self._mm2(area_px)

        # Perimeter
        peri_px = cv2.arcLength(contour, True)
        peri_mm = self._mm(peri_px)

        # Bounding rect (axis-aligned)
        bx, by, bw, bh = cv2.boundingRect(contour)
        bbox_w_mm = self._mm(bw)
        bbox_h_mm = self._mm(bh)

        # Rotated (min-area) rect
        (rx, ry), (rw, rh), angle = cv2.minAreaRect(contour)
        rot_w_mm = self._mm(rw)
        rot_h_mm = self._mm(rh)

        # Major / minor axis from rotated rect
        major_px = max(rw, rh)
        minor_px = min(rw, rh)
        major_mm = self._mm(major_px)
        minor_mm = self._mm(minor_px)

        # Grain length = major axis, width = minor axis
        length_px = major_px
        width_px = minor_px
        length_mm = self._mm(length_px)
        width_mm = self._mm(width_px)

        # Convex hull
        hull = cv2.convexHull(contour)
        hull_area_px = cv2.contourArea(hull)
        hull_area_mm = self._mm2(hull_area_px)
        hull_peri_px = cv2.arcLength(hull, True)
        hull_peri_mm = self._mm(hull_peri_px)

        # Solidity, convexity, circularity
        solidity = self._safe_div(area_px, hull_area_px)
        convexity = self._safe_div(hull_peri_px, peri_px) if peri_px > 0 else 0.0
        circularity = self._safe_div(4 * np.pi * area_px, peri_px ** 2) if peri_px > 0 else 0.0

        # Equivalent diameter
        eq_diam_px = np.sqrt(4 * area_px / np.pi) if area_px > 0 else 0.0
        eq_diam_mm = self._mm(eq_diam_px)

        # Elongation, aspect ratio
        elongation = self._safe_div(minor_px, major_px) if major_px > 0 else 0.0
        aspect_ratio = self._safe_div(major_px, minor_px) if minor_px > 0 else 0.0

        # Eccentricity from fitEllipse
        try:
            (cx_e, cy_e), (a_axis, b_axis), _ = cv2.fitEllipse(contour)
            major_e = max(a_axis, b_axis)
            minor_e = min(a_axis, b_axis)
            if major_e > 0 and minor_e > 0:
                ratio = minor_e / major_e
                val = 1 - ratio ** 2
                eccentricity = float(np.sqrt(max(val, 0.0)))
            else:
                eccentricity = 0.0
        except (cv2.error, Exception):
            eccentricity = 0.0

        # Centroid
        cx = M["m10"] / area_px if area_px > 0 else 0.0
        cy = M["m01"] / area_px if area_px > 0 else 0.0

        # Orientation angle from fitEllipse (normalised to 0-180)
        try:
            (_, _), (_, _), orient = cv2.fitEllipse(contour)
            orientation = orient % 180
        except cv2.error:
            orientation = angle % 180

        # Feret diameters (max and min caliper)
        feret_max, feret_min = self._feret_diameters(contour)
        feret_max_mm = self._mm(feret_max)
        feret_min_mm = self._mm(feret_min)

        # Hu moments (7 values)
        hu = cv2.HuMoments(M).flatten().tolist()
        # Log-transform for better dynamic range
        hu_log = [float(np.sign(h) * np.log1p(abs(h))) if h != 0 else 0.0 for h in hu]

        # Zernike moments (simplified)
        zernike = self._zernike_moments(mask, order=4)

        # Extent (ratio of bbox area to contour area)
        extent = self._safe_div(area_px, bw * bh) if bw * bh > 0 else 0.0

        # Fill ratio
        fill_ratio = self._safe_div(area_px, rw * rh) if rw * rh > 0 else 0.0

        return {
            "label": label,
            # --- area ---
            "area_px": float(area_px),
            "area_mm2": area_mm,
            "hull_area_px": float(hull_area_px),
            "hull_area_mm2": hull_area_mm,
            # --- perimeter ---
            "perimeter_px": float(peri_px),
            "perimeter_mm": peri_mm,
            "hull_perimeter_px": float(hull_peri_px),
            "hull_perimeter_mm": hull_peri_mm,
            # --- length / width ---
            "length_px": float(length_px),
            "length_mm": length_mm,
            "width_px": float(width_px),
            "width_mm": width_mm,
            "major_axis_px": float(major_px),
            "major_axis_mm": major_mm,
            "minor_axis_px": float(minor_px),
            "minor_axis_mm": minor_mm,
            # --- bounding boxes ---
            "bbox_x": int(bx),
            "bbox_y": int(by),
            "bbox_w_px": int(bw),
            "bbox_h_px": int(bh),
            "bbox_w_mm": bbox_w_mm,
            "bbox_h_mm": bbox_h_mm,
            "rot_rect_w_px": float(rw),
            "rot_rect_h_px": float(rh),
            "rot_rect_w_mm": rot_w_mm,
            "rot_rect_h_mm": rot_h_mm,
            "rot_rect_angle": float(angle),
            # --- shape descriptors ---
            "solidity": float(solidity),
            "convexity": float(convexity),
            "circularity": float(circularity),
            "elongation": float(elongation),
            "aspect_ratio": float(aspect_ratio),
            "eccentricity": float(eccentricity),
            "extent": float(extent),
            "fill_ratio": float(fill_ratio),
            # --- diameters ---
            "equivalent_diameter_px": float(eq_diam_px),
            "equivalent_diameter_mm": eq_diam_mm,
            "feret_max_px": float(feret_max),
            "feret_max_mm": feret_max_mm,
            "feret_min_px": float(feret_min),
            "feret_min_mm": feret_min_mm,
            # --- position / orientation ---
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "orientation_angle": float(orientation),
            # --- moments ---
            "hu_moments": hu_log,
            "zernike_moments": zernike,
        }

    # ------------------------------------------------------------------
    # Feret diameters
    # ------------------------------------------------------------------
    @staticmethod
    def _feret_diameters(contour: np.ndarray) -> tuple:
        """Max and min Feret diameters via rotating calipers approximation."""
        pts = contour.reshape(-1, 2)
        if len(pts) < 2:
            return 0.0, 0.0

        # Convex hull points
        hull = cv2.convexHull(contour).reshape(-1, 2)

        # Try rotating calipers
        try:
            (cx, cy), (w, h), _ = cv2.fitEllipse(contour)
            feret_max = max(w, h)
            feret_min = min(w, h)
        except cv2.error:
            # Fallback: max / min pairwise distances
            from scipy.spatial.distance import pdist
            d = pdist(pts)
            feret_max = d.max() if len(d) > 0 else 0.0
            feret_min = d.min() if len(d) > 0 else 0.0

        return float(feret_max), float(feret_min)

    # ------------------------------------------------------------------
    # All grains
    # ------------------------------------------------------------------
    def measure_all(self, grains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Measure every grain and return a list of measurement dicts."""
        results = []
        for g in grains:
            m = self.measure_grain(g["contour"], g["mask"], g["label"])
            results.append(m)
        return results
