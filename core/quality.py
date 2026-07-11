"""
Phase 8 – Image-Level Quality Metrics

Aggregate quality indicators across the entire image:
  • Percentage of broken grains
  • Grain size distribution
  • Uniformity index
  • Average aspect ratio
  • Shape consistency
  • Percentage of abnormal grains
  • Density of grains
  • Grain packing statistics
  • Orientation distribution
"""

import numpy as np
from typing import Dict, Any, List


class QualityAnalyzer:
    """Compute image-level quality metrics from measurements + classification."""

    def __init__(self, pixels_per_mm: float = 0.0):
        self.ppm = pixels_per_mm

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------
    def analyze(
        self,
        measurements: List[Dict[str, Any]],
        classifications: List[Dict[str, Any]],
        image_shape: tuple = (0, 0),
    ) -> Dict[str, Any]:
        if not measurements:
            return {"error": "No grains for quality analysis"}

        total = len(measurements)
        h, w = image_shape[:2]
        image_area_px = h * w if h and w else 0

        # --- Broken grain percentage ---
        broken_count = sum(
            1 for c in classifications if "broken_grain" in c["categories"]
        )
        broken_pct = (broken_count / total) * 100

        # --- Abnormal grain percentage ---
        abnormal_count = sum(
            1 for c in classifications if "abnormal_grain" in c["categories"]
        )
        abnormal_pct = (abnormal_count / total) * 100

        # --- Size distribution ---
        lengths = np.array([m.get("length_px", 0) for m in measurements])
        widths = np.array([m.get("width_px", 0) for m in measurements])
        areas = np.array([m.get("area_px", 0) for m in measurements])

        size_dist = {
            "length_mean": float(np.mean(lengths)),
            "length_std": float(np.std(lengths)) if len(lengths) > 1 else 0.0,
            "width_mean": float(np.mean(widths)),
            "width_std": float(np.std(widths)) if len(widths) > 1 else 0.0,
            "area_mean": float(np.mean(areas)),
            "area_std": float(np.std(areas)) if len(areas) > 1 else 0.0,
        }

        # --- Uniformity index ---
        # CV of length → lower CV = more uniform
        cv_length = float(np.std(lengths) / np.mean(lengths) * 100) if np.mean(lengths) > 0 else 0.0
        cv_area = float(np.std(areas) / np.mean(areas) * 100) if np.mean(areas) > 0 else 0.0
        uniformity = max(0.0, 100.0 - cv_length)

        # --- Average aspect ratio ---
        aspect_ratios = np.array([m.get("aspect_ratio", 0) for m in measurements])
        avg_aspect = float(np.mean(aspect_ratios))

        # --- Shape consistency (inverse of CV of circularity) ---
        circularities = np.array([m.get("circularity", 0) for m in measurements])
        cv_circ = float(np.std(circularities) / np.mean(circularities) * 100) if np.mean(circularities) > 0 else 0.0
        shape_consistency = max(0.0, 100.0 - cv_circ)

        # --- Grain density ---
        total_grain_area = float(np.sum(areas))
        density = float(total_grain_area / image_area_px) if image_area_px > 0 else 0.0

        # --- Packing statistics ---
        # Fraction of image area covered by grains
        packing_ratio = density
        # Average inter-grain spacing (approximate)
        if image_area_px > 0 and total > 0:
            avg_spacing = float(np.sqrt(image_area_px / total) - np.sqrt(np.mean(areas)))
        else:
            avg_spacing = 0.0

        # --- Orientation distribution ---
        orientations = np.array([m.get("orientation_angle", 0) for m in measurements])
        orient_mean = float(np.mean(orientations))
        orient_std = float(np.std(orientations))
        # Circular mean (for angles)
        rad = np.deg2rad(orientations)
        orient_circ_mean = float(np.rad2deg(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))))
        orient_circ_std = float(np.rad2deg(np.sqrt(-2 * np.log(abs(np.mean(np.exp(1j * rad)))))) if len(orientations) > 1 else 0.0)

        # --- Size category distribution ---
        size_cats = {"long_grain": 0, "medium_grain": 0, "short_grain": 0}
        for c in classifications:
            for cat in c["categories"]:
                if cat in size_cats:
                    size_cats[cat] += 1

        return {
            "total_grains": total,
            "broken_pct": float(broken_pct),
            "abnormal_pct": float(abnormal_pct),
            "size_distribution": size_dist,
            "size_category_counts": size_cats,
            "uniformity_index": float(uniformity),
            "cv_length": float(cv_length),
            "cv_area": float(cv_area),
            "average_aspect_ratio": float(avg_aspect),
            "shape_consistency": float(shape_consistency),
            "grain_density": float(density),
            "packing_ratio": float(packing_ratio),
            "avg_inter_grain_spacing_px": float(avg_spacing),
            "orientation": {
                "mean": orient_mean,
                "std": orient_std,
                "circular_mean": orient_circ_mean,
                "circular_std": orient_circ_std,
            },
        }
