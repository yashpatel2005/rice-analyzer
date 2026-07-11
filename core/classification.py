"""
Phase 7 – Grain Classification

Rule-based classification using only extracted measurements.
  • Whole grains
  • Broken grains
  • Long grains
  • Medium grains
  • Short grains
  • Oversized grains
  • Undersized grains
  • Abnormal grains
  • Outliers

All thresholds are configurable via config.py.
"""

import numpy as np
from typing import Dict, Any, List

import config


class GrainClassifier:
    """Classify individual grains based on morphometric thresholds."""

    def __init__(self, thresholds: dict = None, pixels_per_mm: float = 0.0, broken_threshold: float = None):
        self.t = thresholds or config.CLASSIFICATION_THRESHOLDS
        if broken_threshold is not None:
            # Create a copy so we don't mutate the global config
            self.t = self.t.copy()
            self.t["broken_max_length_ratio"] = broken_threshold
            
        self.ppm = pixels_per_mm
        self.use_mm = self.ppm and self.ppm > 0

    # ------------------------------------------------------------------
    # Length helper
    # ------------------------------------------------------------------
    def _get_length(self, grain: Dict[str, Any]) -> float:
        if self.use_mm and grain.get("length_mm") is not None:
            return grain["length_mm"]
        return grain.get("length_px", 0.0)

    def _get_area(self, grain: Dict[str, Any]) -> float:
        if self.use_mm and grain.get("area_mm2") is not None:
            return grain["area_mm2"]
        return grain.get("area_px", 0.0)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    def classify_all(self, measurements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify every grain.  Each grain gets a list of category labels
        and a primary category.
        """
        if not measurements:
            return []

        # Compute median length & area for relative thresholds
        lengths = [self._get_length(m) for m in measurements]
        areas = [self._get_area(m) for m in measurements]
        median_len = float(np.median(lengths)) if lengths else 0.0
        median_area = float(np.median(areas)) if areas else 0.0

        results = []
        for m in measurements:
            cats = []
            length = self._get_length(m)
            area = self._get_area(m)

            # --- Size classification (absolute) ---
            if self.use_mm:
                if length >= self.t["long_grain_min_length"]:
                    cats.append("long_grain")
                elif length >= self.t["medium_grain_min_length"]:
                    cats.append("medium_grain")
                else:
                    cats.append("short_grain")
            else:
                # Pixel mode: use relative thresholds
                if length >= median_len * 1.15:
                    cats.append("long_grain")
                elif length >= median_len * 0.85:
                    cats.append("medium_grain")
                else:
                    cats.append("short_grain")

            # --- Broken grain ---
            if median_len > 0:
                ratio = length / median_len
                if ratio < self.t["broken_max_length_ratio"]:
                    cats.append("broken_grain")
                else:
                    cats.append("whole_grain")
            else:
                cats.append("whole_grain")

            # --- Oversized / undersized ---
            if median_area > 0:
                area_ratio = area / median_area
                if area_ratio > self.t["oversized_max_area_ratio"]:
                    cats.append("oversized_grain")
                elif area_ratio < self.t["undersized_min_area_ratio"]:
                    cats.append("undersized_grain")

            # --- Abnormal shape ---
            solidity = m.get("solidity", 1.0)
            circularity = m.get("circularity", 0.0)
            eccentricity = m.get("eccentricity", 0.0)
            if (
                solidity < self.t["abnormal_solidity_min"]
                or circularity < self.t["abnormal_circularity_max"]
                or eccentricity > self.t["abnormal_eccentricity_max"]
            ):
                cats.append("abnormal_grain")

            # Determine primary category
            primary = self._primary_category(cats)

            results.append({
                "label": m.get("label"),
                "categories": cats,
                "primary_category": primary,
            })

        return results

    # ------------------------------------------------------------------
    # Primary category logic
    # ------------------------------------------------------------------
    @staticmethod
    def _primary_category(cats: List[str]) -> str:
        """Pick the most important category for display."""
        priority = [
            "broken_grain",
            "abnormal_grain",
            "oversized_grain",
            "undersized_grain",
            "long_grain",
            "medium_grain",
            "short_grain",
            "whole_grain",
        ]
        for p in priority:
            if p in cats:
                return p
        return "whole_grain"

    # ------------------------------------------------------------------
    # Summary counts
    # ------------------------------------------------------------------
    @staticmethod
    def category_counts(classifications: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for c in classifications:
            for cat in c["categories"]:
                counts[cat] = counts.get(cat, 0) + 1
        return counts
