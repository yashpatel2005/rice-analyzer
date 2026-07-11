"""
Phase 6 – Dataset-Level Statistical Analysis

Comprehensive statistical analysis across all measured grains:
  • Descriptive statistics (mean, median, mode, min, max, std, var, CV)
  • Quartiles, IQR, percentiles
  • Confidence intervals
  • Outlier detection (IQR + Z-score)
  • Correlation matrix
  • Distribution fitting (normal)
  • Histograms, box plots, KDE, scatter plots
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional

import config


class StatisticalAnalyzer:
    """Compute dataset-level statistics from per-grain measurements."""

    # Primary metrics to analyse
    METRIC_KEYS = [
        "area_px", "perimeter_px", "length_px", "width_px",
        "major_axis_px", "minor_axis_px", "equivalent_diameter_px",
        "feret_max_px", "feret_min_px",
        "solidity", "convexity", "circularity", "elongation",
        "aspect_ratio", "eccentricity", "extent", "fill_ratio",
        "orientation_angle",
    ]

    def __init__(self, pixels_per_mm: float = 0.0):
        self.ppm = pixels_per_mm

    # ------------------------------------------------------------------
    # Descriptive statistics for a single metric
    # ------------------------------------------------------------------
    def _describe(self, values: np.ndarray) -> Dict[str, Any]:
        """Full descriptive statistics for one array of values."""
        if len(values) == 0:
            return {}

        result = {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "mode": float(stats.mode(values, keepdims=False).mode),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
            "variance": float(np.var(values, ddof=1)) if len(values) > 1 else 0.0,
            "std_dev": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "cv": float(np.std(values, ddof=1) / np.mean(values) * 100)
            if np.mean(values) != 0 and len(values) > 1 else 0.0,
            "q1": float(np.percentile(values, 25)),
            "q3": float(np.percentile(values, 75)),
            "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "p5": float(np.percentile(values, 5)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "skewness": float(stats.skew(values)) if len(values) > 2 else 0.0,
            "kurtosis": float(stats.kurtosis(values)) if len(values) > 3 else 0.0,
        }

        # 95 % confidence interval for the mean
        if len(values) > 1:
            ci = stats.t.interval(
                0.95, len(values) - 1,
                loc=np.mean(values),
                scale=stats.sem(values),
            )
            result["ci_95_lower"] = float(ci[0])
            result["ci_95_upper"] = float(ci[1])
        else:
            result["ci_95_lower"] = float(values[0]) if len(values) == 1 else 0.0
            result["ci_95_upper"] = float(values[0]) if len(values) == 1 else 0.0

        # Normal distribution fit test (Shapiro-Wilk)
        if 3 <= len(values) <= 5000:
            try:
                sh_stat, sh_p = stats.shapiro(values)
                result["shapiro_stat"] = float(sh_stat)
                result["shapiro_p"] = float(sh_p)
                result["is_normal"] = bool(sh_p > 0.05)
            except Exception:
                result["shapiro_stat"] = None
                result["shapiro_p"] = None
                result["is_normal"] = None
        else:
            result["shapiro_stat"] = None
            result["shapiro_p"] = None
            result["is_normal"] = None

        return result

    # ------------------------------------------------------------------
    # Outlier detection
    # ------------------------------------------------------------------
    def _detect_outliers(self, values: np.ndarray) -> Dict[str, Any]:
        """IQR and Z-score outlier detection."""
        if len(values) < 4:
            return {"iqr_outliers": [], "zscore_outliers": []}

        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_out = np.where((values < lower) | (values > upper))[0].tolist()

        z = np.abs(stats.zscore(values))
        z_out = np.where(z > 3)[0].tolist()

        return {
            "iqr_outliers": [int(i) for i in iqr_out],
            "zscore_outliers": [int(i) for i in z_out],
            "iqr_lower": float(lower),
            "iqr_upper": float(upper),
        }

    # ------------------------------------------------------------------
    # Correlation matrix
    # ------------------------------------------------------------------
    def correlation_matrix(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute Pearson correlation matrix for numeric columns."""
        numeric_cols = [c for c in self.METRIC_KEYS if c in df.columns]
        if len(numeric_cols) < 2:
            return {}
        corr = df[numeric_cols].corr(method="pearson")
        return corr.round(4).to_dict()

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------
    def analyze(self, measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the complete statistical analysis suite."""
        if not measurements:
            return {"error": "No grains to analyze"}

        df = pd.DataFrame(measurements)

        # Per-metric descriptive statistics
        per_metric = {}
        for key in self.METRIC_KEYS:
            if key not in df.columns:
                continue
            values = df[key].dropna().values
            if len(values) == 0:
                continue
            desc = self._describe(values)
            desc["outliers"] = self._detect_outliers(values)
            per_metric[key] = desc

        # Correlation matrix
        corr = self.correlation_matrix(df)

        # Total count
        total = len(measurements)

        # If calibrated, also compute mm-based stats for key metrics
        mm_stats = {}
        if self.ppm and self.ppm > 0:
            for key in ["length_mm", "width_mm", "area_mm2", "perimeter_mm"]:
                if key in df.columns:
                    vals = df[key].dropna().values
                    if len(vals) > 0:
                        mm_stats[key] = self._describe(vals)

        return {
            "total_grains": total,
            "per_metric": per_metric,
            "correlation": corr,
            "mm_stats": mm_stats,
            "calibrated": self.ppm > 0,
            "pixels_per_mm": self.ppm,
        }
