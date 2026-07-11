"""
Phase 9 – Reporting

Generate publication-quality outputs:
  • CSV files
  • Excel spreadsheets
  • JSON exports
  • Annotated images (every grain labelled + measured)
  • Histograms, box plots, scatter plots, correlation heatmaps, KDE
  • Quality summary tables
"""

import cv2
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, List, Optional
import os

import config

matplotlib.use("Agg")


class ReportGenerator:
    """Create all report artefacts for a single analysis run."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or config.OUTPUT_DIR
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{self.timestamp}"

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    def export_csv(self, measurements: List[Dict[str, Any]]) -> str:
        """Export per-grain measurements to CSV."""
        # Flatten nested fields (hu moments, zernike moments)
        flat = []
        for m in measurements:
            row = {}
            for k, v in m.items():
                if isinstance(v, list):
                    for i, val in enumerate(v):
                        row[f"{k}_{i}"] = val
                else:
                    row[k] = v
            flat.append(row)
        df = pd.DataFrame(flat)
        path = os.path.join(config.CSV_DIR, f"{self.run_id}_measurements.csv")
        df.to_csv(path, index=False)
        return path

    def export_stats_csv(self, stats: Dict[str, Any]) -> str:
        """Export per-metric statistics to CSV."""
        rows = []
        for metric, s in stats.get("per_metric", {}).items():
            row = {"metric": metric}
            row.update(s)
            rows.append(row)
        df = pd.DataFrame(rows)
        path = os.path.join(config.CSV_DIR, f"{self.run_id}_statistics.csv")
        df.to_csv(path, index=False)
        return path

    # ------------------------------------------------------------------
    # Excel
    # ------------------------------------------------------------------
    def export_excel(
        self,
        measurements: List[Dict[str, Any]],
        stats: Dict[str, Any],
        quality: Dict[str, Any],
        classifications: List[Dict[str, Any]],
        grading: Dict[str, Any],
    ) -> str:
        """Export a multi-sheet Excel workbook."""
        path = os.path.join(config.EXCEL_DIR, f"{self.run_id}_report.xlsx")

        # Flatten measurements
        flat_m = []
        for m in measurements:
            row = {}
            for k, v in m.items():
                if isinstance(v, list):
                    for i, val in enumerate(v):
                        row[f"{k}_{i}"] = val
                else:
                    row[k] = v
            flat_m.append(row)
        df_m = pd.DataFrame(flat_m)

        # Merge classification info
        if classifications:
            df_c = pd.DataFrame(classifications)
            df_m = df_m.merge(df_c, on="label", how="left")

        # Stats sheet
        stats_rows = []
        for metric, s in stats.get("per_metric", {}).items():
            row = {"metric": metric}
            row.update(s)
            stats_rows.append(row)
        df_stats = pd.DataFrame(stats_rows)

        # Quality sheet
        df_quality = pd.DataFrame([quality])

        # Grading sheet
        df_grading = pd.DataFrame([grading])

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df_m.to_excel(writer, sheet_name="Measurements", index=False)
            df_stats.to_excel(writer, sheet_name="Statistics", index=False)
            df_quality.to_excel(writer, sheet_name="Quality Metrics", index=False)
            df_grading.to_excel(writer, sheet_name="Grading", index=False)

        return path

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------
    def export_json(
        self,
        measurements: List[Dict[str, Any]],
        stats: Dict[str, Any],
        quality: Dict[str, Any],
        classifications: List[Dict[str, Any]],
        grading: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Export the complete analysis to JSON."""
        data = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "metadata": metadata or {},
            "measurements": measurements,
            "statistics": stats,
            "quality": quality,
            "classifications": classifications,
            "grading": grading,
        }
        path = os.path.join(config.JSON_DIR, f"{self.run_id}_report.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    # ------------------------------------------------------------------
    # Annotated image
    # ------------------------------------------------------------------
    def annotate_image(
        self,
        image: np.ndarray,
        grains: List[Dict[str, Any]],
        measurements: List[Dict[str, Any]],
        classifications: List[Dict[str, Any]],
    ) -> str:
        """Draw contours, labels, and measurements on the image."""
        annotated = image.copy()

        # Category → colour
        cat_colors = {
            "whole_grain": (0, 255, 0),
            "broken_grain": (0, 0, 255),
            "long_grain": (255, 0, 0),
            "medium_grain": (0, 165, 255),
            "short_grain": (255, 255, 0),
            "oversized_grain": (255, 0, 255),
            "undersized_grain": (128, 0, 128),
            "abnormal_grain": (0, 0, 128),
        }

        for i, (grain, meas, cls) in enumerate(
            zip(grains, measurements, classifications)
        ):
            contour = grain["contour"]
            primary = cls.get("primary_category", "whole_grain")
            color = cat_colors.get(primary, (0, 255, 0))

            # Draw contour
            cv2.drawContours(annotated, [contour], -1, color, 2)

            # Draw label number
            cx, cy = int(meas["centroid_x"]), int(meas["centroid_y"])
            cv2.putText(annotated, str(meas["label"]), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Draw length × width near grain
            length = meas.get("length_px", 0)
            width = meas.get("width_px", 0)
            if meas.get("length_mm") is not None:
                text = f"{meas['length_mm']:.1f}x{meas['width_mm']:.1f}mm"
            else:
                text = f"{length:.0f}x{width:.0f}px"
            cv2.putText(annotated, text, (cx - 20, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        path = os.path.join(config.IMAGE_DIR, f"{self.run_id}_annotated.jpg")
        cv2.imwrite(path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return path

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def generate_plots(
        self, measurements: List[Dict[str, Any]], stats: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate all statistical plots and return their paths."""
        plots = {}
        df = pd.DataFrame(measurements)

        # --- Histograms for key metrics ---
        for key in ["length_px", "width_px", "area_px", "perimeter_px",
                     "aspect_ratio", "circularity"]:
            if key not in df.columns:
                continue
            path = self._plot_histogram(df, key)
            plots[f"hist_{key}"] = path

        # --- Box plots ---
        box_keys = [k for k in ["length_px", "width_px", "area_px",
                                "perimeter_px", "equivalent_diameter_px"] if k in df.columns]
        if box_keys:
            path = self._plot_boxplots(df, box_keys)
            plots["boxplots"] = path

        # --- Scatter: length vs width ---
        if "length_px" in df.columns and "width_px" in df.columns:
            path = self._plot_scatter(df, "length_px", "width_px")
            plots["scatter_length_width"] = path

        # --- Correlation heatmap ---
        corr_keys = [k for k in [
            "area_px", "perimeter_px", "length_px", "width_px",
            "aspect_ratio", "circularity", "solidity", "eccentricity"
        ] if k in df.columns]
        if len(corr_keys) >= 2:
            path = self._plot_correlation(df, corr_keys)
            plots["correlation"] = path

        # --- KDE for length ---
        if "length_px" in df.columns and len(df) > 5:
            path = self._plot_kde(df, "length_px")
            plots["kde_length"] = path

        return plots

    def _plot_histogram(self, df: pd.DataFrame, key: str) -> str:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df[key].dropna(), bins=30, edgecolor="black", alpha=0.7, color="#4CAF50")
        ax.set_title(f"Distribution of {key}", fontsize=14, fontweight="bold")
        ax.set_xlabel(key)
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(config.PLOT_DIR, f"{self.run_id}_hist_{key}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_boxplots(self, df: pd.DataFrame, keys: List[str]) -> str:
        fig, ax = plt.subplots(figsize=(10, 6))
        data = [df[k].dropna().values for k in keys]
        bp = ax.boxplot(data, tick_labels=keys, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#4CAF50")
            patch.set_alpha(0.6)
        ax.set_title("Box Plots of Key Metrics", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        path = os.path.join(config.PLOT_DIR, f"{self.run_id}_boxplots.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_scatter(self, df: pd.DataFrame, x: str, y: str) -> str:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df[x], df[y], alpha=0.6, s=30, c="#4CAF50", edgecolors="black", linewidths=0.5)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{y} vs {x}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        # Trend line
        if len(df) > 2:
            z = np.polyfit(df[x], df[y], 1)
            p = np.poly1d(z)
            x_sorted = np.sort(df[x].values)
            ax.plot(x_sorted, p(x_sorted), "r--", alpha=0.7, label=f"y={z[0]:.4f}x+{z[1]:.2f}")
            ax.legend()
        fig.tight_layout()
        path = os.path.join(config.PLOT_DIR, f"{self.run_id}_scatter_{x}_{y}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_correlation(self, df: pd.DataFrame, keys: List[str]) -> str:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df[keys].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0,
                    square=True, linewidths=0.5, ax=ax)
        ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(config.PLOT_DIR, f"{self.run_id}_correlation.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_kde(self, df: pd.DataFrame, key: str) -> str:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(df[key].dropna(), fill=True, ax=ax, color="#4CAF50")
        ax.set_title(f"Kernel Density Estimation – {key}", fontsize=14, fontweight="bold")
        ax.set_xlabel(key)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(config.PLOT_DIR, f"{self.run_id}_kde_{key}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Quality summary table image
    # ------------------------------------------------------------------
    def generate_quality_table(
        self, quality: Dict[str, Any], grading: Dict[str, Any]
    ) -> str:
        """Render a publication-quality summary table as an image."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")

        rows = [
            ["Total Grains", f"{quality.get('total_grains', 0)}"],
            ["Broken Grains", f"{quality.get('broken_pct', 0):.2f} %"],
            ["Abnormal Grains", f"{quality.get('abnormal_pct', 0):.2f} %"],
            ["Uniformity Index", f"{quality.get('uniformity_index', 0):.2f} %"],
            ["CV of Length", f"{quality.get('cv_length', 0):.2f} %"],
            ["Avg Aspect Ratio", f"{quality.get('average_aspect_ratio', 0):.2f}"],
            ["Shape Consistency", f"{quality.get('shape_consistency', 0):.2f} %"],
            ["Grain Density", f"{quality.get('grain_density', 0):.4f}"],
            ["Final Grade", grading.get("grade", "N/A")],
        ]

        table = ax.table(
            cellText=rows,
            colLabels=["Metric", "Value"],
            cellLoc="center",
            loc="center",
            colColours=["#4CAF50", "#4CAF50"],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.8)
        # Style header
        for i in range(2):
            table[0, i].set_text_props(color="white", fontweight="bold")
        # Color grade row
        grade_colors = {
            "Premium": "#2E7D32", "Grade A": "#66BB6A",
            "Grade B": "#FFA726", "Grade C": "#FF7043", "Reject": "#EF5350",
        }
        gc = grade_colors.get(grading.get("grade", ""), "#4CAF50")
        table[len(rows), 1].set_facecolor(gc)
        table[len(rows), 1].set_text_props(color="white", fontweight="bold")

        ax.set_title("Quality Summary", fontsize=16, fontweight="bold", pad=20)
        fig.tight_layout()
        path = os.path.join(config.PLOT_DIR, f"{self.run_id}_quality_table.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
