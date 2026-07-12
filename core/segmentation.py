"""
Phase 4 – Classical Image Segmentation

Pure OpenCV / scikit-image segmentation — no deep learning.
  • Connected-component labelling
  • Contour extraction
  • Watershed segmentation for touching grains
  • Distance-transform based separation
  • Adaptive area filtering based on image resolution

Each individual grain is isolated as a separate labelled object.
"""

import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed as sk_watershed
from typing import Dict, Any, List, Tuple, Optional

import config


class Segmenter:
    """Classical segmentation pipeline producing labelled grain masks."""

    def __init__(self):
        self.min_area = config.MIN_GRAIN_AREA_PX
        self.max_area = config.MAX_GRAIN_AREA_PX
        self.watershed_threshold = config.WATERSHED_DISTANCE_THRESHOLD

    def _adaptive_area_limits(self, binary: np.ndarray) -> Tuple[int, int]:
        """
        Compute adaptive area limits based on image resolution.
        For high-res images (e.g. 4080×3060), the max area must be
        larger because individual grains occupy more pixels.

        Also uses a heuristic: max_area should not exceed 10% of total
        image area (a single grain can't be > 10% of the frame), and
        min_area should be at least 0.0004% of image area.
        """
        total_pixels = binary.shape[0] * binary.shape[1]

        # Scale configured limits relative to a 1920×1080 reference
        ref_pixels = 1920 * 1080
        scale = total_pixels / ref_pixels

        min_area = max(self.min_area, int(30 * scale))
        max_area = max(self.max_area, int(500000 * scale))

        # Hard cap: single grain can't be more than 10% of image
        max_area = min(max_area, int(total_pixels * 0.10))

        return min_area, max_area

    # ------------------------------------------------------------------
    # Connected components
    # ------------------------------------------------------------------
    def connected_components(self, binary: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
        """Label connected components (8-connectivity)."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8
        )
        return labels, num_labels, stats, centroids

    # ------------------------------------------------------------------
    # Filter by area
    # ------------------------------------------------------------------
    def filter_by_area(
        self, labels: np.ndarray, num_labels: int, stats: np.ndarray,
        min_area: Optional[int] = None, max_area: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, List[int]]:
        """Remove components outside the area range."""
        if min_area is None:
            min_area = self.min_area
        if max_area is None:
            max_area = self.max_area

        valid_ids = []
        for i in range(1, num_labels):  # skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                valid_ids.append(i)

        # Build a remapped label image
        filtered = np.zeros_like(labels)
        for new_id, old_id in enumerate(valid_ids, start=1):
            filtered[labels == old_id] = new_id

        return filtered, len(valid_ids), valid_ids

    # ------------------------------------------------------------------
    # Watershed for touching grains
    # ------------------------------------------------------------------
    def watershed_separation(
        self, binary: np.ndarray, min_area: Optional[int] = None, max_area: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Separate touching grains using distance transform + watershed.
        Returns a labelled image and the number of labels.
        """
        if min_area is None:
            min_area = self.min_area
        if max_area is None:
            max_area = self.max_area

        # Distance transform (with optional Canny Edge snapping)
        if getattr(config, "THRESHOLD_METHOD", "otsu") == "canny_watershed":
            edges = cv2.Canny(binary, 50, 150)
            binary_no_edges = cv2.bitwise_and(binary, cv2.bitwise_not(edges))
            dist = cv2.distanceTransform(binary_no_edges, cv2.DIST_L2, 5)
        else:
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # Adaptive min_distance based on expected grain size
        # Larger images → larger grains → larger min_distance
        total_px = binary.shape[0] * binary.shape[1]
        scale = max(1.0, (total_px / (1920 * 1080)) ** 0.5)
        min_dist = max(10, int(10 * scale))

        # Find peaks (seed markers)
        coords = peak_local_max(
            dist, min_distance=min_dist,
            threshold_abs=self.watershed_threshold * dist.max()
        )
        if len(coords) == 0:
            # Fallback: use connected components
            num_labels, labels, _, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
            return labels, num_labels

        markers = np.zeros(binary.shape, dtype=np.int32)
        for i, (y, x) in enumerate(coords, start=1):
            markers[y, x] = i

        # Watershed from skimage (works on 2-D labels)
        labels = sk_watershed(-dist, markers, mask=binary)

        # Filter by area
        num_labels = labels.max()
        valid = []
        for i in range(1, num_labels + 1):
            area = np.sum(labels == i)
            if min_area <= area <= max_area:
                valid.append(i)

        # Remap
        filtered = np.zeros_like(labels)
        for new_id, old_id in enumerate(valid, start=1):
            filtered[labels == old_id] = new_id

        return filtered, len(valid)

    # ------------------------------------------------------------------
    # Contour extraction
    # ------------------------------------------------------------------
    def extract_contours(self, labels: np.ndarray) -> List[Dict[str, Any]]:
        """Extract per-grain contour and mask from a labelled image."""
        grains = []
        num_labels = labels.max()

        for i in range(1, num_labels + 1):
            mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            grains.append({
                "label": i,
                "contour": contour,
                "mask": mask,
            })

        return grains

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def segment(self, binary: np.ndarray, use_watershed: bool = True) -> Dict[str, Any]:
        """
        Complete segmentation pipeline.
        Returns labelled image, grain contours, and debug info.
        """
        # Compute adaptive area limits for this image's resolution
        min_area, max_area = self._adaptive_area_limits(binary)

        # Initial connected components
        labels_raw, num_raw, stats, centroids = self.connected_components(binary)

        # Filter by area (adaptive)
        labels_filtered, num_filtered, valid_ids = self.filter_by_area(
            labels_raw, num_raw, stats, min_area, max_area
        )

        # Watershed separation if requested
        if use_watershed and num_filtered > 0:
            # Reconstruct binary from filtered labels
            filtered_binary = (labels_filtered > 0).astype(np.uint8) * 255
            labels_ws, num_ws = self.watershed_separation(
                filtered_binary, min_area, max_area
            )
            if num_ws > 0:
                labels_final = labels_ws
            else:
                labels_final = labels_filtered
        else:
            labels_final = labels_filtered

        # Extract contours
        grains = self.extract_contours(labels_final)

        return {
            "labels": labels_final,
            "num_grains": len(grains),
            "grains": grains,
            "raw_components": num_raw - 1,  # exclude background
            "filtered_components": num_filtered,
        }
