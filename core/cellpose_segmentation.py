"""
Phase 4c – Cellpose 3 (cyto3) Segmentation - Hardened for Production

Key improvements over the initial version:
  • Graceful initialization — server never crashes if Cellpose/PyTorch unavailable
  • Rice-grain-optimized diameter estimation from image resolution
  • Configurable flow_threshold and cellprob_threshold from config.py
  • Performance timing and structured logging
  • Post-processing: small-fragment removal with adaptive area limits
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import time

import config

logger = logging.getLogger(__name__)


class CellposeSegmenter:
    """Cellpose 3 (cyto3) segmentation hardened for production use."""

    def __init__(self,
                 model_type: str = "cyto3",
                 device: str = None,
                 flow_threshold: float = 0.4,
                 cellprob_threshold: float = 0.0,
                 diameter_estimate: int = 0):
        """
        Initialize Cellpose segmenter.

        Args:
            model_type: Cellpose model type ('cyto3', 'cyto2', 'nuclei')
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
            flow_threshold: Flow error threshold (lower = stricter boundaries)
            cellprob_threshold: Cell probability threshold (higher = fewer but more confident detections)
            diameter_estimate: Expected cell diameter in pixels; 0 = auto-detect
        """
        self.model_type = model_type
        self.device = device
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.diameter_estimate = diameter_estimate
        self._model = None
        self._init_model()

    def _init_model(self):
        """Initialize the Cellpose model with graceful error handling."""
        try:
            from cellpose import models
            import torch

            use_gpu = False
            device_info = "CPU"
            if self.device == 'cuda' or (self.device is None and torch.cuda.is_available()):
                use_gpu = True
                device_info = "CUDA GPU"
            elif self.device == 'mps' or (self.device is None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                use_gpu = True
                device_info = "Apple MPS"

            t0 = time.time()
            self._model = models.CellposeModel(gpu=use_gpu, model_type=self.model_type)
            elapsed = time.time() - t0
            logger.info(
                f"Cellpose {self.model_type} model loaded on {device_info} "
                f"in {elapsed:.2f}s (flow_thresh={self.flow_threshold}, "
                f"cellprob_thresh={self.cellprob_threshold})"
            )
        except ImportError as e:
            logger.error(f"Cellpose or PyTorch not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load Cellpose model: {e}")
            raise

    def _estimate_diameter(self, image: np.ndarray) -> Optional[float]:
        """
        Estimate rice grain diameter from image resolution.

        Rice grains are typically 6-7mm long, 2-2.5mm wide.
        At typical smartphone macro distances (~15cm), grains are roughly
        40-80 pixels wide depending on resolution.

        Returns None to let Cellpose auto-detect, or a float diameter.
        """
        if self.diameter_estimate > 0:
            return float(self.diameter_estimate)

        h, w = image.shape[:2]
        total_pixels = h * w

        # Heuristic: at 12MP (4000x3000), rice grains are ~60px diameter
        # Scale proportionally for other resolutions
        ref_pixels = 4000 * 3000
        scale = (total_pixels / ref_pixels) ** 0.5
        estimated = int(60 * scale)

        # Clamp to reasonable range (20-200 px)
        estimated = max(20, min(200, estimated))

        logger.debug(f"Auto-estimated grain diameter: {estimated}px for {w}x{h} image")
        return float(estimated)

    def segment(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run Cellpose segmentation on a rice grain image.

        Args:
            image: BGR input image (numpy array)

        Returns:
            Dictionary with labels, grains list, binary mask, steps, and counts
        """
        h, w = image.shape[:2]
        t0 = time.time()
        logger.info(f"Starting Cellpose segmentation on {w}x{h} image")

        # Convert to grayscale for Cellpose
        if len(image.shape) == 3 and image.shape[2] == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img = image

        # Estimate diameter
        diameter = self._estimate_diameter(image)

        # Run Cellpose model
        try:
            result = self._model.eval(
                img,
                diameter=diameter,
                channels=[0, 0],
                flow_threshold=self.flow_threshold,
                cellprob_threshold=self.cellprob_threshold,
            )
        except TypeError:
            # Fallback for different Cellpose API versions
            result = self._model.eval(
                img,
                diameter=diameter,
                flow_threshold=self.flow_threshold,
                cellprob_threshold=self.cellprob_threshold,
            )

        masks = result[0]
        t_model = time.time() - t0
        logger.info(f"Cellpose model inference: {t_model:.2f}s")

        # Post-processing: remove small fragments
        t1 = time.time()
        grains, final_labels, raw_count = self._postprocess(masks, h, w)
        t_post = time.time() - t1

        total_elapsed = time.time() - t0
        logger.info(
            f"Cellpose segmentation complete: {len(grains)} grains "
            f"(raw: {raw_count}, model: {t_model:.2f}s, post: {t_post:.2f}s, "
            f"total: {total_elapsed:.2f}s)"
        )

        binary = (final_labels > 0).astype(np.uint8)

        return {
            "labels": final_labels,
            "num_grains": len(grains),
            "grains": grains,
            "binary": binary,
            "raw_components": raw_count,
            "filtered_components": len(grains),
            "steps": {
                "original": image,
                "cellpose_binary": binary * 255,
                "cellpose_labels": self._labels_to_rgb(final_labels),
            }
        }

    def _postprocess(self, masks: np.ndarray, h: int, w: int):
        """
        Post-process Cellpose masks: filter by area, extract contours.

        Returns:
            (grains_list, final_label_image, raw_component_count)
        """
        from skimage.measure import regionprops, label as relabel

        raw_components = int(masks.max())

        # Adaptive area limits based on image resolution
        total_pixels = h * w
        ref_pixels = 1920 * 1080
        scale = total_pixels / ref_pixels
        min_area = max(config.MIN_GRAIN_AREA_PX, int(30 * scale))
        max_area = max(config.MAX_GRAIN_AREA_PX, int(500000 * scale))
        # Cap: single grain can't be more than 10% of image
        max_area = min(max_area, int(total_pixels * 0.10))

        # Remove small/large fragments
        clean_masks = masks.copy()
        for r in regionprops(masks):
            if r.area < min_area or r.area > max_area:
                clean_masks[clean_masks == r.label] = 0

        # Relabel to sequential IDs
        final_labels = relabel(clean_masks)

        # Extract grain data
        grains = []
        for r in regionprops(final_labels):
            grain_mask = (final_labels == r.label).astype(np.uint8) * 255

            # Find contour
            contours, _ = cv2.findContours(
                grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                contour = max(contours, key=cv2.contourArea)
            else:
                continue  # Skip grains without valid contours

            if len(contour) < 5:
                continue  # Need at least 5 points for ellipse fitting

            x, y, w_g, h_g = cv2.boundingRect(contour)

            grains.append({
                "label": r.label,
                "contour": contour,
                "mask": grain_mask,
                "bbox": (x, y, w_g, h_g),
                "centroid": (r.centroid[1], r.centroid[0]),  # (x, y)
                "area": r.area,
            })

        return grains, final_labels, raw_components

    def _labels_to_rgb(self, labels: np.ndarray) -> np.ndarray:
        """Convert label mask to RGB visualization with distinct colors."""
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]

        if len(unique_labels) == 0:
            return np.zeros((*labels.shape, 3), dtype=np.uint8)

        hsv_colors = np.zeros((len(unique_labels), 3), dtype=np.uint8)
        hsv_colors[:, 0] = np.linspace(0, 179, len(unique_labels), dtype=np.uint8)
        hsv_colors[:, 1] = 255
        hsv_colors[:, 2] = 255

        rgb_colors = cv2.cvtColor(
            hsv_colors.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR
        ).reshape(-1, 3)

        rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
        for i, label in enumerate(unique_labels):
            rgb[labels == label] = rgb_colors[i]

        return rgb


def create_cellpose_segmenter(**kwargs) -> Optional[CellposeSegmenter]:
    """
    Factory function to create a CellposeSegmenter.
    Returns None (instead of crashing) if Cellpose cannot be loaded.
    """
    try:
        return CellposeSegmenter(
            model_type=kwargs.get("model_type", config.CELLPOSE_MODEL),
            device=kwargs.get(
                "device",
                config.CELLPOSE_DEVICE if config.CELLPOSE_DEVICE != "auto" else None,
            ),
            flow_threshold=kwargs.get("flow_threshold", config.CELLPOSE_FLOW_THRESHOLD),
            cellprob_threshold=kwargs.get("cellprob_threshold", config.CELLPOSE_CELLPROB_THRESHOLD),
            diameter_estimate=kwargs.get("diameter_estimate", config.CELLPOSE_DIAMETER_ESTIMATE),
        )
    except Exception as e:
        logger.warning(
            f"Cellpose segmenter unavailable — falling back to classical methods. "
            f"Reason: {e}"
        )
        return None