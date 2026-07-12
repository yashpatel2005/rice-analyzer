"""
Phase 4c – Cellpose 3 (cyto3) Segmentation - Masks/Regions only
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

import config

logger = logging.getLogger(__name__)


class CellposeSegmenter:
    """Cellpose 3 (cyto3) segmentation."""

    def __init__(self, 
                 model_type: str = "cyto3",
                 device: str = None):
        """
        Initialize Cellpose segmenter.
        """
        self.model_type = model_type
        self.device = device
        self._model = None
        self._init_model()

    def _init_model(self):
        """Initialize the Cellpose model."""
        try:
            from cellpose import models
            import torch
            
            use_gpu = False
            if self.device == 'cuda' or (self.device is None and torch.cuda.is_available()):
                use_gpu = True
            elif self.device == 'mps' or (self.device is None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                use_gpu = True
                
            self._model = models.CellposeModel(gpu=use_gpu, model_type=self.model_type)
            logger.info(f"Cellpose {self.model_type} model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Cellpose model: {e}")
            raise

    def segment(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run Cellpose segmentation.
        """
        h, w = image.shape[:2]
        logger.info(f"Starting Cellpose segmentation on {w}x{h} image")
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img = image

        try:
            result = self._model.eval(
                img, diameter=None, channels=[0, 0],
                flow_threshold=0.4, cellprob_threshold=0.5,
            )
        except TypeError:
            result = self._model.eval(
                img, diameter=None,
                flow_threshold=0.4, cellprob_threshold=0.5,
            )

        masks = result[0]
        
        # Adaptive area limits based on image resolution
        total_pixels = h * w
        ref_pixels = 1920 * 1080
        scale = total_pixels / ref_pixels
        min_area = max(config.MIN_GRAIN_AREA_PX, int(30 * scale))
        
        from skimage.measure import regionprops, label as relabel
        
        raw_components = masks.max()
        
        clean_masks = masks.copy()
        for r in regionprops(masks):
            if r.area < min_area:
                clean_masks[clean_masks == r.label] = 0
                
        final_labels = relabel(clean_masks)
        
        grains = []
        for r in regionprops(final_labels):
            grain_mask = (final_labels == r.label).astype(np.uint8) * 255
            
            # Find contour
            contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
            else:
                contour = np.array([])
                
            x, y, w_g, h_g = cv2.boundingRect(contour) if len(contour) > 0 else (0,0,0,0)
            
            grains.append({
                "label": r.label,
                "contour": contour,
                "mask": grain_mask,
                "bbox": (x, y, w_g, h_g),
                "centroid": (r.centroid[1], r.centroid[0]), # (x, y)
                "area": r.area,
            })
            
        logger.info(f"Cellpose segmentation complete: {len(grains)} grains detected")
        
        binary = (final_labels > 0).astype(np.uint8)
        
        return {
            "labels": final_labels,
            "num_grains": len(grains),
            "grains": grains,
            "binary": binary,
            "raw_components": raw_components,
            "filtered_components": len(grains),
            "steps": {
                "original": image,
                "binary": binary * 255,
                "labels": self._labels_to_rgb(final_labels)
            }
        }

    def _labels_to_rgb(self, labels: np.ndarray) -> np.ndarray:
        """Convert label mask to RGB visualization."""
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]
        
        if len(unique_labels) == 0:
            return np.zeros((*labels.shape, 3), dtype=np.uint8)
            
        hsv_colors = np.zeros((len(unique_labels), 3), dtype=np.uint8)
        hsv_colors[:, 0] = np.linspace(0, 179, len(unique_labels), dtype=np.uint8)
        hsv_colors[:, 1] = 255
        hsv_colors[:, 2] = 255
        
        rgb_colors = cv2.cvtColor(hsv_colors.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)
        
        rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
        for i, label in enumerate(unique_labels):
            rgb[labels == label] = rgb_colors[i]
            
        return rgb


def create_cellpose_segmenter(**kwargs) -> CellposeSegmenter:
    """Factory function to create a CellposeSegmenter."""
    return CellposeSegmenter(
        model_type=kwargs.get("model_type", config.CELLPOSE_MODEL),
        device=kwargs.get("device", config.CELLPOSE_DEVICE if config.CELLPOSE_DEVICE != "auto" else None)
    )