"""
Phase 4b – Deep Learning / ML Segmentation
Using YOLOv8 for instance segmentation.
This acts as an alternative to `segmentation.py` and `preprocessing.py` when deep learning is toggled ON.
"""
import cv2
import numpy as np
import os
import logging
from typing import Dict, Any, List
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

import config

logger = logging.getLogger(__name__)

class ML_Segmenter:
    """Deep learning segmentation pipeline using YOLOv8 Instance Segmentation."""

    def __init__(self):
        self.model_path = config.YOLO_MODEL_PATH
        self.model = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        if not HAS_ULTRALYTICS:
            logger.warning("Ultralytics package not installed. YOLO segmentation unavailable.")
            return

        if not os.path.exists(self.model_path):
            logger.warning(f"YOLO model not found at {self.model_path}. Please train and place model here.")
            return

        try:
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            logger.info("YOLOv8-Seg model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")

    def segment(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run YOLOv8 instance segmentation.
        Returns the exact same dictionary format as classical Segmenter so it's a drop-in replacement.
        """
        h, w = image.shape[:2]
        labels = np.zeros((h, w), dtype=np.int32)
        grains = []

        if not self.is_loaded:
            logger.warning("Cannot segment with ML: model is not loaded or missing.")
            return {
                "labels": labels,
                "num_grains": 0,
                "grains": grains,
                "raw_components": 0,
                "filtered_components": 0,
            }

        # Inference
        results = self.model(image, verbose=False)
        result = results[0]

        if result.masks is None:
            return {
                "labels": labels,
                "num_grains": 0,
                "grains": grains,
                "raw_components": 0,
                "filtered_components": 0,
            }

        # Extract masks and contours
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.data.cpu().numpy()

        # Resize masks to original image dimensions if necessary
        if masks.shape[1:] != (h, w):
            masks = [cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) for m in masks]
        
        num_grains = len(masks)
        
        for i, mask_arr in enumerate(masks, start=1):
            mask_binary = (mask_arr > 0.5).astype(np.uint8) * 255
            labels[mask_binary == 255] = i

            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            contour = max(contours, key=cv2.contourArea)
            grains.append({
                "label": i,
                "contour": contour,
                "mask": mask_binary,
            })

        return {
            "labels": labels,
            "num_grains": len(grains),
            "grains": grains,
            "raw_components": num_grains,
            "filtered_components": len(grains),
            "steps": {"ml_mask": (labels > 0).astype(np.uint8) * 255} # Dummy for UI visualization
        }
