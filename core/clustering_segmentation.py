"""
Phase 4b – Advanced Clustering Segmentation (No YOLO)
Using K-Means + GrabCut to segment grains mathematically.
"""
import cv2
import numpy as np
import logging
from typing import Dict, Any, List

import config

logger = logging.getLogger(__name__)

class ClusteringSegmenter:
    """Unsupervised segmentation using K-Means and GrabCut."""

    def __init__(self):
        pass

    def segment(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run K-Means clustering in HSV space followed by GrabCut.
        Returns the exact dictionary format as classical Segmenter.
        """
        h, w = image.shape[:2]
        labels_img = np.zeros((h, w), dtype=np.int32)
        grains = []

        # 1. Convert to HSV for better color discrimination
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Reshape for K-Means
        Z = hsv.reshape((-1, 3))
        Z = np.float32(Z)

        # 2. Run K-Means with K=2 (Foreground/Background)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        ret, labels_km, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Reshape labels back to image size
        kmeans_mask = labels_km.reshape((h, w)).astype(np.uint8)

        # Figure out which cluster is background (assumed to touch edges more)
        edge_pixels = np.concatenate([
            kmeans_mask[0, :], kmeans_mask[-1, :],
            kmeans_mask[:, 0], kmeans_mask[:, -1]
        ])
        bg_label = np.bincount(edge_pixels).argmax()
        fg_label = 1 - bg_label

        # Create binary mask from K-Means
        binary_mask = (kmeans_mask == fg_label).astype(np.uint8) * 255

        # Refine with morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Extract connected components from the refined K-Means mask
        num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        # 3. Filter and run GrabCut on each component to refine boundaries
        valid_grain_idx = 1
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < config.MIN_GRAIN_AREA_PX or area > config.MAX_GRAIN_AREA_PX:
                continue

            # Get bounding box for this component
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w_comp = stats[i, cv2.CC_STAT_WIDTH]
            h_comp = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Add padding to bounding box
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + w_comp + pad)
            y2 = min(h, y + h_comp + pad)

            rect = (x1, y1, x2 - x1, y2 - y1)

            # Prepare GrabCut arrays
            gc_mask = np.zeros((h, w), np.uint8)
            
            # Set the component pixels as Probable Foreground (PR_FGD = 3)
            # and the rest of the rect as Probable Background (PR_BGD = 2)
            gc_mask[y1:y2, x1:x2] = cv2.GC_PR_BGD
            gc_mask[labels_cc == i] = cv2.GC_PR_FGD
            
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            try:
                # Run GrabCut
                cv2.grabCut(image, gc_mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
                
                # Extract finalized mask for this grain
                grain_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
                
                # Only keep pixels within the ROI to avoid grabbing global noise
                roi_mask = np.zeros((h, w), np.uint8)
                roi_mask[y1:y2, x1:x2] = grain_mask[y1:y2, x1:x2]
                
                # Find contours of the refined grain
                contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                    
                contour = max(contours, key=cv2.contourArea)
                
                # Update output arrays
                labels_img[roi_mask == 255] = valid_grain_idx
                grains.append({
                    "label": valid_grain_idx,
                    "contour": contour,
                    "mask": roi_mask,
                })
                valid_grain_idx += 1
            except Exception as e:
                logger.error(f"GrabCut failed on component {i}: {e}")
                continue

        return {
            "labels": labels_img,
            "num_grains": len(grains),
            "grains": grains,
            "raw_components": num_labels - 1,
            "filtered_components": len(grains),
            "steps": {"kmeans_mask": binary_mask}
        }
