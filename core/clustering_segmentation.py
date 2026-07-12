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
    """Unsupervised segmentation using Fast K-Means and Watershed."""

    def __init__(self):
        pass

    def segment(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run fast K-Means clustering in HSV space to find foreground/background,
        then apply Watershed to separate touching grains.
        Returns the exact dictionary format as classical Segmenter.
        """
        h_full, w_full = image.shape[:2]
        
        # 1. Fast K-Means on downscaled image
        # Downscale to max 640px for speed
        max_dim = 640
        scale = 1.0
        if max(h_full, w_full) > max_dim:
            scale = max_dim / max(h_full, w_full)
            small_img = cv2.resize(image, (int(w_full * scale), int(h_full * scale)))
        else:
            small_img = image.copy()
            
        h_small, w_small = small_img.shape[:2]
        
        # Convert to HSV and reshape
        hsv_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
        Z = hsv_small.reshape((-1, 3))
        Z = np.float32(Z)

        # Run K-Means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        _, labels_km, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Determine which cluster is rice.
        # Background is usually the larger cluster, or the one that dominates the edges.
        labels_small = labels_km.reshape((h_small, w_small))
        edge_pixels = np.concatenate([
            labels_small[0, :], labels_small[-1, :],
            labels_small[:, 0], labels_small[:, -1]
        ])
        # Background is the most frequent label on the edges
        bg_label = np.bincount(edge_pixels).argmax()
        fg_cluster = 1 - bg_label
        
        # Generate full-resolution mask by applying the centers
        hsv_full = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        dist_0 = np.linalg.norm(hsv_full - centers[0], axis=2)
        dist_1 = np.linalg.norm(hsv_full - centers[1], axis=2)
        
        if fg_cluster == 0:
            binary_mask = (dist_0 < dist_1).astype(np.uint8) * 255
        else:
            binary_mask = (dist_1 < dist_0).astype(np.uint8) * 255
            
        # If binary mask is empty or inverted, fallback to brightness
        if np.count_nonzero(binary_mask) == 0 or np.count_nonzero(binary_mask) > (h_full * w_full * 0.9):
            v_mean_0 = centers[0][2]
            v_mean_1 = centers[1][2]
            fg_cluster = 0 if v_mean_0 > v_mean_1 else 1
            if fg_cluster == 0:
                binary_mask = (dist_0 < dist_1).astype(np.uint8) * 255
            else:
                binary_mask = (dist_1 < dist_0).astype(np.uint8) * 255
            
        # 2. Morphological Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 3. Distance Transform & Watershed
        # Sure background area
        sure_bg = cv2.dilate(binary_mask, kernel, iterations=3)
        
        # Finding sure foreground area using distance transform
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Finding unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        
        # 4. Extract distinct grains
        labels_img = np.zeros((h_full, w_full), dtype=np.int32)
        grains = []
        valid_grain_idx = 1
        
        # Get unique markers (excluding background 1 and boundaries -1)
        unique_markers = np.unique(markers)
        
        for m in unique_markers:
            if m == 1 or m == -1:
                continue
                
            # Create a mask for this specific grain
            grain_mask = np.zeros((h_full, w_full), dtype=np.uint8)
            grain_mask[markers == m] = 255
            
            # Find contours
            contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < config.MIN_GRAIN_AREA_PX or area > config.MAX_GRAIN_AREA_PX:
                continue
                
            labels_img[markers == m] = valid_grain_idx
            grains.append({
                "label": valid_grain_idx,
                "contour": contour,
                "mask": grain_mask,
            })
            valid_grain_idx += 1

        dt_vis = np.zeros_like(dist_transform, dtype=np.uint8)
        if dist_transform.max() > 0:
            dt_vis = (dist_transform / dist_transform.max() * 255).astype(np.uint8)
            
        return {
            "labels": labels_img,
            "num_grains": len(grains),
            "grains": grains,
            "raw_components": len(unique_markers) - 2,
            "filtered_components": len(grains),
            "steps": {"kmeans_mask": binary_mask, "distance_transform": dt_vis}
        }
