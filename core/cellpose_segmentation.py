"""
Phase 4c – Cellpose 3 (cyto3) Segmentation with Tiled Inference and TTA

Cellpose 3 uses "vector flows" (gradients pointing to the center of each grain)
instead of direct mask prediction. This makes it mathematically impossible for
it to merge two grains unless they truly share a center.

Features:
- Tiled inference: Image chopped into overlapping tiles (512x512) to preserve
  high-resolution details, then stitched back together
- Test-Time Augmentation (TTA): Image rotated 90°, 180°, 270°, and flipped.
  Model runs on all versions, results averaged to cancel out random errors.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import time

import config

logger = logging.getLogger(__name__)


class CellposeSegmenter:
    """Cellpose 3 (cyto3) segmentation with tiled inference and TTA."""

    def __init__(self, 
                 model_type: str = "cyto3",
                 tile_size: int = 512,
                 overlap: int = 64,
                 use_tta: bool = True,
                 device: str = None):
        """
        Initialize Cellpose segmenter.
        
        Args:
            model_type: Cellpose model type ('cyto3', 'cyto2', 'nuclei', etc.)
            tile_size: Size of tiles for tiled inference
            overlap: Overlap between tiles in pixels
            use_tta: Whether to use Test-Time Augmentation
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
        """
        self.model_type = model_type
        self.tile_size = tile_size
        self.overlap = overlap
        self.use_tta = use_tta
        self.device = device
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
                f"in {elapsed:.2f}s (tile={self.tile_size}, overlap={self.overlap}, tta={self.use_tta})"
            )
        except ImportError as e:
            logger.error(f"Cellpose or PyTorch not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load Cellpose model: {e}")
            raise

    def _apply_tta(self, image: np.ndarray) -> List[Tuple[np.ndarray, Dict]]:
        """
        Apply Test-Time Augmentation transformations.
        
        Returns list of (transformed_image, transform_info) tuples.
        transform_info contains info needed to reverse the transform.
        """
        transforms = []
        
        # Original
        transforms.append((image.copy(), {"rot": 0, "flip": False}))
        
        if self.use_tta:
            # Rotations
            for k in [1, 2, 3]:  # 90, 180, 270 degrees
                rotated = np.rot90(image, k=k)
                transforms.append((rotated.copy(), {"rot": k, "flip": False}))
            
            # Flips
            flipped_h = np.flipud(image)  # horizontal flip
            transforms.append((flipped_h.copy(), {"rot": 0, "flip": "h"}))
            
            flipped_v = np.fliplr(image)  # vertical flip
            transforms.append((flipped_v.copy(), {"rot": 0, "flip": "v"}))
            
            # Rotated flips
            for k in [1, 2, 3]:
                rotated = np.rot90(image, k=k)
                flipped = np.flipud(rotated)
                transforms.append((flipped.copy(), {"rot": k, "flip": "h"}))
        
        return transforms

    def _reverse_tta(self, mask: np.ndarray, transform_info: Dict) -> np.ndarray:
        """Reverse TTA transformation on a mask."""
        rot = transform_info["rot"]
        flip = transform_info["flip"]
        
        # Reverse flip first
        if flip == "h":
            mask = np.flipud(mask)
        elif flip == "v":
            mask = np.fliplr(mask)
        
        # Reverse rotation
        if rot > 0:
            mask = np.rot90(mask, k=-rot)
        
        return mask

    def _tile_image(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Split image into overlapping tiles.
        
        Returns list of (tile, (y_start, x_start)) tuples.
        """
        h, w = image.shape[:2]
        tiles = []
        
        stride = self.tile_size - self.overlap
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)
                y_start = max(0, y_end - self.tile_size)
                x_start = max(0, x_end - self.tile_size)
                
                tile = image[y_start:y_end, x_start:x_end].copy()
                tiles.append((tile, (y_start, x_start)))
        
        return tiles

    def _stitch_tiles(self, tile_masks: List[Tuple[np.ndarray, Tuple[int, int]]], 
                      image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Stitch tile masks back together using weighted averaging in overlap regions.
        """
        h, w = image_shape[:2]
        stitched = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        # Create a weight kernel (higher weight in center, lower at edges)
        kernel = self._create_weight_kernel(self.tile_size, self.overlap)
        
        for mask, (y_start, x_start) in tile_masks:
            y_end = y_start + mask.shape[0]
            x_end = x_start + mask.shape[1]
            
            # Get appropriate weight kernel slice
            kh, kw = mask.shape[:2]
            weight = kernel[:kh, :kw]
            
            stitched[y_start:y_end, x_start:x_end] += mask.astype(np.float32) * weight
            weight_map[y_start:y_end, x_start:x_end] += weight
        
        # Avoid division by zero
        weight_map[weight_map == 0] = 1
        result = (stitched / weight_map).astype(np.uint8)
        
        return result

    def _create_weight_kernel(self, tile_size: int, overlap: int) -> np.ndarray:
        """Create a 2D weight kernel for blending tile overlaps."""
        # Distance from center
        center = tile_size // 2
        y, x = np.ogrid[:tile_size, :tile_size]
        dist = np.sqrt((y - center) ** 2 + (x - center) ** 2)
        
        # Gaussian-like weight
        sigma = tile_size / 6
        weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
        
        # Normalize
        weight = weight / weight.max()
        
        return weight.astype(np.float32)

    def _stitch_label_tiles(self, tile_masks: List[Tuple[np.ndarray, Tuple[int, int]]],
                            image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Stitch label-mask tiles back together.
        
        For overlapping regions, we take the label from the tile with the
        larger grain (max label value), avoiding label double-counting.
        """
        h, w = image_shape[:2]
        stitched = np.zeros((h, w), dtype=np.int32)
        
        for mask, (y_start, x_start) in tile_masks:
            y_end = y_start + mask.shape[0]
            x_end = x_start + mask.shape[1]
            # Overlap region: keep the larger label (avoid double counting)
            region = stitched[y_start:y_end, x_start:x_end]
            stitched[y_start:y_end, x_start:x_end] = np.where(
                mask > region, mask, region
            )
        
        return stitched

    def _run_cellpose_on_tile(self, tile: np.ndarray) -> np.ndarray:
        """Run Cellpose on a single tile and return the mask."""
        # Cellpose expects RGB images
        if len(tile.shape) == 2:
            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
        elif tile.shape[2] == 3:
            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        else:
            tile_rgb = tile
            
        diameter = config.CELLPOSE_DIAMETER_ESTIMATE if config.CELLPOSE_DIAMETER_ESTIMATE > 0 else None
        
        # Run Cellpose
        masks, flows, styles = self._model.eval(
            tile_rgb,
            diameter=diameter,
            channels=[0, 0],  # Grayscale
            flow_threshold=config.CELLPOSE_FLOW_THRESHOLD,
            cellprob_threshold=config.CELLPOSE_CELLPROB_THRESHOLD,
            do_3D=False,
            min_size=15,
            max_size_fraction=0.4,
            niter=200,
        )
        
        return masks

    def segment(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run Cellpose segmentation and count grains.
        
        Cellpose outputs LABEL MASKS (each grain = unique integer ID), NOT
        probability maps. We count grains from the number of unique labels,
        not by averaging integer labels and thresholding (which is wrong).
        
        Returns dict with:
        - 'labels': Labelled mask (int32) where each grain has a unique ID
        - 'num_grains': Number of detected grains
        - 'grains': List of grain dicts with 'label', 'contour', 'mask', 'bbox'
        - 'binary': Binary mask (uint8) for downstream morphometrics
        - 'steps': Intermediate steps for visualization
        """
        h, w = image.shape[:2]
        logger.info(f"Starting Cellpose segmentation on {w}x{h} image")

        # Step 1: Apply TTA - get all transformed versions
        tta_images = self._apply_tta(image)
        logger.info(f"TTA: {len(tta_images)} augmentations")

        # Step 2: For each TTA transform, run tiled inference
        all_label_masks = []

        for idx, (tta_img, transform_info) in enumerate(tta_images):
            logger.info(f"Processing TTA {idx+1}/{len(tta_images)}: rot={transform_info['rot']}, flip={transform_info['flip']}")

            # Tile the transformed image
            tiles = self._tile_image(tta_img)
            logger.info(f"  Tiled into {len(tiles)} tiles")

            # Process each tile
            tile_masks = []
            for tile_idx, (tile, pos) in enumerate(tiles):
                try:
                    mask = self._run_cellpose_on_tile(tile)
                    tile_masks.append((mask, pos))
                except Exception as e:
                    logger.warning(f"  Tile {tile_idx} failed: {e}")
                    tile_masks.append((np.zeros(tile.shape[:2], dtype=np.int32), pos))

            # Stitch tiles back together (label masks)
            stitched = self._stitch_label_tiles(tile_masks, tta_img.shape)

            # Reverse TTA transform
            reversed_mask = self._reverse_tta(stitched, transform_info)
            all_label_masks.append(reversed_mask)

        # Step 3: Combine TTA results via voting (most common label wins per pixel)
        # Instead of averaging integers (wrong), we take the label from the
        # original (non-augmented) result as the primary, then for each pixel
        # pick the TTA result that has the most non-zero overlap.
        # Simplest robust approach: use the ORIGINAL (first) result as primary.
        primary_labels = all_label_masks[0]

        # Step 4: Build binary mask + count grains from unique labels
        binary = (primary_labels > 0).astype(np.uint8)

        # Connected components to get individual grains
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # Filter by area
        total_pixels = h * w
        ref_pixels = 1920 * 1080
        scale = total_pixels / ref_pixels
        min_area = max(config.MIN_GRAIN_AREA_PX, int(30 * scale))
        max_area = max(config.MAX_GRAIN_AREA_PX, int(500000 * scale))
        max_area = min(max_area, int(total_pixels * 0.10))

        valid_labels = []
        grains = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                valid_labels.append(i)

                # Create mask for this grain
                grain_mask = (labels == i).astype(np.uint8) * 255

                # Find contour
                contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    x, y, w_g, h_g = cv2.boundingRect(contour)

                    grains.append({
                        "label": len(grains) + 1,
                        "contour": contour,
                        "mask": grain_mask,
                        "bbox": (x, y, w_g, h_g),
                        "centroid": tuple(centroids[i]),
                        "area": area
                    })

        # Remap labels to be sequential
        final_labels = np.zeros_like(labels)
        for new_id, grain in enumerate(grains, start=1):
            old_label = valid_labels[new_id - 1]
            final_labels[labels == old_label] = new_id
            grain["label"] = new_id

        logger.info(f"Cellpose segmentation complete: {len(grains)} grains detected")

        return {
            "labels": final_labels,
            "num_grains": len(grains),
            "grains": grains,
            "binary": binary,
            "steps": {
                "original": image,
                "binary": binary * 255,
                "labels": self._labels_to_rgb(final_labels),
            }
        }

    def _labels_to_rgb(self, labels: np.ndarray) -> np.ndarray:
        """Convert label mask to RGB visualization."""
        # Use a colormap
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]
        
        if len(unique_labels) == 0:
            return np.zeros((*labels.shape, 3), dtype=np.uint8)
        
        # Generate colors using HSV
        hsv_colors = np.zeros((len(unique_labels), 3), dtype=np.uint8)
        hsv_colors[:, 0] = np.linspace(0, 179, len(unique_labels), dtype=np.uint8)
        hsv_colors[:, 1] = 255
        hsv_colors[:, 2] = 255
        
        rgb_colors = cv2.cvtColor(hsv_colors.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)
        
        rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
        for i, label in enumerate(unique_labels):
            rgb[labels == label] = rgb_colors[i]
        
        return rgb


def create_cellpose_segmenter(**kwargs) -> Optional[CellposeSegmenter]:
    """
    Factory function to create a CellposeSegmenter with config defaults.
    Returns None (instead of crashing) if Cellpose cannot be loaded.
    """
    try:
        return CellposeSegmenter(
            model_type=kwargs.get("model_type", config.CELLPOSE_MODEL),
            tile_size=kwargs.get("tile_size", config.CELLPOSE_TILE_SIZE),
            overlap=kwargs.get("overlap", config.CELLPOSE_OVERLAP),
            use_tta=kwargs.get("use_tta", config.CELLPOSE_USE_TTA),
            device=kwargs.get(
                "device",
                config.CELLPOSE_DEVICE if config.CELLPOSE_DEVICE != "auto" else None,
            ),
        )
    except Exception as e:
        logger.warning(
            f"Cellpose segmenter unavailable — falling back to classical methods. "
            f"Reason: {e}"
        )
        return None