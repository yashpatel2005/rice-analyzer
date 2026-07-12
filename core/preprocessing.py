"""
Phase 3 – Image Preprocessing

All classical OpenCV preprocessing steps:
  • Noise reduction (Gaussian / bilateral / median)
  • Contrast enhancement (CLAHE)
  • Background normalization
  • Grayscale conversion
  • Otsu / adaptive thresholding with auto-polarity detection
  • Morphological opening & closing
  • Robust hole filling
  • Convex-hull gap bridging for translucent grains

Handles both scenarios:
  - Bright grains on dark background (standard lab setup)
  - Dark grains on light background (casual photos)

IMPORTANT: For black-background images with translucent rice,
uses a low threshold + convex-hull fill to avoid splitting single
grains into fragments.
"""

import cv2
import numpy as np
from typing import Dict, Any

import config


class Preprocessor:
    """Pipeline of classical preprocessing operators applied in sequence."""

    def __init__(self):
        self.blur_kernel = config.GAUSSIAN_BLUR_KERNEL
        self.clahe_clip = config.CLAHE_CLIP_LIMIT
        self.clahe_grid = config.CLAHE_GRID_SIZE
        self.morph_kernel = config.MORPH_KERNEL_SIZE
        self.morph_iter = config.MORPH_ITERATIONS
        self.block_size = None  # set dynamically from process()

    # ------------------------------------------------------------------
    # Individual steps
    # ------------------------------------------------------------------
    def denoise(self, image: np.ndarray, method: str = "gaussian") -> np.ndarray:
        """Apply noise reduction."""
        if method == "gaussian":
            return cv2.GaussianBlur(image, self.blur_kernel, 0)
        elif method == "bilateral":
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == "median":
            return cv2.medianBlur(image, 5)
        elif method == "nonlocal":
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return image

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """CLAHE contrast enhancement on the L channel of LAB."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge([l_channel, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def normalize_background(
        self, image: np.ndarray, kernel_size: int = 51
    ) -> np.ndarray:
        """
        Background normalization via morphological top-hat.
        Extracts foreground objects (bright regions) while suppressing
        uneven illumination. Uses only top-hat (not bottom-hat) to avoid
        shifting the histogram and destroying contrast on dark backgrounds.
        """
        gray_check = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean_val = float(np.mean(gray_check))

        # If the image is predominantly dark (lab setup with matte black bg),
        # top-hat alone isolates bright foreground objects cleanly.
        # If the image is bright/mixed, we skip normalization to preserve contrast.
        if mean_val < 120:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
            # Scale up the tophat result to use the full dynamic range
            if len(tophat.shape) == 3:
                gray_th = cv2.cvtColor(tophat, cv2.COLOR_BGR2GRAY)
            else:
                gray_th = tophat
            max_val = float(np.max(gray_th))
            if max_val > 0:
                scale = min(255.0 / max_val, 4.0)  # cap scaling
                tophat = np.clip(tophat.astype(np.float32) * scale, 0, 255).astype(np.uint8)
            return tophat
        else:
            # Bright background — skip top-hat, just return as-is
            return image

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _detect_polarity(self, gray: np.ndarray) -> str:
        """
        Detect whether grains are bright on dark or dark on bright.
        Uses border vs center analysis.
        """
        h, w = gray.shape
        border_width = max(h // 20, w // 20, 10)

        # Sample border pixels
        border_mask = np.zeros_like(gray, dtype=bool)
        border_mask[:border_width, :] = True
        border_mask[-border_width:, :] = True
        border_mask[:, :border_width] = True
        border_mask[:, -border_width:] = True

        border_mean = float(np.mean(gray[border_mask]))
        center_mean = float(np.mean(gray[~border_mask]))

        # Also check overall image brightness
        overall_mean = float(np.mean(gray))

        if border_mean < 80 and overall_mean < 120:
            return "bright_on_dark"
        elif border_mean > 150:
            return "dark_on_bright"
        elif border_mean < center_mean - 20:
            return "bright_on_dark"
        else:
            return "auto"

    def _is_black_background(self, gray: np.ndarray) -> bool:
        """Check if the image has a predominantly black background."""
        h, w = gray.shape
        border_width = max(h // 15, w // 15, 15)

        # Sample border pixels
        border_pixels = np.concatenate([
            gray[:border_width, :].ravel(),
            gray[-border_width:, :].ravel(),
            gray[:, :border_width].ravel(),
            gray[:, -border_width:].ravel(),
        ])
        border_mean = float(np.mean(border_pixels))
        overall_mean = float(np.mean(gray))

        # Black background: borders are dark and overall image is dark
        return border_mean < 50 and overall_mean < 100

    def threshold(self, gray: np.ndarray, method: str = None) -> np.ndarray:
        """
        Binarise the grayscale image with automatic polarity detection.
        Ensures grains end up as white (255) regardless of original contrast.
        """
        method = method or config.THRESHOLD_METHOD
        polarity = self._detect_polarity(gray)

        if self._is_black_background(gray):
            # Safe low threshold for translucent rice
            _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
            return binary

        if method == "adaptive":
            # CLAHE first for uneven illumination
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray)
            
            bs = self.block_size or config.THRESHOLD_BLOCK_SIZE
            # OpenCV requires blockSize to be odd and > 1
            bs = int(bs)
            if bs < 3:
                bs = 3
            if bs % 2 == 0:
                bs += 1
            binary = cv2.adaptiveThreshold(
                gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, bs, config.THRESHOLD_C
            )
            # Ensure polarity is correct
            white_pct = np.sum(binary == 255) / (gray.shape[0] * gray.shape[1])
            if white_pct > 0.5:
                binary = cv2.bitwise_not(binary)
                
        else: # otsu
            if polarity == "bright_on_dark":
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return binary

    def morphological_cleanup(self, binary: np.ndarray) -> np.ndarray:
        """
        Opening (remove noise) then closing (fill small holes).
        Uses a LARGE closing kernel to bridge translucent gaps inside grains.
        """
        h, w = binary.shape[:2]
        total_px = h * w
        scale = max(1.0, (total_px / (1920 * 1080)) ** 0.5)

        # Small kernel for opening (remove noise/dust)
        k_open = max(3, int(3 * scale))
        if k_open % 2 == 0:
            k_open += 1
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # Larger kernel for closing — this is critical to bridge translucent
        # gaps that would otherwise split a single grain into fragments
        k_close = max(9, int(12 * scale))
        if k_close % 2 == 0:
            k_close += 1
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        return closed

    def fill_holes(self, binary: np.ndarray) -> np.ndarray:
        """
        Fill interior holes in foreground objects using flood-fill.
        Robust: checks that the border is actually background before filling.
        """
        if not config.FILL_HOLES:
            return binary

        h, w = binary.shape[:2]

        # Safety check: if the borders are mostly foreground (white),
        # flood-filling from (0,0) would be useless or destructive.
        border_pixels = np.concatenate([
            binary[0, :], binary[-1, :],
            binary[:, 0], binary[:, -1],
        ])
        border_white_ratio = np.sum(border_pixels == 255) / len(border_pixels)

        if border_white_ratio > 0.5:
            return binary

        # Standard flood-fill from corners
        floodfill = binary.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Try flood-filling from all four corners for robustness
        for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
            if floodfill[seed[1], seed[0]] == 0:
                cv2.floodFill(floodfill, mask, seed, 255)

        inverted = cv2.bitwise_not(floodfill)
        return binary | inverted

    def convex_hull_fill(self, binary: np.ndarray) -> np.ndarray:
        """
        For each connected component, replace its mask with its convex hull.
        This bridges internal translucent gaps that morphological closing
        cannot reach, ensuring each grain is a single solid blob.
        """
        # Find all contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        result = np.zeros_like(binary)
        for cnt in contours:
            if len(cnt) < 3:
                continue
            hull = cv2.convexHull(cnt)
            cv2.drawContours(result, [hull], -1, 255, -1)

        return result

    def apply_high_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Force image to high-contrast B&W.
        Equivalent to 0 saturation (grayscale) and aggressive contrast stretch.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        high_contrast = clahe.apply(gray)

        return cv2.cvtColor(high_contrast, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def process(self, image: np.ndarray, contrast_boost: bool = False, block_size: int = None) -> Dict[str, Any]:
        """
        Run the complete preprocessing pipeline and return every
        intermediate result so the UI can display them.
        """
        self.block_size = block_size
        steps = {}

        # 0. Optional High-Contrast B&W Boost
        if contrast_boost:
            image = self.apply_high_contrast(image)
            steps["high_contrast"] = image

        # 1. Denoise
        denoised = self.denoise(image, method="gaussian")
        steps["denoised"] = denoised

        # 2. Contrast enhancement
        enhanced = self.enhance_contrast(denoised)
        steps["enhanced"] = enhanced

        # 3. Background normalization
        normalized = self.normalize_background(enhanced)
        steps["normalized"] = normalized

        # 4. Grayscale
        gray = self.to_grayscale(normalized)
        steps["grayscale"] = gray

        # 5. Threshold (with auto-polarity detection + black-bg fast path)
        binary = self.threshold(gray)
        steps["binary"] = binary

        # 6. Morphological cleanup (larger closing to bridge translucent gaps)
        cleaned = self.morphological_cleanup(binary)
        steps["morphological"] = cleaned

        # 7. Fill holes (flood-fill)
        filled = self.fill_holes(cleaned)
        steps["filled"] = filled

        # 8. Convex-hull fill — makes each grain's interior solid,
        #    eliminating any remaining translucent splits
        hulled = self.convex_hull_fill(filled)
        steps["convex_hull"] = hulled

        return {
            "binary": hulled,
            "steps": steps,
        }
