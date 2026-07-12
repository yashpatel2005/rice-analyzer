"""
Global configuration for the Rice Grain Morphometric Analysis System.
All thresholds, defaults, and paths live here so the whole pipeline
can be tuned from one place.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CALIBRATION_DIR = os.path.join(BASE_DIR, "calibration")

CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
EXCEL_DIR = os.path.join(OUTPUT_DIR, "excel")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

for _d in (UPLOAD_DIR, OUTPUT_DIR, CALIBRATION_DIR,
           CSV_DIR, EXCEL_DIR, JSON_DIR, IMAGE_DIR, PLOT_DIR):
    os.makedirs(_d, exist_ok=True)

# ------------------------------------------------------------------
# Camera defaults
# ------------------------------------------------------------------
CAMERA_INDEX = 0
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_AUTO_EXPOSURE = True
CAPTURE_EXPOSURE = -7          # OpenCV exposure prop (log-scale)
CAPTURE_GAIN = 0
CAPTURE_WHITE_BALANCE = 4600
CAPTURE_FOCUS = 0
CAPTURE_FORMAT = ".jpg"
CAPTURE_JPEG_QUALITY = 95

# ------------------------------------------------------------------
# Preprocessing defaults
# ------------------------------------------------------------------
GAUSSIAN_BLUR_KERNEL = (5, 5)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
MORPH_KERNEL_SIZE = 5
MORPH_ITERATIONS = 2

# ------------------------------------------------------------------
# Segmentation defaults
# ------------------------------------------------------------------
THRESHOLD_METHOD = "otsu"          # "otsu" | "adaptive"
THRESHOLD_BLOCK_SIZE = 35
THRESHOLD_C = 5
MIN_GRAIN_AREA_PX = 50             # discard blobs smaller than this
MAX_GRAIN_AREA_PX = 500000         # discard blobs larger than this
WATERSHED_DISTANCE_THRESHOLD = 0.1 # fraction of max distance-transform peak
FILL_HOLES = True

# ------------------------------------------------------------------
# Calibration defaults
# ------------------------------------------------------------------
# Pixels-per-millimetre; updated after checkerboard / known-object calibration
PIXELS_PER_MM = 0.0                # 0 means "not calibrated" → px-only mode
CHECKERBOARD_SQUARE_MM = 25.0
CHECKERBOARD_PATTERN = (7, 10)     # inner corners

# ------------------------------------------------------------------
# Classification thresholds (mm where calibrated, px otherwise)
# ------------------------------------------------------------------
CLASSIFICATION_THRESHOLDS = {
    "long_grain_min_length": 6.0,
    "medium_grain_min_length": 5.0,
    "short_grain_max_length": 5.0,
    "broken_max_length_ratio": 0.75,  # < 75 % of median length → broken (adjustable via UI)
    "oversized_max_area_ratio": 1.8,  # > 180 % of median area → oversized
    "undersized_min_area_ratio": 0.4, # < 40 % of median area → undersized
    "abnormal_solidity_min": 0.85,
    "abnormal_circularity_max": 0.3,
    "abnormal_eccentricity_max": 0.95,
}

# ------------------------------------------------------------------
# Grading rules (configurable)
# ------------------------------------------------------------------
GRADING_RULES = {
    "premium": {
        "max_broken_pct": 2.0,
        "min_uniformity": 90.0,
        "max_abnormal_pct": 1.0,
        "max_cv_length": 10.0,
    },
    "grade_a": {
        "max_broken_pct": 5.0,
        "min_uniformity": 80.0,
        "max_abnormal_pct": 3.0,
        "max_cv_length": 15.0,
    },
    "grade_b": {
        "max_broken_pct": 10.0,
        "min_uniformity": 70.0,
        "max_abnormal_pct": 5.0,
        "max_cv_length": 20.0,
    },
    "grade_c": {
        "max_broken_pct": 15.0,
        "min_uniformity": 60.0,
        "max_abnormal_pct": 8.0,
        "max_cv_length": 25.0,
    },
    "reject": {
        "max_broken_pct": 100.0,
        "min_uniformity": 0.0,
        "max_abnormal_pct": 100.0,
        "max_cv_length": 100.0,
    },
}

# ------------------------------------------------------------------
# Flask
# ------------------------------------------------------------------
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5050
FLASK_DEBUG = True
MAX_CONTENT_LENGTH = 64 * 1024 * 1024  # 64 MB upload limit
