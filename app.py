"""
Rice Grain Morphometric Analysis System
Main Flask Application

Endpoints:
  /                     → Dashboard
  /camera               → Camera control page
  /analysis             → Analysis page
  /reports              → Reports page
  /dashboard            → Power BI-style dashboard
  /settings             → Settings page

  GET  /api/camera/info              → Camera properties
  GET  /api/camera/calibration_status → Calibration status (no camera open)
  GET  /api/camera/preview           → Live preview frame (JPEG)
  POST /api/camera/capture     → Capture a frame
  POST /api/camera/calibrate   → Calibrate from uploaded checkerboard/known object
  POST /api/camera/calibrate/manual → Set manual pixels-per-mm

  POST /api/analyze            → Upload an image and run the full pipeline
  POST /api/analyze/captured   → Run the pipeline on the last captured image

  GET  /api/reports            → List all report runs
  GET  /api/reports/latest     → Latest analysis result
  GET  /api/download/<cat>/<file> → Download a file
  GET  /api/preview/<cat>/<file>  → Preview a file
  GET  /api/dashboard/data     → Dashboard-ready JSON data
  GET  /api/export/powerbi    → Power BI-ready Excel file

  GET  /api/settings           → Get current settings
  POST /api/settings           → Update settings
"""

import os
import io
import json
import base64
import threading
from datetime import datetime
from typing import Dict, Any
import traceback

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ExifTags
from flask import (
    Flask, render_template, request, jsonify, send_file,
    send_from_directory, abort, url_for
)
from flask_cors import CORS

import config
from core.camera import CameraManager
from core.preprocessing import Preprocessor
from core.segmentation import Segmenter
from core.clustering_segmentation import ClusteringSegmenter
from core.cellpose_segmentation import CellposeSegmenter, create_cellpose_segmenter
from core.measurement import GrainMeasurer
from core.statistics import StatisticalAnalyzer
from core.classification import GrainClassifier
from core.quality import QualityAnalyzer
from core.reporting import ReportGenerator
from core.grading import GradingEngine

# ------------------------------------------------------------------
# Flask app
# ------------------------------------------------------------------
from flask.json.provider import DefaultJSONProvider

class NumpyJSONProvider(DefaultJSONProvider):
    """Handle numpy types, NaN/Inf, and tuples in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            return val if np.isfinite(val) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)

app = Flask(__name__)
app.json = NumpyJSONProvider(app)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# Global singletons
camera_mgr = CameraManager()
preprocessor = Preprocessor()
segmenter = Segmenter()
clustering_segmenter = ClusteringSegmenter()
cellpose_segmenter = create_cellpose_segmenter() if config.USE_CELLPOSE else None
reporter = ReportGenerator()
grader = GradingEngine()

# Store the last analysis result for the reports page
last_analysis: Dict[str, Any] = {}


# ==================================================================
#  API HEALTH / STATUS
# ==================================================================
@app.route("/")
@app.route("/health")
def health_check():
    """Return backend status details."""
    return jsonify({
        "status": "online",
        "service": "Rice Analyzer API Core",
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "camera_connected": camera_mgr.is_calibrated() or bool(camera_mgr.get_properties())
    })


# ==================================================================
#  CAMERA API
# ==================================================================
@app.route("/api/camera/info")
def camera_info():
    """Return all auto-detected camera properties, then release the camera."""
    if not camera_mgr.discover_and_close():
        return jsonify({
            "success": False,
            "error": "No camera found. Please connect a camera and try again.",
        }), 404

    props = camera_mgr.get_properties()
    optimal = camera_mgr.get_optimal_config()
    return jsonify({
        "success": True,
        "properties": props,
        "optimal_config": optimal,
        "is_calibrated": camera_mgr.is_calibrated(),
        "pixels_per_mm": camera_mgr.pixels_per_mm,
        "calibration_data": camera_mgr.calibration_data,
    })


@app.route("/api/camera/calibration_status")
def camera_calibration_status():
    """Return calibration status without powering on the camera."""
    return jsonify({
        "success": True,
        "is_calibrated": camera_mgr.is_calibrated(),
        "pixels_per_mm": camera_mgr.pixels_per_mm,
        "calibration_data": camera_mgr.calibration_data,
    })


@app.route("/api/camera/preview")
def camera_preview():
    """Capture a single frame and return it as JPEG for live preview."""
    frame = camera_mgr.get_preview()
    if frame is None:
        return jsonify({"success": False, "error": "Camera not available"}), 404

    # Encode to JPEG
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buf).decode("utf-8")
    return jsonify({
        "success": True,
        "image": f"data:image/jpeg;base64,{img_b64}",
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/camera/capture", methods=["POST"])
def camera_capture():
    """Capture a frame and save it."""
    result = camera_mgr.capture(save=True)
    if not result.get("success"):
        return jsonify(result), 500

    # Encode the frame for display
    frame = result["frame"]
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    img_b64 = base64.b64encode(buf).decode("utf-8")

    return jsonify({
        "success": True,
        "image": f"data:image/jpeg;base64,{img_b64}",
        "metadata": result["metadata"],
    })


@app.route("/api/camera/calibrate", methods=["POST"])
def camera_calibrate():
    """Calibrate from uploaded checkerboard images or known object."""
    method = request.form.get("method", "checkerboard")

    if method == "checkerboard":
        files = request.files.getlist("images")
        if not files:
            return jsonify({"success": False, "error": "No images uploaded"}), 400

        paths = []
        for f in files:
            fname = f"calib_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{f.filename}"
            fpath = os.path.join(config.CALIBRATION_DIR, fname)
            f.save(fpath)
            paths.append(fpath)

        square_size = float(request.form.get("square_size_mm", config.CHECKERBOARD_SQUARE_MM))
        result = camera_mgr.calibrate_checkerboard(paths, square_size)
        return jsonify(result)

    elif method == "known_object":
        f = request.files.get("image")
        if not f:
            return jsonify({"success": False, "error": "No image uploaded"}), 400

        fname = f"calib_obj_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{f.filename}"
        fpath = os.path.join(config.CALIBRATION_DIR, fname)
        f.save(fpath)

        obj_length = float(request.form.get("object_length_mm", 10.0))
        result = camera_mgr.calibrate_known_object(fpath, obj_length)
        return jsonify(result)

    return jsonify({"success": False, "error": "Unknown calibration method"}), 400


@app.route("/api/camera/calibrate/manual", methods=["POST"])
def camera_calibrate_manual():
    """Manually set pixels-per-mm."""
    data = request.get_json()
    ppm = float(data.get("pixels_per_mm", 0))
    if ppm <= 0:
        return jsonify({"success": False, "error": "pixels_per_mm must be > 0"}), 400
    result = camera_mgr.set_manual_calibration(ppm)
    return jsonify(result)


# ==================================================================
#  ANALYSIS API
# ==================================================================
@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    """Upload an image and run the full analysis pipeline."""
    global last_analysis

    try:
        # Get the image
        file = request.files.get("image")
        if file:
            # Save uploaded file
            fname = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{file.filename}"
            fpath = os.path.join(config.UPLOAD_DIR, fname)
            file.save(fpath)
            image = cv2.imread(fpath)
        else:
            # Check for captured image path
            img_path = request.form.get("image_path")
            if img_path and os.path.exists(img_path):
                image = cv2.imread(img_path)
                fpath = img_path
            else:
                return jsonify({"success": False, "error": "No image provided"}), 400

        if image is None:
            return jsonify({"success": False, "error": "Cannot read image file"}), 400

        # Get parameters
        use_watershed = request.form.get("use_watershed", "true").lower() == "true"
        ppm = float(request.form.get("pixels_per_mm", camera_mgr.pixels_per_mm))
        
        # Advanced settings overrides
        contrast_boost = request.form.get("contrast_boost", "false").lower() == "true"
        use_clustering = request.form.get("use_clustering", str(config.USE_CLUSTERING)).lower() == "true"
        use_cellpose = request.form.get("use_cellpose", str(config.USE_CELLPOSE)).lower() == "true"
        
        broken_threshold = request.form.get("broken_threshold")
        broken_threshold = float(broken_threshold) if broken_threshold else None
        
        block_size = request.form.get("block_size")
        block_size = int(block_size) if block_size else None
        
        # EXIF Auto-calibration if ppm is 0
        if ppm == 0:
            ppm = _guess_ppm_from_exif(fpath, image.shape)

        # Run the pipeline
        result = _run_pipeline(
            image, fpath, use_watershed, ppm, 
            contrast_boost=contrast_boost, 
            use_clustering=use_clustering,
            use_cellpose=use_cellpose,
            broken_threshold=broken_threshold, 
            block_size=block_size
        )
        last_analysis = result
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/analyze/captured", methods=["POST"])
def analyze_captured():
    """Run the pipeline on the last captured image."""
    global last_analysis

    data = request.get_json() or {}
    img_path = data.get("image_path")
    if not img_path or not os.path.exists(img_path):
        return jsonify({"success": False, "error": "No captured image found"}), 400

    image = cv2.imread(img_path)
    if image is None:
        return jsonify({"success": False, "error": "Cannot read captured image"}), 400

    use_watershed = data.get("use_watershed", True)
    ppm = float(data.get("pixels_per_mm", camera_mgr.pixels_per_mm))
    
    contrast_boost = data.get("contrast_boost", False)
    use_clustering = data.get("use_clustering", config.USE_CLUSTERING)
    use_cellpose = data.get("use_cellpose", config.USE_CELLPOSE)
    broken_threshold = data.get("broken_threshold")
    if broken_threshold is not None:
        broken_threshold = float(broken_threshold)
    block_size = data.get("block_size")
    if block_size is not None:
        block_size = int(block_size)
        
    if ppm == 0:
        ppm = _guess_ppm_from_exif(img_path, image.shape)

    result = _run_pipeline(
        image, img_path, use_watershed, ppm,
        contrast_boost=contrast_boost,
        use_clustering=use_clustering,
        use_cellpose=use_cellpose,
        broken_threshold=broken_threshold,
        block_size=block_size
    )
    last_analysis = result
    return jsonify(result)


def _sanitize_json(obj):
    """Recursively replace NaN/Inf with None and convert numpy types to Python."""
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def _guess_ppm_from_exif(fpath: str, shape: tuple) -> float:
    """
    Attempt to calculate an approximate pixels_per_mm from EXIF data.
    Uses focal length and sensor size to estimate the field of view,
    then assumes a typical macro distance of 15cm for smartphone cameras.
    If EXIF is unavailable, falls back to a resolution-based heuristic.
    """
    try:
        # Default fallback heuristic based on resolution (approx 40 px/mm at 12MP)
        h, w = shape[:2]
        total_pixels = h * w
        base_ppm = 40.0 * ((total_pixels / (4000 * 3000)) ** 0.5)

        with Image.open(fpath) as img:
            exif = img.getexif()
            if not exif:
                return base_ppm

            # Extract EXIF tags
            # FocalLength (37386) - in mm
            # FocalLengthIn35mmFilm (41989) - equivalent focal length in 35mm format
            # Sensor size estimation from common smartphone sensors
            
            focal_length = exif.get(37386)  # FocalLength
            focal_35mm = exif.get(41989)    # FocalLengthIn35mmFilm
            
            # If we have focal length in 35mm equivalent, we can estimate sensor size
            # Typical smartphone: ~4.3mm focal length, 1/2.55" sensor (5.76mm x 4.29mm)
            # 35mm equivalent is typically ~26mm for main camera
            
            if focal_35mm and focal_35mm > 0:
                # Estimate sensor width from 35mm equivalent
                # 35mm film width = 36mm
                # focal_35mm / focal_length = 36 / sensor_width
                if focal_length and focal_length > 0:
                    sensor_width_mm = 36.0 * focal_length / focal_35mm
                else:
                    # Assume typical smartphone sensor width ~5.76mm
                    sensor_width_mm = 5.76
            elif focal_length and focal_length > 0:
                # Assume typical smartphone: 4.3mm focal length, 1/2.55" sensor
                sensor_width_mm = 5.76
            else:
                return base_ppm
            
            # Image width in pixels
            img_width_px = img.width
            
            # Assume typical macro distance of 15cm (150mm) for rice photography
            # Field of view width at 150mm = sensor_width * (distance / focal_length)
            if focal_length and focal_length > 0:
                fov_width_mm = sensor_width_mm * (150.0 / focal_length)
            else:
                # Fallback: assume 26mm equivalent at 150mm
                fov_width_mm = 36.0 * (150.0 / 26.0)
            
            # Pixels per mm = image width in pixels / FOV width in mm
            ppm = img_width_px / fov_width_mm
            
            # Sanity check: clamp to reasonable range
            ppm = max(10.0, min(200.0, ppm))
            
            return ppm

    except Exception as e:
        print(f"EXIF calibration failed: {e}")
        return 40.0


def _run_pipeline(
    image: np.ndarray,
    image_path: str,
    use_watershed: bool = True,
    ppm: float = 0.0,
    contrast_boost: bool = False,
    use_clustering: bool = False,
    use_cellpose: bool = False,
    broken_threshold: float = None,
    block_size: int = None
) -> Dict[str, Any]:
    """Run the complete analysis pipeline on an image."""
    t0 = datetime.now()

    # Phase 3 & 4 – Preprocessing & Segmentation
    if use_cellpose and cellpose_segmenter is not None:
        # Use Cellpose 3 (cyto3) Segmentation
        seg_result = cellpose_segmenter.segment(image)
        binary = np.zeros_like(image[:, :, 0])  # Dummy binary for the rest of the pipeline
        pre_result = {"steps": seg_result.get("steps", {})}
    elif use_clustering:
        # Use Advanced K-Means + GrabCut Clustering
        seg_result = clustering_segmenter.segment(image)
        binary = np.zeros_like(image[:, :, 0])  # Dummy binary for the rest of the pipeline
        pre_result = {"steps": seg_result.get("steps", {})}
    else:
        # Classical OpenCV Pipeline
        pre_result = preprocessor.process(image, contrast_boost=contrast_boost, block_size=block_size)
        binary = pre_result["binary"]
        seg_result = segmenter.segment(binary, use_watershed=use_watershed)

    grains = seg_result["grains"]

    if not grains:
        # Return a successful result with 0 grains instead of failing
        t1 = datetime.now()
        
        # Preprocessing previews
        step_previews = {}
        for name, step_img in pre_result["steps"].items():
            if len(step_img.shape) == 2:
                step_img = cv2.cvtColor(step_img, cv2.COLOR_GRAY2BGR)
            _, buf = cv2.imencode(".jpg", step_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            step_previews[name] = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
            
        # Use original image as annotated preview
        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
        
        return {
            "success": True,
            "num_grains": 0,
            "elapsed_seconds": round((t1 - t0).total_seconds(), 2),
            "measurements": [],
            "statistics": {},
            "classifications": [],
            "category_counts": {},
            "quality": {},
            "grading": {"grade": "None (No grains detected)", "grade_key": "reject", "criteria_checks": []},
            "files": {},
            "step_previews": step_previews,
            "annotated_preview": annotated_b64,
            "plot_previews": {},
            "quality_table_preview": None,
            "metadata": {
                "analysis_time": t1.isoformat(),
                "calibrated": ppm > 0,
                "pixels_per_mm": ppm,
                "use_watershed": use_watershed,
                "contrast_boost": contrast_boost,
                "use_clustering": use_clustering,
                "broken_threshold": broken_threshold,
                "block_size": block_size
            }
        }

    # Phase 5 – Measurement
    measurer = GrainMeasurer(pixels_per_mm=ppm)
    measurements = measurer.measure_all(grains)

    # Sanitize NaN/Inf values for JSON safety
    measurements = _sanitize_json(measurements)

    # Phase 6 – Statistics
    stat_analyzer = StatisticalAnalyzer(pixels_per_mm=ppm)
    stats = stat_analyzer.analyze(measurements)

    # Phase 7 – Classification
    classifier = GrainClassifier(pixels_per_mm=ppm, broken_threshold=broken_threshold)
    classifications = classifier.classify_all(measurements)
    category_counts = classifier.category_counts(classifications)

    # Phase 8 – Quality
    quality_analyzer = QualityAnalyzer(pixels_per_mm=ppm)
    quality = quality_analyzer.analyze(measurements, classifications, image.shape)

    # Phase 10 – Grading
    grading = grader.grade(quality, stats)

    # Phase 9 – Reporting
    # CSV
    csv_path = reporter.export_csv(measurements)
    stats_csv_path = reporter.export_stats_csv(stats)

    # Annotated image
    annotated_path = reporter.annotate_image(image, grains, measurements, classifications)

    # Plots - Disabled to fix severe backend latency and OOM crashes
    # plots = reporter.generate_plots(measurements, stats)
    plots = {}

    # Quality table - Disabled for performance
    # quality_table_path = reporter.generate_quality_table(quality, grading)
    quality_table_path = ""

    # Excel
    excel_path = reporter.export_excel(measurements, stats, quality, classifications, grading)

    # JSON
    metadata = {
        "image_path": image_path,
        "image_shape": list(image.shape),
        "analysis_time": datetime.now().isoformat(),
        "calibrated": ppm > 0,
        "pixels_per_mm": ppm,
        "use_watershed": use_watershed,
    }
    json_path = reporter.export_json(
        measurements, stats, quality, classifications, grading, metadata
    )

    t1 = datetime.now()
    elapsed = (t1 - t0).total_seconds()

    # Prepare web-friendly previews
    # Preprocessing steps as base64
    step_previews = {}
    for name, img in pre_result["steps"].items():
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        step_previews[name] = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

    # Annotated image as base64
    annotated_img = cv2.imread(annotated_path)
    _, buf = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

    # Plot previews
    plot_previews = {}
    for name, path in plots.items():
        img = cv2.imread(path)
        if img is not None:
            _, buf = cv2.imencode(".png", img)
            plot_previews[name] = f"data:image/png;base64,{base64.b64encode(buf).decode('utf-8')}"

    # Quality table preview (Disabled)
    quality_table_b64 = None
    if quality_table_path:
        qt_img = cv2.imread(quality_table_path)
        if qt_img is not None:
            _, buf = cv2.imencode(".png", qt_img)
            quality_table_b64 = f"data:image/png;base64,{base64.b64encode(buf).decode('utf-8')}"

    return {
        "success": True,
        "run_id": reporter.run_id,
        "elapsed_seconds": round(elapsed, 2),
        "image_path": image_path,
        "metadata": metadata,
        # Phase 4
        "num_grains": len(grains),
        "raw_components": seg_result["raw_components"],
        "filtered_components": seg_result["filtered_components"],
        # Phase 5
        "measurements": measurements,
        # Phase 6
        "statistics": stats,
        # Phase 7
        "classifications": classifications,
        "category_counts": category_counts,
        # Phase 8
        "quality": quality,
        # Phase 10
        "grading": grading,
        # Phase 9 – file paths
        "files": {
            "csv": os.path.basename(csv_path),
            "stats_csv": os.path.basename(stats_csv_path),
            "excel": os.path.basename(excel_path),
            "json": os.path.basename(json_path),
            "annotated_image": os.path.basename(annotated_path),
            "quality_table": os.path.basename(quality_table_path),
            "plots": {k: os.path.basename(v) for k, v in plots.items()},
        },
        # Previews
        "step_previews": step_previews,
        "annotated_preview": annotated_b64,
        "plot_previews": plot_previews,
        "quality_table_preview": quality_table_b64,
    }


# ==================================================================
#  REPORTS API
# ==================================================================
@app.route("/api/reports")
def list_reports():
    """List all available report files."""
    reports = {"csv": [], "excel": [], "json": [], "images": [], "plots": []}

    for subdir, key in [(config.CSV_DIR, "csv"), (config.EXCEL_DIR, "excel"),
                        (config.JSON_DIR, "json"), (config.IMAGE_DIR, "images"),
                        (config.PLOT_DIR, "plots")]:
        if os.path.isdir(subdir):
            for f in sorted(os.listdir(subdir), reverse=True):
                fpath = os.path.join(subdir, f)
                if os.path.isfile(fpath):
                    reports[key].append({
                        "name": f,
                        "size": os.path.getsize(fpath),
                        "modified": datetime.fromtimestamp(
                            os.path.getmtime(fpath)
                        ).isoformat(),
                    })

    return jsonify({"success": True, "reports": reports})


@app.route("/api/reports/latest")
def latest_report():
    """Return the last analysis result."""
    if not last_analysis:
        return jsonify({"success": False, "error": "No analysis has been run yet"}), 404
    return jsonify(last_analysis)


@app.route("/api/download/<category>/<filename>")
def download_file(category, filename):
    """Download a specific report file."""
    dir_map = {
        "csv": config.CSV_DIR,
        "excel": config.EXCEL_DIR,
        "json": config.JSON_DIR,
        "images": config.IMAGE_DIR,
        "plots": config.PLOT_DIR,
    }
    directory = dir_map.get(category)
    if not directory:
        abort(404)
    fpath = os.path.join(directory, filename)
    if not os.path.exists(fpath):
        abort(404)
    return send_file(fpath, as_attachment=True, download_name=filename)


@app.route("/api/preview/<category>/<filename>")
def preview_file(category, filename):
    """Preview a file inline (for images and plots)."""
    dir_map = {
        "images": config.IMAGE_DIR,
        "plots": config.PLOT_DIR,
    }
    directory = dir_map.get(category)
    if not directory:
        abort(404)
    fpath = os.path.join(directory, filename)
    if not os.path.exists(fpath):
        abort(404)
    return send_file(fpath)


@app.route("/api/dashboard/data")
def dashboard_data():
    """Return the latest analysis data formatted for the Power BI-style dashboard."""
    if not last_analysis or not last_analysis.get("success"):
        return jsonify({"success": False, "error": "No analysis has been run yet"}), 404

    data = last_analysis
    measurements = data.get("measurements", [])
    classifications = data.get("classifications", [])
    quality = data.get("quality", {})
    grading = data.get("grading", {})
    stats = data.get("statistics", {})
    metadata = data.get("metadata", {})

    # Detect calibration: if pixels_per_mm is 0 or not set, use pixel units
    calibrated = metadata.get("calibrated", False)
    unit_suffix = "mm" if calibrated else "px"

    # Build grain-level dataset — fall back to pixel values when uncalibrated
    grain_data = []
    for m, c in zip(measurements, classifications):
        length_val = m.get("length_mm") if calibrated else m.get("length_px")
        width_val = m.get("width_mm") if calibrated else m.get("width_px")
        area_val = m.get("area_mm2") if calibrated else m.get("area_px")

        grain_data.append({
            "label": m.get("label"),
            "area_mm2": area_val,
            "length_mm": length_val,
            "width_mm": width_val,
            "aspect_ratio": m.get("aspect_ratio"),
            "circularity": m.get("circularity"),
            "solidity": m.get("solidity"),
            "eccentricity": m.get("eccentricity"),
            "orientation_angle": m.get("orientation_angle"),
            "primary_category": c.get("primary_category"),
            "categories": ", ".join(c.get("categories", [])),
            "is_broken": "broken_grain" in c.get("categories", []),
            "is_whole": "whole_grain" in c.get("categories", []),
            "is_long": "long_grain" in c.get("categories", []),
            "is_medium": "medium_grain" in c.get("categories", []),
            "is_short": "short_grain" in c.get("categories", []),
        })

    # Build category distribution
    category_dist = data.get("category_counts", {})

    # Build length/width/area binned distributions
    def bin_values(values, bins=10):
        if not values:
            return [], []
        values = [v for v in values if v is not None and isinstance(v, (int, float))]
        if not values:
            return [], []
        min_v, max_v = min(values), max(values)
        if min_v == max_v:
            return [f"{min_v:.2f}"], [len(values)]
        step = (max_v - min_v) / bins
        ranges = [f"{min_v + i*step:.2f}-{min_v + (i+1)*step:.2f}" for i in range(bins)]
        counts = [0] * bins
        for v in values:
            idx = min(int((v - min_v) / step), bins - 1)
            counts[idx] += 1
        return ranges, counts

    lengths = [g["length_mm"] for g in grain_data if g["length_mm"] is not None]
    widths = [g["width_mm"] for g in grain_data if g["width_mm"] is not None]
    areas = [g["area_mm2"] for g in grain_data if g["area_mm2"] is not None]

    length_bins, length_counts = bin_values(lengths, 8)
    width_bins, width_counts = bin_values(widths, 8)
    area_bins, area_counts = bin_values(areas, 8)

    # Build scatter data
    scatter_data = [
        {"x": g["length_mm"], "y": g["width_mm"], "category": g["primary_category"]}
        for g in grain_data
        if g["length_mm"] is not None and g["width_mm"] is not None
    ]

    # KPI cards — fall back to pixel stats when uncalibrated
    if calibrated:
        mm_stats = stats.get("mm_stats", {})
        mean_length = mm_stats.get("length_mm", {}).get("mean")
        mean_width = mm_stats.get("width_mm", {}).get("mean")
        mean_area = mm_stats.get("area_mm2", {}).get("mean")
        cv_length = mm_stats.get("length_mm", {}).get("cv")
    else:
        px_stats = stats.get("px_stats", stats.get("mm_stats", {}))
        mean_length = px_stats.get("length_px", {}).get("mean")
        mean_width = px_stats.get("width_px", {}).get("mean")
        mean_area = px_stats.get("area_px", {}).get("mean")
        cv_length = px_stats.get("length_px", {}).get("cv")

    kpis = {
        "total_grains": len(measurements),
        "broken_pct": quality.get("broken_pct", 0),
        "uniformity_index": quality.get("uniformity_index", 0),
        "grade": grading.get("grade", "N/A"),
        "mean_length": mean_length,
        "mean_width": mean_width,
        "mean_area": mean_area,
        "cv_length": cv_length,
    }

    return jsonify({
        "success": True,
        "run_id": data.get("run_id"),
        "kpis": kpis,
        "grain_data": grain_data,
        "category_distribution": category_dist,
        "length_distribution": {"labels": length_bins, "values": length_counts},
        "width_distribution": {"labels": width_bins, "values": width_counts},
        "area_distribution": {"labels": area_bins, "values": area_counts},
        "scatter_data": scatter_data,
        "quality": quality,
        "grading": grading,
        "annotated_preview": data.get("annotated_preview"),
        "files": data.get("files", {}),
        "calibrated": calibrated,
        "unit_suffix": unit_suffix,
    })


@app.route("/api/export/powerbi")
def export_powerbi():
    """Export a Power BI-ready Excel workbook with multiple clean tables."""
    if not last_analysis or not last_analysis.get("success"):
        return jsonify({"success": False, "error": "No analysis has been run yet"}), 404

    data = last_analysis
    measurements = data.get("measurements", [])
    classifications = data.get("classifications", [])
    quality = data.get("quality", {})
    grading = data.get("grading", {})

    # Grain facts table — the main star schema fact table
    grain_facts = []
    for m, c in zip(measurements, classifications):
        grain_facts.append({
            "GrainID": m.get("label"),
            "Area_mm2": m.get("area_mm2"),
            "Length_mm": m.get("length_mm"),
            "Width_mm": m.get("width_mm"),
            "Perimeter_mm": m.get("perimeter_mm"),
            "AspectRatio": m.get("aspect_ratio"),
            "Circularity": m.get("circularity"),
            "Solidity": m.get("solidity"),
            "Eccentricity": m.get("eccentricity"),
            "Orientation_deg": m.get("orientation_angle"),
            "PrimaryCategory": c.get("primary_category"),
            "IsWhole": "whole_grain" in c.get("categories", []),
            "IsBroken": "broken_grain" in c.get("categories", []),
            "IsLong": "long_grain" in c.get("categories", []),
            "IsMedium": "medium_grain" in c.get("categories", []),
            "IsShort": "short_grain" in c.get("categories", []),
            "IsOversized": "oversized_grain" in c.get("categories", []),
            "IsUndersized": "undersized_grain" in c.get("categories", []),
            "IsAbnormal": "abnormal_grain" in c.get("categories", []),
        })

    # Category dimension
    category_dim = []
    for cat, count in data.get("category_counts", {}).items():
        category_dim.append({"Category": cat, "Count": count})

    # Quality snapshot
    quality_df = pd.DataFrame([quality])

    # Grading snapshot
    grading_df = pd.DataFrame([grading])

    # Run metadata
    metadata = data.get("metadata", {})
    meta_df = pd.DataFrame([{
        "RunID": data.get("run_id"),
        "AnalysisTime": metadata.get("analysis_time"),
        "Calibrated": metadata.get("calibrated"),
        "PixelsPerMM": metadata.get("pixels_per_mm"),
        "UseWatershed": metadata.get("use_watershed"),
        "TotalGrains": data.get("num_grains"),
    }])

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(grain_facts).to_excel(writer, sheet_name="GrainFacts", index=False)
        pd.DataFrame(category_dim).to_excel(writer, sheet_name="CategoryDim", index=False)
        quality_df.to_excel(writer, sheet_name="QualitySnapshot", index=False)
        grading_df.to_excel(writer, sheet_name="GradingSnapshot", index=False)
        meta_df.to_excel(writer, sheet_name="RunMetadata", index=False)

    output.seek(0)
    filename = f"{data.get('run_id', 'report')}_powerbi.xlsx"
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=filename,
    )


# ==================================================================
#  SETTINGS API
# ==================================================================
@app.route("/api/settings", methods=["GET"])
def get_settings():
    """Return current configuration."""
    return jsonify({
        "success": True,
        "settings": {
            "gaussian_blur_kernel": config.GAUSSIAN_BLUR_KERNEL,
            "clahe_clip_limit": config.CLAHE_CLIP_LIMIT,
            "morph_kernel_size": config.MORPH_KERNEL_SIZE,
            "morph_iterations": config.MORPH_ITERATIONS,
            "threshold_method": config.THRESHOLD_METHOD,
            "min_grain_area_px": config.MIN_GRAIN_AREA_PX,
            "max_grain_area_px": config.MAX_GRAIN_AREA_PX,
            "watershed_distance_threshold": config.WATERSHED_DISTANCE_THRESHOLD,
            "pixels_per_mm": camera_mgr.pixels_per_mm,
            "is_calibrated": camera_mgr.is_calibrated(),
            "classification_thresholds": config.CLASSIFICATION_THRESHOLDS,
            "grading_rules": config.GRADING_RULES,
            "capture_width": config.CAPTURE_WIDTH,
            "capture_height": config.CAPTURE_HEIGHT,
        },
    })


@app.route("/api/settings", methods=["POST"])
def update_settings():
    """Update configuration values at runtime."""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    updates = []

    if "gaussian_blur_kernel" in data:
        config.GAUSSIAN_BLUR_KERNEL = tuple(data["gaussian_blur_kernel"])
        preprocessor.blur_kernel = config.GAUSSIAN_BLUR_KERNEL
        updates.append("gaussian_blur_kernel")

    if "clahe_clip_limit" in data:
        config.CLAHE_CLIP_LIMIT = float(data["clahe_clip_limit"])
        preprocessor.clahe_clip = config.CLAHE_CLIP_LIMIT
        updates.append("clahe_clip_limit")

    if "morph_kernel_size" in data:
        config.MORPH_KERNEL_SIZE = int(data["morph_kernel_size"])
        preprocessor.morph_kernel = config.MORPH_KERNEL_SIZE
        updates.append("morph_kernel_size")

    if "morph_iterations" in data:
        config.MORPH_ITERATIONS = int(data["morph_iterations"])
        preprocessor.morph_iter = config.MORPH_ITERATIONS
        updates.append("morph_iterations")

    if "threshold_method" in data:
        config.THRESHOLD_METHOD = data["threshold_method"]
        updates.append("threshold_method")

    if "min_grain_area_px" in data:
        config.MIN_GRAIN_AREA_PX = int(data["min_grain_area_px"])
        segmenter.min_area = config.MIN_GRAIN_AREA_PX
        updates.append("min_grain_area_px")

    if "max_grain_area_px" in data:
        config.MAX_GRAIN_AREA_PX = int(data["max_grain_area_px"])
        segmenter.max_area = config.MAX_GRAIN_AREA_PX
        updates.append("max_grain_area_px")

    if "watershed_distance_threshold" in data:
        config.WATERSHED_DISTANCE_THRESHOLD = float(data["watershed_distance_threshold"])
        segmenter.watershed_threshold = config.WATERSHED_DISTANCE_THRESHOLD
        updates.append("watershed_distance_threshold")

    if "classification_thresholds" in data:
        config.CLASSIFICATION_THRESHOLDS.update(data["classification_thresholds"])
        updates.append("classification_thresholds")

    if "grading_rules" in data:
        for grade, rules in data["grading_rules"].items():
            if grade in config.GRADING_RULES:
                config.GRADING_RULES[grade].update(rules)
        updates.append("grading_rules")

    if "capture_width" in data:
        config.CAPTURE_WIDTH = int(data["capture_width"])
        updates.append("capture_width")

    if "capture_height" in data:
        config.CAPTURE_HEIGHT = int(data["capture_height"])
        updates.append("capture_height")

    return jsonify({"success": True, "updated": updates})


# ==================================================================
#  MAIN
# ==================================================================
def test_module_imports():
    """Pytest marker: ensures core modules import cleanly."""
    __import__("Rice Analyzer.core.grading")
    __import__("Rice Analyzer.core.classification")
    __import__("Rice Analyzer.core.preprocessing")
    __import__("Rice Analyzer.core.segmentation")


if __name__ == "__main__":
    print("=" * 60)
    print("  Rice Grain Morphometric Analysis System")
    print("  Starting Flask server...")
    print("=" * 60)
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=False,
    )
