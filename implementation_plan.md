# Advanced Settings, Auto-Calibration, and Preprocessing Update

This plan addresses your three requests: enforcing high-contrast B&W processing, adding EXIF-based auto-calibration, and providing live "Advanced Settings" to adjust thresholds (like the broken grain threshold) directly from the UI.

## Open Questions
- **Auto-Calibration Accuracy**: Estimating real-world size (mm) from just a photo's EXIF data is an approximation because we don't know the exact distance from the camera to the rice. I will use a standard assumption (e.g., standard smartphone focal length at a typical 15cm macro distance). Is this acceptable for a baseline auto-calibration?
- **High-Contrast B&W**: You mentioned turning saturation to 0 and contrast to 100%. I will implement this as a contrast-stretching step on the grayscale image before the analysis pipeline runs. 

## Proposed Changes

### Preprocessing & Computer Vision Pipeline

#### [MODIFY] `core/preprocessing.py`
- Add a high-contrast enhancement step at the very beginning of the pipeline (equivalent to 0 saturation and 100% contrast).
- Update the `Preprocessor` to accept dynamic parameters for thresholding (e.g., adaptive block size).

#### [MODIFY] `core/classification.py`
- Expose the `broken_ratio_threshold` (currently hardcoded to 0.75). If a grain's length is less than this ratio of the average length, it is marked as broken. Exposing this will allow you to fix instances where whole rice is marked as broken by lowering the threshold.

### Backend Updates

#### [MODIFY] `app.py`
- Update the `/api/analyze` endpoint to accept new JSON parameters:
  - `broken_threshold`
  - `contrast_boost` (toggle)
- Implement **EXIF Auto-Calibration**: Extract EXIF data from uploaded images using the `Pillow` library. Use the focal length and image resolution to approximate `pixels_per_mm` if no manual calibration has been done.

### Frontend Updates

#### [MODIFY] `templates/analysis.html`
- Add an "Advanced Settings" collapsible panel below the file upload area.
- Add a slider for **Broken Grain Threshold** (e.g., adjust from 0.5 to 0.9).
- Add a slider for **Adaptive Threshold Block Size** (to fix region marking issues).
- Send these settings dynamically via the API when the user clicks "Run Analysis".

## Verification Plan
1. **Automated Tests**: I will upload a test image via the API passing different `broken_threshold` values to verify that the number of broken grains changes dynamically.
2. **Manual Verification**: I will ask you to upload an image using the new Advanced Settings UI and adjust the slider to verify that whole grains are no longer misclassified as broken.
