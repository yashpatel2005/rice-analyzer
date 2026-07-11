# 🌾 Rice Grain Morphometric Analysis System

A complete end-to-end automated computer vision system for rice grain morphometric analysis using **classical image processing** (OpenCV, scikit-image) — no deep learning required.

Built for controlled imaging environments with a matte black background and bright rice grains.

---

## ✨ Features

| Phase | Feature |
|-------|---------|
| **1** | Camera auto-detection & property discovery |
| **2** | Automated image capture with full metadata logging |
| **3** | OpenCV preprocessing: denoising, CLAHE, background normalization, thresholding, morphology |
| **4** | Classical segmentation: connected components, contours, distance transform, watershed |
| **5** | 25+ morphometric measurements per grain (px + mm when calibrated) |
| **6** | Full statistical analysis: mean, median, mode, std, CV, quartiles, IQR, percentiles, confidence intervals, outliers |
| **7** | Rule-based grain classification: whole, broken, long, medium, short, oversized, undersized, abnormal |
| **8** | Image-level quality metrics: broken %, uniformity, aspect ratio, shape consistency, density, orientation |
| **9** | Reports: CSV, Excel, JSON, annotated images, histograms, box plots, scatter plots, correlation heatmaps, KDE |
| **10** | Transparent rule-based grading: Premium → Grade A → Grade B → Grade C → Reject |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the application

```bash
python app.py
```

### 3. Open the web interface

Visit: [http://localhost:5050](http://localhost:5050)

---

## 🖥️ Web Interface

The system ships with a production-grade web UI:

- **Dashboard** — system overview, quick actions, latest results
- **Camera** — live preview, auto-detected camera properties, calibration tools (camera only powers on during capture/preview)
- **Analysis** — upload/capture images, run the full pipeline, explore results
- **Reports** — browse and download all generated files
- **Power BI Dashboard** — interactive dashboard with KPI cards, slicers, charts, and grain-level table
- **Settings** — tune preprocessing, segmentation, classification & grading thresholds

---

## 📷 Camera Calibration

Three calibration methods are supported:

1. **Checkerboard** — upload multiple checkerboard images
2. **Known object** — upload an image containing an object of known length
3. **Manual** — directly enter pixels-per-mm

Calibration data is saved to `calibration/calibration.json` and reused across sessions.

---

## 🔧 Configuration

All tunable parameters live in `config.py`:

- Preprocessing defaults
- Segmentation thresholds
- Classification thresholds
- Grading rules
- Camera defaults
- Output paths

You can also adjust them at runtime through the **Settings** page.

---

## 📊 Output Files

All outputs are saved under `outputs/`:

```
outputs/
├── csv/              # Per-grain measurements + statistics
├── excel/            # Multi-sheet Excel report
├── json/             # Complete analysis as JSON
├── images/           # Annotated images
└── plots/            # Statistical plots
```

---

## 🧪 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/camera/info` | GET | Auto-detected camera properties |
| `/api/camera/preview` | GET | Live preview frame |
| `/api/camera/capture` | POST | Capture a frame |
| `/api/camera/calibrate` | POST | Calibrate from checkerboard / known object |
| `/api/camera/calibrate/manual` | POST | Set manual pixels-per-mm |
| `/api/analyze` | POST | Upload image and run full pipeline |
| `/api/analyze/captured` | POST | Analyze last captured image |
| `/api/reports` | GET | List all report files |
| `/api/reports/latest` | GET | Latest analysis result |
| `/api/download/<cat>/<file>` | GET | Download a report file |
| `/api/dashboard/data` | GET | Dashboard-ready JSON data |
| `/api/export/powerbi` | GET | Power BI-ready Excel workbook |
| `/api/settings` | GET / POST | Read / update settings |

---

## 🛠️ Tech Stack

- **Python 3.11+**
- **OpenCV** — camera, preprocessing, segmentation, measurement
- **NumPy / SciPy** — numerical & statistical computation
- **pandas / openpyxl** — data handling & Excel export
- **scikit-image / scikit-learn** — watershed, regionprops, distribution tests
- **matplotlib / seaborn** — publication-quality plots
- **Flask** — web backend & REST API
- **Bootstrap 5** — responsive frontend

---

## 📝 Notes

- The system assumes a **controlled matte black background** with bright foreground rice grains.
- Deep learning is intentionally avoided; the pipeline is fully explainable and rule-based.
- All grading decisions are transparent with criterion-by-criterion pass/fail logs.

---

## 📄 License

MIT License — free for research and industrial use.
