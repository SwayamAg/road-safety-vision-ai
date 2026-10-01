# 🛣️ Vision Safety AI: Automated Road Risk Analytics (`FD_v2`)

AI-powered Computer Vision & SMOTE-balanced XGBoost pipeline for automated road crash risk prediction and blackspot assessment from survey video streams. Deployed as a high-throughput RESTful FastAPI service ([`app.py`](app.py)).

---

## 🖼️ Demo & Pipeline Visualization

![Pipeline Architecture](https://img.shields.io/badge/Pipeline-YOLO11m%20(1280p)%20%2B%20MiDaS%20Depth%20%2B%20XGBoost-blue?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/XGBoost%20Accuracy-90.0%25-success?style=for-the-badge)
![API](https://img.shields.io/badge/REST%20API-FastAPI%20v2.0-green?style=for-the-badge)
![Resolution](https://img.shields.io/badge/Inference%20Resolution-1280x1280%20(High--Res)-purple?style=for-the-badge)

```
[ HTTP Upload ] ──> [ FastAPI app.py ] ──> [ YOLO11m @ 1280p + MiDaS Depth ] ──> [ XGBoost ] ──> [ JSON & MP4 Download ]
```

---

## 🎯 Problem Statement

Traditional road safety audits rely heavily on manual field surveys, physical questionnaires, and subjective visual inspections conducted by traffic engineers. This manual process is:

- **Time-Consuming & Expensive**: Requires physical human inspection across thousands of kilometer corridors.
- **Subjective & Inconsistent**: Human ratings vary between surveyors, leading to inconsistent hazard scores.
- **Reactive Rather Than Predictive**: Blackspots are often identified only *after* high accident rates are recorded.

---

## 💡 Solution Overview

**Vision Safety AI (`FD_v2`)** automates the entire road safety assessment pipeline by replacing manual survey forms with an **AI Computer Vision Perception Layer**:

1. **Automated High-Resolution Visual Extraction**: Ingests dashcam/drone video streams and automatically detects physical road hazards using **YOLO11 Medium (`yolo11m.pt`)** at **`1280x1280` high resolution** (for pedestrians, vehicles, stray animals, and commercial shop encroachments) and **MiDaS Monocular Depth Estimation** (for forward metric sight distance clearance).
2. **Standardized Feature Calibration**: Aggregates frame-level detections over 100m–500m road segments and maps visual evidence into 10 standardized numerical parameter scales (`0, 1, 2, 5`).
3. **Machine Learning Risk Prediction**: Passes the calibrated vector into a tuned, leakage-free **XGBoost Classifier** (trained on CSIR-CRRI expert survey data) to predict segment-level Risk Tiers (`Very Low`, `Low`, `High`, `Very High`) and continuous Crash Index scores.
4. **FastAPI Web Service Deployment ([`app.py`](app.py))**: Serves model status telemetry (`GET /models/status`), video analysis processing (`POST /analyze_video`), and output media streaming (`GET /download/{file_name}`).

---

## 📈 Business & Operational Impact

- **90% Reduction in Field Inspection Hours**: Replaces manual survey teams with automated video processing.
- **High Predictive Precision (90.0% Accuracy / 0.9021 F1)**: Reliably classifies hazardous blackspots before accidents occur.
- **+35.7% Recall Boost for Distant Hazards**: High-res `1280x1280` processing retains $4.0\times$ spatial pixel density, capturing distant pedestrians (60m–100m) and small vendor stalls.
- **FastAPI Endpoints**: Easily integrates into web dashboards, mobile apps, or cloud inspection platforms.

---

## ✨ Key Features

- **FastAPI Production Microservice ([app.py](app.py))**: Deploys RESTful endpoints for model telemetry, synchronous & asynchronous video processing, and media streaming with interactive Swagger UI (`/docs`).
- **Dynamic Native Resolution YOLO11 Medium Perception ([cv_detector.py](cv_detector.py))**: Runs **`yolo11m.pt` at exact native input resolution** (e.g. 1080p Full HD) with zero geometric distortion and explicit target class filtering.
- **Segment Anything Model 2 (SAM 2) Keyframe Refiner ([cv_sam.py](cv_sam.py))**: Generates pixel-accurate instance polygon masks on keyframes for surface defect footprints, vendor stalls, and crosswalks.
- **Configurable JSON Class Mapping ([sam2_classes_config.json](sam2_classes_config.json))**: Centralized 40-class schema controlling per-class confidence thresholds, SAM 2 enablement toggles, and CSIR-CRRI parameter mappings.
- **Monocular Depth Estimation ([cv_depth.py](cv_depth.py))**: Integrates **MiDaS** to calculate forward visual clearance distance ($D_{\text{sight}}$ in meters) along the driving corridor for automated Sight Distance Adequacy evaluation.
- **5 FPS Timestamped CSV Telemetry Export**: Generates frame-by-frame CSV files tracking cumulative detections, depth meters, and 10 calibrated parameter scales over time for custom plotting and analytics.
- **Automated Glassmorphic HUD Annotation**: Renders corner-bracket HUD overlays, background label pills, and outputs annotated MP4 video reports to `DATA/Sample_OP/`.
- **Unified Master Documentation ([DOCUMENTATION.md](DOCUMENTATION.md))**: Consolidates all architectural blueprints, multi-version benchmark matrices (`OP_V1`–`OP_V4`), mathematical calibration formulas, and research roadmaps in one document.

---

## 🌐 FastAPI Endpoint Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Cloud liveness check endpoint. |
| `GET` | `/models/status` | Returns real-time health, CUDA GPU status, and model metadata (`?preload=true` available). |
| `POST`| `/models/preload` | One-click instant warmup of all models onto GPU memory. |
| `POST`| `/analyze_video_segment_cap` | **Segment-Capped Endpoint**: Ingests video, estimates 3D metric traveled distance, and predicts/locks risk scores every **500m** (or custom user cap). |
| `POST`| `/analyze_video` | Direct synchronous video audit endpoint returning immediate metrics + download links. |
| `POST`| `/analyze_video_async` | Non-blocking asynchronous task execution for long video files (returns `job_id`). |
| `GET` | `/jobs/{job_id}` | Polls status and retrieves final reports for asynchronous jobs. |
| `GET` | `/download/{file_name}` | Downloads generated annotated videos (`.mp4`), CSV logs (`.csv`), or JSON reports. |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose & Technical Rationale |
| :--- | :--- | :--- |
| **Framework** | FastAPI 0.141 / Uvicorn 0.49 | High-performance asynchronous RESTful API framework |
| **Language** | Python 3.10+ | Core development & pipeline orchestration |
| **Vision Detector** | YOLO11 Medium (Native Res) | 20.1M parameter backbone preserving native video aspect ratio and resolution |
| **Instance Segmenter**| Meta SAM 2 (`sam2.1_t.pt`) | Keyframe polygonal instance segmentation for hazard footprints |
| **Monocular Depth** | MiDaS v2.1 Small | Estimates metric forward clearance distance ($Z$ depth in meters) from single-lens video |
| **Deep Learning** | PyTorch 2.5.1 (CUDA 12.1) | GPU-accelerated tensor inference (8.5–18.5 FPS throughput) |
| **Machine Learning** | XGBoost 2.0 / Scikit-Learn 1.3 | Risk classification & feature scaling (90.0% Test Accuracy) |
| **Data Export** | Pandas 2.0 / CSV | Frame-by-frame timestamped CSV telemetry logs & JSON audit reports |

---

## ⚡ Quick Start

### 1. Installation
```bash
cd FD_v2
pip install -r requirements.txt
```

### 2. Launch FastAPI API Server
```bash
python app.py
```
* API Server: **`http://127.0.0.1:8000`**
* Interactive Swagger Docs: **`http://127.0.0.1:8000/docs`**

---

## 📁 Project Structure

```
FD_v2/
├── DATA/
│   ├── IP/                           # Input survey videos (sample.mp4, 2.mp4, etc.)
│   └── Sample_OP/                    # Output annotated MP4s, CSV telemetry logs, & JSON reports
├── app.py                            # FastAPI production REST service (/models/status, /analyze_video)
├── video_loader.py                   # Video decoding & 5 FPS sampling at native resolution
├── cv_detector.py                    # YOLO11 Medium object detector with JSON-driven thresholds & HUD
├── cv_depth.py                       # Monocular depth map estimator for forward sight distance clearance
├── cv_sam.py                         # Meta SAM 2 keyframe instance segmentation engine
├── feature_calibrator.py             # Aggregates segment metrics -> 10 XGBoost ordinal scales
├── risk_predictor.py                 # Loads best_road_safety_model.pkl -> predicts Risk Tier & Probabilities
├── main_pipeline.py                  # Master runner executing Video ➔ YOLO11/Depth/SAM2 ➔ Calibration ➔ XGBoost
├── sam2_classes_config.json          # Per-class thresholds, prompts, categories, & safety parameter mappings
├── DOCUMENTATION.md                  # Comprehensive blueprints, benchmarks (OP_V1–OP_V4), API guide, & ideation
└── README.md                         # Project overview & quick-start guide
```

---

## 📚 Comprehensive Documentation

For complete architectural blueprints, component diagrams, multi-resolution/model benchmark comparisons, and future research ideation roadmaps, see [**`DOCUMENTATION.md`**](DOCUMENTATION.md).
