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

- **FastAPI Production Microservice ([app.py](app.py))**: Deploys RESTful endpoints for model telemetry, video processing, and media streaming with interactive Swagger UI (`/docs`).
- **High-Resolution YOLO11 Medium Perception ([cv_detector.py](cv_detector.py))**: Runs **`yolo11m.pt` @ `1280x1280` resolution** with explicit target class filtering (eliminating false positives).
- **Monocular Depth Estimation ([cv_depth.py](cv_depth.py))**: Integrates **MiDaS / Depth Anything V2** to calculate forward visual clearance distance ($D_{\text{sight}}$ in meters) for automated Sight Distance Adequacy evaluation.
- **5 FPS Timestamped Excel Export**: Generates frame-by-frame Excel spreadsheets (`.xlsx`) and CSV files tracking cumulative detections, depth meters, and 10 calibrated parameter scales over time for custom plotting.
- **Automated Glassmorphic HUD Annotation**: Renders corner-bracket HUD overlays, background label pills, and outputs annotated MP4 video reports to `DATA/OP_V4/`.

---

## 🌐 FastAPI Endpoint Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/models/status` | **Endpoint 1**: Returns real-time health, CUDA GPU status, and model metadata for all 3 pre-trained engines. |
| `POST` | `/analyze_video` | **Endpoint 2**: Ingests uploaded MP4 video, executes Vision AI + XGBoost analytics, and returns audit metrics + download links. |
| `GET` | `/download/{file_name}` | Downloads generated annotated videos (`.mp4`), Excel logs (`.xlsx`), CSVs, or JSON reports. |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose & Technical Rationale |
| :--- | :--- | :--- |
| **Framework** | FastAPI 0.141 / Uvicorn 0.49 | High-performance asynchronous RESTful API framework |
| **Language** | Python 3.10+ | Core development & pipeline orchestration |
| **Vision Detector** | YOLO11 Medium @ 1280p | Upgraded 20.1M parameter backbone @ 1280x1280 resolution for max small-object recall |
| **Monocular Depth** | MiDaS v3.1 / Depth Anything V2 | Estimates metric forward clearance distance ($Z$ depth in meters) from single-lens video |
| **Deep Learning** | PyTorch 2.5.1 (CUDA 12.1) | GPU-accelerated tensor inference (8.5–12.6 FPS throughput) |
| **Machine Learning** | XGBoost 2.0 / Scikit-Learn 1.3 | Risk classification & feature scaling (90.0% Test Accuracy) |
| **Data Export** | OpenPyXL 3.1 / Pandas 2.0 | Frame-by-frame timestamped Excel spreadsheet (`.xlsx`) & CSV export |

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
│   ├── IP/                           # Input survey videos (1.mp4, 2.mp4)
│   ├── OP_V1/                        # Baseline YOLOv8 output baseline (640p)
│   ├── OP_V2/                        # Upgraded YOLO11 outputs (640p)
│   ├── OP_V3/                        # High-Res YOLO11 outputs (1280p)
│   └── OP_V4/                        # Gold-Standard YOLO11 Medium (1280p + HUD)
├── app.py                            # FastAPI production REST service (/models/status, /analyze_video)
├── video_loader.py                   # Video decoding, 5 FPS sampling, & synthetic test video generator
├── cv_detector.py                    # YOLO11 Medium object detector with tactical HUD annotations
├── cv_depth.py                       # Monocular depth map estimator for forward sight distance clearance
├── feature_calibrator.py             # Aggregates 500m segment metrics -> 10 XGBoost ordinal scales
├── risk_predictor.py                 # Loads best_road_safety_model.pkl -> predicts Risk Tier & Probabilities
├── main_pipeline.py                  # Master runner executing Video ➔ YOLO11/Depth ➔ Calibration ➔ XGBoost
├── FastAPI_Deployment_Guide.md       # API endpoint documentation, request/response schemas, & curl examples
├── Detection_Precision_Benchmark.md  # OP_V3 vs OP_V4 precision & explicit class filtering benchmark report
├── Resolution_Benchmark_Comparison.md# OP_V2 (640p) vs OP_V3 (1280p) spatial density & recall benchmark
├── Model_Benchmark_Comparison.md     # OP_V1 (YOLOv8) vs OP_V2 (YOLO11) performance comparison report
├── TO-DO.md                          # Development roadmap & milestone checkmarks
├── Architecture_Workflow.md          # Mermaid architecture blueprints & detailed sequence diagrams
└── README.md                         # Project overview & execution guide
```

---

## 🔮 Future Ideas for Maximum Precision Ideation

To push perception precision to maximum research-grade accuracy in future iterations, we recommend the following model upgrades:

1. **Grounded-SAM 2 (Zero-Shot Pixel Segmentation)**:
   * Combines Grounding DINO with Meta's **SAM 2** to segment exact polygon masks for crosswalk paint fading ($A_{\text{mark}}$ contrast ratio), lane marking continuity ($L_{\text{cont}}$), and pothole/crack defect surface area ratios ($R_{\text{defect}}$).
2. **ByteTrack + YOLO11-Pose (Trajectory & Time-To-Collision)**:
   * Integrates **ByteTrack** multi-object tracking to compute vehicle/pedestrian trajectory vectors and extract exact **Time-To-Collision (TTC < 1.5s)** for active near-miss conflict detection.
3. **Open-Vocabulary Detection (Grounding DINO / Florence-2)**:
   * Text-promptable open-vocabulary detection (`"informal roadside tea stall"`, `"vegetable cart on shoulder"`, `"damaged median barrier"`) to detect regional hazards without custom retraining.
4. **Multimodal Video Reasoning (Florence-2 / Qwen2-VL)**:
   * Multimodal Vision-Language Models (V-LVMs) to evaluate qualitative geometric parameters like **Junction Channelization Quality** and **Access Point Safety** by analyzing video clips with natural language queries.
