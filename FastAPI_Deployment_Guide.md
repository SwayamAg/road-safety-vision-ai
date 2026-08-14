# 🌐 FastAPI Production Deployment Guide (`app.py`)

This document details the RESTful API endpoints, request/response schemas, deployment instructions, and model telemetry provided by the **FastAPI Road Safety Vision-AI Service** ([`app.py`](app.py)).

---

## 🚀 1. API Architecture Overview

The FastAPI service encapsulates the end-to-end perception and inference pipeline:

```
[ HTTP Client / Frontend (React, Vue, Next.js, HTML/JS on ANY origin) ]
          │
          ├───► GET /models/status  ───► Returns GPU, CUDA, YOLO11, & XGBoost Health
          │
          ├───► POST /analyze_video ───► Ingests MP4 ──> YOLO11m (1280p) + MiDaS ──> XGBoost
          │                                                                │
          │                                                                ▼
          └───◄ GET /download/{file} ◄── Returns MP4 Video, Excel Logs, & JSON Reports
```

> [!NOTE]
> **Universal CORS Enabled**: The API is configured with `CORSMiddleware` (`allow_origins=["*"]`, `allow_origin_regex=r"https?://.*"`), allowing cross-origin AJAX/fetch requests from **any website**, local dev server (`localhost:3000`, `127.0.0.1:5500`), or production cloud domain.

---

## 📡 2. Endpoint Specifications

### 1. Model Telemetry Endpoint: `GET /models/status`
* **Description**: Returns real-time health, CUDA GPU acceleration status, and model metadata for all pre-trained engines.
* **Sample Response**:
```json
{
  "status": "HEALTHY",
  "cuda_acceleration": {
    "available": true,
    "device": "NVIDIA GeForce RTX 4060 Laptop GPU",
    "torch_version": "2.5.1+cu121"
  },
  "models": {
    "object_detector": {
      "name": "YOLO11 Medium",
      "weights": "yolo11m.pt",
      "parameters": "20.1M",
      "inference_resolution": "1280x1280",
      "status": "LOADED"
    },
    "monocular_depth": {
      "name": "MiDaS Monocular Depth",
      "variant": "MiDaS_small",
      "task": "Sight Distance Metric Clearance (Meters)",
      "status": "LOADED"
    },
    "risk_classifier": {
      "name": "Tuned XGBoost Classifier",
      "artifact": "best_road_safety_model.pkl",
      "input_features": 10,
      "test_accuracy": "90.0%",
      "macro_f1": 0.9021,
      "status": "LOADED"
    }
  }
}
```

---

### 2. Video Perception & Risk Analytics Endpoint: `POST /analyze_video`
* **Description**: Accepts an uploaded road survey video (`.mp4`, `.avi`, `.mov`), runs YOLO11 Medium detection @ 1280p + MiDaS Depth estimation + XGBoost inference, and returns complete audit metrics alongside continuous Risk Score and download URLs.
* **Form Parameters**:
  - `file`: Video File Binary (`Multipart/Form-Data`)
  - `sample_fps`: `float` (Default: `5.0` FPS)
  - `imgsz`: `int` (Default: `1280`)
* **Sample Response**:
```json
{
  "message": "Road Safety Vision-AI Audit Completed Successfully",
  "download_urls": {
    "annotated_video": "/download/upload_1785671995_1_annotated.mp4",
    "excel_log": "/download/upload_1785671995_1_frame_by_frame_detections.xlsx",
    "csv_log": "/download/upload_1785671995_1_frame_by_frame_detections.csv",
    "json_report": "/download/upload_1785671995_1_risk_report.json"
  },
  "audit_summary": {
    "video_metadata": {
      "resolution": [1920, 1080],
      "native_fps": 29.78,
      "total_frames": 9977,
      "duration_sec": 334.99
    },
    "pipeline_config": {
      "model_weights": "yolo11m.pt",
      "inference_resolution": "1280x1280"
    },
    "risk_prediction": {
      "predicted_risk_tier": "Very Low",
      "risk_score": 1.14,
      "risk_score_max": 10.0,
      "description": "Safe Corridor - Normal Operations",
      "probabilities": {
        "Very Low": 0.9702,
        "Low": 0.0127,
        "High": 0.0159,
        "Very High": 0.0012
      }
    }
  }
}
```

---

### 3. File Download Endpoint: `GET /download/{file_name}`
* **Description**: Streams output media or audit documents directly to the client.
* **Supported Downloads**:
  - `*_annotated.mp4` (Annotated Video Stream)
  - `*_frame_by_frame_detections.xlsx` (Timestamped 5 FPS Excel Spreadsheet)
  - `*_frame_by_frame_detections.csv` (CSV Log)
  - `*_risk_report.json` (Structured JSON Audit Document)

---

## ⚡ 3. How to Run & Deploy the API Server

### 1. Launch Local API Server (Uvicorn)
```bash
cd FD_v2
python app.py
```
* The API will start on **`http://127.0.0.1:8000`**
* Interactive Swagger UI documentation is available at **`http://127.0.0.1:8000/docs`**

### 2. Test Endpoints via `curl`
```bash
# Check Model Health & Telemetry
curl http://127.0.0.1:8000/models/status

# Upload & Analyze Video Stream
curl -X POST "http://127.0.0.1:8000/analyze_video?sample_fps=5.0&imgsz=1280" \
     -F "file=@DATA/IP/1.mp4"
```
