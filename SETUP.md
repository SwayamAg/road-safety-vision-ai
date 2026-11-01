# 🛠️ System Setup, Requirements & Installation Guide

This document provides complete instructions for setting up, installing dependencies, configuring hardware acceleration, and deploying the **Road Safety Vision-AI (`FD_v2`)** system across local development, Docker containers, and cloud environments.

---

## 📑 Table of Contents
1. [Prerequisites & System Requirements](#1-prerequisites--system-requirements)
2. [Local Development Setup](#2-local-development-setup)
3. [PyTorch Hardware Acceleration (CPU vs. CUDA GPU)](#3-pytorch-hardware-acceleration-cpu-vs-cuda-gpu)
4. [Pre-Trained Model Weights Verification](#4-pre-trained-model-weights-verification)
5. [Environment Variables & Configuration](#5-environment-variables--configuration)
6. [Running the Application](#6-running-the-application)
7. [Docker Container Setup](#7-docker-container-setup)
8. [Cloud Deployment (Railway & Render)](#8-cloud-deployment-railway--render)
9. [Troubleshooting & FAQs](#9-troubleshooting--faqs)

---

## 1. Prerequisites & System Requirements

### Operating System Compatibility
- **Windows**: Windows 10 / 11 (64-bit)
- **Linux**: Ubuntu 20.04 / 22.04 / 24.04 LTS, Debian 11 / 12
- **macOS**: macOS 12 Monterey or newer (Apple Silicon M1/M2/M3 or Intel)

### Core Software Requirements
| Component | Minimum Version | Recommended Version |
| :--- | :--- | :--- |
| **Python** | 3.10.0 | **3.10.12** |
| **Pip** | 23.0+ | Latest |
| **Git** | 2.30+ | Latest |
| **FFmpeg** | 4.4+ | 6.0+ (Required for video decoding & headless OpenCV) |

### Hardware Recommendations
- **CPU-Only Mode (Cloud Tiers / Laptops)**:
  - 4 vCPUs / Cores
  - 4 GB RAM (Minimum 2 GB with dynamic resolution downscaling to `imgsz=640`)
  - Inference Speed: ~2–5 FPS
- **GPU-Accelerated Mode (Production / Real-Time Audit)**:
  - NVIDIA GPU with **CUDA 11.8 or 12.1+** support (GTX 1660 Ti, RTX 3060/4070/4090, A10, T4, V100)
  - 6 GB+ VRAM (8 GB+ recommended for native 1080p high-resolution processing)
  - Inference Speed: **12–25 FPS**

---

## 2. Local Development Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/<your-repo>/road-safety-vision-ai.git
cd road-safety-vision-ai
```

### Step 2: Create a Dedicated Virtual Environment
Using a clean virtual environment prevents package version conflicts with system-level packages:

```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
python -m venv venv
.\venv\Scripts\activate.bat
```

### Step 3: Upgrade Pip
```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 4: Install Python Dependencies
```bash
pip install -r requirements.txt
```

---

## 3. PyTorch Hardware Acceleration (CPU vs. CUDA GPU)

The default `requirements.txt` points to the **PyTorch CPU wheel index** for lightweight, memory-efficient deployment on standard cloud instances (e.g., Render, Railway free/starter tiers).

### Option A: Standard CPU Mode (Default)
Already pre-configured in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Option B: NVIDIA CUDA GPU Acceleration (Recommended for Production)
If you have an NVIDIA GPU, install the CUDA-accelerated PyTorch binaries directly for a **$5\times$ to $10\times$ speedup**:

```bash
# For CUDA 12.1 (Standard for RTX 30xx, 40xx, A10, T4):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (Legacy GPU instances):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU Detection:
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 4. Pre-Trained Model Weights Verification

All pre-trained deep learning weights and ML artifacts must reside inside the `models/` directory:

| Filename | Framework | Task Description | Approximate Size |
| :--- | :--- | :--- | :--- |
| **`models/yolo11m.pt`** | Ultralytics YOLO11 Medium | Multi-class hazard detection & traffic bounding boxes | ~40.7 MB |
| **`models/midas_v21_small_256.pt`** | Intel ISL MiDaS v2.1 Small | Dense metric depth, sight distance & inter-vehicle ranging | ~85.8 MB |
| **`models/sam2.1_t.pt`** | Meta SAM 2.1 Tiny | Keyframe polygon instance segmentation for defect areas | ~78.1 MB |
| **`models/best_road_safety_model.pkl`** | XGBoost 2.0 + Scikit-Learn | Calibrated 10-parameter road crash risk classifier | ~355 KB |
| **`models/model_classes.pkl`** | Scikit-Learn LabelEncoder | Risk tier categorical label mapping (`Very Low` to `Very High`) | ~1 KB |

### Automatic Downloads
- If `models/yolo11m.pt` is not present locally, Ultralytics will auto-download it on the first execution.
- If `models/midas_v21_small_256.pt` is not present, MiDaS will fetch the weights via PyTorch Hub cache.

---

## 5. Environment Variables & Configuration

Copy the example environment configuration template:
```bash
# Linux / macOS
cp .env.example .env

# Windows
copy .env.example .env
```

### Key Configurable Parameters (`.env`)
```ini
# Server Network Settings
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info

# Storage Directories
MODEL_DIR=models
DATA_IP_DIR=DATA/IP
DATA_OP_DIR=DATA/OP

# Hardware Acceleration ('cuda' or 'cpu')
DEVICE=cuda

# Dashcam Geometry Defaults
CAMERA_HEIGHT_M=1.35
CAMERA_FOV_DEG=65.0

# Road Safety Audit Spatial Capping
DEFAULT_SEGMENT_CAP_M=500.0
DEFAULT_SAMPLE_FPS=5.0
DEFAULT_KEYFRAME_INTERVAL=15
```

---

## 6. Running the Application

### Mode 1: Offline Batch Video Analysis (CLI)
Process a road survey video directly from the terminal without starting the web server:

```bash
# Basic run on sample video:
python main_pipeline.py --video DATA/IP/1.mp4

# Advanced run with custom 500m segment cap, native resolution, and 5 FPS sampling:
python main_pipeline.py \
  --video DATA/IP/1.mp4 \
  --output_dir DATA/OP \
  --cap 500.0 \
  --fps 5.0 \
  --height 1.35 \
  --fov 65.0
```

#### Generated Output Artifacts:
- **`DATA/OP/*_annotated.mp4`**: Video with corner brackets, inter-vehicle distance badges (`7.4m`), and 500m telemetry HUD.
- **`DATA/OP/*_frame_by_frame_detections.csv`**: Timestamped telemetry log with vehicle distance records.
- **`DATA/OP/*_risk_report.json`**: Segment-by-segment XGBoost risk scores and recommendations.

---

### Mode 2: FastAPI REST Microservice
Start the production API server:

```bash
python app.py
```
Or using Uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Swagger Documentation**: `http://127.0.0.1:8000/docs`
- **ReDoc Interactive UI**: `http://127.0.0.1:8000/redoc`
- **Liveness Health Check**: `http://127.0.0.1:8000/health`
- **Model Status & Telemetry**: `http://127.0.0.1:8000/models/status`

---

### Mode 3: Automated API Test Suite
Run the built-in automated test suite to verify endpoints:
```bash
python test_api.py
```

---

## 7. Docker Container Setup

The included multi-stage `Dockerfile` packages all Debian system libraries (`libgl1`, `ffmpeg`, `libgomp1`), dependencies, and models.

### Step 1: Build Docker Image
```bash
docker build -t road-safety-vision-ai:latest .
```

### Step 2: Run Docker Container
```bash
# Standard CPU container:
docker run -d \
  --name road-safety-ai \
  -p 8000:8000 \
  -v $(pwd)/DATA/OP:/app/DATA/OP \
  road-safety-vision-ai:latest

# GPU-accelerated container (requires NVIDIA Container Toolkit):
docker run -d \
  --name road-safety-ai \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/DATA/OP:/app/DATA/OP \
  road-safety-vision-ai:latest
```

### Step 3: Test Container Health
```bash
curl http://localhost:8000/health
```

---

## 8. Cloud Deployment (Railway & Render)

### Deploying on Railway.app
1. Link your GitHub repository in the [Railway Dashboard](https://railway.app).
2. Railway automatically detects `railway.toml` and `Dockerfile`.
3. Set environment variable `PORT=8000` (Railway injects `$PORT` dynamically).
4. Under **Settings -> Healthcheck**, ensure path is `/health` with timeout `180s`.

### Deploying on Render.com
1. Connect repository in [Render Dashboard](https://render.com).
2. Select **Web Service** -> Blueprint (uses [`render.yaml`](render.yaml)).
3. Plan: **Standard** (2 GB+ RAM recommended for video processing).
4. Build Command: `pip install --upgrade pip && pip install -r requirements.txt`.
5. Start Command: `python -m uvicorn app:app --host 0.0.0.0 --port $PORT`.

---

## 9. Troubleshooting & FAQs

### Q1: `ImportError: libGL.so.1: cannot open shared object file` (Linux / Docker)
**Fix**: Install headless OpenGL bindings:
```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0 libgomp1
```
Or ensure `opencv-python-headless` is used instead of standard `opencv-python`.

### Q2: Out of Memory (OOM) during video processing on low-tier CPUs
**Fix**: 
1. Pass downscaled optical resolution: `--imgsz 640` or query parameter `?imgsz=640`.
2. Lower the sampling frame rate from `5.0` to `2.0` FPS: `?sample_fps=2.0`.
3. Increase `keyframe_interval` from `15` to `30` to invoke SAM 2 less frequently.

### Q3: `No module named 'torchvision.transforms.functional_tensor'`
**Fix**: Caused by version mismatch with older MiDaS transforms. Ensure `torchvision>=0.16.0` and `timm>=1.0.0` are installed as specified in `requirements.txt`.

### Q4: Video codec error writing `.mp4` output
**Fix**: The pipeline defaults to `mp4v` codec which is universally supported across Linux and Windows without proprietary licensing issues. If required for HTML5 browser streaming, convert the output using FFmpeg:
```bash
ffmpeg -i DATA/OP/input_annotated.mp4 -vcodec libx264 -crf 23 DATA/OP/input_h264.mp4
```

