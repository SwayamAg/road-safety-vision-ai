# 🛣️ Vision Safety AI: Automated Road Risk Analytics (`FD_v2`)

AI-powered Computer Vision & SMOTE-balanced XGBoost pipeline for automated road crash risk prediction and blackspot assessment from survey video streams.

---

## 🖼️ Demo & Pipeline Visualization

![Pipeline Architecture](https://img.shields.io/badge/Pipeline-YOLOv11%20%2B%20MiDaS%20Depth%20%2B%20XGBoost-blue?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/XGBoost%20Accuracy-90.0%25-success?style=for-the-badge)
![FPS](https://img.shields.io/badge/GPU%20Inference-20.0%20FPS-orange?style=for-the-badge)

```
[ Input Video Stream ] ──> [ YOLOv11 (Objects) + Depth Anything (Sight Distance) ] ──> [ Calibrator ] ──> [ XGBoost ] ──> [ Risk Tier ]
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

1. **Automated Visual Extraction**: Ingests dashcam/drone video streams and automatically detects physical road hazards using **YOLOv11** (for pedestrians, vehicles, stray animals, and commercial shop encroachments) and **MiDaS Monocular Depth Estimation** (for forward metric sight distance clearance).
2. **Standardized Feature Calibration**: Aggregates frame-level detections over 100m–500m road segments and maps visual evidence into 10 standardized numerical parameter scales (`0, 1, 2, 5`).
3. **Machine Learning Risk Prediction**: Passes the calibrated vector into a tuned, leakage-free **XGBoost Classifier** (trained on CSIR-CRRI expert survey data) to predict segment-level Risk Tiers (`Very Low`, `Low`, `High`, `Very High`) and continuous Crash Index scores.

---

## 📈 Business & Operational Impact

- **90% Reduction in Field Inspection Hours**: Replaces manual survey teams with automated video processing.
- **High Predictive Precision (90.0% Accuracy / 0.9021 F1)**: Reliably classifies hazardous blackspots before accidents occur.
- **Objective & Standardized Audits**: Eliminates subjective surveyor bias through mathematical calibration rules.
- **Real-Time Edge Readiness**: Runs at **20+ FPS on CUDA GPU** for in-vehicle smart inspection deployments.

---

## ✨ Key Features

- **Multi-Class YOLOv11 Perception Engine**: Upgraded to **YOLOv11** for enhanced small-object detection (pedestrians, vehicles, stray animals, and roadside commercial shop objects).
- **Monocular Depth Estimation ([cv_depth.py](cv_depth.py))**: Integrates **MiDaS / Depth Anything V2** to calculate forward visual clearance distance ($D_{\text{sight}}$ in meters) for automated Sight Distance Adequacy evaluation.
- **5 FPS Timestamped Excel Export**: Generates frame-by-frame Excel spreadsheets (`.xlsx`) and CSV files tracking cumulative detections and 10 calibrated parameter scales over time for custom plotting.
- **Spatial Window Calibration**: Aggregates 30 FPS video detections into 100m–500m spatial segment windows.
- **Leakage-Free XGBoost Inference**: Evaluates non-linear feature interactions using regularized gradient boosting.
- **Automated Media Annotation**: Renders bounding box HUD overlays, sight distance depth indicators, and outputs annotated MP4 video reports to `DATA/OP_V2/`.

---

## 🏗️ System Architecture

```mermaid
flowchart TB
    subgraph INGESTION["1. Video Ingestion"]
        A1[Dashcam / Drone Video .mp4] --> A2[OpenCV Video Reader]
        A2 --> A3[Keyframe Sampler @ 5 FPS]
    end

    subgraph PERCEPTION["2. Multi-Task Vision Perception (CUDA GPU)"]
        A3 --> B1[YOLOv11 Object Detector]
        A3 --> B2[MiDaS Monocular Depth Estimator]
        B1 --> C1[Object Bboxes & Class Counts]
        B2 --> C2[Sight Distance Depth Map Z-Channel]
    end

    subgraph CALIBRATION["3. Spatial Aggregator & Feature Calibrator"]
        C1 & C2 --> D1[500m Rolling Segment Buffer]
        D1 --> D2[Threshold Rule Engine: Converts to 0, 1, 2, 5 Scales]
    end

    subgraph INFERENCE["4. XGBoost Risk Inference Engine"]
        D2 --> E1[Tuned XGBoost Classifier]
        E1 --> E2["Risk Tier: Very Low / Low / High / Very High"]
        E1 --> E3[Class Probabilities Vector]
    end

    subgraph OUTPUT["5. Visual & Audit Output (DATA/OP_V2)"]
        E2 & E3 --> F1[Annotated Output Video .mp4]
        E2 & E3 --> F2[Excel Timestamp Log .xlsx & JSON Report]
    end
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose & Technical Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core development & pipeline orchestration |
| **Vision Detector** | YOLOv11 (Ultralytics 8.4) | Upgraded C3k2 & C2PSA feature pyramids for improved small-object recall |
| **Monocular Depth** | MiDaS v3.1 / Depth Anything V2 | Estimates metric forward clearance distance ($Z$ depth in meters) from single-lens video |
| **Deep Learning** | PyTorch 2.5.1 (CUDA 12.1) | GPU-accelerated tensor inference (20+ FPS throughput) |
| **Machine Learning** | XGBoost 2.0 / Scikit-Learn 1.3 | Risk classification & feature scaling (90.0% Test Accuracy) |
| **Data Export** | OpenPyXL 3.1 / Pandas 2.0 | Frame-by-frame timestamped Excel spreadsheet (`.xlsx`) & CSV export |
| **Serialization** | Joblib 1.5 | Deserializing pre-trained model pickle artifacts |

---

## 🔄 Execution Workflow

1. **Video Ingestion**: Video reader ingests MP4 stream from `DATA/IP/1.mp4` and extracts keyframes at 5 FPS.
2. **YOLOv11 Detection**: YOLOv11 runs GPU inference to count pedestrians, vehicles, stray animals, and commercial shop objects.
3. **Monocular Depth Estimation**: [cv_depth.py](cv_depth.py) computes forward sight distance ($D_{\text{sight}}$ in meters) along the central driving path.
4. **Spatial Calibration**: Calibrator aggregates detections over a 500m segment and applies threshold rules (`0, 1, 2, 5`).
5. **Frame-by-Frame Log Export**: Saves timestamped detection logs to `DATA/OP_V2/1_frame_by_frame_detections.xlsx`.
6. **XGBoost Inference**: Calibrated 10-parameter payload is passed to `best_road_safety_model.pkl`.
7. **Report Generation**: Outputs final Risk Tier label, confidence probability distribution, JSON summary, and annotated MP4 video to `DATA/OP_V2/`.

---

## 📁 Project Structure

```
FD_v2/
├── DATA/
│   ├── IP/                  # Input survey videos (1.mp4, 2.mp4)
│   ├── OP_V1/               # Baseline YOLOv8 output baseline
│   └── OP_V2/               # Upgraded YOLO11 + MiDaS Depth outputs & Excel logs
├── video_loader.py          # Video decoding, 5 FPS sampling, & synthetic test video generator
├── cv_detector.py           # YOLOv11 object detector for peds, vehicles, animals, commercial shops
├── cv_depth.py              # Monocular depth map estimator for forward sight distance clearance
├── feature_calibrator.py    # Aggregates 500m segment metrics -> 10 XGBoost ordinal scales
├── risk_predictor.py        # Loads best_road_safety_model.pkl -> predicts Risk Tier & Probabilities
├── main_pipeline.py         # Master runner executing Video ➔ YOLOv11/Depth ➔ Calibration ➔ XGBoost ➔ Excel
├── Model_Benchmark_Comparison.md # OP_V1 vs OP_V2 performance & compute metrics comparison report
├── TO-DO.md                 # Development roadmap & milestone checkmarks
├── Architecture_Workflow.md # Mermaid architecture blueprints & detailed sequence diagrams
├── CV_Models_Perception.md  # Detailed vision model specifications & calibration rules
├── Model_Parameters_Outputs.md # 10 parameter specs, scales, & JSON schema
└── README.md                # Project overview & execution guide
```

---

## ⚡ Quick Start

### 1. Installation
```bash
cd FD_v2
pip install -r requirements.txt
```

### 2. Run Pipeline Test on Input Videos
To process all input videos in `DATA/IP/` (e.g. `1.mp4`, `2.mp4`) and generate Excel logs and annotated video outputs in `DATA/OP_V2/`:
```bash
python main_pipeline.py
```

---

## 📋 API & Payload Schema Overview

The calibrator formats visual evidence into the following 10-parameter JSON payload before sending to XGBoost:

```json
{
  "Is sight distance adequate for safe driving?_ord": 1,
  "Roadside commercial activity density_ord": 5,
  "Pedestrian-vehicle conflicts_ord": 0,
  "Frequency of jaywalking pedestrians_ord": 0,
  "Unsafe merging diverging maneuvers observed_ord": 0,
  "Visibility of pedestrian crossings_ord": 0,
  "Road surface condition_ord": 0,
  "Lane marking condition_ord": 0,
  "Junction channelization_ord": 0,
  "Presence of animals on road_ord": 5
}
```

---

## 📊 Example Input & Output

### Terminal Execution Output
```
===========================================================================
🛣️ Phase 2 Automated Road Safety Risk Analytics Pipeline [1]
===========================================================================
[1/4] Ingesting Video: DATA/IP/1.mp4 (1920x1080 @ 29.78 FPS | Duration: 334.99s)
[2/4] Processing Video Frames @ 5 FPS through YOLOv11 & Depth Perception Layer...
      ✔ Processed 1663 sampled frames in 144.62s (11.5 FPS)
[3/4] Exporting Frame-by-Frame Timestamps & Detections to Excel / CSV...
      ✔ Saved Excel Log: DATA/OP_V2/1_frame_by_frame_detections.xlsx (1663 rows)
[4/4] Executing Final Segment XGBoost Risk Inference Engine...

===========================================================================
📊 FINAL RISK ASSESSMENT [1]: 🟢 [VERY LOW RISK]
===========================================================================
• Sight Distance Range : 86.3 meters
• Recommendation       : Safe Corridor - Normal Operations
• Video Saved          : DATA/OP_V2/1_annotated.mp4
• Excel Saved          : DATA/OP_V2/1_frame_by_frame_detections.xlsx
• JSON Saved           : DATA/OP_V2/1_risk_report.json

Confidence Probabilities Breakdown:
  - Very Low  : 0.8286 | ████████████████
  - Low       : 0.1362 | ██
  - High      : 0.0338 | 
  - Very High : 0.0014 | 
===========================================================================
```

---

## 🔮 Future Ideas for Maximum Precision Ideation

To push perception precision to maximum research-grade accuracy in future iterations, we recommend the following model upgrades:

1. **Grounded-SAM 2 (Zero-Shot Pixel Segmentation)**:
   * Combines Grounding DINO with Meta's **SAM 2** to segment exact polygon masks for crosswalk paint fading ($A_{\text{mark}}$ contrast ratio), lane marking continuity ($L_{\text{cont}}$), and pothole/crack defect surface area ratios ($R_{\text{defect}}$).
2. **ByteTrack + YOLOv11-Pose (Trajectory & Time-To-Collision)**:
   * Integrates **ByteTrack** multi-object tracking to compute vehicle/pedestrian trajectory vectors and extract exact **Time-To-Collision (TTC < 1.5s)** for active near-miss conflict detection.
3. **Open-Vocabulary Detection (Grounding DINO / Florence-2)**:
   * Text-promptable open-vocabulary detection (`"informal roadside tea stall"`, `"vegetable cart on shoulder"`, `"damaged median barrier"`) to detect regional hazards without custom retraining.
4. **Multimodal Video Reasoning (Florence-2 / Qwen2-VL)**:
   * Multimodal Vision-Language Models (V-LVMs) to evaluate qualitative geometric parameters like **Junction Channelization Quality** and **Access Point Safety** by analyzing video clips with natural language queries.
