# 💡 Phase 2 Ideation: Vision-Based Automated Road Safety Analytics

## 🎯 Executive Vision
The primary objective of **Phase 2 (`FD_v2`)** is to transition from manual, survey-based road parameter data collection to an **Automated Computer Vision (CV) Perception Pipeline**. 

In Phase 1 ([Final_Draft](file:///c:/SWAYAMs/PROJ/Road_Safety_params/Final_Draft/summary.md)), a leakage-free, SMOTE-balanced **XGBoost Classifier** was developed and tuned, achieving **90.0% accuracy** in predicting road crash risk tiers (`Very Low`, `Low`, `High`, `Very High`) based on 10 core road safety parameters.

In Phase 2, we eliminate human survey dependency by ingesting **dashcam, drone, or roadside mobile inspection video footage**. An integrated ensemble of specialized Computer Vision models will automatically extract, quantify, and map visual evidence directly into the 10 XGBoost input parameters, enabling **real-time segment-wise road safety risk predictions**.

---

## 🏗️ End-to-End System Pipeline

```
┌─────────────────────────────────────────────────────────┐
│              1. Video Ingestion Module                  │
│       (Dashcam / Drone / Inspection Video Stream)       │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│        2. Multi-Model CV Perception Layer               │
│   ┌───────────────────┬─────────────────────────────┐   │
│   │ Object Detection  │ Semantic Segmentation       │   │
│   │ (YOLOv8 / YOLOv11)│ (U-Net / Segment Anything)  │   │
│   ├───────────────────┼─────────────────────────────┤   │
│   │ Depth Estimation  │ Object Tracking & Motion    │   │
│   │ (Depth Anything)  │ (ByteTrack / DeepSORT)      │   │
│   └───────────────────┴─────────────────────────────┘   │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│         3. Spatial-Temporal Aggregation Engine          │
│     (Aggregates 30fps frames into 100m–500m segments)    │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│        4. Feature Calibration & Mapper Engine           │
│   (Maps pixel counts / TTC / counts ──> Ordinal 0-5)    │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│        5. Tuned XGBoost Risk Inference Engine           │
│    (Outputs Crash Index Score & Risk Classification)    │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│         6. Dashboard & Geographic Risk Mapping          │
│    (Generates segment-level hazard heatmaps & alerts)   │
└─────────────────────────────────────────────────────────┘
```

---

## 🔑 Core Technical Challenges & Solutions

### 1. Spatial-Temporal Resolution Gap
* **Challenge**: Video models output frame-by-frame predictions (e.g., 30 frames per second), whereas road risk is evaluated over spatial segments (e.g., 100m, 500m, or 1km road stretches).
* **Solution**: Implement a rolling spatial aggregation window tied to vehicle speed / GPS telemetry (or frame count estimation) that aggregates frame-level detections using windowed maximums, weighted averages, and event frequency metrics.

### 2. Scale Calibration (Continuous Visual Metrics ➔ Ordinal Model Inputs)
* **Challenge**: CV models output continuous numerical values (e.g., 4 commercial shop bounding boxes per frame, 1.2m crosswalk distance, 8.5m sight distance depth map), but XGBoost was trained on discrete ordinal scales (`0, 1, 2, 5`).
* **Solution**: Build an explicit Calibration Layer with empirical threshold rules and Sigmoidal/Quantile mapping functions to standardize CV outputs into the exact integer scales required by XGBoost.

### 3. Edge Compute Efficiency vs. Accuracy
* **Challenge**: Running multiple heavy vision models (Depth, Segmentation, YOLO, Tracking) simultaneously can bottleneck processing pipelines.
* **Solution**: Implement a cascaded multi-task architecture using TensorRT/ONNX-optimized backbones, running lightweight detection models per frame and triggering heavy depth/segmentation tasks on keyframes or sampled intervals (e.g., 5–10 fps).

---

## 📌 Document Structure in `FD_v2/`

The Phase 2 project architecture is documented across 4 dedicated files:

1. **[Ideation.md](file:///c:/SWAYAMs/PROJ/Road_Safety_params/FD_v2/Ideation.md)** *(This Document)* — High-level vision, core pipeline objectives, and technical challenge strategies.
2. **[Model_Parameters_Outputs.md](file:///c:/SWAYAMs/PROJ/Road_Safety_params/FD_v2/Model_Parameters_Outputs.md)** — Detailed specification of all 10 input parameters, numerical scales, target variables, and risk tier thresholds.
3. **[CV_Models_Perception.md](file:///c:/SWAYAMs/PROJ/Road_Safety_params/FD_v2/CV_Models_Perception.md)** — Deep dive into the specific Computer Vision models, pre-trained architectures, tracking algorithms, and calibration thresholds for extracting each parameter from video.
4. **[Architecture_Workflow.md](file:///c:/SWAYAMs/PROJ/Road_Safety_params/FD_v2/Architecture_Workflow.md)** — Comprehensive Mermaid diagrams illustrating component interactions, data flows, edge inference pipelines, and API schemas.
