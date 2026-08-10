# 📋 Phase 2 Development Roadmap & TO-DO List (`FD_v2`)

This document outlines the step-by-step development roadmap for **Phase 2 (`FD_v2`)**: building an end-to-end Vision AI & Computer Vision perception system to automatically extract the 10 XGBoost road safety parameters from video footage.

---

## 📌 Development Milestones & Task Status

### 🏁 Milestone 1: Environment & Model Artifact Linking
- [x] **Task 1.1**: Define Phase 2 architecture, parameter specs, and workflow diagrams ([Architecture_Workflow.md](Architecture_Workflow.md)).
- [x] **Task 1.2**: Create Phase 2 environment config ([requirements.txt](requirements.txt) & [.gitignore](.gitignore)).
- [x] **Task 1.3**: Sync `Phase2` branch with GitHub remote repository.
- [x] **Task 1.4**: Verify Python environment dependencies (`torch`, `ultralytics`, `opencv-python`, `joblib`).
- [x] **Task 1.5**: Set up test video input source (`1.mp4` & `2.mp4`).

---

### 📹 Milestone 2: Core Perception Module Development
- [x] **Task 2.1**: **[video_loader.py](video_loader.py)**
  - Implemented `VideoFrameReader` class to decode MP4 streams.
  - Implemented keyframe sampling (5 fps extraction) and frame resizing (640x640 / 1080p).
- [x] **Task 2.2**: **[cv_detector.py](cv_detector.py)**
  - Upgraded to **YOLOv11** (`yolov11n.pt`) for multi-class object detection.
  - Implemented detection and HUD annotation for pedestrians, vehicles, stray animals, and commercial roadside objects.
- [x] **Task 2.3**: **[cv_depth.py](cv_depth.py)** *(Short-Term Implemented)*
  - Implemented **SightDistanceDepthEstimator** using monocular depth maps to calculate metric forward sight distance clearance ($D_{\text{sight}}$ in meters).
- [ ] **Task 2.4**: **`cv_tracker.py`** *(Optional Trajectory Tracker Module)*
  - Tracking logic integrated into multi-frame accumulator.

---

### ⚙️ Milestone 3: Feature Calibration & Aggregation Engine
- [x] **Task 3.1**: **[feature_calibrator.py](feature_calibrator.py)**
  - Implemented rolling spatial window buffer (aggregates frame metrics over 100m–500m segments).
- [x] **Task 3.2**: Implemented Calibration Rules Engine:
  - Converts continuous visual metrics (shop counts, depth meters, TTC near-misses) into the 10 integer ordinal scales (`0, 1, 2, 5`).
- [x] **Task 3.3**: Validated output schema against standard 10-parameter JSON dictionary payload.

---

### 🌲 Milestone 4: XGBoost Risk Inference Engine
- [x] **Task 4.1**: **[risk_predictor.py](risk_predictor.py)**
  - Loaded pre-trained `best_road_safety_model.pkl`.
  - Passes calibrated 10-parameter dictionary vector to produce:
    - **Risk Tier Label** (`Very Low`, `Low`, `High`, `Very High`)
    - **Continuous Crash Index Score** (Range `0.00` – `10.00`)
    - **Class Probability Vector** (`[P_VeryLow, P_Low, P_High, P_VeryHigh]`)

---

### 🚀 Milestone 5: End-to-End Pipeline & Excel Export
- [x] **Task 5.1**: **[main_pipeline.py](main_pipeline.py)**
  - Connected Video ➔ YOLOv11 & Depth Perception ➔ Calibration ➔ XGBoost Inference in a unified CLI runner.
- [x] **Task 5.2**: Exported 5 FPS timestamped Excel spreadsheets (`1_frame_by_frame_detections.xlsx` & `2_frame_by_frame_detections.xlsx`).
- [ ] **Task 5.3**: Commit & push Phase 2 codebase updates to GitHub `Phase2` branch *(Awaiting User Approval)*.
