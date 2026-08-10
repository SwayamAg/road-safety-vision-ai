# 📸 Computer Vision Models & Perception Pipeline (`FD_v2`)

This document details the Computer Vision (CV) architectures, pre-trained models, detection algorithms, technical rationales, and threshold calibration logic used to extract the 10 XGBoost input parameters automatically from road survey video streams.

---

## 🚀 1. Implemented Short-Term Vision AI Models & Technical Rationale

The perception system deploys a hybrid multi-task perception stack:

```
                              ┌─────────────────────────────────────────┐
                              │  Input Road Video (30 fps / 1080p Stream)│
                              └────────────────────┬────────────────────┘
                                                   │
         ┌────────────────────────┬────────────────┴───────────────┬────────────────────────┐
         │                        │                                │                        │
         ▼                        ▼                                ▼                        ▼
┌──────────────────┐    ┌──────────────────┐            ┌──────────────────┐    ┌──────────────────┐
│ Object Detection │    │ Monocular Depth  │            │ Multi-Object     │    │ Semantic         │
│  (YOLOv11x)      │    │  (MiDaS / Depth) │            │ Tracking (MOT)   │    │ Segmentation     │
│ [Short-Term Impl]│    │ [Short-Term Impl]│            │ (ByteTrack / TTC)│    │ (Grounded-SAM 2) │
└────────┬─────────┘    └────────┬─────────┘            └────────┬─────────┘    └────────┬─────────┘
         │                        │                                │                        │
         └────────────────────────┴────────────────┬───────────────┴────────────────────────┘
                                                   │
                                                   ▼
                                ┌──────────────────────────────────────┐
                                │ Feature Extraction & Calibration Engine│
                                └──────────────────────────────────────┘
```

### A. Multi-Class Object Detection: **YOLOv11x (Short-Term Implemented Upgrade)**
* **Why YOLOv11 over YOLOv8**:
  1. **Enhanced Small-Object Feature Pyramids**: YOLOv11 introduces **C3k2** and **C2PSA** feature extraction blocks that preserve spatial resolution for distant pedestrian, animal, and roadside shop detections.
  2. **Superior Speed/Accuracy Trade-off**: Delivers higher mAP50-95 object recall while running at high real-time throughput (**~20+ FPS on CUDA GPU**).
* **Target Classes**: `person` (pedestrians), `car`, `motorcycle`, `bus`, `truck` (vehicles), `cow`, `dog`, `horse`, `sheep` (animals), `commercial_shop` (roadside friction objects).

### B. Monocular Metric Depth Estimation: **Depth Anything V2 / MiDaS (Short-Term Implemented)**
* **Why Monocular Depth**: Traditional depth estimation requires dual-camera stereo rigs or expensive LiDAR units. Monocular depth models infer metric distance $Z$ (in meters) directly from standard single-lens dashcam video.
* **Role**: Computes forward obstacle clearance distance along the driving corridor ($D_{\text{sight}}$ in meters) to automate **Sight Distance Adequacy**.

---

## 🔬 2. Parameter-by-Parameter CV Perception & Calibration Matrix

### 1. Sight Distance Adequacy (`Is sight distance adequate for safe driving?_ord`)
* **Primary CV Task**: Monocular Depth Estimation & Forward Clearance Mapping
* **Architecture**: **Depth Anything V2** / **MiDaS v3.1** ([cv_depth.py](cv_depth.py))
* **Extraction Logic**: 
  1. Extract central driving corridor depth values from frame tensor.
  2. Calculate un-occluded forward clearance distance $D_{\text{sight}}$ in meters along driver horizon path.
* **Calibration Rules**:
  $$\text{Sight\_Distance\_Score} = \begin{cases} 0 \quad (\text{Adequate}) & \text{if } D_{\text{sight}} \ge 100\text{m} \\ 1 \quad (\text{Partly}) & \text{if } 50\text{m} \le D_{\text{sight}} < 100\text{m} \\ 2 \quad (\text{No / Blind}) & \text{if } D_{\text{sight}} < 50\text{m} \end{cases}$$

---

### 2. Roadside Commercial Activity Density (`Roadside commercial activity density_ord`)
* **Primary CV Task**: Multi-Class Object Detection
* **Architecture**: **YOLOv11x** ([cv_detector.py](cv_detector.py))
* **Target Classes**: `shop`, `vendor_stall`, `billboard_encroachment`, `parked_commercial_vehicle`
* **Extraction Logic**: Count non-duplicated roadside commercial bounding boxes per 100m window ($N_{\text{shops}}$).
* **Calibration Rules**:
  $$\text{Commercial\_Density\_Score} = \begin{cases} 0 & \text{if } N_{\text{shops}} = 0 \\ 2 & \text{if } 1 \le N_{\text{shops}} \le 2 \\ 3 & \text{if } 3 \le N_{\text{shops}} \le 5 \\ 5 & \text{if } N_{\text{shops}} > 5 \end{cases}$$

---

### 3. Pedestrian–Vehicle Conflicts (`Pedestrian–vehicle conflicts_ord`)
* **Primary CV Task**: Multi-Object Tracking (MOT) & Trajectory Conflict Analysis
* **Architecture**: **YOLOv11x** + **ByteTrack** / **DeepSORT**
* **Extraction Logic**: Track pedestrian and vehicle vectors to calculate Time-To-Collision ($\text{TTC} = \frac{\|\vec{P}(t) - \vec{V}(t)\|}{\Delta v}$). Count near-misses where $\text{TTC} < 1.5\text{s}$.
* **Calibration Rules**:
  $$\text{Pedestrian\_Conflict\_Score} = \begin{cases} 0 & \text{if } C_{\text{ped}} = 0 \\ 2 & \text{if } 1 \le C_{\text{ped}} \le 3 \\ 5 & \text{if } C_{\text{ped}} > 3 \end{cases}$$

---

### 4. Frequency of Jaywalking Pedestrians (`Frequency of jaywalking pedestrians_ord`)
* **Primary CV Task**: Crosswalk Masking + Pedestrian Trajectory Tracking
* **Architecture**: **YOLOv11-Pose** + **Grounded-SAM 2**
* **Extraction Logic**: Segment zebra crosswalk masks ($M_{\text{crosswalk}}$) and count pedestrians crossing outside designated crosswalk masks ($N_{\text{jaywalk}}$).
* **Calibration Rules**:
  $$\text{Jaywalking\_Score} = \begin{cases} 0 & \text{if } N_{\text{jaywalk}} = 0 \text{ per min} \\ 2 & \text{if } 1 \le N_{\text{jaywalk}} \le 3 \text{ per min} \\ 5 & \text{if } N_{\text{jaywalk}} > 3 \text{ per min} \end{cases}$$

---

### 5. Presence of Animals on Road (`Presence of animals on road_ord`)
* **Primary CV Task**: Multi-Class Stray Animal Detection
* **Architecture**: **YOLOv11x** ([cv_detector.py](cv_detector.py))
* **Target Classes**: `cow`, `dog`, `horse`, `sheep`, `stray_animal`
* **Extraction Logic**: Count stray animals on active carriageway per 500m window ($N_{\text{animal}}$).
* **Calibration Rules**:
  $$\text{Animal\_Presence\_Score} = \begin{cases} 0 & \text{if } N_{\text{animal}} = 0 \\ 1 & \text{if } N_{\text{animal}} = 1 \quad (\text{Rare}) \\ 5 & \text{if } N_{\text{animal}} \ge 2 \quad (\text{Frequent}) \end{cases}$$

---

## 🔮 3. Future Maximum Precision Ideations & Roadmap

To achieve absolute state-of-the-art precision in future iterations, we recommend the following model upgrades:

### 1. **Grounded-SAM 2 (Zero-Shot Pixel Segmentation)**
* **Future Application**: Combines Grounding DINO with Meta's SAM 2 to generate pixel-exact masks for crosswalk fading ($A_{\text{mark}}$ contrast ratio) and pothole/crack defect surface area ratios ($R_{\text{defect}}$).

### 2. **Grounding DINO & Florence-2 (Open-Vocabulary Encroachment Detection)**
* **Future Application**: Replaces COCO bounding boxes with text-promptable open-vocabulary detection (`"informal roadside tea stall"`, `"vegetable cart on shoulder"`, `"damaged median barrier"`) to detect regional hazards without custom retraining.

### 3. **Video Vision-Language Models (Qwen2-VL / Florence-2-Large)**
* **Future Application**: Multimodal reasoning models that evaluate qualitative road safety parameters like **Junction Channelization Quality** by analyzing video clips with natural language prompts.
