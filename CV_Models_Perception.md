# 📸 Computer Vision Models & Perception Pipeline (`FD_v2`)

This document provides a technical specification of the Computer Vision (CV) architectures, pre-trained models, detection algorithms, and threshold calibration logic used to extract the 10 XGBoost input parameters automatically from road survey video streams.

---

## 🛠️ Overview of CV Task Framework

The visual perception system is divided into **4 Core Computer Vision Sub-Systems**:

```
                              ┌─────────────────────────────────────────┐
                              │  Input Road Video (30 fps / 1080p Stream)│
                              └────────────────────┬────────────────────┘
                                                   │
         ┌────────────────────────┬────────────────┴───────────────┬────────────────────────┐
         │                        │                                │                        │
         ▼                        ▼                                ▼                        ▼
┌──────────────────┐    ┌──────────────────┐            ┌──────────────────┐    ┌──────────────────┐
│  Multi-Class     │    │  Semantic        │            │  Monocular       │    │ Multi-Object     │
│  Object Detect   │    │  Segmentation    │            │  Depth Map       │    │ Tracking (MOT)   │
│ (YOLOv8 / v11)   │    │ (U-Net / SAM 2)  │            │ (Depth Anything) │    │(ByteTrack/DeepSORT)│
└────────┬─────────┘    └────────┬─────────┘            └────────┬─────────┘    └────────┬─────────┘
         │                        │                                │                        │
         └────────────────────────┴────────────────┬───────────────┴────────────────────────┘
                                                   │
                                                   ▼
                                ┌──────────────────────────────────────┐
                                │ Feature Extraction & Calibration Engine│
                                └──────────────────────────────────────┘
```

---

## 🔬 Parameter-by-Parameter CV Perception & Calibration Specification

### 1. Sight Distance Adequacy (`Is sight distance adequate for safe driving?_ord`)
* **Primary CV Task**: Monocular Depth Estimation & Obstacle Horizon Mapping
* **Recommended Architecture**: **Depth Anything V2** (Small/Base backbone) or **MiDaS v3.1**
* **Extraction Logic**: 
  1. Generate metric depth map per frame estimating distance $Z$ (in meters) along the center driving corridor.
  2. Compute max un-occluded depth range $D_{\text{sight}}$ within the ego-lane path.
* **Calibration Threshold Rules**:
  $$\text{Sight\_Distance\_Score} = \begin{cases} 0 \quad (\text{Adequate}) & \text{if } D_{\text{sight}} \ge 100\text{m} \\ 1 \quad (\text{Partly}) & \text{if } 50\text{m} \le D_{\text{sight}} < 100\text{m} \\ 2 \quad (\text{No / Blind}) & \text{if } D_{\text{sight}} < 50\text{m} \end{cases}$$

---

### 2. Roadside Commercial Activity Density (`Roadside commercial activity density_ord`)
* **Primary CV Task**: Multi-Class Object Detection
* **Recommended Architecture**: **YOLOv8x** / **YOLOv11x** (Fine-tuned on roadside urban assets dataset)
* **Target Classes**: `shop`, `vendor_stall`, `billboard_encroachment`, `parked_commercial_vehicle`
* **Extraction Logic**:
  1. Define a lateral Region of Interest (ROI) mask corresponding to roadside shoulders.
  2. Count non-duplicated roadside commercial bounding boxes per 100m road segment ($N_{\text{shops}}$).
* **Calibration Threshold Rules**:
  $$\text{Commercial\_Density\_Score} = \begin{cases} 0 & \text{if } N_{\text{shops}} = 0 \\ 2 & \text{if } 1 \le N_{\text{shops}} \le 2 \\ 3 & \text{if } 3 \le N_{\text{shops}} \le 5 \\ 5 & \text{if } N_{\text{shops}} > 5 \end{cases}$$

---

### 3. Pedestrian–Vehicle Conflicts (`Pedestrian–vehicle conflicts_ord`)
* **Primary CV Task**: Multi-Object Tracking (MOT) & Trajectory Conflict Analysis
* **Recommended Architecture**: **YOLOv8** + **ByteTrack** / **DeepSORT**
* **Extraction Logic**:
  1. Track pedestrian vectors $\vec{P}(t)$ and vehicle vectors $\vec{V}(t)$.
  2. Compute Time-To-Collision (TTC) for intersecting trajectories: $\text{TTC} = \frac{\|\vec{P}(t) - \vec{V}(t)\|}{\Delta v}$.
  3. Count critical near-miss conflict events where $\text{TTC} < 1.5\text{ seconds}$ ($C_{\text{ped}}$).
* **Calibration Threshold Rules**:
  $$\text{Pedestrian\_Conflict\_Score} = \begin{cases} 0 & \text{if } C_{\text{ped}} = 0 \text{ conflicts/segment} \\ 2 & \text{if } 1 \le C_{\text{ped}} \le 3 \text{ conflicts/segment} \\ 5 & \text{if } C_{\text{ped}} > 3 \text{ conflicts/segment} \end{cases}$$

---

### 4. Frequency of Jaywalking Pedestrians (`Frequency of jaywalking pedestrians_ord`)
* **Primary CV Task**: Crosswalk Segmentation + Pedestrian Position Mapping
* **Recommended Architecture**: **YOLOv8-Pose / Det** + **Segment Anything (SAM 2)** / **U-Net**
* **Extraction Logic**:
  1. Segment marked zebra crosswalk regions ($M_{\text{crosswalk}}$).
  2. Detect pedestrians crossing the carriageway ($N_{\text{crossing}}$).
  3. Count pedestrians crossing OUTSIDE the crosswalk mask ($N_{\text{jaywalk}}$).
* **Calibration Threshold Rules**:
  $$\text{Jaywalking\_Score} = \begin{cases} 0 & \text{if } N_{\text{jaywalk}} = 0 \text{ per minute} \\ 2 & \text{if } 1 \le N_{\text{jaywalk}} \le 3 \text{ per minute} \\ 5 & \text{if } N_{\text{jaywalk}} > 3 \text{ per minute} \end{cases}$$

---

### 5. Unsafe Merging / Diverging Maneuvers (`Unsafe merging diverging maneuvers observed_ord`)
* **Primary CV Task**: Vehicle Velocity Tracking & Optical Flow Displacement
* **Recommended Architecture**: **RAFT / Dense Optical Flow** + **YOLOv8 + ByteTrack**
* **Extraction Logic**:
  1. Track vehicle heading angles $\theta(t)$ and lateral acceleration $\frac{d^2 y}{dt^2}$ at merge points.
  2. Detect sudden cut-ins, sharp lane crosses without signal, or lateral clearance violations ($M_{\text{unsafe}}$).
* **Calibration Threshold Rules**:
  $$\text{Unsafe\_Merging\_Score} = \begin{cases} 0 & \text{if } M_{\text{unsafe}} = 0 \\ 2 & \text{if } 1 \le M_{\text{unsafe}} \le 2 \\ 5 & \text{if } M_{\text{unsafe}} > 2 \end{cases}$$

---

### 6. Visibility of Pedestrian Crossings (`Visibility of pedestrian crossings_ord`)
* **Primary CV Task**: Semantic Segmentation & Local Contrast Evaluation
* **Recommended Architecture**: **U-Net** / **DeepLabV3+** (Trained on road marking dataset)
* **Extraction Logic**:
  1. Detect crosswalk presence in frame.
  2. Calculate paint mask area ratio $A_{\text{mark}}$ and average pixel contrast ratio $C_{\text{ratio}}$ against surrounding asphalt.
* **Calibration Threshold Rules**:
  $$\text{Crosswalk\_Visibility\_Score} = \begin{cases} 0 \quad (\text{Good}) & \text{if } A_{\text{mark}} > 70\% \text{ and } C_{\text{ratio}} \ge 0.6 \\ 1 \quad (\text{Poor}) & \text{if } 30\% \le A_{\text{mark}} \le 70\% \text{ or } C_{\text{ratio}} < 0.6 \\ 2 \quad (\text{Absent}) & \text{if no crosswalk detected or } A_{\text{mark}} < 30\% \end{cases}$$

---

### 7. Road Surface Condition (`Road surface condition_ord`)
* **Primary CV Task**: Pavement Defect Detection & Surface Roughness Classification
* **Recommended Architecture**: **YOLOv8-Seg** / **ResNet-50 Classifier** (Fine-tuned on RDD2022 Pothole Dataset)
* **Target Classes**: `pothole`, `alligator_crack`, `longitudinal_crack`, `rutting`
* **Extraction Logic**:
  1. Segment road pavement ROI.
  2. Calculate total defect surface area ratio $R_{\text{defect}} = \frac{\text{Area}_{\text{defects}}}{\text{Area}_{\text{road}}}$.
* **Calibration Threshold Rules**:
  $$\text{Surface\_Condition\_Score} = \begin{cases} 0 \quad (\text{Good}) & \text{if } R_{\text{defect}} < 2\% \\ 1 \quad (\text{Minor}) & \text{if } 2\% \le R_{\text{defect}} \le 10\% \\ 2 \quad (\text{Damaged}) & \text{if } R_{\text{defect}} > 10\% \end{cases}$$

---

### 8. Lane Marking Condition (`Lane marking condition_ord`)
* **Primary CV Task**: Line Detection & Structural Continuity Tracking
* **Recommended Architecture**: **CLRNet** / **SCNN (Spatial CNN)** / **Hough Transform + U-Net**
* **Extraction Logic**:
  1. Detect left and right ego-lane boundary lines.
  2. Measure lane line continuity ratio $L_{\text{cont}} = \frac{\text{Detected Line Length}}{\text{Expected Corridor Length}}$.
* **Calibration Threshold Rules**:
  $$\text{Lane\_Marking\_Score} = \begin{cases} 0 \quad (\text{Clear}) & \text{if } L_{\text{cont}} \ge 80\% \\ 1 \quad (\text{Faded}) & \text{if } 40\% \le L_{\text{cont}} < 80\% \\ 2 \quad (\text{Absent}) & \text{if } L_{\text{cont}} < 40\% \end{cases}$$

---

### 9. Junction Channelization (`Junction channelization_ord`)
* **Primary CV Task**: Junction Layout Detection & Geometric Island Segmentation
* **Recommended Architecture**: **YOLOv8** (Intersection Objects) + **Mask R-CNN**
* **Target Classes**: `median_island`, `roundabout`, `channelizing_curb`, `unstructured_intersection`
* **Extraction Logic**:
  1. Identify intersection approach frames.
  2. Check for physical or painted channelizing islands separating turning flows.
* **Calibration Threshold Rules**:
  $$\text{Channelization\_Score} = \begin{cases} 0 \quad (\text{Proper}) & \text{if physical/painted islands guide all turns} \\ 1 \quad (\text{Poor}) & \text{if partial or damaged channelization} \\ 2 \quad (\text{Absent}) & \text{if open, unchannelized intersection} \end{cases}$$

---

### 10. Presence of Animals on Road (`Presence of animals on road_ord`)
* **Primary CV Task**: Multi-Class Animal Object Detection
* **Recommended Architecture**: **YOLOv8x** (COCO animal classes + custom stray animal dataset)
* **Target Classes**: `cow`, `dog`, `horse`, `sheep`, `stray_animal`
* **Extraction Logic**:
  1. Detect animals inside or within 2 meters of the active carriageway.
  2. Count animal detection instances per 500m segment ($N_{\text{animal}}$).
* **Calibration Threshold Rules**:
  $$\text{Animal\_Presence\_Score} = \begin{cases} 0 & \text{if } N_{\text{animal}} = 0 \\ 1 & \text{if } N_{\text{animal}} = 1 \quad (\text{Rare}) \\ 5 & \text{if } N_{\text{animal}} \ge 2 \quad (\text{Frequent}) \end{cases}$$

---

## 🎛️ Summary Table of CV Models & Frameworks

| Model Category | Primary Framework | Pre-trained Weights / Dataset | Target Output |
| :--- | :--- | :--- | :--- |
| **Object Detection** | YOLOv8x / YOLOv11x | COCO + Roadside Encroachment / RDD2022 | Shops, Animals, Potholes, Vehicles, Pedestrians |
| **Monocular Depth** | Depth Anything V2 / MiDaS | Outdoor Metric Depth Weights | Distance Map ($Z$ depth in meters) |
| **Segmentation** | U-Net / SAM 2 / DeepLabV3+ | Cityscapes / VPGNet | Crosswalk Masks, Lane Lines, Pavement Defects |
| **Object Tracking** | ByteTrack / DeepSORT | MOT20 Weights | Pedestrian & Vehicle Trajectories, TTC calculation |
| **Lane Detection** | CLRNet / Ultra-Fast-Lane | TuSimple / CULane | Line Continuity & Fading Ratios |
