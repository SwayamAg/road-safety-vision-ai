# 📊 Model Parameters & Expected Output Variables (`FD_v2`)

This document defines the exact specification for the **10 input parameters** required by the tuned XGBoost model and the expected **output risk classifications and scores**. Any Computer Vision perception model deployed in Phase 2 must ultimately extract, calibrate, and format visual evidence into this exact schema.

---

## 📥 1. The 10 Required Input Parameters

The XGBoost model expects an input vector of 10 feature variables. Each feature represents a specific road safety parameter, mapped to a standardized ordinal/intensity scale:

| # | Parameter Name | Variable Identifier in Model | Model Split Weight (%) | Valid Integer Scale | Feature Description & Scale Definitions |
| :-: | :--- | :--- | :---: | :---: | :--- |
| **1** | **Sight Distance Adequacy** | `Is sight distance adequate for safe driving?_ord` | **19.57%** | `0, 1, 2` | Measures visual range along driver path.<br>• `0` = Adequate (Clear > 100m)<br>• `1` = Partly Compromised (50m–100m)<br>• `2` = Inadequate / Blind (< 50m) |
| **2** | **Roadside Commercial Activity Density** | `Roadside commercial activity density_ord` | **17.56%** | `0, 2, 3, 5` | Commercial shop encroachment & lateral friction.<br>• `0` = None<br>• `2` = Low (1–2 shops/100m)<br>• `3` = Medium (3–5 shops/100m)<br>• `5` = High (> 5 shops/100m) |
| **3** | **Pedestrian–Vehicle Conflicts** | `Pedestrian–vehicle conflicts_ord` | **17.41%** | `0, 2, 5` | Active interaction & trajectory conflicts.<br>• `0` = None (No near-misses)<br>• `2` = Few (Occasional near-miss)<br>• `5` = Frequent (High risk TTC < 1.5s) |
| **4** | **Frequency of Jaywalking Pedestrians** | `Frequency of jaywalking pedestrians_ord` | **9.02%** | `0, 2, 5` | Pedestrian crossing outside crosswalks.<br>• `0` = None<br>• `2` = Occasional (1–3 per min)<br>• `5` = Frequent (> 3 per min) |
| **5** | **Unsafe Merging / Diverging Maneuvers** | `Unsafe merging diverging maneuvers observed_ord` | **7.44%** | `0, 2, 5` | Flow instability & abrupt lane changes.<br>• `0` = None<br>• `2` = Few<br>• `5` = Frequent |
| **6** | **Visibility of Pedestrian Crossings** | `Visibility of pedestrian crossings_ord` | **6.69%** | `0, 1, 2` | Prominence & visual clarity of zebra marks.<br>• `0` = Good (High contrast)<br>• `1` = Poor (Faded paint)<br>• `2` = Absent / Missing |
| **7** | **Road Surface Condition** | `Road surface condition_ord` | **6.66%** | `0, 1, 2` | Pavement structural integrity & defects.<br>• `0` = Good (Smooth)<br>• `1` = Minor defects (Cracks/patches)<br>• `2` = Damaged (Severe potholes) |
| **8** | **Lane Marking Condition** | `Lane marking condition_ord` | **5.92%** | `0, 1, 2` | Delineation quality of lane boundaries.<br>• `0` = Clear (Delineated)<br>• `1` = Faded<br>• `2` = Absent |
| **9** | **Junction Channelization** | `Junction channelization_ord` | **4.98%** | `0, 1, 2` | Guidance islands/curbs at intersections.<br>• `0` = Proper (Channelized)<br>• `1` = Poor<br>• `2` = Absent |
| **10**| **Presence of Animals on Road** | `Presence of animals on road_ord` | **4.75%** | `0, 1, 5` | Encroachment of stray animals.<br>• `0` = None<br>• `1` = Rare<br>• `5` = Frequent |

---

## 📤 2. Expected Model Output Variables

The output layer of the inference engine produces two primary outputs:

### A. Primary Output: Road Crash Risk Level (Classification)
The XGBoost Classifier maps the 10-parameter input vector into **4 Logical Risk Tiers**:

| Class Index | Risk Tier Label | Visual Alert Color | Safety Score Interval (CI Range) | Operational Definition & Recommended Action |
| :-: | :--- | :---: | :---: | :--- |
| `0` | 🟢 **Very Low Risk** | Green | `[0.00 – 1.96]` | Safe corridor. Baseline infrastructure adequacy. No immediate intervention required. |
| `1` | 🟡 **Low Risk** | Yellow | `[2.07 – 3.21]` | Acceptable corridor. Minor friction or light wear present. Schedule routine maintenance. |
| `2` | 🟠 **High Risk** | Orange | `[7.21 – 7.95]` | Vulnerable corridor. Significant pedestrian conflicts or degraded markings. High priority intervention. |
| `3` | 🔴 **Very High Risk** | Red | `[8.00 – 9.00]` | Hazardous blackspot. Severe sight distance constraints, high jaywalking, or commercial encroachment. Emergency safety audit required. |

### B. Secondary Output: Predicted Crash Index Score (Regression / Probabilistic)
* **Continuous Index Range**: `[0.00 – 10.00]`
* **Probability Distribution**: Class confidence probabilities vector (e.g., `[P_VeryLow, P_Low, P_High, P_VeryHigh]`).

---

## 📋 3. Standard Model JSON Payload Schema

Any video processing pipeline module must assemble and POST the following JSON payload structure to the inference engine:

```json
{
  "segment_metadata": {
    "video_id": "ROAD_SURVEY_2026_08_02_CAM01",
    "segment_id": "SEG_KM_14_500",
    "start_chainage_km": 14.00,
    "end_chainage_km": 14.50,
    "timestamp": "2026-08-02T16:50:00Z"
  },
  "extracted_parameters": {
    "Is sight distance adequate for safe driving?_ord": 0,
    "Roadside commercial activity density_ord": 2,
    "Pedestrian–vehicle conflicts_ord": 0,
    "Frequency of jaywalking pedestrians_ord": 0,
    "Unsafe merging diverging maneuvers observed_ord": 2,
    "Visibility of pedestrian crossings_ord": 2,
    "Road surface condition_ord": 0,
    "Lane marking condition_ord": 0,
    "Junction channelization_ord": 1,
    "Presence of animals on road_ord": 0
  }
}
```

---

## ⚙️ 4. Excluded Candidate Variables (Why CV Focuses Only on the Top 10)

During Phase 1 feature selection, composite indices (`Infrastructure_Quality_Index`, `Roadside_Friction_Score`, `Total_Conflict_Load`, `Visibility_Risk`) and non-discriminative raw features (e.g., `Shoulder condition`, `Two-wheeler conflicts`, `Signage adequacy`) were dropped due to multicollinearity ($|r| > 0.90$) or low split-gain importance.

Therefore, **CV models in Phase 2 DO NOT need to waste compute resources extracting secondary variables** like shoulder width or weather codes, focusing 100% of GPU compute on extracting the 10 high-gain parameters listed above.
