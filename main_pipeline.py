import os
import sys
import cv2
import time
import json
import numpy as np
import pandas as pd
from video_loader import VideoFrameReader
from cv_detector import RoadSafetyDetector
from cv_depth import SightDistanceDepthEstimator
from feature_calibrator import FeatureCalibrator
from risk_predictor import RiskPredictor

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def format_timestamp(seconds: float) -> str:
    """Converts timestamp in seconds to MM:SS.s format."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:04.1f}"

def run_road_safety_pipeline(video_path: str, output_dir: str = "DATA/OP_V3", imgsz: int = 1280):
    """
    Master End-to-End Execution Pipeline for Phase 2 Road Safety Risk Analytics.
    Processes video streams @ imgsz=1280 resolution using YOLOv11 + PyTorch MiDaS Depth,
    exports frame-by-frame Excel/CSV, calibrates XGBoost inputs, predicts Risk Tier,
    and outputs annotated MP4 video & JSON report in DATA/OP_V3/.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    
    output_annotated_path = os.path.join(output_dir, f"{video_basename}_annotated.mp4")
    output_json_path = os.path.join(output_dir, f"{video_basename}_risk_report.json")
    output_excel_path = os.path.join(output_dir, f"{video_basename}_frame_by_frame_detections.xlsx")
    output_csv_path = os.path.join(output_dir, f"{video_basename}_frame_by_frame_detections.csv")
    
    print("=" * 75)
    print(f"🛣️ Phase 2 Automated Road Safety Risk Analytics Pipeline [{video_basename}] @ {imgsz}x{imgsz}")
    print("=" * 75)
    
    # 1. Initialize Video Reader
    reader = VideoFrameReader(video_path)
    info = reader.get_info()
    print(f"\n[1/4] Ingesting Video: {info['source_path']}")
    print(f"      • Resolution: {info['resolution'][0]}x{info['resolution'][1]}")
    print(f"      • Native FPS: {info['native_fps']} | Duration: {info['duration_sec']}s ({info['total_frames']} frames)")
    
    # 2. Initialize Computer Vision Perception Engines (YOLOv11 @ imgsz=1280 + Depth Estimator)
    detector = RoadSafetyDetector(model_weights="yolo11n.pt", imgsz=imgsz)
    depth_estimator = SightDistanceDepthEstimator()
    predictor = RiskPredictor(model_dir="../Ready dataset")
    
    print(f"\n[2/4] Processing Video Frames @ 5 FPS through High-Res YOLOv11 ({imgsz}p) & Depth Perception...")
    
    cum_metrics = {
        "commercial_count": 0,
        "pedestrian_count": 0,
        "animal_count": 0,
        "vehicle_count": 0,
        "avg_sight_distance_m": 100.0
    }
    
    frame_sight_distances = []
    frame_logs = []
    
    # Set up VideoWriter for annotated output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_annotated_path, fourcc, 5.0, (640, 640))
    
    processed_frames = 0
    t_start = time.time()
    
    for frame_idx, timestamp, orig_frame, resized_frame in reader.extract_frames(sample_fps=5.0):
        # Run YOLOv11 Object Detection at High Resolution (imgsz=1280)
        dets = detector.detect_frame(resized_frame)
        
        # Estimate Metric Sight Distance via Monocular Depth Map
        frame_sight_m = depth_estimator.estimate_sight_distance(resized_frame)
        frame_sight_distances.append(frame_sight_m)
        
        # Accumulate cumulative object counts
        cum_metrics["commercial_count"] += dets["commercial_count"]
        cum_metrics["pedestrian_count"] += dets["pedestrian_count"]
        cum_metrics["animal_count"] += dets["animal_count"]
        cum_metrics["vehicle_count"] += dets["vehicle_count"]
        cum_metrics["avg_sight_distance_m"] = float(np.median(frame_sight_distances))
        
        # Calibrate intermediate payload at this timestamp
        calibrated_at_frame = FeatureCalibrator.calibrate_segment_metrics(cum_metrics)
        pred_at_frame = predictor.predict_risk(calibrated_at_frame)
        
        # Record frame log entry for Excel export
        log_entry = {
            "frame_index": frame_idx,
            "timestamp_sec": round(timestamp, 2),
            "timestamp_formatted": format_timestamp(timestamp),
            "estimated_sight_distance_m": frame_sight_m,
            "pedestrians_in_frame": dets["pedestrian_count"],
            "vehicles_in_frame": dets["vehicle_count"],
            "animals_in_frame": dets["animal_count"],
            "commercial_in_frame": dets["commercial_count"],
            "cum_pedestrians": cum_metrics["pedestrian_count"],
            "cum_vehicles": cum_metrics["vehicle_count"],
            "cum_animals": cum_metrics["animal_count"],
            "cum_commercial": cum_metrics["commercial_count"],
            "sight_distance_ord": calibrated_at_frame["Is sight distance adequate for safe driving?_ord"],
            "commercial_density_ord": calibrated_at_frame["Roadside commercial activity density_ord"],
            "pedestrian_conflicts_ord": calibrated_at_frame["Pedestrian-vehicle conflicts_ord"],
            "jaywalking_frequency_ord": calibrated_at_frame["Frequency of jaywalking pedestrians_ord"],
            "unsafe_merging_ord": calibrated_at_frame["Unsafe merging diverging maneuvers observed_ord"],
            "crosswalk_visibility_ord": calibrated_at_frame["Visibility of pedestrian crossings_ord"],
            "road_surface_ord": calibrated_at_frame["Road surface condition_ord"],
            "lane_marking_ord": calibrated_at_frame["Lane marking condition_ord"],
            "junction_channelization_ord": calibrated_at_frame["Junction channelization_ord"],
            "animal_presence_ord": calibrated_at_frame["Presence of animals on road_ord"],
            "predicted_risk_tier": pred_at_frame["predicted_risk_tier"],
            "prob_very_low": pred_at_frame["probabilities"].get("Very Low", 0.0),
            "prob_low": pred_at_frame["probabilities"].get("Low", 0.0),
            "prob_high": pred_at_frame["probabilities"].get("High", 0.0),
            "prob_very_high": pred_at_frame["probabilities"].get("Very High", 0.0)
        }
        frame_logs.append(log_entry)
        
        # Annotate HUD & write to output video
        annotated_frame = detector.annotate_frame(resized_frame, dets)
        
        # Add Sight Distance HUD overlay
        cv2.putText(annotated_frame, f"Sight Dist: {frame_sight_m:.1f}m", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
        writer.write(annotated_frame)
        processed_frames += 1

    writer.release()
    t_elapsed = time.time() - t_start
    print(f"      ✔ Processed {processed_frames} sampled frames in {t_elapsed:.2f}s ({processed_frames/t_elapsed:.1f} FPS)")
    
    # 3. Export Frame-by-Frame Excel & CSV Files
    print("\n[3/4] Exporting Frame-by-Frame Timestamps & Detections to Excel / CSV...")
    df_logs = pd.DataFrame(frame_logs)
    
    # Save Excel and CSV
    df_logs.to_excel(output_excel_path, index=False, engine='openpyxl')
    df_logs.to_csv(output_csv_path, index=False)
    print(f"      ✔ Saved Excel Log: {output_excel_path} ({len(df_logs)} rows)")
    print(f"      ✔ Saved CSV Log  : {output_csv_path}")
    
    # 4. Final Calibrated Segment Payload & XGBoost Inference
    print("\n[4/4] Executing Final Segment XGBoost Risk Inference Engine...")
    final_calibrated_payload = FeatureCalibrator.calibrate_segment_metrics(cum_metrics)
    final_result = predictor.predict_risk(final_calibrated_payload)
    
    # Save JSON Summary Report
    audit_report = {
        "video_metadata": info,
        "pipeline_config": {
            "model_weights": "yolo11n.pt",
            "inference_resolution": f"{imgsz}x{imgsz}",
            "depth_model": "MiDaS_small"
        },
        "cv_detections_summary": {
            "processed_frames": processed_frames,
            "processing_time_sec": round(t_elapsed, 2),
            "fps_throughput": round(processed_frames / t_elapsed, 1),
            "median_sight_distance_m": round(cum_metrics["avg_sight_distance_m"], 1),
            "total_pedestrians_detected": cum_metrics["pedestrian_count"],
            "total_vehicles_detected": cum_metrics["vehicle_count"],
            "total_animals_detected": cum_metrics["animal_count"],
            "total_commercial_objects_detected": cum_metrics["commercial_count"]
        },
        "xgboost_calibrated_payload": final_calibrated_payload,
        "risk_prediction": final_result,
        "output_files": {
            "annotated_video": output_annotated_path,
            "json_report": output_json_path,
            "excel_log": output_excel_path,
            "csv_log": output_csv_path
        }
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(audit_report, f, indent=2)
        
    # Print Console Report
    print("\n" + "=" * 75)
    print(f"📊 FINAL RISK ASSESSMENT [{video_basename}]: {final_result['emoji']} [{final_result['predicted_risk_tier'].upper()} RISK]")
    print("=" * 75)
    print(f"• Inference Resolution: {imgsz}x{imgsz}")
    print(f"• Sight Distance Range : {cum_metrics['avg_sight_distance_m']:.1f} meters")
    print(f"• Recommendation       : {final_result['description']}")
    print(f"• Video Saved          : {output_annotated_path}")
    print(f"• Excel Saved          : {output_excel_path}")
    print(f"• JSON Saved           : {output_json_path}")
    print("\nConfidence Probabilities Breakdown:")
    for tier, prob in final_result["probabilities"].items():
        bar = "█" * int(prob * 20)
        print(f"  - {tier:<10}: {prob:.4f} | {bar}")
    print("=" * 75)
    return audit_report


def process_all_input_videos(input_dir: str = "DATA/IP", output_dir: str = "DATA/OP_V3", imgsz: int = 1280):
    """
    Sequentially processes all MP4 videos in DATA/IP/ (e.g. 1.mp4, 2.mp4)
    at imgsz=1280 resolution and saves output videos, Excel/CSV timestamp logs, and JSON reports in DATA/OP_V3/.
    """
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    video_files.sort()
    
    print(f"\n🚀 Found {len(video_files)} input videos in {input_dir}: {video_files}")
    for video in video_files:
        video_full_path = os.path.join(input_dir, video)
        run_road_safety_pipeline(video_full_path, output_dir=output_dir, imgsz=imgsz)
        print("\n")


if __name__ == "__main__":
    process_all_input_videos(input_dir="DATA/IP", output_dir="DATA/OP_V3", imgsz=1280)
