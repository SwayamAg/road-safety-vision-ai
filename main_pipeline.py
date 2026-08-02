import os
import sys
import cv2
import time
from video_loader import VideoFrameReader
from cv_detector import RoadSafetyDetector
from feature_calibrator import FeatureCalibrator
from risk_predictor import RiskPredictor

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def run_road_safety_pipeline(video_path: str, output_annotated_path: str = "output_road_safety_annotated.mp4"):
    """
    Master End-to-End Execution Pipeline for Phase 2 Road Safety Risk Analytics.
    Executes Video Ingestion ➔ CV Object Perception ➔ Spatial Feature Calibration ➔ XGBoost Inference.
    """
    print("=" * 70)
    print("🛣️ Phase 2 Automated Road Safety Risk Analytics Pipeline")
    print("=" * 70)
    
    # 1. Initialize Video Reader
    reader = VideoFrameReader(video_path)
    info = reader.get_info()
    print(f"\n[1/4] Ingesting Video: {info['source_path']}")
    print(f"      • Resolution: {info['resolution'][0]}x{info['resolution'][1]}")
    print(f"      • Native FPS: {info['native_fps']} | Duration: {info['duration_sec']}s ({info['total_frames']} frames)")
    
    # 2. Initialize Computer Vision Detector & Model Predictor
    detector = RoadSafetyDetector(model_weights="yolov8n.pt")
    predictor = RiskPredictor(model_dir="../Ready dataset")
    
    print("\n[2/4] Processing Video Frames through CV Perception Layer...")
    
    aggregated_metrics = {
        "commercial_count": 0,
        "pedestrian_count": 0,
        "animal_count": 0,
        "vehicle_count": 0,
        "avg_sight_distance_m": 85.0 # Mock forward sight distance estimate
    }
    
    # Set up VideoWriter for annotated output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_annotated_path, fourcc, 5.0, (640, 640))
    
    processed_frames = 0
    t_start = time.time()
    
    for frame_idx, timestamp, orig_frame, resized_frame in reader.extract_frames(sample_fps=5.0):
        # Run Object Detection
        dets = detector.detect_frame(resized_frame)
        
        # Accumulate metrics
        aggregated_metrics["commercial_count"] += dets["commercial_count"]
        aggregated_metrics["pedestrian_count"] += dets["pedestrian_count"]
        aggregated_metrics["animal_count"] += dets["animal_count"]
        aggregated_metrics["vehicle_count"] += dets["vehicle_count"]
        
        # Annotate HUD
        annotated_frame = detector.annotate_frame(resized_frame, dets)
        writer.write(annotated_frame)
        processed_frames += 1

    writer.release()
    t_elapsed = time.time() - t_start
    print(f"      ✔ Processed {processed_frames} sampled frames in {t_elapsed:.2f}s ({processed_frames/t_elapsed:.1f} FPS)")
    
    # 3. Calibrate Features to 10 XGBoost Inputs
    print("\n[3/4] Calibrating Frame Detections into 10 XGBoost Input Scales...")
    calibrated_payload = FeatureCalibrator.calibrate_segment_metrics(aggregated_metrics)
    
    # 4. Predict Road Safety Risk Tier via XGBoost
    print("\n[4/4] Executing XGBoost Risk Inference Engine...")
    result = predictor.predict_risk(calibrated_payload)
    
    # Report Results
    print("\n" + "=" * 70)
    print(f"📊 FINAL SEGMENT RISK ASSESSMENT RESULT: {result['emoji']} [{result['predicted_risk_tier'].upper()} RISK]")
    print("=" * 70)
    print(f"• Recommendation : {result['description']}")
    print(f"• Annotations Saved: {output_annotated_path}")
    print("\nConfidence Probabilities Breakdown:")
    for tier, prob in result["probabilities"].items():
        bar = "█" * int(prob * 20)
        print(f"  - {tier:<10}: {prob:.4f} | {bar}")
    print("=" * 70)
    return result


if __name__ == "__main__":
    test_video = "sample_dashcam_test.mp4"
    if not os.path.exists(test_video):
        VideoFrameReader.generate_synthetic_test_video(test_video)
        
    run_road_safety_pipeline(test_video)
