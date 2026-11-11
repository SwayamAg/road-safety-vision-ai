import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = r'c:\SWAYAMs\PROJ\Road_Safety_params\FD_v2'
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'DATA', 'OP')

TARGET_VIDEOS = [
    os.path.join(SCRIPT_DIR, 'DATA', 'IP', 'H_VIDEO_17092026084824_1789615104328', 'H_VIDEO_17092026090210_1789615930354.mp4'),
    os.path.join(SCRIPT_DIR, 'DATA', 'IP', 'Sample_2 (2).mp4')
]

sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from cv_detector import RoadSafetyDetector
from cv_depth import SightDistanceDepthEstimator
from cv_sam import RoadSafetySAMSegmenter
from cv_distance_odometry import VehicleEgoOdometryTracker
from risk_predictor import RiskPredictor
from main_pipeline import run_road_safety_pipeline

def main():
    print('=' * 80)
    print('  ROAD SAFETY VISION-AI PIPELINE BATCH RUNNER')
    print('  Output Directory:', OUTPUT_DIR)
    print('=' * 80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Verify inputs exist
    for idx, v in enumerate(TARGET_VIDEOS, 1):
        if not os.path.exists(v):
            print(f'[ERROR] Target video {idx} not found: {v}')
            sys.exit(1)
        else:
            print(f'  Target {idx}: {v} ({os.path.getsize(v) / (1024*1024):.1f} MB)')

    print('\n[INITIALIZING MODELS ONCE FOR BATCH EFFICIENCY]')
    t0 = time.time()
    detector = RoadSafetyDetector(model_weights=os.path.join(SCRIPT_DIR, 'models', 'yolo11m.pt'))
    depth_estimator = SightDistanceDepthEstimator()
    sam_segmenter = RoadSafetySAMSegmenter(model_weights=os.path.join(SCRIPT_DIR, 'models', 'sam2.1_t.pt'))
    predictor = RiskPredictor(model_dir=os.path.join(SCRIPT_DIR, 'models'))
    print(f'Models initialized in {time.time() - t0:.2f}s\n')

    results = []

    for idx, v_path in enumerate(TARGET_VIDEOS, 1):
        v_name = os.path.basename(v_path)
        print('\n' + '#' * 80)
        print(f'  PROCESSING VIDEO {idx}/{len(TARGET_VIDEOS)}: {v_name}')
        print('#' * 80 + '\n')

        odometry_tracker = VehicleEgoOdometryTracker(
            camera_fov_deg=65.0,
            camera_height_m=1.35,
            target_segment_length_m=500.0
        )

        v_t0 = time.time()
        audit = run_road_safety_pipeline(
            video_path=v_path,
            output_dir=OUTPUT_DIR,
            model_weights=os.path.join(SCRIPT_DIR, 'models', 'yolo11m.pt'),
            sam_weights=os.path.join(SCRIPT_DIR, 'models', 'sam2.1_t.pt'),
            imgsz=None,
            sample_fps=5.0,
            keyframe_interval=15,
            segment_cap_m=500.0,
            detector=detector,
            depth_estimator=depth_estimator,
            sam_segmenter=sam_segmenter,
            odometry_tracker=odometry_tracker,
            predictor=predictor
        )
        elapsed = time.time() - v_t0
        results.append({
            'video_path': v_path,
            'filename': v_name,
            'elapsed_sec': round(elapsed, 1),
            'audit': audit
        })

    print('\n' + '=' * 80)
    print('  BATCH PROCESSING COMPLETE')
    print('=' * 80)
    for r in results:
        print(f"\n▶ Video: {r['filename']} (took {r['elapsed_sec']}s)")
        print(f"  Distance Traveled : {r['audit']['distance_and_odometry_summary']['total_distance_traveled_m']} m")
        print(f"  Avg Speed         : {r['audit']['distance_and_odometry_summary']['average_speed_kmh']} km/h")
        assessment = r['audit']['overall_corridor_risk_assessment']
        print(f"  Overall Risk      : {assessment['emoji']} [{assessment['predicted_risk_tier'].upper()}] - Score: {assessment['risk_score']}/10")
        print("  Generated Files   :")
        for k, p in r['audit']['output_files'].items():
            print(f"    • {k}: {p} (exists: {os.path.exists(p)})")
    print('\n' + '=' * 80)

if __name__ == '__main__':
    main()
