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
from cv_sam import RoadSafetySAMSegmenter
from cv_distance_odometry import VehicleEgoOdometryTracker
from feature_calibrator import FeatureCalibrator
from risk_predictor import RiskPredictor

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def format_timestamp(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds % 60
    return str(mins).zfill(2) + ':' + f'{secs:04.1f}'

def run_road_safety_pipeline(
    video_path: str, 
    output_dir: str = 'DATA/OP', 
    model_weights: str = 'models/yolo11m.pt',
    sam_weights: str = 'models/sam2.1_t.pt',
    imgsz: int = None,
    sample_fps: float = 5.0,
    keyframe_interval: int = 15,
    max_frames: int = None,
    segment_cap_m: float = 500.0,
    target_segment_length_m: float = None,
    detector: RoadSafetyDetector = None,
    depth_estimator: SightDistanceDepthEstimator = None,
    sam_segmenter: RoadSafetySAMSegmenter = None,
    odometry_tracker: VehicleEgoOdometryTracker = None,
    predictor: RiskPredictor = None
):
    os.makedirs(output_dir, exist_ok=True)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    
    # Resolve configurable segment cap (default: 500.0m unless specified)
    if target_segment_length_m is not None:
        segment_cap_m = float(target_segment_length_m)
    else:
        segment_cap_m = float(segment_cap_m)
        
    output_annotated_path = os.path.join(output_dir, video_basename + '_annotated.mp4')
    output_json_path = os.path.join(output_dir, video_basename + '_risk_report.json')
    output_csv_path = os.path.join(output_dir, video_basename + '_frame_by_frame_detections.csv')
    
    reader = VideoFrameReader(video_path)
    info = reader.get_info()
    res_str = f"{info['resolution'][0]}x{info['resolution'][1]}"
    
    print('=' * 75)
    print(f'Option B Road Safety Vision-AI Pipeline [{video_basename}] @ Native Resolution ({res_str})')
    print(f'   • YOLO11m ({model_weights}) | Metric Odometry ({segment_cap_m:.0f}m Cap) | SAM 2 ({sam_weights})')
    print('=' * 75)
    
    print('\n[1/4] Ingesting Video: ' + str(info['source_path']))
    print('      • Resolution: ' + res_str)
    print('      • Native FPS: ' + str(info['native_fps']) + ' | Duration: ' + str(info['duration_sec']) + 's (' + str(info['total_frames']) + ' frames)')
    print(f'      • Segment Risk Cap: Every {segment_cap_m:.0f}m traveled')
    
    if detector is None:
        detector = RoadSafetyDetector(model_weights=model_weights, imgsz=imgsz)
    if depth_estimator is None:
        depth_estimator = SightDistanceDepthEstimator()
    if sam_segmenter is None:
        sam_segmenter = RoadSafetySAMSegmenter(model_weights=sam_weights)
    if odometry_tracker is None:
        odometry_tracker = VehicleEgoOdometryTracker(target_segment_length_m=segment_cap_m)
    if predictor is None:
        predictor = RiskPredictor(model_dir='models')
    
    print(f'\n[2/4] Processing Video Frames @ {sample_fps} FPS (Predicting Risk at every {segment_cap_m:.0f}m cap)...')
    
    # Overall cumulative video metrics
    cum_metrics = {
        'commercial_count': 0,
        'pedestrian_count': 0,
        'animal_count': 0,
        'vehicle_count': 0,
        'avg_sight_distance_m': 100.0
    }
    
    # Active Segment State (resets at each segment_cap_m threshold)
    current_segment_idx = 1
    current_segment_metrics = {
        'commercial_count': 0,
        'pedestrian_count': 0,
        'animal_count': 0,
        'vehicle_count': 0,
        'sight_distances': []
    }
    
    completed_segments_list = []
    latest_completed_risk = None
    milestone_alert_until_frame = -1
    milestone_alert_text = ""
    
    frame_sight_distances = []
    frame_speeds = []
    frame_logs = []
    total_sam_masks_generated = 0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_annotated_path, fourcc, sample_fps, (reader.width, reader.height))
    
    processed_frames = 0
    prev_timestamp = 0.0
    t_start = time.time()
    
    for frame_idx, timestamp, orig_frame, proc_frame in reader.extract_frames(sample_fps=sample_fps, target_size=None):
        if max_frames is not None and processed_frames >= max_frames:
            print('      • Reached max_frames limit (' + str(max_frames) + '). Finalizing segment analytics...')
            break
            
        dt = timestamp - prev_timestamp if processed_frames > 0 else (1.0 / sample_fps)
        prev_timestamp = timestamp
        
        # 1. YOLO11 Object Detection
        dets = detector.detect_frame(proc_frame)
        
        # 2. Metric Depth Estimation & 3D Keypoint Ego-Odometry (Option 1)
        depth_map, frame_sight_m = depth_estimator.compute_depth_map(proc_frame)
        frame_sight_distances.append(frame_sight_m)
        
        odom_metrics = odometry_tracker.update(
            frame=proc_frame, 
            depth_map_m=depth_map, 
            dynamic_boxes=dets['boxes'], 
            dt=dt
        )
        frame_speeds.append(odom_metrics['current_speed_kmh'])
        
        # Accumulate to active segment
        current_segment_metrics['commercial_count'] += dets['commercial_count']
        current_segment_metrics['pedestrian_count'] += dets['pedestrian_count']
        current_segment_metrics['animal_count'] += dets['animal_count']
        current_segment_metrics['vehicle_count'] += dets['vehicle_count']
        current_segment_metrics['sight_distances'].append(frame_sight_m)
        
        # Accumulate to overall video totals
        cum_metrics['commercial_count'] += dets['commercial_count']
        cum_metrics['pedestrian_count'] += dets['pedestrian_count']
        cum_metrics['animal_count'] += dets['animal_count']
        cum_metrics['vehicle_count'] += dets['vehicle_count']
        cum_metrics['avg_sight_distance_m'] = float(np.median(frame_sight_distances))
        
        # 3. Check if Segment Cap Milestone was crossed (e.g. every 500m)
        if odom_metrics.get("milestone_500m_triggered", False):
            seg_id = odom_metrics["completed_500m_segments"]
            start_m = (seg_id - 1) * segment_cap_m
            end_m = seg_id * segment_cap_m
            
            # Predict risk for the completed segment window
            seg_calib_input = {
                'commercial_count': current_segment_metrics['commercial_count'],
                'pedestrian_count': current_segment_metrics['pedestrian_count'],
                'animal_count': current_segment_metrics['animal_count'],
                'vehicle_count': current_segment_metrics['vehicle_count'],
                'avg_sight_distance_m': float(np.median(current_segment_metrics['sight_distances'])) if current_segment_metrics['sight_distances'] else 100.0
            }
            seg_calibrated = FeatureCalibrator.calibrate_segment_metrics(seg_calib_input)
            seg_prediction = predictor.predict_risk(seg_calibrated)
            
            seg_record = {
                'segment_index': seg_id,
                'start_distance_m': round(start_m, 1),
                'end_distance_m': round(end_m, 1),
                'range_str': f"{start_m:.0f}m - {end_m:.0f}m",
                'calibrated_features': seg_calibrated,
                'risk_prediction': seg_prediction,
                'timestamp_sec': round(timestamp, 2),
                'timestamp_formatted': format_timestamp(timestamp)
            }
            completed_segments_list.append(seg_record)
            latest_completed_risk = seg_record
            
            # Show milestone alert banner for 4 seconds
            milestone_alert_until_frame = processed_frames + int(sample_fps * 4.0)
            milestone_alert_text = f"★ {segment_cap_m:.0f}m CAP: SEGMENT #{seg_id} [{start_m:.0f}m-{end_m:.0f}m] RISK: {seg_prediction['predicted_risk_tier'].upper()} ({seg_prediction['risk_score']}/10) ★"
            
            print(f"\n[CAP REACHED] Traveled {end_m:.1f}m! Completed Segment #{seg_id} [{start_m:.0f}m - {end_m:.0f}m]: {seg_prediction['emoji']} {seg_prediction['predicted_risk_tier']} (Score: {seg_prediction['risk_score']}/10)")
            
            # Reset active segment metrics for the next stretch
            current_segment_idx = seg_id + 1
            current_segment_metrics = {
                'commercial_count': 0,
                'pedestrian_count': 0,
                'animal_count': 0,
                'vehicle_count': 0,
                'sight_distances': []
            }
            
        # 4. Compute provisional real-time risk for current active segment
        seg_calib_provisional = {
            'commercial_count': current_segment_metrics['commercial_count'],
            'pedestrian_count': current_segment_metrics['pedestrian_count'],
            'animal_count': current_segment_metrics['animal_count'],
            'vehicle_count': current_segment_metrics['vehicle_count'],
            'avg_sight_distance_m': float(np.median(current_segment_metrics['sight_distances'])) if current_segment_metrics['sight_distances'] else 100.0
        }
        calibrated_at_frame = FeatureCalibrator.calibrate_segment_metrics(seg_calib_provisional)
        pred_at_frame = predictor.predict_risk(calibrated_at_frame)
        
        # 5. Keyframe-triggered SAM 2 segmentation
        is_keyframe = (processed_frames % keyframe_interval == 0) and (len(dets['boxes']) > 0)
        sam_mask_count = 0
        if is_keyframe:
            sam_res = sam_segmenter.segment_keyframe(proc_frame, dets)
            annotated_frame = sam_res['annotated_frame']
            sam_mask_count = sam_res['mask_count']
            total_sam_masks_generated += sam_mask_count
            annotated_frame = detector.annotate_frame(annotated_frame, dets)
        else:
            annotated_frame = detector.annotate_frame(proc_frame, dets)
            
        # 6. Render Distance Telemetry + Segment Risk Score HUD onto Video
        segment_risk_info = {
            'segment_cap_m': segment_cap_m,
            'current_segment_idx': current_segment_idx,
            'latest_completed_risk': latest_completed_risk,
            'current_provisional_prediction': pred_at_frame,
            'milestone_alert_active': (processed_frames <= milestone_alert_until_frame),
            'milestone_alert_text': milestone_alert_text
        }
        annotated_frame = odometry_tracker.draw_distance_hud(
            annotated_frame, 
            odom_metrics, 
            segment_risk_info=segment_risk_info
        )
        
        log_entry = {
            'frame_index': frame_idx,
            'timestamp_sec': round(timestamp, 2),
            'timestamp_formatted': format_timestamp(timestamp),
            'is_sam2_keyframe': is_keyframe,
            'sam2_masks_refined': sam_mask_count,
            'cumulative_distance_m': odom_metrics['cumulative_distance_m'],
            'vehicle_speed_kmh': odom_metrics['current_speed_kmh'],
            'current_segment_index': current_segment_idx,
            'segment_progress_m': odom_metrics['segment_progress_m'],
            'segment_progress_pct': odom_metrics['segment_progress_pct'],
            'completed_segments_count': odom_metrics['completed_500m_segments'],
            'latest_completed_segment_id': latest_completed_risk['segment_index'] if latest_completed_risk else 0,
            'latest_completed_segment_tier': latest_completed_risk['risk_prediction']['predicted_risk_tier'] if latest_completed_risk else "None",
            'latest_completed_segment_score': latest_completed_risk['risk_prediction']['risk_score'] if latest_completed_risk else 0.0,
            'odometry_3d_inliers': odom_metrics['tracked_static_inliers'],
            'estimated_sight_distance_m': frame_sight_m,
            'pedestrians_in_frame': dets['pedestrian_count'],
            'vehicles_in_frame': dets['vehicle_count'],
            'animals_in_frame': dets['animal_count'],
            'commercial_in_frame': dets['commercial_count'],
            'active_seg_pedestrians': current_segment_metrics['pedestrian_count'],
            'active_seg_vehicles': current_segment_metrics['vehicle_count'],
            'active_seg_animals': current_segment_metrics['animal_count'],
            'active_seg_commercial': current_segment_metrics['commercial_count'],
            'active_seg_provisional_tier': pred_at_frame['predicted_risk_tier'],
            'active_seg_provisional_score': pred_at_frame['risk_score']
        }
        frame_logs.append(log_entry)
        
        # Sight distance badge dynamically positioned above bottom edge
        sd_y = max(30, reader.height - 20)
        cv2.putText(annotated_frame, 'SIGHT DISTANCE: ' + str(round(frame_sight_m, 1)) + 'm', (20, sd_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
                    
        writer.write(annotated_frame)
        processed_frames += 1

    writer.release()
    t_elapsed = time.time() - t_start
    print('      ✔ Processed ' + str(processed_frames) + ' sampled frames in ' + str(round(t_elapsed, 2)) + 's (' + str(round(processed_frames/t_elapsed, 1)) + ' FPS)')
    print('      ✔ Generated ' + str(total_sam_masks_generated) + ' high-precision SAM 2 instance masks on keyframes')
    print('      ✔ Total Traveled Distance: ' + str(round(odometry_tracker.cumulative_distance_m, 1)) + f' meters ({len(completed_segments_list)} completed x {segment_cap_m:.0f}m caps)')
    
    print('\n[3/4] Exporting Frame-by-Frame Timestamps & Detections to CSV...')
    df_logs = pd.DataFrame(frame_logs)
    df_logs.to_csv(output_csv_path, index=False)
    print('      ✔ Saved CSV Log  : ' + str(output_csv_path) + ' (' + str(len(df_logs)) + ' rows)')
    
    print('\n[4/4] Finalizing Per-Segment & Overall Risk Audit Reports...')
    # If final segment has traversed distance and wasn't closed by an exact cap, evaluate it
    final_partial_segment = None
    if len(current_segment_metrics['sight_distances']) > 0:
        partial_calib = FeatureCalibrator.calibrate_segment_metrics({
            'commercial_count': current_segment_metrics['commercial_count'],
            'pedestrian_count': current_segment_metrics['pedestrian_count'],
            'animal_count': current_segment_metrics['animal_count'],
            'vehicle_count': current_segment_metrics['vehicle_count'],
            'avg_sight_distance_m': float(np.median(current_segment_metrics['sight_distances']))
        })
        partial_prediction = predictor.predict_risk(partial_calib)
        start_m = (current_segment_idx - 1) * segment_cap_m
        end_m = odometry_tracker.cumulative_distance_m
        final_partial_segment = {
            'segment_index': current_segment_idx,
            'start_distance_m': round(start_m, 1),
            'end_distance_m': round(end_m, 1),
            'range_str': f"{start_m:.0f}m - {end_m:.0f}m (Partial: {end_m - start_m:.1f}m)",
            'calibrated_features': partial_calib,
            'risk_prediction': partial_prediction
        }
        
    overall_calibrated_payload = FeatureCalibrator.calibrate_segment_metrics(cum_metrics)
    overall_result = predictor.predict_risk(overall_calibrated_payload)
    
    audit_report = {
        'video_metadata': info,
        'pipeline_config': {
            'detection_model': model_weights,
            'depth_and_odometry': f'MiDaS_small + 3D Metric Keypoint VO ({segment_cap_m:.0f}m cap)',
            'sam_model': sam_weights,
            'inference_resolution': res_str,
            'segment_cap_m': segment_cap_m
        },
        'distance_and_odometry_summary': {
            'total_distance_traveled_m': round(odometry_tracker.cumulative_distance_m, 1),
            'segment_cap_m': segment_cap_m,
            'completed_caps_count': len(completed_segments_list),
            'average_speed_kmh': round(float(np.mean(frame_speeds)) if len(frame_speeds) > 0 else 0.0, 1),
            'max_speed_kmh': round(float(np.max(frame_speeds)) if len(frame_speeds) > 0 else 0.0, 1)
        },
        'completed_segments_risk_evaluations': completed_segments_list,
        'final_partial_segment_evaluation': final_partial_segment,
        'overall_corridor_risk_assessment': overall_result,
        'output_files': {
            'annotated_video': output_annotated_path,
            'json_report': output_json_path,
            'csv_log': output_csv_path
        }
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(audit_report, f, indent=2)
        
    print('\n' + '=' * 75)
    print(f'SEGMENT RISK AUDIT REPORT [{video_basename}] (Cap: Every {segment_cap_m:.0f}m)')
    print('=' * 75)
    print('• Total Traveled Distance : ' + str(round(odometry_tracker.cumulative_distance_m, 1)) + ' meters')
    print('• Completed Full Segments : ' + str(len(completed_segments_list)))
    
    for seg in completed_segments_list:
        p = seg['risk_prediction']
        print(f"  ▶ Segment #{seg['segment_index']} [{seg['range_str']}]: {p['emoji']} [{p['predicted_risk_tier'].upper()}] - Score: {p['risk_score']}/10")
        
    if final_partial_segment:
        p = final_partial_segment['risk_prediction']
        print(f"  ▶ Active Segment #{final_partial_segment['segment_index']} [{final_partial_segment['range_str']}]: {p['emoji']} [{p['predicted_risk_tier'].upper()}] - Score: {p['risk_score']}/10")
        
    print('• Video Saved             : ' + output_annotated_path)
    print('• CSV Saved               : ' + output_csv_path)
    print('• JSON Saved              : ' + output_json_path)
    print('=' * 75)
    return audit_report

def process_target_videos(videos=['sample.mp4', '1.mp4'], input_dir='DATA/IP', output_dir='DATA/OP', segment_cap_m=500.0):
    detector = RoadSafetyDetector(model_weights='models/yolo11m.pt', imgsz=1280)
    depth_estimator = SightDistanceDepthEstimator()
    sam_segmenter = RoadSafetySAMSegmenter(model_weights='models/sam2.1_t.pt')
    predictor = RiskPredictor(model_dir='models')
    
    for v in videos:
        v_path = os.path.join(input_dir, v)
        if os.path.exists(v_path):
            odometry_tracker = VehicleEgoOdometryTracker(target_segment_length_m=segment_cap_m)
            run_road_safety_pipeline(
                video_path=v_path,
                output_dir=output_dir,
                segment_cap_m=segment_cap_m,
                detector=detector,
                depth_estimator=depth_estimator,
                sam_segmenter=sam_segmenter,
                odometry_tracker=odometry_tracker,
                predictor=predictor
            )
            print('\n')
        else:
            print('Warning: Video ' + str(v_path) + ' not found.')

if __name__ == '__main__':
    process_target_videos()
