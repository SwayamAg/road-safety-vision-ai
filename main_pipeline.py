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
    output_dir: str = 'DATA/OP_V6', 
    model_weights: str = 'models/yolo11m.pt',
    sam_weights: str = 'models/sam2.1_t.pt',
    imgsz: int = None,
    sample_fps: float = 5.0,
    keyframe_interval: int = 15,
    max_frames: int = None,
    detector: RoadSafetyDetector = None,
    depth_estimator: SightDistanceDepthEstimator = None,
    sam_segmenter: RoadSafetySAMSegmenter = None,
    predictor: RiskPredictor = None
):
    os.makedirs(output_dir, exist_ok=True)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    
    output_annotated_path = os.path.join(output_dir, video_basename + '_annotated.mp4')
    output_json_path = os.path.join(output_dir, video_basename + '_risk_report.json')
    output_csv_path = os.path.join(output_dir, video_basename + '_frame_by_frame_detections.csv')
    
    reader = VideoFrameReader(video_path)
    info = reader.get_info()
    res_str = f"{info['resolution'][0]}x{info['resolution'][1]}"
    
    print('=' * 75)
    print(f'Option B Road Safety Vision-AI Pipeline [{video_basename}] @ Native Resolution ({res_str})')
    print('   • YOLO11m (' + model_weights + ') | MiDaS Depth | SAM 2 (' + sam_weights + ')')
    print('=' * 75)
    
    print('\n[1/4] Ingesting Video: ' + str(info['source_path']))
    print('      • Resolution: ' + res_str)
    print('      • Native FPS: ' + str(info['native_fps']) + ' | Duration: ' + str(info['duration_sec']) + 's (' + str(info['total_frames']) + ' frames)')
    
    if detector is None:
        detector = RoadSafetyDetector(model_weights=model_weights, imgsz=imgsz)
    if depth_estimator is None:
        depth_estimator = SightDistanceDepthEstimator()
    if sam_segmenter is None:
        sam_segmenter = RoadSafetySAMSegmenter(model_weights=sam_weights)
    if predictor is None:
        predictor = RiskPredictor(model_dir='models')
    
    print('\n[2/4] Processing Video Frames @ ' + str(sample_fps) + ' FPS through YOLO11m + MiDaS + SAM 2...')
    
    cum_metrics = {
        'commercial_count': 0,
        'pedestrian_count': 0,
        'animal_count': 0,
        'vehicle_count': 0,
        'avg_sight_distance_m': 100.0
    }
    
    frame_sight_distances = []
    frame_logs = []
    total_sam_masks_generated = 0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_annotated_path, fourcc, sample_fps, (reader.width, reader.height))
    
    processed_frames = 0
    t_start = time.time()
    
    for frame_idx, timestamp, orig_frame, proc_frame in reader.extract_frames(sample_fps=sample_fps, target_size=None):
        if max_frames is not None and processed_frames >= max_frames:
            print('      • Reached max_frames limit (' + str(max_frames) + '). Finalizing segment analytics...')
            break
            
        dets = detector.detect_frame(proc_frame)
        
        frame_sight_m = depth_estimator.estimate_sight_distance(proc_frame)
        frame_sight_distances.append(frame_sight_m)
        
        cum_metrics['commercial_count'] += dets['commercial_count']
        cum_metrics['pedestrian_count'] += dets['pedestrian_count']
        cum_metrics['animal_count'] += dets['animal_count']
        cum_metrics['vehicle_count'] += dets['vehicle_count']
        cum_metrics['avg_sight_distance_m'] = float(np.median(frame_sight_distances))
        
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
            
        calibrated_at_frame = FeatureCalibrator.calibrate_segment_metrics(cum_metrics)
        pred_at_frame = predictor.predict_risk(calibrated_at_frame)
        
        log_entry = {
            'frame_index': frame_idx,
            'timestamp_sec': round(timestamp, 2),
            'timestamp_formatted': format_timestamp(timestamp),
            'is_sam2_keyframe': is_keyframe,
            'sam2_masks_refined': sam_mask_count,
            'estimated_sight_distance_m': frame_sight_m,
            'pedestrians_in_frame': dets['pedestrian_count'],
            'vehicles_in_frame': dets['vehicle_count'],
            'animals_in_frame': dets['animal_count'],
            'commercial_in_frame': dets['commercial_count'],
            'cum_pedestrians': cum_metrics['pedestrian_count'],
            'cum_vehicles': cum_metrics['vehicle_count'],
            'cum_animals': cum_metrics['animal_count'],
            'cum_commercial': cum_metrics['commercial_count'],
            'sight_distance_ord': calibrated_at_frame['Is sight distance adequate for safe driving?_ord'],
            'commercial_density_ord': calibrated_at_frame['Roadside commercial activity density_ord'],
            'pedestrian_conflicts_ord': calibrated_at_frame['Pedestrian-vehicle conflicts_ord'],
            'jaywalking_frequency_ord': calibrated_at_frame['Frequency of jaywalking pedestrians_ord'],
            'unsafe_merging_ord': calibrated_at_frame['Unsafe merging diverging maneuvers observed_ord'],
            'crosswalk_visibility_ord': calibrated_at_frame['Visibility of pedestrian crossings_ord'],
            'road_surface_ord': calibrated_at_frame['Road surface condition_ord'],
            'lane_marking_ord': calibrated_at_frame['Lane marking condition_ord'],
            'junction_channelization_ord': calibrated_at_frame['Junction channelization_ord'],
            'animal_presence_ord': calibrated_at_frame['Presence of animals on road_ord'],
            'predicted_risk_tier': pred_at_frame['predicted_risk_tier'],
            'frame_risk_score': pred_at_frame['risk_score'],
            'prob_very_low': pred_at_frame['probabilities'].get('Very Low', 0.0),
            'prob_low': pred_at_frame['probabilities'].get('Low', 0.0),
            'prob_high': pred_at_frame['probabilities'].get('High', 0.0),
            'prob_very_high': pred_at_frame['probabilities'].get('Very High', 0.0)
        }
        frame_logs.append(log_entry)
        
        # Position sight distance badge dynamically above bottom edge
        sd_y = max(30, reader.height - 20)
        cv2.putText(annotated_frame, 'SIGHT DISTANCE: ' + str(round(frame_sight_m, 1)) + 'm', (20, sd_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
                    
        writer.write(annotated_frame)
        processed_frames += 1

    writer.release()
    t_elapsed = time.time() - t_start
    print('      ✔ Processed ' + str(processed_frames) + ' sampled frames in ' + str(round(t_elapsed, 2)) + 's (' + str(round(processed_frames/t_elapsed, 1)) + ' FPS)')
    print('      ✔ Generated ' + str(total_sam_masks_generated) + ' high-precision SAM 2 instance masks on keyframes')
    
    print('\n[3/4] Exporting Frame-by-Frame Timestamps & Detections to CSV...')
    df_logs = pd.DataFrame(frame_logs)
    df_logs.to_csv(output_csv_path, index=False)
    print('      ✔ Saved CSV Log  : ' + str(output_csv_path) + ' (' + str(len(df_logs)) + ' rows)')
    
    print('\n[4/4] Executing Final Segment XGBoost Risk Inference Engine...')
    final_calibrated_payload = FeatureCalibrator.calibrate_segment_metrics(cum_metrics)
    final_result = predictor.predict_risk(final_calibrated_payload)
    
    audit_report = {
        'video_metadata': info,
        'pipeline_config': {
            'detection_model': model_weights,
            'depth_model': 'MiDaS_small',
            'sam_model': sam_weights,
            'inference_resolution': res_str
        },
        'cv_detections_summary': {
            'processed_frames': processed_frames,
            'processing_time_sec': round(t_elapsed, 2),
            'fps_throughput': round(processed_frames / t_elapsed, 1),
            'median_sight_distance_m': round(cum_metrics['avg_sight_distance_m'], 1),
            'total_pedestrians_detected': cum_metrics['pedestrian_count'],
            'total_vehicles_detected': cum_metrics['vehicle_count'],
            'total_animals_detected': cum_metrics['animal_count'],
            'total_commercial_objects_detected': cum_metrics['commercial_count'],
            'total_sam2_masks_generated': total_sam_masks_generated
        },
        'xgboost_calibrated_payload': final_calibrated_payload,
        'risk_prediction': final_result,
        'output_files': {
            'annotated_video': output_annotated_path,
            'json_report': output_json_path,
            'csv_log': output_csv_path
        }
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(audit_report, f, indent=2)
        
    print('\n' + '=' * 75)
    print('FINAL RISK ASSESSMENT [' + video_basename + ']: ' + final_result['emoji'] + ' [' + final_result['predicted_risk_tier'].upper() + ' RISK]')
    print('=' * 75)
    print('• Vision-AI Models     : YOLO11m + MiDaS Depth + SAM 2 (' + sam_weights + ')')
    print('• Crash Index Score    : ' + str(final_result['risk_score']) + ' / ' + str(final_result['risk_score_max']))
    print('• Sight Distance Range : ' + str(round(cum_metrics['avg_sight_distance_m'], 1)) + ' meters')
    print('• Video Saved          : ' + output_annotated_path)
    print('• CSV Saved            : ' + output_csv_path)
    print('• JSON Saved           : ' + output_json_path)
    print('\nConfidence Probabilities Breakdown:')
    for tier, prob in final_result['probabilities'].items():
        bar = '█' * int(prob * 20)
        print('  - ' + tier.ljust(10) + ': ' + f'{prob:.4f}' + ' | ' + bar)
    print('=' * 75)
    return audit_report

def process_target_videos(videos=['sample.mp4', '1.mp4'], input_dir='DATA/IP', output_dir='DATA/OP_V6'):
    detector = RoadSafetyDetector(model_weights='models/yolo11m.pt', imgsz=1280)
    depth_estimator = SightDistanceDepthEstimator()
    sam_segmenter = RoadSafetySAMSegmenter(model_weights='models/sam2.1_t.pt')
    predictor = RiskPredictor(model_dir='models')
    
    for v in videos:
        v_path = os.path.join(input_dir, v)
        if os.path.exists(v_path):
            run_road_safety_pipeline(
                video_path=v_path,
                output_dir=output_dir,
                detector=detector,
                depth_estimator=depth_estimator,
                sam_segmenter=sam_segmenter,
                predictor=predictor
            )
            print('\n')
        else:
            print('Warning: Video ' + str(v_path) + ' not found.')

if __name__ == '__main__':
    process_target_videos()
