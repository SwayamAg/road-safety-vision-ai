import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

class VehicleEgoOdometryTracker:
    """
    Monocular Metric 3D Keypoint Visual Odometry Engine for Option 1 Distance Measurement.
    
    Combines dense metric depth maps with dynamic-masked static keypoint tracking to:
      1. Calculate frame-to-frame metric vehicle displacement (Delta Z in meters).
      2. Compute vehicle speed (km/h) with temporal smoothing.
      3. Track continuous cumulative distance traveled from 0m to video end.
      4. Manage 500m road safety spatial segments and milestone triggers.
      5. Render real-time visual HUD overlays with progress bars and telemetry.
    """
    
    def __init__(
        self,
        camera_fov_deg: float = 65.0,
        target_segment_length_m: float = 500.0,
        max_speed_kmh: float = 160.0,
        min_tracked_points: int = 12
    ):
        self.fov_deg = camera_fov_deg
        self.target_segment_length_m = float(target_segment_length_m)
        self.max_speed_kmh = float(max_speed_kmh)
        self.min_tracked_points = int(min_tracked_points)
        
        # State tracking
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_depth_m: Optional[np.ndarray] = None
        self.prev_points_2d: Optional[np.ndarray] = None
        self.prev_mask: Optional[np.ndarray] = None
        
        # Cumulative metrics
        self.cumulative_distance_m: float = 0.0
        self.current_speed_kmh: float = 0.0
        self.smoothed_speed_kmh: float = 0.0
        self.completed_segments: int = 0
        self.last_milestone_segment: int = 0
        
        # Camera intrinsics cache
        self.fx: float = 0.0
        self.fy: float = 0.0
        self.cx: float = 0.0
        self.cy: float = 0.0
        self.K: Optional[np.ndarray] = None
        
        # Feature detection params (Shi-Tomasi)
        self.feature_params = dict(
            maxCorners=250,
            qualityLevel=0.015,
            minDistance=12,
            blockSize=7
        )
        # Lucas-Kanade optical flow params
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def _init_intrinsics(self, height: int, width: int):
        """Initializes camera intrinsic matrix from standard dashcam FOV."""
        fov_rad = np.radians(self.fov_deg)
        focal_len = float((width / 2.0) / np.tan(fov_rad / 2.0))
        self.fx = focal_len
        self.fy = focal_len
        self.cx = float(width / 2.0)
        self.cy = float(height / 2.0)
        self.K = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def build_static_mask(self, frame_shape: Tuple[int, int], dynamic_boxes: List[Dict]) -> np.ndarray:
        """
        Creates a binary mask where dynamic moving objects (vehicles, pedestrians, animals)
        and sky/hood zones are masked out (0), leaving only static background (1).
        """
        h, w = frame_shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Mask out upper sky (top 15%) and extreme car hood (bottom 4%)
        mask[0:int(h * 0.15), :] = 0
        mask[int(h * 0.96):, :] = 0
        
        # Dynamic categories to exclude
        dynamic_categories = {"vehicle", "pedestrian", "animal"}
        
        for item in dynamic_boxes:
            category = item.get("category", "").lower()
            label = item.get("label", "").lower()
            
            is_dynamic = (
                category in dynamic_categories or 
                label in {"car", "truck", "bus", "motorcycle", "bicycle", "person", "dog", "cat", "cow", "horse"}
            )
            
            if is_dynamic and "bbox" in item:
                x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
                # Add 8% spatial safety margin around dynamic objects
                pad_w = int(max(4, (x2 - x1) * 0.08))
                pad_h = int(max(4, (y2 - y1) * 0.08))
                
                bx1 = max(0, x1 - pad_w)
                by1 = max(0, y1 - pad_h)
                bx2 = min(w, x2 + pad_w)
                by2 = min(h, y2 + pad_h)
                
                mask[by1:by2, bx1:bx2] = 0
                
        return mask

    def _unproject_to_3d(self, points_2d: np.ndarray, depth_map_m: np.ndarray) -> np.ndarray:
        """Unprojects 2D image coordinates (u, v) into 3D metric camera coordinates (X, Y, Z)."""
        h, w = depth_map_m.shape[:2]
        u = np.clip(np.round(points_2d[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.round(points_2d[:, 1]).astype(int), 0, h - 1)
        
        z = depth_map_m[v, u]
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        return np.column_stack([x, y, z])

    def _estimate_rigid_transform_3d(
        self, 
        pts3d_prev: np.ndarray, 
        pts3d_curr: np.ndarray, 
        max_ransac_iters: int = 100, 
        inlier_thresh_m: float = 0.45
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Estimates 3D-to-3D rigid body transformation (R, t) using Umeyama SVD with RANSAC.
        Returns: (Rotation matrix R, Translation vector t, Inlier count)
        """
        n = pts3d_prev.shape[0]
        if n < 4:
            return np.eye(3), np.zeros(3), 0
            
        best_inliers = 0
        best_R = np.eye(3)
        best_t = np.zeros(3)
        
        for _ in range(max_ransac_iters):
            idx = np.random.choice(n, 4, replace=False)
            src_sample = pts3d_prev[idx]
            dst_sample = pts3d_curr[idx]
            
            src_mean = np.mean(src_sample, axis=0)
            dst_mean = np.mean(dst_sample, axis=0)
            
            H = (src_sample - src_mean).T @ (dst_sample - dst_mean)
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[2, :] *= -1
                R = Vt.T @ U.T
                
            t = dst_mean - (R @ src_mean)
            
            # Count inliers
            transformed = (R @ pts3d_prev.T).T + t
            errors = np.linalg.norm(transformed - pts3d_curr, axis=1)
            inliers = np.sum(errors < inlier_thresh_m)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_R = R
                best_t = t
                if best_inliers > int(0.75 * n):
                    break
                    
        return best_R, best_t, best_inliers

    def update(
        self, 
        frame: np.ndarray, 
        depth_map_m: np.ndarray, 
        dynamic_boxes: List[Dict], 
        dt: float
    ) -> Dict:
        """
        Processes a single video frame to update 3D odometry, distance, speed, and segment milestones.
        
        Args:
            frame: Input BGR image (H, W, 3).
            depth_map_m: Dense metric depth map (H, W) in meters.
            dynamic_boxes: List of object detection dicts from YOLO/SAM.
            dt: Time delta in seconds since previous frame.
            
        Returns:
            Dictionary with distance, speed, and segment status metrics.
        """
        h, w = frame.shape[:2]
        if self.K is None:
            self._init_intrinsics(h, w)
            
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        static_mask = self.build_static_mask((h, w), dynamic_boxes)
        
        delta_d_m = 0.0
        inliers_count = 0
        speed_kmh = 0.0
        segment_event = False
        
        if self.prev_gray is not None and self.prev_depth_m is not None and dt > 1e-4:
            # Detect or reuse points on static mask in previous frame
            det_mask = self.prev_mask if self.prev_mask is not None else static_mask
            prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=det_mask, **self.feature_params)
            
            if prev_pts is not None and len(prev_pts) >= self.min_tracked_points:
                prev_pts_2d = prev_pts.reshape(-1, 2)
                
                # Forward Optical Flow
                curr_pts_2d, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, curr_gray, prev_pts_2d, None, **self.lk_params
                )
                
                # Backward validation Flow
                back_pts_2d, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
                    curr_gray, self.prev_gray, curr_pts_2d, None, **self.lk_params
                )
                
                # Filter valid points
                fwd_stat = (status_fwd.reshape(-1) == 1)
                bwd_stat = (status_bwd.reshape(-1) == 1)
                fb_dist = np.linalg.norm(prev_pts_2d - back_pts_2d.reshape(-1, 2), axis=1)
                valid = fwd_stat & bwd_stat & (fb_dist < 2.0)
                
                # Also ensure points stay within static mask in current frame
                curr_valid_pts = curr_pts_2d[valid].reshape(-1, 2)
                prev_valid_pts = prev_pts_2d[valid]
                
                if len(curr_valid_pts) >= self.min_tracked_points:
                    u_curr = np.clip(np.round(curr_valid_pts[:, 0]).astype(int), 0, w - 1)
                    v_curr = np.clip(np.round(curr_valid_pts[:, 1]).astype(int), 0, h - 1)
                    mask_valid = (static_mask[v_curr, u_curr] > 0)
                    
                    p0 = prev_valid_pts[mask_valid]
                    p1 = curr_valid_pts[mask_valid]
                    
                    if len(p0) >= self.min_tracked_points:
                        # Lift to 3D metric coordinates
                        pts3d_0 = self._unproject_to_3d(p0, self.prev_depth_m)
                        pts3d_1 = self._unproject_to_3d(p1, depth_map_m)
                        
                        # Filter points with realistic depth
                        valid_z = (pts3d_0[:, 2] > 1.5) & (pts3d_0[:, 2] < 120.0) & (pts3d_1[:, 2] > 1.5) & (pts3d_1[:, 2] < 120.0)
                        pts3d_0 = pts3d_0[valid_z]
                        pts3d_1 = pts3d_1[valid_z]
                        
                        if len(pts3d_0) >= self.min_tracked_points:
                            R_est, t_est, inliers_count = self._estimate_rigid_transform_3d(pts3d_0, pts3d_1)
                            
                            if inliers_count >= self.min_tracked_points:
                                # Forward camera motion is along negative Z translation of environment
                                # Delta Z forward = -t_est[2]
                                forward_disp = -float(t_est[2])
                                lateral_disp = float(t_est[0])
                                
                                # Road ego-motion is predominantly forward
                                candidate_disp = float(np.sqrt(max(0.0, forward_disp)**2 + (0.3 * lateral_disp)**2))
                                
                                # Physical limit validation
                                max_disp_possible = (self.max_speed_kmh / 3.6) * dt
                                if 0.0 <= candidate_disp <= max_disp_possible:
                                    delta_d_m = candidate_disp
                                    speed_kmh = (delta_d_m / dt) * 3.6
                                    
        # Update smoothed speed and cumulative metrics
        alpha = 0.25 if delta_d_m > 0 else 0.1
        self.smoothed_speed_kmh = (alpha * speed_kmh) + ((1.0 - alpha) * self.smoothed_speed_kmh)
        self.current_speed_kmh = round(self.smoothed_speed_kmh, 1)
        self.cumulative_distance_m += float(delta_d_m)
        
        # Segment accounting (e.g. 500m intervals)
        completed_segs = int(self.cumulative_distance_m // self.target_segment_length_m)
        if completed_segs > self.last_milestone_segment:
            segment_event = True
            self.last_milestone_segment = completed_segs
            self.completed_segments = completed_segs
            
        segment_progress_m = self.cumulative_distance_m % self.target_segment_length_m
        segment_progress_pct = (segment_progress_m / self.target_segment_length_m) * 100.0
        
        # Cache current frame for next step
        self.prev_gray = curr_gray
        self.prev_depth_m = depth_map_m
        self.prev_mask = static_mask
        
        return {
            "delta_distance_m": round(delta_d_m, 2),
            "cumulative_distance_m": round(self.cumulative_distance_m, 1),
            "current_speed_kmh": self.current_speed_kmh,
            "segment_progress_m": round(segment_progress_m, 1),
            "segment_progress_pct": round(segment_progress_pct, 1),
            "completed_500m_segments": completed_segs,
            "milestone_500m_triggered": segment_event,
            "tracked_static_inliers": inliers_count
        }

    def draw_distance_hud(
        self, 
        frame: np.ndarray, 
        metrics: Dict, 
        segment_risk_info: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Renders a dual-panel HUD banner:
          1. Left Panel : Vehicle Telemetry (traveled distance, speed, 500m segment progress bar, 3D lock).
          2. Right Panel: 500m Segment Risk Prediction (score, tier, and status per 500m cap).
          3. Center Alert Banner: Prominently displayed when a 500m milestone cap is reached.
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.44, min(0.60, (w / 1920.0) * 0.52))
        sub_scale = max(0.38, min(0.50, (w / 1920.0) * 0.44))
        
        # -------------------------------------------------------------
        # 1. Left Telemetry HUD Card
        # -------------------------------------------------------------
        card1_w = int(w * 0.47)
        card1_h = 86
        c1_x1, c1_y1 = 15, 15
        c1_x2, c1_y2 = c1_x1 + card1_w, c1_y1 + card1_h
        
        overlay = annotated.copy()
        cv2.rectangle(overlay, (c1_x1, c1_y1), (c1_x2, c1_y2), (18, 22, 28), -1)
        cv2.rectangle(overlay, (c1_x1, c1_y1), (c1_x2, c1_y2), (0, 200, 255), 1)
        
        cum_dist = metrics.get("cumulative_distance_m", 0.0)
        speed = metrics.get("current_speed_kmh", 0.0)
        seg_pct = metrics.get("segment_progress_pct", 0.0)
        seg_m = metrics.get("segment_progress_m", 0.0)
        completed_segs = metrics.get("completed_500m_segments", 0)
        seg_cap = self.target_segment_length_m
        active_seg_num = completed_segs + 1
        inliers = metrics.get("tracked_static_inliers", 0)
        
        # -------------------------------------------------------------
        # 2. Right Segment Risk Assessment Card
        # -------------------------------------------------------------
        card2_w = int(w * 0.48)
        card2_h = 86
        c2_x1 = max(c1_x2 + 10, w - card2_w - 15)
        c2_y1 = 15
        c2_x2, c2_y2 = c2_x1 + card2_w, c2_y1 + card2_h
        
        # Determine risk color and text
        risk_color = (0, 220, 100) # Default Green (Very Low)
        risk_title = f"SEGMENT RISK AUDIT ({seg_cap:.0f}m CAP)"
        risk_score_line = f"Active Seg #{active_seg_num}: Accumulating Metrics"
        risk_sub_line = f"Cap Trigger in {max(0.0, seg_cap - seg_m):.1f}m"
        
        if segment_risk_info is not None:
            latest = segment_risk_info.get("latest_completed_risk")
            provisional = segment_risk_info.get("current_provisional_prediction")
            
            if latest is not None:
                pred = latest.get("risk_prediction", {})
                tier = pred.get("predicted_risk_tier", "Low")
                score = pred.get("risk_score", 0.0)
                seg_idx = latest.get("segment_index", 1)
                r_range = latest.get("range_str", "")
                
                # Color code
                if tier == "Very Low":
                    risk_color = (0, 220, 80)
                elif tier == "Low":
                    risk_color = (0, 230, 210)
                elif tier == "High":
                    risk_color = (0, 140, 255)
                else: # Very High
                    risk_color = (40, 40, 255)
                    
                risk_title = f"COMPLETED SEGMENT #{seg_idx} [{r_range}] RISK"
                risk_score_line = f"LOCKED TIER: {tier.upper()}  |  SCORE: {score:.2f} / 10.0"
                risk_sub_line = f"Active Seg #{active_seg_num}: {seg_m:.1f}m / {seg_cap:.0f}m ({seg_pct:.1f}%)"
            elif provisional is not None:
                tier = provisional.get("predicted_risk_tier", "Very Low")
                score = provisional.get("risk_score", 0.0)
                if tier == "Very Low":
                    risk_color = (0, 220, 80)
                elif tier == "Low":
                    risk_color = (0, 230, 210)
                elif tier == "High":
                    risk_color = (0, 140, 255)
                else:
                    risk_color = (40, 40, 255)
                risk_score_line = f"PROVISIONAL TIER: {tier.upper()} ({score:.2f}/10)"
                risk_sub_line = f"Evaluating ongoing {seg_cap:.0f}m window ({seg_pct:.1f}%)"
                
        cv2.rectangle(overlay, (c2_x1, c2_y1), (c2_x2, c2_y2), (18, 22, 28), -1)
        cv2.rectangle(overlay, (c2_x1, c2_y1), (c2_x2, c2_y2), risk_color, 2)
        cv2.addWeighted(overlay, 0.82, annotated, 0.18, 0, annotated)
        
        # Render Left Card Text
        dist_text = f"TRAVELED: {cum_dist:5.1f}m  |  SPEED: {speed:4.1f} km/h"
        seg_text = f"SEGMENT #{active_seg_num} ({seg_cap:.0f}m cap): {seg_m:4.1f}m ({seg_pct:4.1f}%)"
        status_text = f"3D Lock: {inliers} pts"
        
        cv2.putText(annotated, dist_text, (c1_x1 + 12, c1_y1 + 25), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, seg_text, (c1_x1 + 12, c1_y1 + 47), font, sub_scale, (200, 240, 255), 1, cv2.LINE_AA)
        
        # Progress Bar on Left Card
        bar_x1, bar_y = c1_x1 + 12, c1_y1 + 60
        bar_w, bar_h = int(card1_w * 0.65), 11
        cv2.rectangle(annotated, (bar_x1, bar_y), (bar_x1 + bar_w, bar_y + bar_h), (45, 50, 60), -1)
        fill_w = int((seg_pct / 100.0) * bar_w)
        fill_w = max(0, min(bar_w, fill_w))
        bar_color = (255, 200, 0) if seg_pct < 85 else (0, 230, 100)
        if fill_w > 0:
            cv2.rectangle(annotated, (bar_x1, bar_y), (bar_x1 + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(annotated, (bar_x1, bar_y), (bar_x1 + bar_w, bar_y + bar_h), (110, 120, 130), 1)
        cv2.putText(annotated, status_text, (bar_x1 + bar_w + 12, bar_y + 9), font, sub_scale, (120, 240, 120), 1, cv2.LINE_AA)
        
        # Render Right Card Text
        cv2.putText(annotated, risk_title, (c2_x1 + 12, c2_y1 + 25), font, sub_scale, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated, risk_score_line, (c2_x1 + 12, c2_y1 + 49), font, font_scale, risk_color, 2, cv2.LINE_AA)
        cv2.putText(annotated, risk_sub_line, (c2_x1 + 12, c2_y1 + 71), font, sub_scale, (180, 200, 210), 1, cv2.LINE_AA)
        
        # -------------------------------------------------------------
        # 3. Center Milestone Alert Banner (when cap threshold is crossed)
        # -------------------------------------------------------------
        if segment_risk_info is not None and segment_risk_info.get("milestone_alert_active", False):
            banner_text = segment_risk_info.get("milestone_alert_text", "MILESTONE REACHED")
            banner_w = int(w * 0.70)
            banner_h = 40
            bx1 = int((w - banner_w) / 2)
            by1 = c1_y2 + 12
            bx2 = bx1 + banner_w
            by2 = by1 + banner_h
            
            b_overlay = annotated.copy()
            cv2.rectangle(b_overlay, (bx1, by1), (bx2, by2), (20, 25, 35), -1)
            cv2.rectangle(b_overlay, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
            cv2.addWeighted(b_overlay, 0.88, annotated, 0.12, 0, annotated)
            
            text_size = cv2.getTextSize(banner_text, font, font_scale * 1.05, 2)[0]
            tx = bx1 + int((banner_w - text_size[0]) / 2)
            ty = by1 + int((banner_h + text_size[1]) / 2)
            cv2.putText(annotated, banner_text, (tx, ty), font, font_scale * 1.05, (0, 255, 255), 2, cv2.LINE_AA)
            
        return annotated


if __name__ == "__main__":
    from video_loader import VideoFrameReader
    from cv_depth import SightDistanceDepthEstimator
    from cv_detector import RoadSafetyDetector
    import os
    
    print("Testing VehicleEgoOdometryTracker (Option 1)...")
    test_video = "DATA/IP/1.mp4"
    if not os.path.exists(test_video):
        VideoFrameReader.generate_synthetic_test_video(test_video)
        
    reader = VideoFrameReader(test_video)
    depth_est = SightDistanceDepthEstimator()
    detector = RoadSafetyDetector()
    tracker = VehicleEgoOdometryTracker(target_segment_length_m=500.0)
    
    prev_time = 0.0
    for frame_idx, t_sec, orig, resized in reader.extract_frames(sample_fps=2.0):
        dt = t_sec - prev_time if frame_idx > 0 else 0.5
        prev_time = t_sec
        
        depth_map, sight_m = depth_est.compute_depth_map(resized)
        det_result = detector.detect_frame(resized)
        
        metrics = tracker.update(resized, depth_map, det_result["boxes"], dt=dt)
        print(f"[Frame {frame_idx:3d} @ {t_sec:4.1f}s] Distance: {metrics['cumulative_distance_m']:5.1f}m | Speed: {metrics['current_speed_kmh']:4.1f} km/h | 500m Progress: {metrics['segment_progress_pct']}% | Inliers: {metrics['tracked_static_inliers']}")
        
        if frame_idx >= 10:
            break
