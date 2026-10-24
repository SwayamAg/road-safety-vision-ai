import cv2
import torch
import numpy as np

class SightDistanceDepthEstimator:
    """
    Monocular Metric Depth Estimation Engine for Phase 2 Road Safety Perception.
    Integrates MiDaS / Metric Depth transforms to:
      1. Calculate forward visual clearance distance (meters) along driver horizon path.
      2. Generate dense 2D metric depth maps (meters) for 3D Keypoint Ego-Odometry (500m distance tracking).
    """
    def __init__(self, model_type: str = "MiDaS_small", weights_path: str = "models/midas_v21_small_256.pt"):
        import os
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Depth Estimation Model [{model_type}] on device: {self.device.type.upper()}")
        
        try:
            # Automatically trust Intel-ISL & Ross Wightman repos to prevent container EOF prompt
            if hasattr(torch.hub, "_TRUSTED_REPO_OWNERS"):
                owners = set(torch.hub._TRUSTED_REPO_OWNERS)
                owners.update(["intel-isl", "rwightman", "facebookresearch"])
                torch.hub._TRUSTED_REPO_OWNERS = tuple(owners)
                
            # Load PyTorch Hub MiDaS monocular depth architecture
            self.midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            
            # Load local pre-trained weights if available for offline Railway.app deployment
            if os.path.exists(weights_path):
                print(f"Loading local MiDaS weights from: {weights_path}")
                state_dict = torch.load(weights_path, map_location=str(self.device), weights_only=False)
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                self.midas.load_state_dict(state_dict, strict=False)
                
            self.midas.to(self.device)
            self.midas.eval()
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = midas_transforms.small_transform if "small" in model_type else midas_transforms.dpt_transform
            self.is_loaded = True
        except Exception as e:
            print(f"Warning: Depth Estimation loading fallback ({e}). Using contrast depth fallback.")
            self.is_loaded = False

    def compute_depth_map(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Infers dense metric depth map (in meters) and calculates forward sight distance.
        
        Returns:
            metric_depth_map: 2D numpy array (H, W) float32 with depth values in meters.
            sight_distance_m: Forward line-of-sight clearance in meters.
        """
        h, w, _ = frame.shape
        roi_x1, roi_x2 = int(w * 0.3), int(w * 0.7)
        roi_y1, roi_y2 = int(h * 0.25), int(h * 0.75)
        
        if self.is_loaded:
            try:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_batch = self.transform(img_rgb).to(self.device)
                
                with torch.no_grad():
                    prediction = self.midas(input_batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img_rgb.shape[:2],
                        mode="bicubic",
                        align_corners=False
                    ).squeeze()
                    
                raw_disp = prediction.cpu().numpy().astype(np.float32)
                raw_disp = np.maximum(raw_disp, 1e-4)
                
                # Corridor relative clearance calculation
                corridor_disp = raw_disp[roi_y1:roi_y2, roi_x1:roi_x2]
                min_val = np.percentile(corridor_disp, 5)
                max_val = np.percentile(corridor_disp, 95)
                disp_range = max(1e-5, max_val - min_val)
                relative_clearance = (np.median(corridor_disp) - min_val) / disp_range
                sight_distance_m = float(np.clip(20.0 + (130.0 * (1.0 - relative_clearance)), 15.0, 180.0))
                
                # Calibrate inverse disparity to metric depth (meters) using physical ground plane projection
                # Road corridor: y from 52% to 68% height, center 40% width
                h_cam = 1.35  # standard automotive dashcam height (meters)
                fov_rad = np.radians(65.0)
                fy = float((w / 2.0) / np.tan(fov_rad / 2.0))
                v_horiz = float(0.46 * h)
                
                v_rows = np.linspace(int(h * 0.52), int(h * 0.68), 12).astype(int)
                inv_z_ground = [(v - v_horiz) / (h_cam * fy) for v in v_rows]
                corridor_disp = [float(np.median(raw_disp[v, int(w * 0.35):int(w * 0.65)])) for v in v_rows]
                
                try:
                    a, b = np.polyfit(corridor_disp, inv_z_ground, 1)
                    if a < 1e-6 or np.isnan(a) or np.isnan(b):
                        a, b = 0.0035, -0.18
                except Exception:
                    a, b = 0.0035, -0.18
                    
                inv_z_map = np.maximum(1.0 / 150.0, a * raw_disp + b)
                metric_depth = np.clip(1.0 / inv_z_map, 1.0, 150.0).astype(np.float32)
                
                return metric_depth, round(sight_distance_m, 1)
            except Exception as err:
                print(f"Depth inference error: {err}")
                
        # Heuristic fallback
        corridor_bgr = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        gray = cv2.cvtColor(corridor_bgr, cv2.COLOR_BGR2GRAY)
        std_dev = float(np.std(gray))
        sight_distance_m = float(np.clip(40.0 + (std_dev * 1.8), 25.0, 140.0))
        
        # Approximate depth gradient from bottom to horizon
        y_indices = np.linspace(3.0, sight_distance_m, h, dtype=np.float32)[:, None]
        fallback_depth = np.repeat(y_indices, w, axis=1)
        return fallback_depth, round(sight_distance_m, 1)

    def estimate_sight_distance(self, frame: np.ndarray) -> float:
        """
        Legacy wrapper returning forward sight distance clearance (meters).
        """
        _, sight_dist = self.compute_depth_map(frame)
        return sight_dist


if __name__ == "__main__":
    from video_loader import VideoFrameReader
    import os
    
    print("Testing SightDistanceDepthEstimator with Metric Depth...")
    test_video = "DATA/IP/1.mp4"
    if not os.path.exists(test_video):
        VideoFrameReader.generate_synthetic_test_video(test_video)
        
    reader = VideoFrameReader(test_video)
    estimator = SightDistanceDepthEstimator()
    
    for frame_idx, t_sec, orig, resized in reader.extract_frames(sample_fps=1.0):
        depth_map, dist_m = estimator.compute_depth_map(resized)
        print(f"[Timestamp {t_sec:.1f}s] Sight Dist: {dist_m}m | Depth Map Mean: {depth_map.mean():.1f}m (min={depth_map.min():.1f}m, max={depth_map.max():.1f}m)")
        if frame_idx >= 5:
            break
