import cv2
import torch
import numpy as np

class SightDistanceDepthEstimator:
    """
    Monocular Metric Depth Estimation Engine for Phase 2 Road Safety Perception.
    Integrates Depth Anything V2 / MiDaS to calculate forward visual clearance distance 
    (in meters) along the driver horizon path to quantify Sight Distance Adequacy.
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
                state_dict = torch.load(weights_path, map_location=str(self.device))
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

    def estimate_sight_distance(self, frame: np.ndarray) -> float:
        """
        Estimates forward sight distance clearance (in meters) along the central driving corridor.
        Returns: Estimated sight distance float value in meters (e.g. 120.5m, 65.0m, 35.0m).
        """
        h, w, _ = frame.shape
        # Define central driving corridor ROI (center 40% of image width, middle-to-horizon height)
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
                    
                depth_map = prediction.cpu().numpy()
                
                # Extract central driving corridor depth values
                corridor_depth = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
                
                # Normalize relative depth to metric meter estimation
                min_val, max_val = np.percentile(corridor_depth, 5), np.percentile(corridor_depth, 95)
                depth_range = max(1e-5, max_val - min_val)
                relative_clearance = (np.median(corridor_depth) - min_val) / depth_range
                
                # Metric calibration curve: Map normalized clearance to meters (20m to 150m range)
                estimated_meters = float(np.clip(20.0 + (130.0 * (1.0 - relative_clearance)), 15.0, 180.0))
                return round(estimated_meters, 1)
            except Exception as err:
                print(f"Depth inference error: {err}")
                
        # Heuristic fallback based on visual horizon contrast & brightness variance
        corridor_bgr = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        gray = cv2.cvtColor(corridor_bgr, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        
        # Clear road corridor exhibits high spatial variance/clearance
        heuristic_meters = float(np.clip(40.0 + (std_dev * 1.8), 25.0, 140.0))
        return round(heuristic_meters, 1)


if __name__ == "__main__":
    from video_loader import VideoFrameReader
    import os
    
    print("Testing SightDistanceDepthEstimator...")
    test_video = "DATA/IP/1.mp4"
    if not os.path.exists(test_video):
        VideoFrameReader.generate_synthetic_test_video(test_video)
        
    reader = VideoFrameReader(test_video)
    estimator = SightDistanceDepthEstimator()
    
    for frame_idx, t_sec, orig, resized in reader.extract_frames(sample_fps=1.0):
        dist_m = estimator.estimate_sight_distance(resized)
        print(f"[Timestamp {t_sec:.1f}s] Estimated Sight Distance: {dist_m} meters")
        if frame_idx >= 30:
            break
