import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO

class RoadSafetyDetector:
    """
    Multi-Class Object Detector for Phase 2 Road Safety Perception.
    Leverages YOLO11 Medium (yolo11m.pt) with explicit class filtering to detect 
    pedestrians, vehicles, stray animals, and commercial roadside friction objects.
    Renders high-legibility HUD video annotations.
    """
    # Explicit COCO Target Class Maps
    COCO_PEDESTRIAN_CLASSES = {0: 'person'}
    COCO_VEHICLE_CLASSES = {1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    COCO_ANIMAL_CLASSES = {15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow'}
    COCO_COMMERCIAL_CLASSES = {9: 'traffic light', 11: 'stop sign', 13: 'bench', 25: 'umbrella', 56: 'chair', 58: 'potted plant', 60: 'dining table'}
    
    def __init__(self, model_weights: str = "models/yolo11m.pt", conf_threshold: float = None, imgsz: int = None, config_path: str = "sam2_classes_config.json"):
        import json
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.imgsz = imgsz
        
        # Load external JSON class configuration
        self.config_path = config_path
        self.class_configs = {}
        self.name_to_config = {}
        self.global_sam2_params = {}
        self.default_conf_threshold = conf_threshold if conf_threshold is not None else 0.20
        
        if not os.path.exists(self.config_path):
            alt_config = os.path.join("..", config_path)
            if os.path.exists(alt_config):
                self.config_path = alt_config
                
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                    self.global_sam2_params = cfg_data.get("global_sam2_parameters", cfg_data.get("sam2_global_settings", {}))
                    for item in cfg_data.get("classes", []):
                        cid = item.get("class_id")
                        cname = item.get("name", "").lower().strip()
                        self.class_configs[cid] = item
                        if cname:
                            self.name_to_config[cname] = item
                            # Add common hyphen/underscore and COCO aliases
                            self.name_to_config[cname.replace("_", " ")] = item
                            self.name_to_config[cname.replace(" ", "_")] = item
                            self.name_to_config[cname.replace("-", " ")] = item
                            self.name_to_config[cname.replace("-", "_")] = item
                            
                    # Alias specific COCO labels to rich config classes
                    if "dining table" not in self.name_to_config and "table" in self.name_to_config:
                        self.name_to_config["dining table"] = self.name_to_config["table"]
                    if "potted plant" not in self.name_to_config and "potted_plant" in self.name_to_config:
                        self.name_to_config["potted plant"] = self.name_to_config["potted_plant"]
                    if "motorcycle" in self.name_to_config and "motorbike" not in self.name_to_config:
                        self.name_to_config["motorbike"] = self.name_to_config["motorcycle"]
                        
                print(f"Loaded {len(self.class_configs)} class definitions and SAM2 global params from [{self.config_path}]")
            except Exception as e:
                print(f"Warning: Could not parse class config ({e}). Using default class maps.")
                
        target_weights = model_weights
        if not os.path.exists(target_weights):
            alt_weights = os.path.join("models", os.path.basename(model_weights))
            if os.path.exists(alt_weights):
                target_weights = alt_weights
                
        res_info = f"imgsz={self.imgsz}" if self.imgsz is not None else "Dynamic Native Resolution"
        print(f"Loading YOLO11 Medium Model [{target_weights}] @ {res_info} on device: {self.device.upper()}")
        self.model = YOLO(target_weights)
        self.conf_threshold = self.default_conf_threshold

    def detect_frame(self, frame: np.ndarray, imgsz=None) -> dict:
        """
        Runs object detection on a single RGB/BGR frame using JSON-configured target classes and thresholds.
        Ignores non-road COCO classes (backpacks, cell phones, ties, etc.) to eliminate false positives.
        Uses exact input frame resolution (height, width) dynamically when imgsz is not specified.
        """
        infer_imgsz = imgsz if imgsz is not None else (self.imgsz if self.imgsz is not None else (frame.shape[0], frame.shape[1]))
        results = self.model.predict(source=frame, conf=0.10, imgsz=infer_imgsz, device=self.device, verbose=False)[0]
        
        detections = {
            "pedestrian_count": 0,
            "vehicle_count": 0,
            "animal_count": 0,
            "commercial_count": 0,
            "boxes": []
        }
        
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            label = self.model.names[cls_id]
            norm_label = label.lower().strip()
            
            # Match against JSON configuration if loaded
            cfg = None
            if self.name_to_config:
                cfg = self.name_to_config.get(norm_label)
                if not cfg and norm_label.replace(" ", "_") in self.name_to_config:
                    cfg = self.name_to_config.get(norm_label.replace(" ", "_"))
                if not cfg and norm_label.replace("_", " ") in self.name_to_config:
                    cfg = self.name_to_config.get(norm_label.replace("_", " "))
                    
            if cfg is not None:
                min_conf = cfg.get("conf_threshold", self.conf_threshold)
                if conf < min_conf:
                    continue
                category = cfg.get("category", "vehicle")
                sam2_enabled = cfg.get("sam2_enabled", True)
                description = cfg.get("description", "")
                min_area = cfg.get("sam2_min_area_pixels", cfg.get("min_area_pixels", 80))
                color = tuple(cfg.get("color_bgr", [0, 255, 255]))
                associated_param = cfg.get("associated_safety_param", "")
            else:
                # Fallback to built-in dictionaries
                is_pedestrian = cls_id in self.COCO_PEDESTRIAN_CLASSES
                is_vehicle = cls_id in self.COCO_VEHICLE_CLASSES
                is_animal = cls_id in self.COCO_ANIMAL_CLASSES
                is_commercial = cls_id in self.COCO_COMMERCIAL_CLASSES
                
                if not (is_pedestrian or is_vehicle or is_animal or is_commercial):
                    continue
                if conf < self.conf_threshold:
                    continue
                    
                if is_pedestrian:
                    category = "pedestrian"
                elif is_vehicle:
                    category = "vehicle"
                elif is_animal:
                    category = "animal"
                else:
                    category = "commercial"
                    
                sam2_enabled = True
                description = label
                min_area = 80
                color = None
                associated_param = ""
                
            if category == "pedestrian":
                detections["pedestrian_count"] += 1
            elif category == "vehicle":
                detections["vehicle_count"] += 1
            elif category == "animal":
                detections["animal_count"] += 1
            elif category == "commercial":
                detections["commercial_count"] += 1
                
            detections["boxes"].append({
                "class_id": cls_id,
                "label": label,
                "category": category,
                "confidence": round(conf, 3),
                "bbox": [round(c, 1) for c in xyxy],
                "sam2_enabled": sam2_enabled,
                "description": description,
                "min_area_pixels": min_area,
                "color_bgr": color,
                "associated_safety_param": associated_param
            })
            
        return detections

    def annotate_frame(self, frame: np.ndarray, detections: dict) -> np.ndarray:
        """
        Renders sleek autonomous-dashboard HUD annotations on the frame.
        Features corner-bracket bounding boxes, semi-transparent label pills, and dark header panel.
        """
        annotated = frame.copy()
        h, w, _ = annotated.shape
        
        # 1. Draw Bounding Boxes with Color-Coded Category Scheme
        for det in detections["boxes"]:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            label_text = f"{det['label']} {int(det['confidence'] * 100)}%"
            category = det["category"]
            
            # Color Palette (BGR):
            # Blue/Cyan = Pedestrians, Emerald Green = Vehicles, Red/Crimson = Animals, Gold/Yellow = Commercial
            if category == "animal":
                color = (0, 0, 255)
            elif category == "pedestrian":
                color = (255, 165, 0)
            elif category == "vehicle":
                color = (0, 235, 120)
            else:
                color = (0, 215, 255)
                
            # Draw Corner Brackets
            line_len = min(20, int(min(x2 - x1, y2 - y1) * 0.25))
            thickness = 2
            
            # Corners
            cv2.line(annotated, (x1, y1), (x1 + line_len, y1), color, thickness)
            cv2.line(annotated, (x1, y1), (x1, y1 + line_len), color, thickness)
            
            cv2.line(annotated, (x2, y1), (x2 - line_len, y1), color, thickness)
            cv2.line(annotated, (x2, y1), (x2, y1 + line_len), color, thickness)
            
            cv2.line(annotated, (x1, y2), (x1 + line_len, y2), color, thickness)
            cv2.line(annotated, (x1, y2), (x1, y2 - line_len), color, thickness)
            
            cv2.line(annotated, (x2, y2), (x2 - line_len, y2), color, thickness)
            cv2.line(annotated, (x2, y2), (x2, y2 - line_len), color, thickness)
            
            # Draw Label Background Pill for 100% Legibility
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            text_bg_y1 = max(0, y1 - th - 8)
            text_bg_y2 = max(th + 8, y1)
            
            cv2.rectangle(annotated, (x1, text_bg_y1), (x1 + tw + 10, text_bg_y2), color, cv2.FILLED)
            cv2.putText(annotated, label_text, (x1 + 5, text_bg_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            
        # 2. Draw Top HUD Dashboard Panel (Semi-Transparent Dark Overlay)
        hud_height = max(40, int(h * 0.08))
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_height), (15, 15, 25), cv2.FILLED)
        cv2.addWeighted(overlay, 0.70, annotated, 0.30, 0, annotated)
        
        font_scale = max(0.4, min(0.65, w / 1280.0 * 0.65))
        y_pos = int(hud_height * 0.65)
        
        # HUD Text Strings
        hud_title = "SAFETY AUDIT | "
        hud_peds = f"Peds: {detections['pedestrian_count']}"
        hud_vehs = f" | Vehs: {detections['vehicle_count']}"
        hud_anim = f" | Anim: {detections['animal_count']}"
        hud_comm = f" | Comm: {detections['commercial_count']}"
        
        # Calculate dynamic x positions
        x = 15
        cv2.putText(annotated, hud_title, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(hud_title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        x += tw
        
        cv2.putText(annotated, hud_peds, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 165, 0), 2, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(hud_peds, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        x += tw
        
        cv2.putText(annotated, hud_vehs, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 235, 120), 2, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(hud_vehs, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        x += tw
        
        cv2.putText(annotated, hud_anim, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 100, 255), 2, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(hud_anim, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        x += tw
        
        cv2.putText(annotated, hud_comm, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 215, 255), 2, cv2.LINE_AA)
        
        return annotated


if __name__ == "__main__":
    from video_loader import VideoFrameReader
    
    print("Testing RoadSafetyDetector with YOLO11 Medium...")
    test_video_path = "DATA/IP/1.mp4"
    if not os.path.exists(test_video_path):
        VideoFrameReader.generate_synthetic_test_video(test_video_path)
        
    reader = VideoFrameReader(test_video_path)
    detector = RoadSafetyDetector(model_weights="yolo11m.pt", imgsz=1280)
    
    for frame_idx, timestamp, orig_frame, resized_frame in reader.extract_frames(sample_fps=2.0):
        dets = detector.detect_frame(resized_frame)
        print(f"[Frame {frame_idx:03d} | {timestamp:.1f}s] Detections -> Peds: {dets['pedestrian_count']}, Vehicles: {dets['vehicle_count']}, Animals: {dets['animal_count']}, Commercial: {dets['commercial_count']}")
        if frame_idx >= 30:
            break
