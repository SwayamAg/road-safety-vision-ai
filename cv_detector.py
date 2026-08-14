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
    
    def __init__(self, model_weights: str = "yolo11m.pt", conf_threshold: float = 0.20, imgsz: int = 1280):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.imgsz = imgsz
        print(f"Loading YOLO11 Medium Model [{model_weights}] @ imgsz={self.imgsz} on device: {self.device.upper()}")
        self.model = YOLO(model_weights)
        self.conf_threshold = conf_threshold

    def detect_frame(self, frame: np.ndarray) -> dict:
        """
        Runs object detection on a single RGB/BGR frame using explicit target class filtering.
        Ignores non-road COCO classes (backpacks, cell phones, ties, etc.) to eliminate false positives.
        """
        results = self.model.predict(source=frame, conf=self.conf_threshold, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        
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
            
            is_pedestrian = cls_id in self.COCO_PEDESTRIAN_CLASSES
            is_vehicle = cls_id in self.COCO_VEHICLE_CLASSES
            is_animal = cls_id in self.COCO_ANIMAL_CLASSES
            is_commercial = cls_id in self.COCO_COMMERCIAL_CLASSES
            
            # Skip non-target COCO classes (eliminates spurious background boxes)
            if not (is_pedestrian or is_vehicle or is_animal or is_commercial):
                continue
                
            if is_pedestrian:
                detections["pedestrian_count"] += 1
                category = "pedestrian"
            elif is_vehicle:
                detections["vehicle_count"] += 1
                category = "vehicle"
            elif is_animal:
                detections["animal_count"] += 1
                category = "animal"
            elif is_commercial:
                detections["commercial_count"] += 1
                category = "commercial"
                
            detections["boxes"].append({
                "class_id": cls_id,
                "label": label,
                "category": category,
                "confidence": round(conf, 3),
                "bbox": [round(c, 1) for c in xyxy]
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
        hud_height = 55
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_height), (15, 15, 25), cv2.FILLED)
        cv2.addWeighted(overlay, 0.70, annotated, 0.30, 0, annotated)
        
        # HUD Text Strings
        hud_peds = f"Peds: {detections['pedestrian_count']}"
        hud_vehs = f"Vehs: {detections['vehicle_count']}"
        hud_anim = f"Animals: {detections['animal_count']}"
        hud_comm = f"Commercial: {detections['commercial_count']}"
        
        # Draw Color-Coded HUD Metrics
        cv2.putText(annotated, f"ROAD SAFETY AUDIT HUD | ", (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, hud_peds, (270, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, f" | {hud_vehs}", (365, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 235, 120), 2, cv2.LINE_AA)
        cv2.putText(annotated, f" | {hud_anim}", (470, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f" | {hud_comm}", (580, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2, cv2.LINE_AA)
        
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
