import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO

class RoadSafetyDetector:
    """
    Multi-Class Object Detector for Phase 2 Road Safety Perception.
    Leverages YOLOv8 / YOLOv11 to detect pedestrians, vehicles, stray animals,
    and roadside commercial activity objects in road survey video frames.
    """
    COCO_ANIMAL_CLASSES = {14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe'}
    COCO_VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    COCO_PEDESTRIAN_CLASSES = {0: 'person'}
    
    def __init__(self, model_weights: str = "yolo11n.pt", conf_threshold: float = 0.25):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading YOLOv11 Model [{model_weights}] on device: {self.device.upper()}")
        self.model = YOLO(model_weights)
        self.conf_threshold = conf_threshold

    def detect_frame(self, frame: np.ndarray) -> dict:
        """
        Runs object detection on a single RGB/BGR frame.
        Returns aggregated counts and detailed bounding box detections.
        """
        results = self.model.predict(source=frame, conf=self.conf_threshold, device=self.device, verbose=False)[0]
        
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
            
            if is_pedestrian:
                detections["pedestrian_count"] += 1
            elif is_vehicle:
                detections["vehicle_count"] += 1
            elif is_animal:
                detections["animal_count"] += 1
            else:
                # Roadside objects / shop proxies
                detections["commercial_count"] += 1
                
            detections["boxes"].append({
                "class_id": cls_id,
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [round(c, 1) for c in xyxy]
            })
            
        return detections

    def annotate_frame(self, frame: np.ndarray, detections: dict) -> np.ndarray:
        """
        Draws colored bounding boxes and visual count overlays on the input frame.
        """
        annotated = frame.copy()
        for det in detections["boxes"]:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            label = f"{det['label']} {det['confidence']}"
            
            # Color coding: Red = Animal, Blue = Pedestrian, Green = Vehicle, Yellow = Commercial
            if det["class_id"] in self.COCO_ANIMAL_CLASSES:
                color = (0, 0, 255)
            elif det["class_id"] in self.COCO_PEDESTRIAN_CLASSES:
                color = (255, 100, 0)
            elif det["class_id"] in self.COCO_VEHICLE_CLASSES:
                color = (0, 255, 0)
            else:
                color = (0, 255, 255)
                
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        # Top HUD bar
        hud_text = f"Peds: {detections['pedestrian_count']} | Vehicles: {detections['vehicle_count']} | Animals: {detections['animal_count']} | Commercial: {detections['commercial_count']}"
        cv2.putText(annotated, hud_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return annotated


if __name__ == "__main__":
    from video_loader import VideoFrameReader
    
    print("Testing RoadSafetyDetector...")
    test_video_path = "sample_dashcam_test.mp4"
    if not os.path.exists(test_video_path):
        VideoFrameReader.generate_synthetic_test_video(test_video_path)
        
    reader = VideoFrameReader(test_video_path)
    detector = RoadSafetyDetector(model_weights="yolov8n.pt")
    
    for frame_idx, timestamp, orig_frame, resized_frame in reader.extract_frames(sample_fps=2.0):
        dets = detector.detect_frame(resized_frame)
        print(f"[Frame {frame_idx:03d} | {timestamp:.1f}s] Detections -> Peds: {dets['pedestrian_count']}, Vehicles: {dets['vehicle_count']}, Animals: {dets['animal_count']}")
