import cv2
import os
import time
import numpy as np

class VideoFrameReader:
    """
    Video ingestion and pre-processing engine for Phase 2 Road Safety Analytics.
    Handles video decoding, keyframe sampling, and resolution normalization.
    """
    def __init__(self, source_path: str):
        self.source_path = source_path
        if not os.path.exists(source_path) and not isinstance(source_path, int):
            raise FileNotFoundError(f"Video source not found: {source_path}")
            
        self.cap = cv2.VideoCapture(source_path)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video stream: {source_path}")
            
        self.native_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_sec = self.total_frames / self.native_fps if self.total_frames > 0 else 0.0

    def get_info(self) -> dict:
        return {
            "source_path": self.source_path,
            "resolution": (self.width, self.height),
            "native_fps": round(self.native_fps, 2),
            "total_frames": self.total_frames,
            "duration_sec": round(self.duration_sec, 2)
        }

    def extract_frames(self, sample_fps: float = 5.0, target_size: tuple = None):
        """
        Generator that samples frames at a specified FPS rate.
        If target_size is None, keeps original input resolution without downgrading.
        Yields: (frame_index, timestamp_sec, original_frame, proc_frame)
        """
        frame_interval = max(1, int(round(self.native_fps / sample_fps)))
        frame_idx = 0
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / self.native_fps
                proc_frame = frame if target_size is None else cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
                yield frame_idx, timestamp, frame, proc_frame
                
            frame_idx += 1
            
        self.cap.release()

    @staticmethod
    def generate_synthetic_test_video(output_path: str = "sample_dashcam_test.mp4", duration_sec: int = 5, fps: int = 30):
        """
        Generates a synthetic road survey test video with animated road, lane lines, and mock objects.
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))
        
        num_frames = duration_sec * fps
        for i in range(num_frames):
            # Create dark asphalt road canvas
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            frame[200:, :] = (50, 50, 50)  # Asphalt
            frame[0:200, :] = (200, 150, 100) # Sky horizon
            
            # Draw perspective road lanes
            cv2.line(frame, (400, 200), (100, 720), (255, 255, 255), 4) # Left lane
            cv2.line(frame, (880, 200), (1180, 720), (255, 255, 255), 4) # Right lane
            
            # Animated dashed center line
            offset = (i * 15) % 80
            for y in range(200 + offset, 720, 80):
                scale = (y - 200) / 520
                x = int(640)
                length = int(30 * scale)
                cv2.line(frame, (x, y), (x, min(720, y + length)), (0, 255, 255), max(2, int(6 * scale)))
                
            # Draw mock roadside commercial shop / animal obstacles
            if i % 30 < 15:
                # Roadside shop box on left
                cv2.rectangle(frame, (150, 350), (300, 500), (0, 100, 250), -1)
                cv2.putText(frame, "Shop", (170, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
            # Telemetry text overlay
            cv2.putText(frame, f"Dashcam Frame: {i+1}/{num_frames} | Time: {i/fps:.2f}s", 
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            out.write(frame)
            
        out.release()
        return output_path


if __name__ == "__main__":
    print("Testing VideoFrameReader...")
    test_video_path = VideoFrameReader.generate_synthetic_test_video()
    print(f"Generated synthetic test video: {test_video_path}")
    
    reader = VideoFrameReader(test_video_path)
    print("Video Info:", reader.get_info())
    
    extracted_count = 0
    for frame_idx, t_sec, orig, resized in reader.extract_frames(sample_fps=5.0):
        extracted_count += 1
        
    print(f"Extracted {extracted_count} frames at 5 FPS sampling rate.")
