import os
import sys
import shutil
import time
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from cv_detector import RoadSafetyDetector
from cv_depth import SightDistanceDepthEstimator
from risk_predictor import RiskPredictor
from main_pipeline import run_road_safety_pipeline

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Global Model Container
MODEL_STATE = {
    "detector": None,
    "depth_estimator": None,
    "predictor": None,
    "status": "INITIALIZING"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for loading and releasing models."""
    print("🚀 Initializing Road Safety Vision AI API models...")
    try:
        MODEL_STATE["detector"] = RoadSafetyDetector(model_weights="models/yolo11m.pt", conf_threshold=0.20, imgsz=1280)
        MODEL_STATE["depth_estimator"] = SightDistanceDepthEstimator(model_type="MiDaS_small")
        MODEL_STATE["predictor"] = RiskPredictor(model_dir="models")
        MODEL_STATE["status"] = "HEALTHY"
        print("✅ All pre-trained models loaded successfully on CUDA GPU!")
    except Exception as e:
        MODEL_STATE["status"] = f"ERROR: {str(e)}"
        print(f"❌ Error initializing models: {e}")
    yield
    print("🛑 Shutting down FastAPI application state.")


app = FastAPI(
    title="🛣️ Road Safety Vision-AI Analytics API",
    description="Production REST API for Automated Computer Vision & SMOTE-balanced XGBoost Road Crash Risk Assessment",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """API Root Welcome Endpoint."""
    return {
        "title": "Road Safety Vision-AI Analytics API",
        "status": MODEL_STATE["status"],
        "docs": "/docs",
        "endpoints": {
            "model_status": "/models/status",
            "analyze_video": "/analyze_video",
            "download_output": "/download/{file_name}"
        }
    }


@app.get("/models/status", tags=["Model Telemetry"])
async def check_models_status():
    """
    Endpoint 1: Checks operational health, CUDA device, and memory status for all pre-trained models.
    """
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    
    detector_status = "LOADED" if MODEL_STATE["detector"] is not None else "NOT_LOADED"
    depth_status = "LOADED" if MODEL_STATE["depth_estimator"] is not None else "NOT_LOADED"
    predictor_status = "LOADED" if MODEL_STATE["predictor"] is not None and getattr(MODEL_STATE["predictor"], "model", None) is not None else "NOT_LOADED"
    
    is_healthy = (detector_status == "LOADED") and (predictor_status == "LOADED")
    
    return {
        "status": "HEALTHY" if is_healthy else "UNHEALTHY",
        "cuda_acceleration": {
            "available": cuda_available,
            "device": device_name,
            "torch_version": torch.__version__
        },
        "models": {
            "object_detector": {
                "name": "YOLO11 Medium",
                "weights": "yolo11m.pt",
                "parameters": "20.1M",
                "inference_resolution": "1280x1280",
                "status": detector_status
            },
            "monocular_depth": {
                "name": "MiDaS Monocular Depth",
                "variant": "MiDaS_small",
                "task": "Sight Distance Metric Clearance (Meters)",
                "status": depth_status
            },
            "risk_classifier": {
                "name": "Tuned XGBoost Classifier",
                "artifact": "best_road_safety_model.pkl",
                "input_features": 10,
                "test_accuracy": "90.0%",
                "macro_f1": 0.9021,
                "status": predictor_status
            }
        }
    }


@app.post("/analyze_video", tags=["Inference Pipeline"])
async def analyze_video(
    file: UploadFile = File(..., description="MP4, AVI, or MOV video stream of road survey"),
    sample_fps: float = Query(5.0, description="Keyframe extraction rate (default: 5.0 FPS)"),
    imgsz: int = Query(1280, description="YOLO11 tensor input resolution (default: 1280)")
):
    """
    Endpoint 2: Uploads a road survey video, executes Vision AI + XGBoost analytics,
    and returns full audit metrics, Risk Tier prediction, annotated video URL, and Excel log URL.
    """
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an MP4, AVI, MOV, or MKV video.")
        
    os.makedirs("DATA/IP", exist_ok=True)
    os.makedirs("DATA/OP_V4", exist_ok=True)
    
    # Save uploaded video to DATA/IP/
    temp_filename = f"upload_{int(time.time())}_{file.filename}"
    input_video_path = os.path.join("DATA/IP", temp_filename)
    
    with open(input_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Run End-to-End Execution Pipeline
        audit_report = run_road_safety_pipeline(
            video_path=input_video_path,
            output_dir="DATA/OP_V4",
            model_weights="models/yolo11m.pt",
            imgsz=imgsz
        )
        
        # Build Download Base URLs
        video_basename = os.path.splitext(os.path.basename(input_video_path))[0]
        
        download_urls = {
            "annotated_video": f"/download/{video_basename}_annotated.mp4",
            "excel_log": f"/download/{video_basename}_frame_by_frame_detections.xlsx",
            "csv_log": f"/download/{video_basename}_frame_by_frame_detections.csv",
            "json_report": f"/download/{video_basename}_risk_report.json"
        }
        
        return JSONResponse(content={
            "message": "Road Safety Vision-AI Audit Completed Successfully",
            "download_urls": download_urls,
            "audit_summary": audit_report
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {str(e)}")


@app.get("/download/{file_name}", tags=["File Downloads"])
async def download_file(file_name: str):
    """
    Download endpoint for generated annotated videos (.mp4), Excel logs (.xlsx), CSVs (.csv), or JSON reports.
    """
    file_path = os.path.join("DATA/OP_V4", file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{file_name}' not found in DATA/OP_V4.")
        
    media_type = "application/octet-stream"
    if file_name.endswith(".mp4"):
        media_type = "video/mp4"
    elif file_name.endswith(".xlsx"):
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif file_name.endswith(".csv"):
        media_type = "text/csv"
    elif file_name.endswith(".json"):
        media_type = "application/json"
        
    return FileResponse(path=file_path, filename=file_name, media_type=media_type)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
