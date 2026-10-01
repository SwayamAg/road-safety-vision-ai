import os
import sys
import uuid
import shutil
import time
import torch
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from cv_detector import RoadSafetyDetector
from cv_depth import SightDistanceDepthEstimator
from cv_sam import RoadSafetySAMSegmenter
from cv_distance_odometry import VehicleEgoOdometryTracker
from risk_predictor import RiskPredictor
from main_pipeline import run_road_safety_pipeline

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Global Model Container & Async Jobs Store
MODEL_STATE = {
    "detector": None,
    "depth_estimator": None,
    "sam_segmenter": None,
    "predictor": None,
    "status": "HEALTHY"
}

ASYNC_JOBS = {}

def get_model_state():
    """Lazily loads and caches models on first inference request to keep startup RAM minimal."""
    import gc
    if MODEL_STATE["detector"] is None:
        print("🧠 [On-Demand] Loading YOLO11 Medium detector...")
        MODEL_STATE["detector"] = RoadSafetyDetector(model_weights="models/yolo11m.pt", conf_threshold=0.20, imgsz=None)
        gc.collect()
        
    if MODEL_STATE["depth_estimator"] is None:
        print("🧠 [On-Demand] Loading MiDaS depth estimator...")
        MODEL_STATE["depth_estimator"] = SightDistanceDepthEstimator(model_type="MiDaS_small")
        gc.collect()
        
    if MODEL_STATE["sam_segmenter"] is None:
        print("🧠 [On-Demand] Loading SAM 2 keyframe segmenter...")
        MODEL_STATE["sam_segmenter"] = RoadSafetySAMSegmenter(model_weights="models/sam2.1_t.pt")
        gc.collect()
        
    if MODEL_STATE["predictor"] is None:
        print("🧠 [On-Demand] Loading XGBoost risk classifier...")
        MODEL_STATE["predictor"] = RiskPredictor(model_dir="models")
        gc.collect()
        
    return MODEL_STATE


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fast non-blocking startup: binds immediately to $PORT with minimal initial RAM footprint."""
    print("🚀 Starting Road Safety Vision AI API microservice...")
    torch.set_num_threads(2)
    MODEL_STATE["status"] = "HEALTHY"
    print("✅ Web service ready and listening for incoming requests!")
    yield
    print("🛑 Shutting down FastAPI application state.")


app = FastAPI(
    title="🛣️ Road Safety Vision-AI Analytics API",
    description="Production REST API for Automated Computer Vision, 3D Distance Odometry, and Segment-Capped XGBoost Road Crash Risk Assessment",
    version="2.1.0",
    lifespan=lifespan
)

# Enable Universal CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


@app.get("/health", tags=["General"])
async def health():
    """
    ### Health Check Endpoint
    Quick cloud liveness probe returning API service status.
    - **Input Params (IP)**: None.
    - **Returns**: `{"status": "healthy", "service": "..."}`
    """
    return {"status": "healthy", "service": "Road Safety Vision-AI Analytics API"}


@app.get("/", tags=["General"])
async def root():
    """
    ### API Root Welcome Endpoint
    Returns system status, documentation link, and catalog of all available perception endpoints.
    - **Input Params (IP)**: None.
    """
    return {
        "title": "Road Safety Vision-AI Analytics API",
        "status": MODEL_STATE["status"],
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "model_status": "/models/status",
            "analyze_video_segment_cap": "/analyze_video_segment_cap (NEW: Evaluates risk at every 500m or custom cap)",
            "analyze_video_sync": "/analyze_video",
            "analyze_video_async": "/analyze_video_async",
            "check_job_status": "/jobs/{job_id}",
            "download_output": "/download/{file_name}"
        }
    }


@app.get("/models/status", tags=["Model Telemetry"])
async def check_models_status(
    preload: bool = Query(False, description="Set to true to immediately load all neural models into GPU memory.")
):
    """
    ### Model Telemetry & Hardware Acceleration Status
    Inspects GPU availability (CUDA), PyTorch version, and loading status of all neural backbones.
    - **Input Params (IP)**: `preload` (Optional bool: if True, immediately warms up all models on GPU).
    """
    if preload:
        get_model_state()
        
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    
    detector_status = "LOADED" if MODEL_STATE["detector"] is not None else "STANDBY (Loads on first inference)"
    depth_status = "LOADED" if MODEL_STATE["depth_estimator"] is not None else "STANDBY (Loads on first inference)"
    sam_status = "LOADED" if MODEL_STATE["sam_segmenter"] is not None else "STANDBY (Loads on first inference)"
    predictor_status = "LOADED" if MODEL_STATE["predictor"] is not None and getattr(MODEL_STATE["predictor"], "model", None) is not None else "STANDBY (Loads on first inference)"
    
    all_loaded = (MODEL_STATE["detector"] is not None) and (MODEL_STATE["predictor"] is not None)
    overall_status = "HEALTHY (MODELS_LOADED)" if all_loaded else "HEALTHY (STANDBY_READY)"
    
    return {
        "status": overall_status,
        "note": "Models are lazy-loaded to preserve memory until the first inference request, or call /models/preload.",
        "cuda_acceleration": {
            "available": cuda_available,
            "device": device_name,
            "torch_version": torch.__version__
        },
        "models": {
            "object_detector": {
                "name": "YOLO11 Medium",
                "weights": "yolo11m.pt",
                "status": detector_status
            },
            "monocular_depth_and_odometry": {
                "name": "MiDaS Dense Metric Depth + 3D Keypoint VO",
                "variant": "MiDaS_small",
                "task": "Sight Distance & 500m Traveled Distance Odometry",
                "status": depth_status
            },
            "sam2_segmenter": {
                "name": "Segment Anything 2 (SAM 2.1)",
                "weights": "sam2.1_t.pt",
                "status": sam_status
            },
            "risk_classifier": {
                "name": "Tuned SMOTE-Balanced XGBoost Risk Classifier",
                "artifact": "best_road_safety_model.pkl",
                "status": predictor_status
            }
        }
    }


@app.post("/models/preload", tags=["Model Telemetry"])
async def preload_models():
    """
    ### ⚡ Warm Up / Preload All Models onto GPU
    Loads YOLO11, MiDaS Depth, SAM 2, and XGBoost models immediately into GPU memory
    so that subsequent video inference requests start with 0 initialization latency.
    - **Input Params (IP)**: None.
    """
    t0 = time.time()
    get_model_state()
    elapsed = time.time() - t0
    return {
        "status": "HEALTHY",
        "message": f"All models successfully warmed up and loaded into GPU in {elapsed:.2f}s",
        "models_status": {
            "yolo11_detector": "LOADED",
            "midas_depth": "LOADED",
            "sam2_segmenter": "LOADED",
            "xgboost_risk": "LOADED"
        }
    }


def _run_background_pipeline(
    job_id: str, 
    input_video_path: str, 
    imgsz: Optional[int], 
    sample_fps: float, 
    max_frames: int,
    segment_cap_m: float
):
    """Background worker task for processing long videos asynchronously."""
    try:
        ASYNC_JOBS[job_id]["status"] = "PROCESSING"
        limit = None if max_frames <= 0 else max_frames
        models = get_model_state()
        
        odometry_tracker = VehicleEgoOdometryTracker(target_segment_length_m=segment_cap_m)
        
        audit_report = run_road_safety_pipeline(
            video_path=input_video_path,
            output_dir="DATA/OP",
            model_weights="models/yolo11m.pt",
            sam_weights="models/sam2.1_t.pt",
            imgsz=imgsz,
            sample_fps=sample_fps,
            max_frames=limit,
            segment_cap_m=segment_cap_m,
            detector=models.get("detector"),
            depth_estimator=models.get("depth_estimator"),
            sam_segmenter=models.get("sam_segmenter"),
            odometry_tracker=odometry_tracker,
            predictor=models.get("predictor")
        )
        
        video_basename = os.path.splitext(os.path.basename(input_video_path))[0]
        download_urls = {
            "annotated_video": f"/download/{video_basename}_annotated.mp4",
            "csv_log": f"/download/{video_basename}_frame_by_frame_detections.csv",
            "json_report": f"/download/{video_basename}_risk_report.json"
        }
        
        ASYNC_JOBS[job_id]["status"] = "COMPLETED"
        ASYNC_JOBS[job_id]["download_urls"] = download_urls
        ASYNC_JOBS[job_id]["audit_summary"] = audit_report
        ASYNC_JOBS[job_id]["completed_at"] = time.time()
        
    except Exception as e:
        ASYNC_JOBS[job_id]["status"] = "FAILED"
        ASYNC_JOBS[job_id]["error"] = str(e)


# =============================================================================
# NEW DEDICATED ENDPOINT: Distance Segment-Capped Risk Audit
# =============================================================================
@app.post("/analyze_video_segment_cap", tags=["Inference Pipeline"])
async def analyze_video_segment_cap(
    file: UploadFile = File(
        ..., 
        description="[IP Param: file] Road survey video stream (MP4, AVI, MOV, or MKV format)."
    ),
    sample_fps: float = Query(
        5.0, 
        description="[IP Param: sample_fps] Keyframe extraction rate in frames per second (Default: 5.0 FPS). Controls temporal sampling resolution for 3D visual odometry."
    ),
    segment_cap_m: float = Query(
        500.0, 
        description="[IP Param: segment_cap_m] Spatial distance cap in meters (Default: 500.0m). Model calculates traveled vehicle distance and executes a dedicated risk score prediction every time this metric cap is traversed, locking the score on the video HUD."
    ),
    imgsz: Optional[int] = Query(
        None, 
        description="[IP Param: imgsz] Optical inference resolution (Default: None for dynamic native resolution e.g. 1920/2304; pass 640 or 1280 to accelerate low-tier CPUs)."
    ),
    max_frames: int = Query(
        0, 
        description="[IP Param: max_frames] Maximum sampled frames to evaluate (Default: 0 for full video without early stopping; set e.g. 50 for quick preview)."
    )
):
    """
    ### 🎯 Segment-Capped Road Safety Perception Endpoint
    Analyzes uploaded road footage and evaluates crash risk at **every 500 meters** (or custom user-defined distance cap).
    
    #### 📋 Short Overview of Input (IP) Parameters:
    - **`file`**: Raw dashcam or survey video (`.mp4`, `.avi`, `.mov`, `.mkv`).
    - **`sample_fps`**: Sampling frequency (Default: **5.0 FPS**). Governs optical flow resolution and odometry accuracy.
    - **`segment_cap_m`**: Distance milestone cap in meters (Default: **500.0m**). Every time cumulative vehicle motion covers this distance, a segment risk prediction is calculated, displayed on video HUD, and recorded in the audit log.
    - **`imgsz`**: Spatial resolution for YOLO11 (Default: **Native**). Downscale to 640/1280 for higher FPS on CPUs.
    - **`max_frames`**: Early termination ceiling (Default: **0** = unconstrained full video).
    
    #### 📦 Returns:
    - Annotated video download URL with persistent dual-card HUD + 500m milestone badges.
    - CSV log URL with frame-by-frame distance, speed, and segment progression.
    - Comprehensive JSON audit report detailing all individual 500m segments and risk scores.
    """
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an MP4, AVI, MOV, or MKV video.")
        
    os.makedirs("DATA/IP", exist_ok=True)
    os.makedirs("DATA/OP", exist_ok=True)
    
    temp_filename = f"upload_{int(time.time())}_{file.filename}"
    input_video_path = os.path.join("DATA/IP", temp_filename)
    
    with open(input_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        limit = None if max_frames <= 0 else max_frames
        models = get_model_state()
        odometry_tracker = VehicleEgoOdometryTracker(target_segment_length_m=segment_cap_m)
        
        audit_report = await run_in_threadpool(
            run_road_safety_pipeline,
            video_path=input_video_path,
            output_dir="DATA/OP",
            model_weights="models/yolo11m.pt",
            sam_weights="models/sam2.1_t.pt",
            imgsz=imgsz,
            sample_fps=sample_fps,
            max_frames=limit,
            segment_cap_m=segment_cap_m,
            detector=models.get("detector"),
            depth_estimator=models.get("depth_estimator"),
            sam_segmenter=models.get("sam_segmenter"),
            odometry_tracker=odometry_tracker,
            predictor=models.get("predictor")
        )
        
        video_basename = os.path.splitext(os.path.basename(input_video_path))[0]
        download_urls = {
            "annotated_video": f"/download/{video_basename}_annotated.mp4",
            "csv_log": f"/download/{video_basename}_frame_by_frame_detections.csv",
            "json_report": f"/download/{video_basename}_risk_report.json"
        }
        
        return JSONResponse(content={
            "message": f"Segment-Capped Audit Completed Successfully ({segment_cap_m:.0f}m intervals)",
            "input_parameters_summary": {
                "sample_fps": sample_fps,
                "segment_cap_m": segment_cap_m,
                "imgsz": imgsz if imgsz else "Native Resolution",
                "max_frames": max_frames if max_frames > 0 else "Full Video"
            },
            "download_urls": download_urls,
            "completed_segments_count": len(audit_report.get("completed_segments_risk_evaluations", [])),
            "audit_summary": audit_report
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {str(e)}")


@app.post("/analyze_video", tags=["Inference Pipeline"])
async def analyze_video(
    file: UploadFile = File(
        ..., 
        description="[IP Param: file] Road survey video file (MP4, AVI, MOV, MKV)."
    ),
    sample_fps: float = Query(
        5.0, 
        description="[IP Param: sample_fps] Keyframe sampling rate (Default: 5.0 FPS)."
    ),
    segment_cap_m: float = Query(
        500.0, 
        description="[IP Param: segment_cap_m] Distance cap in meters for periodic risk prediction (Default: 500.0m)."
    ),
    imgsz: Optional[int] = Query(
        None, 
        description="[IP Param: imgsz] Optical inference resolution (Default: None for Native; 640/1280 for fast CPU mode)."
    ),
    max_frames: int = Query(
        0, 
        description="[IP Param: max_frames] Max keyframes to process (Default: 0 for entire video)."
    )
):
    """
    ### ⚡ Synchronous Video Audit Endpoint
    Processes uploaded road video directly and returns risk analytics + download links.
    
    #### 📋 Short Overview of Input (IP) Parameters:
    - **`file`**: Survey video stream.
    - **`sample_fps`**: Sampling frequency in frames per second (Default: **5.0 FPS**).
    - **`segment_cap_m`**: Distance milestone in meters (Default: **500.0m**) for segment risk evaluation.
    - **`imgsz`**: Optional spatial downsampling for YOLO11 (Default: **Native**).
    - **`max_frames`**: Maximum frames ceiling (Default: **0** = full video).
    """
    return await analyze_video_segment_cap(
        file=file,
        sample_fps=sample_fps,
        segment_cap_m=segment_cap_m,
        imgsz=imgsz,
        max_frames=max_frames
    )


@app.post("/analyze_video_async", tags=["Inference Pipeline"])
async def analyze_video_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(
        ..., 
        description="[IP Param: file] Long road survey video stream (MP4, AVI, MOV, MKV)."
    ),
    sample_fps: float = Query(
        5.0, 
        description="[IP Param: sample_fps] Sampling rate (Default: 5.0 FPS)."
    ),
    segment_cap_m: float = Query(
        500.0, 
        description="[IP Param: segment_cap_m] Distance interval for risk calculations (Default: 500.0m)."
    ),
    imgsz: Optional[int] = Query(
        None, 
        description="[IP Param: imgsz] Inference resolution (Default: Native)."
    ),
    max_frames: int = Query(
        0, 
        description="[IP Param: max_frames] Max frames (Default: 0 for entire video)."
    )
):
    """
    ### ⏳ Asynchronous Background Audit Endpoint
    Submits a long survey video to the background queue, instantly returning a `job_id` within 50ms to prevent cloud gateway timeouts.
    
    #### 📋 Short Overview of Input (IP) Parameters:
    - **`file`**: Video file to upload.
    - **`sample_fps`**: Frame sampling rate (Default: **5.0 FPS**).
    - **`segment_cap_m`**: Segment distance cap (Default: **500.0m**).
    - **`imgsz`**: YOLO11 resolution (Default: **Native**).
    - **`max_frames`**: Frame count limit (Default: **0** = unconstrained).
    """
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an MP4, AVI, MOV, or MKV video.")
        
    os.makedirs("DATA/IP", exist_ok=True)
    os.makedirs("DATA/OP", exist_ok=True)
    
    job_id = str(uuid.uuid4())[:8]
    temp_filename = f"job_{job_id}_{file.filename}"
    input_video_path = os.path.join("DATA/IP", temp_filename)
    
    with open(input_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    ASYNC_JOBS[job_id] = {
        "job_id": job_id,
        "status": "QUEUED",
        "created_at": time.time(),
        "video_filename": file.filename,
        "parameters": {
            "sample_fps": sample_fps,
            "segment_cap_m": segment_cap_m,
            "imgsz": imgsz,
            "max_frames": max_frames
        }
    }
    
    background_tasks.add_task(
        _run_background_pipeline, 
        job_id, 
        input_video_path, 
        imgsz, 
        sample_fps, 
        max_frames,
        segment_cap_m
    )
    
    return {
        "message": "Video audit task submitted successfully to background worker",
        "job_id": job_id,
        "status": "QUEUED",
        "input_parameters_summary": {
            "sample_fps": sample_fps,
            "segment_cap_m": segment_cap_m,
            "imgsz": imgsz if imgsz else "Native Resolution",
            "max_frames": max_frames if max_frames > 0 else "Full Video"
        },
        "check_status_url": f"/jobs/{job_id}"
    }


@app.get("/jobs/{job_id}", tags=["Inference Pipeline"])
async def get_job_status(job_id: str):
    """
    ### Poll Background Audit Task Status
    Retrieves real-time processing progress, completion status, and download links for an asynchronous job.
    - **Input Params (IP)**: `job_id` (string returned by `/analyze_video_async`).
    """
    if job_id not in ASYNC_JOBS:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found.")
    return ASYNC_JOBS[job_id]


@app.get("/download/{file_name}", tags=["File Downloads"])
async def download_file(file_name: str):
    """
    ### Download Processed Perception Artifacts
    Streams generated annotated videos (.mp4), telemetry CSV logs (.csv), or JSON risk reports.
    - **Input Params (IP)**: `file_name` (e.g. `Sample_1_annotated.mp4`).
    """
    file_path = os.path.join("DATA/OP", file_name)
    if not os.path.exists(file_path):
        alt_path = os.path.join("DATA/OP_V4", file_name)
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            raise HTTPException(status_code=404, detail=f"File '{file_name}' not found in DATA/OP.")
        
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
