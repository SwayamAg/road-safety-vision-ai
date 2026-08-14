import os
import sys
import time
import requests
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://127.0.0.1:8085"

def test_api_endpoints():
    print("=" * 75)
    print("🧪 AUTOMATED FASTAPI ENDPOINT SUITE TESTING")
    print("=" * 75)
    
    # 1. Test Root Endpoint
    print("\n[Test 1/4] Testing GET / (Root Welcome Endpoint)...")
    res = requests.get(f"{BASE_URL}/")
    assert res.status_code == 200, f"Root endpoint failed: {res.status_code}"
    print(f"      ✔ Response (200 OK): {res.json()['title']} | Status: {res.json()['status']}")
    
    # 2. Test Model Telemetry Endpoint
    print("\n[Test 2/4] Testing GET /models/status (Endpoint 1: Model Health & Telemetry)...")
    res = requests.get(f"{BASE_URL}/models/status")
    assert res.status_code == 200, f"Model status endpoint failed: {res.status_code}"
    data = res.json()
    print(f"      ✔ Service Health       : {data['status']}")
    print(f"      ✔ GPU Acceleration     : {data['cuda_acceleration']['device']}")
    print(f"      ✔ YOLO11 Detector      : {data['models']['object_detector']['weights']} ({data['models']['object_detector']['status']})")
    print(f"      ✔ MiDaS Depth Estimator: {data['models']['monocular_depth']['variant']} ({data['models']['monocular_depth']['status']})")
    print(f"      ✔ XGBoost Classifier   : {data['models']['risk_classifier']['artifact']} ({data['models']['risk_classifier']['status']})")
    
    # 3. Test Video Perception & Inference Endpoint (POST /analyze_video)
    sample_video = "DATA/IP/1.mp4"
    if not os.path.exists(sample_video):
        print(f"❌ Sample video '{sample_video}' not found for testing.")
        return
        
    print(f"\n[Test 3/4] Testing POST /analyze_video (Endpoint 2: Uploading & Processing '{sample_video}')...")
    t0 = time.time()
    with open(sample_video, "rb") as f:
        files = {"file": (os.path.basename(sample_video), f, "video/mp4")}
        res = requests.post(f"{BASE_URL}/analyze_video?sample_fps=5.0&imgsz=1280", files=files)
        
    t_el = time.time() - t0
    assert res.status_code == 200, f"Analyze video endpoint failed: {res.status_code} - {res.text}"
    audit_res = res.json()
    print(f"      ✔ Processing Completed in {t_el:.2f}s!")
    print(f"      ✔ Predicted Risk Tier: {audit_res['audit_summary']['risk_prediction']['predicted_risk_tier']}")
    print(f"      ✔ Crash Risk Score   : {audit_res['audit_summary']['risk_prediction']['risk_score']} / 10.0")
    print(f"      ✔ Download URLs      : {audit_res['download_urls']}")
    
    # 4. Test File Streaming Download Endpoint (GET /download/{file_name})
    json_url = audit_res['download_urls']['json_report']
    file_name = json_url.replace("/download/", "")
    print(f"\n[Test 4/4] Testing GET /download/{file_name} (Download Streaming Endpoint)...")
    res = requests.get(f"{BASE_URL}{json_url}")
    assert res.status_code == 200, f"Download endpoint failed: {res.status_code}"
    print(f"      ✔ Successfully retrieved download payload ({len(res.content)} bytes)")
    
    print("\n" + "=" * 75)
    print("🎉 ALL FASTAPI ENDPOINTS PASSED VERIFICATION SUITE (100% HEALTHY)!")
    print("=" * 75)

if __name__ == "__main__":
    test_api_endpoints()
