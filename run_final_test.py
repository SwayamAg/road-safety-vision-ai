import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = r'c:\SWAYAMs\PROJ\Road_Safety_params\FD_v2'
OUTPUT_DIR  = r'c:\SWAYAMs\PROJ\Road_Safety_params\FD_v2\DATA\Final_test'
INPUT_VIDEO = os.path.join(SCRIPT_DIR, 'DATA', 'IP', 'Sample_1.mp4')

print('=' * 70)
print('  FD_v2  -  Final_test  runner')
print(f'  Input  : {INPUT_VIDEO}')
print(f'  Output : {OUTPUT_DIR}')
print('=' * 70)

if not os.path.exists(INPUT_VIDEO):
    print(f'[ERROR] Input video not found: {INPUT_VIDEO}')
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from main_pipeline import run_road_safety_pipeline

audit = run_road_safety_pipeline(
    video_path=INPUT_VIDEO,
    output_dir=OUTPUT_DIR,
    model_weights=os.path.join(SCRIPT_DIR, 'models', 'yolo11m.pt'),
    sam_weights=os.path.join(SCRIPT_DIR, 'models', 'sam2.1_t.pt'),
    imgsz=None,
    sample_fps=5.0,
    keyframe_interval=15,
    max_frames=None,
)

print('\n[DONE] Output files written to:', OUTPUT_DIR)
print('  Annotated video :', audit['output_files']['annotated_video'])
print('  CSV telemetry   :', audit['output_files']['csv_log'])
print('  JSON report     :', audit['output_files']['json_report'])
