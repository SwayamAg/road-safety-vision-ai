import os
import cv2
import torch
import numpy as np
from ultralytics import SAM

class RoadSafetySAMSegmenter:
    def __init__(self, model_weights='models/sam2.1_t.pt', config_path='sam2_classes_config.json'):
        import json
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config_path = config_path
        self.global_params = {
            'mask_alpha_opacity': 0.42,
            'contour_thickness': 2,
            'min_mask_area_pixels_global': 64
        }
        
        if not os.path.exists(self.config_path):
            alt_config = os.path.join('..', config_path)
            if os.path.exists(alt_config):
                self.config_path = alt_config
                
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    cfg_data = json.load(f)
                    self.global_params.update(cfg_data.get('global_sam2_parameters', cfg_data.get('sam2_global_settings', {})))
            except Exception as e:
                print(f'Warning: Could not parse SAM2 config params ({e}).')

        target_weights = model_weights
        if not os.path.exists(target_weights):
            alt_weights = os.path.join('models', os.path.basename(model_weights))
            if os.path.exists(alt_weights):
                target_weights = alt_weights
            elif os.path.exists('models/sam2.1_t.pt'):
                target_weights = 'models/sam2.1_t.pt'
            elif os.path.exists('models/sam2_t.pt'):
                target_weights = 'models/sam2_t.pt'
            elif os.path.exists('sam2.1_t.pt'):
                target_weights = 'sam2.1_t.pt'
            elif os.path.exists('mobile_sam.pt'):
                target_weights = 'mobile_sam.pt'

        print('Loading SAM 2 Keyframe Segmenter [' + str(target_weights) + '] on device: ' + self.device.upper())
        try:
            self.model = SAM(target_weights)
            self.is_loaded = True
        except Exception as e:
            print('Warning: Could not initialize SAM model (' + str(e) + '). Fallback to bounding box overlay.')
            self.is_loaded = False

    def segment_keyframe(self, frame: np.ndarray, detections: dict) -> dict:
        all_boxes = detections.get('boxes', [])
        boxes = [b for b in all_boxes if b.get('sam2_enabled', True)]
        
        if not self.is_loaded or len(boxes) == 0:
            return {
                'masks': [],
                'mask_count': 0,
                'annotated_frame': frame.copy()
            }

        h, w, _ = frame.shape
        prompt_boxes = [det['bbox'] for det in boxes]
        
        try:
            results = self.model(frame, bboxes=prompt_boxes, verbose=False)[0]
            raw_masks = results.masks.data if results.masks is not None else None
        except Exception as err:
            print('SAM prediction warning: ' + str(err))
            raw_masks = None

        sam_masks = []
        annotated = frame.copy()
        mask_overlay = np.zeros_like(frame, dtype=np.uint8)

        color_map = {
            'pedestrian': (0, 165, 255),
            'vehicle': (0, 235, 120),
            'animal': (0, 0, 255),
            'commercial': (0, 215, 255)
        }

        min_global_area = self.global_params.get('min_mask_area_pixels_global', 64)
        contour_thick = self.global_params.get('contour_thickness', 2)
        alpha = self.global_params.get('mask_alpha_opacity', 0.42)

        if raw_masks is not None:
            for idx, mask_tensor in enumerate(raw_masks):
                det_meta = boxes[idx] if idx < len(boxes) else {'category': 'vehicle', 'label': 'object'}
                category = det_meta.get('category', 'vehicle')
                min_area = det_meta.get('min_area_pixels', min_global_area)
                configured_color = det_meta.get('color_bgr')
                color = tuple(configured_color) if configured_color else color_map.get(category, (255, 255, 0))
                
                mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                if mask_np.shape != (h, w):
                    mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

                mask_bool = mask_np > 0
                pixel_area = int(np.sum(mask_bool))
                
                # Filter out tiny spurious masks based on JSON min_area_pixels
                if pixel_area < min_area:
                    continue
                    
                mask_overlay[mask_bool] = color

                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated, contours, -1, color, contour_thick, lineType=cv2.LINE_AA)

                sam_masks.append({
                    'category': category,
                    'label': det_meta.get('label', 'object'),
                    'description': det_meta.get('description', ''),
                    'associated_param': det_meta.get('associated_safety_param', ''),
                    'pixel_area': pixel_area
                })

            has_masks = np.any(mask_overlay > 0, axis=2)
            if np.any(has_masks):
                annotated[has_masks] = cv2.addWeighted(annotated, 1.0 - alpha, mask_overlay, alpha, 0)[has_masks]

            badge_text = 'SAM 2 KEYFRAME REFINE (' + str(len(sam_masks)) + ' MASKS)'
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (w - tw - 25, 15), (w - 10, 15 + th + 14), (40, 10, 80), cv2.FILLED)
            cv2.putText(annotated, badge_text, (w - tw - 18, 15 + th + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 180, 255), 1, cv2.LINE_AA)

        return {
            'masks': sam_masks,
            'mask_count': len(sam_masks),
            'annotated_frame': annotated
        }

if __name__ == '__main__':
    segmenter = RoadSafetySAMSegmenter()
    test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_dets = {
        'boxes': [
            {'category': 'pedestrian', 'label': 'person', 'bbox': [100, 100, 250, 350]},
            {'category': 'vehicle', 'label': 'car', 'bbox': [300, 200, 500, 450]}
        ]
    }
    out = segmenter.segment_keyframe(test_frame, dummy_dets)
    print('SAM 2 Segmenter self-test completed: ' + str(out['mask_count']) + ' masks generated.')
