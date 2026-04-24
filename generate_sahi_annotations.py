import os
import argparse
import json
import torch
import cv2
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image as PILImage

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import Datapoint, Image as SAMImage, FindQueryLoaded, InferenceMetadata
from sam3.train.data.collator import collate_fn_api
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

# ----------------------------------------------------------------------
# ADVANCED MASK NMS & STITCHING ALGORITHMS
# ----------------------------------------------------------------------

def merge_overlapping_masks_tensor(binary_masks, scores, iou_threshold=0.3, iom_threshold=0.6):
    """LOCAL NMS: Merges over-segmented sub-parts inside a single patch."""
    if len(binary_masks) == 0:
        return binary_masks, scores
        
    sorted_indices = torch.argsort(scores, descending=True)
    binary_masks = binary_masks[sorted_indices]
    scores = scores[sorted_indices]
    
    merged_masks = []
    merged_scores = []
    used = torch.zeros(len(binary_masks), dtype=torch.bool)
    areas = binary_masks.sum(dim=(1,2))
    
    for i in range(len(binary_masks)):
        if used[i]: continue
        
        current_mask = binary_masks[i].clone()
        current_score = scores[i]
        current_area = areas[i]
        used[i] = True
        
        for j in range(i + 1, len(binary_masks)):
            if used[j]: continue
            
            intersection = (current_mask & binary_masks[j]).sum()
            if intersection == 0: continue
            
            union = current_area + areas[j] - intersection
            iou = intersection / union if union > 0 else 0
            iom = intersection / min(current_area, areas[j])
            
            if iou > iou_threshold or iom > iom_threshold:
                current_mask = current_mask | binary_masks[j]
                current_score = torch.max(current_score, scores[j])
                current_area = current_mask.sum()
                used[j] = True
                
        merged_masks.append(current_mask)
        merged_scores.append(current_score)
        
    return torch.stack(merged_masks), torch.stack(merged_scores)

def stitch_global_contours(contours, scores, min_area, iom_threshold=0.5, iou_threshold=0.3):
    """
    GLOBAL NMS & STITCHER:
    1. Absorbs nested masks.
    2. Stitches cut leaves at seams using "Crop Box IoM" logic.
    """
    valid_objects = []
    for c, s in zip(contours, scores):
        area = cv2.contourArea(c)
        if area > 10: # Drop absolute pixel noise
            valid_objects.append({
                'contour': c,
                'score': s,
                'bbox': cv2.boundingRect(c),
                'area': area
            })
            
    # Sort by highest confidence
    valid_objects.sort(key=lambda x: x['score'], reverse=True)
    used = [False] * len(valid_objects)
    keep = []
    
    for i in range(len(valid_objects)):
        if used[i]: continue
        used[i] = True
        
        obj_i = valid_objects[i]
        
        # RECURSIVE MERGE: If it expands, we must check it against all others again
        merged_in_this_pass = True
        while merged_in_this_pass:
            merged_in_this_pass = False
            x1, y1, w1, h1 = obj_i['bbox']
            
            for j in range(i + 1, len(valid_objects)):
                if used[j]: continue
                
                obj_j = valid_objects[j]
                x2, y2, w2, h2 = obj_j['bbox']
                
                # Fast bbox overlap check
                if not (x1 < x2+w2 and x1+w1 > x2 and y1 < y2+h2 and y1+h1 > y2):
                    continue
                    
                # Full pixel-level overlap calculation
                min_x, min_y = min(x1, x2), min(y1, y2)
                max_x, max_y = max(x1+w1, x2+w2), max(y1+h1, y2+h2)
                w_canvas, h_canvas = max_x - min_x, max_y - min_y
                
                mask_i = np.zeros((h_canvas, w_canvas), dtype=np.uint8)
                mask_j = np.zeros((h_canvas, w_canvas), dtype=np.uint8)
                
                cv2.fillPoly(mask_i, [obj_i['contour'] - [min_x, min_y]], 1)
                cv2.fillPoly(mask_j, [obj_j['contour'] - [min_x, min_y]], 1)
                
                area_i = mask_i.sum()
                area_j = mask_j.sum()
                intersection = np.logical_and(mask_i, mask_j).sum()
                
                if intersection > 0:
                    # 1. Global overlaps (Catches nested masks / Матрешки)
                    iom = intersection / min(area_i, area_j) if min(area_i, area_j) > 0 else 0
                    iou = intersection / (area_i + area_j - intersection) if (area_i + area_j - intersection) > 0 else 0
                    
                    # 2. Local Crop Box overlap (Catches tiny cut pieces on seams)
                    ix1 = max(x1, x2) - min_x
                    iy1 = max(y1, y2) - min_y
                    ix2 = min(x1+w1, x2+w2) - min_x
                    iy2 = min(y1+h1, y2+h2) - min_y
                    
                    crop_i = mask_i[iy1:iy2, ix1:ix2]
                    crop_j = mask_j[iy1:iy2, ix1:ix2]
                    
                    crop_area_i = crop_i.sum()
                    crop_area_j = crop_j.sum()
                    crop_intersection = np.logical_and(crop_i, crop_j).sum()
                    
                    crop_iom = crop_intersection / min(crop_area_i, crop_area_j) if min(crop_area_i, crop_area_j) > 0 else 0
                    
                    # Merge Logic
                    if iom > iom_threshold or iou > iou_threshold or (crop_iom > 0.5 and crop_intersection > 50):
                        # Combine masks
                        merged_mask = np.logical_or(mask_i, mask_j).astype(np.uint8)
                        new_contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if new_contours:
                            biggest = max(new_contours, key=cv2.contourArea)
                            obj_i['contour'] = biggest + [min_x, min_y]
                            obj_i['area'] = cv2.contourArea(obj_i['contour'])
                            obj_i['bbox'] = cv2.boundingRect(obj_i['contour'])
                            obj_i['score'] = max(obj_i['score'], obj_j['score'])
                            
                            used[j] = True
                            merged_in_this_pass = True
                            break # Restart loop since obj_i expanded and might touch new neighbors
                            
        # Final Area Check
        if obj_i['area'] >= min_area:
            keep.append(obj_i)
            
    return keep

# ----------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------

def get_slice_bboxes(image_w, image_h, slice_size=800, overlap_ratio=0.2):
    step = int(slice_size * (1 - overlap_ratio))
    if step <= 0: raise ValueError("Overlap ratio must be < 1.0")
    bboxes = []
    y_min = 0
    while y_min < image_h:
        y_max = min(y_min + slice_size, image_h)
        x_min = 0
        while x_min < image_w:
            x_max = min(x_min + slice_size, image_w)
            bboxes.append([x_min, y_min, x_max, y_max])
            if x_max >= image_w: break
            x_min += step
        if y_max >= image_h: break
        y_min += step
    return bboxes

def create_coco_skeleton():
    return {
        "info": {"description": "Auto-generated by SAM3 SAHI inference"},
        "categories": [{"id": 1, "name": "leaf"}],
        "images": [],
        "annotations": []
    }

def run_sahi_inference_and_save_coco(
    model, image_paths, text_prompt, output_json_path, device, 
    resolution=1008, prob_threshold=0.3, min_area=500, 
    slice_size=800, overlap_ratio=0.2
):
    transform = ComposeAPI([
        RandomResizeAPI(sizes=[resolution], max_size=resolution, square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    coco_data = create_coco_skeleton()
    img_id_counter = 1
    ann_id_counter = 1
    
    model.eval()
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc=f"SAHI Inference ({Path(output_json_path).name})"):
            pil_img = PILImage.open(img_path).convert("RGB")
            orig_w, orig_h = pil_img.size
            
            coco_data["images"].append({
                "id": img_id_counter,
                "file_name": Path(img_path).name,
                "width": orig_w,
                "height": orig_h
            })
            
            slice_bboxes = get_slice_bboxes(orig_w, orig_h, slice_size=slice_size, overlap_ratio=overlap_ratio)
            global_contours = []
            global_scores = []
            
            for patch_box in slice_bboxes:
                px1, py1, px2, py2 = patch_box
                patch_w = px2 - px1
                patch_h = py2 - py1
                
                patch_img = pil_img.crop((px1, py1, px2, py2))
                sam_image = SAMImage(data=patch_img, objects=[], size=[patch_h, patch_w])
                
                query = FindQueryLoaded(
                    query_text=text_prompt, image_id=0, object_ids_output=[], is_exhaustive=True, query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=0, original_image_id=0, original_category_id=1,
                        original_size=[patch_w, patch_h], object_id=0, frame_index=0,
                    )
                )
                
                dp = transform(Datapoint(images=[sam_image], find_queries=[query]))
                batch = collate_fn_api([dp], dict_key="input", with_seg_masks=False)["input"]
                batch = copy_data_to_device(batch, device)
                
                outputs = model(batch)
                last_output = outputs[-1]

                pred_logits = last_output['pred_logits']
                pred_masks = last_output.get('pred_masks', None)

                if pred_masks is not None:
                    out_probs = pred_logits.sigmoid()
                    scores = out_probs[0, :, :].max(dim=-1)[0]
                    keep_idx = scores > prob_threshold

                    if keep_idx.any():
                        valid_masks = pred_masks[0, keep_idx].sigmoid() > 0.5
                        valid_scores = scores[keep_idx]
                        
                        merged_masks, merged_scores = merge_overlapping_masks_tensor(
                            valid_masks, valid_scores, iou_threshold=0.3, iom_threshold=0.6
                        )
                        
                        for mask_tensor, score in zip(merged_masks, merged_scores):
                            mask_resized = torch.nn.functional.interpolate(
                                mask_tensor.unsqueeze(0).unsqueeze(0).float(), 
                                size=(patch_h, patch_w), 
                                mode='nearest'
                            ).squeeze() > 0.5
                            
                            mask_np = mask_resized.cpu().numpy().astype(np.uint8)
                            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            for c in contours:
                                if cv2.contourArea(c) > 10:
                                    c_shifted = c + np.array([px1, py1])
                                    global_contours.append(c_shifted)
                                    global_scores.append(score.item())
                
                del batch, outputs, pred_logits, pred_masks
                torch.cuda.empty_cache()

            # --- GLOBAL STITCHING & FILTERING ---
            if global_contours:
                stitched_objects = stitch_global_contours(global_contours, global_scores, min_area=min_area)
                
                for obj in stitched_objects:
                    x, y, w, h = obj['bbox']
                    
                    coco_data["annotations"].append({
                        "id": ann_id_counter,
                        "image_id": img_id_counter,
                        "category_id": 1,
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "area": int(obj['area']),
                        "segmentation": [obj['contour'].flatten().tolist()],
                        "iscrowd": 0
                    })
                    ann_id_counter += 1
                    
            img_id_counter += 1

    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco_data, f)
    print(f"[OK] Saved {len(coco_data['annotations'])} objects (Cleaned & Stitched).")

def main():
    parser = argparse.ArgumentParser(description="SAM3 SAHI Inference (Slicing Aided Hyper Inference)")
    parser.add_argument("--config", type=str, required=True, help="Path to custom_config.yaml")
    parser.add_argument("--image-dir", type=str, required=True, help="Folder with high-res test images")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt (e.g., 'leaf')")
    parser.add_argument("--weights", type=str, default=None, help="Path to .pt LoRA weights (overrides config)")
    parser.add_argument("--out-dir", type=str, default="results_json", help="Output directory for JSON files")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold")
    
    parser.add_argument("--slice-size", type=int, default=800, help="Size of the sliding window")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio between slices")
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_min_area = cfg.get('inference', {}).get('min_area', 500)
    
    image_paths = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) 
                   if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png'}]

    model = build_sam3_image_model(
        device=device.type, compile=False, load_from_HF=True, 
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz", eval_mode=True
    )
    model.to(device)

    base_json = os.path.join(args.out_dir, "sahi_base_annotations.coco.json")
    print(f"\n🚀 Starting SAHI Inference with BASE model...")
    run_sahi_inference_and_save_coco(
        model, image_paths, args.prompt, base_json, device, 
        prob_threshold=args.threshold, min_area=config_min_area,
        slice_size=args.slice_size, overlap_ratio=args.overlap
    )

    print("\n🔗 Applying LoRA layers...")
    l_cfg = cfg['lora']
    lora_config = LoRAConfig(
        rank=l_cfg['rank'], alpha=l_cfg['alpha'], dropout=l_cfg.get('dropout', 0.1),
        target_modules=l_cfg['target_modules'], 
        apply_to_vision_encoder=l_cfg.get('apply_to_vision_encoder', False),
        apply_to_text_encoder=l_cfg.get('apply_to_text_encoder', False),
        apply_to_geometry_encoder=l_cfg.get('apply_to_geometry_encoder', False),
        apply_to_detr_encoder=l_cfg.get('apply_to_detr_encoder', False),
        apply_to_detr_decoder=l_cfg.get('apply_to_detr_decoder', True),
        apply_to_mask_decoder=l_cfg.get('apply_to_mask_decoder', True)
    )
    apply_lora_to_model(model, lora_config)

    w_path = args.weights or os.path.join(cfg.get('output', {}).get('output_dir', ''), "best_lora_weights.pt")
    print(f"💾 Loading LoRA weights from: {w_path}")
    load_lora_weights(model, w_path)
    model.to(device)

    lora_json = os.path.join(args.out_dir, "sahi_lora_annotations.coco.json")
    print(f"\n🚀 Starting SAHI Inference with LORA model...")
    run_sahi_inference_and_save_coco(
        model, image_paths, args.prompt, lora_json, device, 
        prob_threshold=args.threshold, min_area=config_min_area,
        slice_size=args.slice_size, overlap_ratio=args.overlap
    )
    
if __name__ == "__main__":
    main()