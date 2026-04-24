import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

def process_split(split_path, output_split_path, context_factor, min_area, category_name):
    
    ann_file = split_path / "_annotations.coco.json"
    if not ann_file.exists():
        return False

    output_split_path.mkdir(parents=True, exist_ok=True)
    out_img_dir = output_split_path / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_file, 'r') as f:
        coco = json.load(f)

    orig_total = len(coco.get("annotations", []))
    orig_small = sum(1 for a in coco.get("annotations", []) if a.get("area", float('inf')) < min_area)

    new_coco = {
        "info": coco.get("info", {"description": f"Cropped Dataset - {split_path.name}"}),
        "categories": [{"id": 1, "name": category_name}],
        "images": [],
        "annotations": []
    }

    img_to_anns = {}
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    out_img_id = 1
    out_ann_id = 1
    dropped_during_crop = 0

    images_list = coco.get("images", [])
    if not images_list:
        return False

    for img_info in tqdm(images_list, desc=f"Processing {split_path.name}"):
        img_file_name = img_info["file_name"]
        
        possible_paths = [
            split_path / img_file_name,
            split_path / "images" / img_file_name,
            split_path / os.path.basename(img_file_name)
        ]
        
        img_path = next((p for p in possible_paths if p.exists()), None)
        if img_path is None:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        h_img, w_img = image.shape[:2]
        annotations = img_to_anns.get(img_info["id"], [])

        for idx, ann in enumerate(annotations):
            if ann.get("area", float('inf')) < min_area:
                continue

            seg = ann.get('segmentation', [])
            if not seg or not isinstance(seg, list):
                continue

            instance_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            for poly in seg:
                if len(poly) >= 6:
                    pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(instance_mask, [pts], 1)

            contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)

            cx, cy = x + w / 2.0, y + h / 2.0
            crop_size = max(w, h) * context_factor
            
            x1 = int(max(0, cx - crop_size / 2))
            y1 = int(max(0, cy - crop_size / 2))
            x2 = int(min(w_img, cx + crop_size / 2))
            y2 = int(min(h_img, cy + crop_size / 2))

            if x2 <= x1 or y2 <= y1:
                continue

            cropped_image = image[y1:y2, x1:x2]
            cropped_mask = instance_mask[y1:y2, x1:x2]

            new_contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not new_contours:
                continue
                
            new_main_contour = max(new_contours, key=cv2.contourArea)
            area = cv2.contourArea(new_main_contour)
            
            if area < min_area:
                dropped_during_crop += 1
                continue

            new_x, new_y, new_w, new_h = cv2.boundingRect(new_main_contour)

            orig_name = Path(img_file_name).stem
            new_file_name = f"{orig_name}_obj{idx}.jpg"
            cv2.imwrite(str(out_img_dir / new_file_name), cropped_image)

            new_coco["images"].append({
                "id": out_img_id,
                "file_name": f"images/{new_file_name}",
                "width": int(x2 - x1),
                "height": int(y2 - y1)
            })

            new_coco["annotations"].append({
                "id": out_ann_id,
                "image_id": out_img_id,
                "category_id": 1,
                "bbox": [float(new_x), float(new_y), float(new_w), float(new_h)],
                "area": float(area),
                "segmentation": [new_main_contour.flatten().tolist()],
                "iscrowd": 0
            })

            out_img_id += 1
            out_ann_id += 1

    with open(output_split_path / "_annotations.coco.json", 'w') as f:
        json.dump(new_coco, f)
        
    print(f"  [+] {split_path.name}: Saved {len(new_coco['images'])} crops (Original: {orig_total}, Skipped <{min_area}px: {orig_small + dropped_during_crop})")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object-Centric Dataset Cropper")
    parser.add_argument("--data", "-d", type=str, default="data", help="Base dataset folder should have train/valid/test folders")
    parser.add_argument("--output", "-o", type=str, default="data_cropped", help="Folder for new dataset")
    parser.add_argument("--context", "-c", type=float, default=1.5, help="Crop multiplier")
    parser.add_argument("--min-area", type=int, default=500, help="Min mask area")
    parser.add_argument("--category", type=str, default="leaf", help="Class name")
    
    args = parser.parse_args()
    
    input_base = Path(args.data)
    output_base = Path(args.output)
    
    if not input_base.exists():
        print(f"[-] Error: Directory '{input_base}' not found.")
        exit(1)

    print(f"=== Start slicing ===")
    print(f"Input folder: {input_base}")
    print(f"Output folder: {output_base}")
    print(f"Context: {args.context}x | Min Area: {args.min_area}px\n")
    
    splits = ["train", "valid", "test"]
    processed_any = False
    
    for split in splits:
        split_path = input_base / split
        if split_path.exists():
            out_split_path = output_base / split
            success = process_split(split_path, out_split_path, args.context, args.min_area, args.category)
            if success:
                processed_any = True
        else:
            print(f"  [-] {split}: folder not found, skip.")
                
    if processed_any:
        print(f"\n[+] Done! New dataset saved to: {output_base}")
    else:
        print("\n[-] No folder can be processed.")