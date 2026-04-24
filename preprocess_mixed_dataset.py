import json
import cv2
import numpy as np
import shutil
import argparse
import random
from pathlib import Path
from tqdm import tqdm

def merge_and_split_data(raw_dir_path, train_ratio=0.8, val_ratio=0.1):
    raw_dir = Path(raw_dir_path)
    if not raw_dir.exists():
        print(f"[-] Directory {raw_dir} not found. Please create it and add your data.")
        return None, None, None

    all_images = []
    img_id_to_anns = {}
    
    global_img_id = 1
    global_ann_id = 1

    coco_info = {}
    coco_categories = []

    print("--- Step 1: Aggregating Data ---")
    
    json_files = []
    for p in raw_dir.rglob("*.json"):
        name_lower = p.name.lower()
        if "annotation" in name_lower or "coco" in name_lower or "cleaned" in name_lower:
            json_files.append(p)

    if not json_files:
        print(f"[-] No valid JSON annotation files found inside {raw_dir}")
        return None, None, None

    for batch_idx, ann_file in enumerate(json_files):
        batch_dir = ann_file.parent
        img_dir = batch_dir / "images"

        if not img_dir.exists():
            print(f"[!] Skipping {batch_dir}: 'images' subfolder not found.")
            continue

        print(f"\n[+] Reading batch {batch_idx + 1}: {batch_dir.relative_to(raw_dir)}")
        print(f"    Using JSON: {ann_file.name}")
        
        with open(ann_file, 'r') as f:
            coco = json.load(f)

        if not coco_info: coco_info = coco.get("info", {})
        if not coco_categories: coco_categories = coco.get("categories", [])

        img_id_map = {}
        batch_name = f"batch_{batch_idx}" 

        missing_images_count = 0
        found_images_count = 0

        for img in coco.get("images", []):
            source_path = img_dir / img["file_name"]

            if not source_path.exists():
                missing_images_count += 1
                continue

            found_images_count += 1
            old_id = img["id"]
            new_id = global_img_id
            img_id_map[old_id] = new_id

            img["source_path"] = str(source_path)
            img["batch_name"] = batch_name 
            img["id"] = new_id

            all_images.append(img)
            img_id_to_anns[new_id] = []
            global_img_id += 1

        print(f"    -> Valid images added: {found_images_count} (Missing: {missing_images_count})")

        for ann in coco.get("annotations", []):
            old_img_id = ann["image_id"]
            if old_img_id not in img_id_map:
                continue 
            
            new_img_id = img_id_map[old_img_id]
            ann["id"] = global_ann_id
            ann["image_id"] = new_img_id
            img_id_to_anns[new_img_id].append(ann)
            global_ann_id += 1

    if not all_images:
        print("\n[-] No valid images found in the raw directory.")
        return None, None, None

    print(f"\n--- Step 2: Splitting Data (Total original images: {len(all_images)}) ---")
    random.seed(42)
    random.shuffle(all_images)

    total_imgs = len(all_images)
    train_cutoff = int(total_imgs * train_ratio)
    val_cutoff = int(total_imgs * (train_ratio + val_ratio))

    splits = {
        "train": all_images[:train_cutoff],
        "valid": all_images[train_cutoff:val_cutoff],
        "test": all_images[val_cutoff:]
    }
    
    print(f" Train: {len(splits['train'])} images")
    print(f" Valid: {len(splits['valid'])} images")
    print(f" Test:  {len(splits['test'])} images")

    base_coco = {
        "info": coco_info,
        "categories": [{"id": 1, "name": "leaf"}]
    }

    return splits, img_id_to_anns, base_coco


def get_crop_bboxes(image_w, image_h, crop_size=1000, overlap=0.2):
    """
    Generates sliding window bounding boxes of size crop_size x crop_size.
    Ensures strict 1000x1000 crops by pushing the window back if it hits the edge.
    """
    stride = int(crop_size * (1 - overlap))
    if stride <= 0: raise ValueError("Overlap must be < 1.0")

    bboxes = []
    y = 0
    while y < image_h:
        y2 = min(y + crop_size, image_h)
        y1 = max(0, y2 - crop_size) # Force height to be crop_size if possible
        
        x = 0
        while x < image_w:
            x2 = min(x + crop_size, image_w)
            x1 = max(0, x2 - crop_size) # Force width to be crop_size if possible
            
            bboxes.append((x1, y1, x2, y2))
            
            if x2 == image_w: break
            x += stride
        if y2 == image_h: break
        y += stride
        
    return bboxes

def process_split(split_name, images_list, img_id_to_anns, base_coco, min_area=500, crop_size=1000, overlap=0.2):
    out_dir = Path(f"data_mixed/{split_name}")

    if out_dir.exists():
        print(f"[!] Deleting old folder {out_dir}...")
        shutil.rmtree(out_dir)
        
    img_out_dir = out_dir / "images"
    img_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Step 3: Processing {split_name} (Mix: 1x1 Full Res + {crop_size}x{crop_size} Crops) ---")

    new_coco = {
        "info": base_coco.get("info", {}),
        "categories": base_coco.get("categories", []),
        "images": [],
        "annotations": []
    }

    out_img_id_counter = 1
    out_ann_id_counter = 1
    
    total_full_anns = 0
    total_crop_anns = 0

    for img_info in tqdm(images_list, desc=f"Rendering {split_name}"):
        source_path = img_info["source_path"]
        batch_name = img_info["batch_name"]
        orig_name = Path(img_info['file_name']).stem
        
        image = cv2.imread(source_path)
        if image is None:
            continue

        h_img, w_img = image.shape[:2]
        unified_img_id = img_info["id"]
        anns = img_id_to_anns.get(unified_img_id, [])

        # ==========================================
        # PASS 1: FULL RESOLUTION (1x1)
        # ==========================================
        full_filename = f"{batch_name}_{orig_name}_full.jpg"
        cv2.imwrite(str(img_out_dir / full_filename), image)

        new_coco["images"].append({
            "id": out_img_id_counter,
            "file_name": f"images/{full_filename}",
            "width": w_img,
            "height": h_img
        })

        for ann in anns:
            seg = ann.get('segmentation', [])
            if not seg or not isinstance(seg, list): continue

            # For 1x1, we just filter by area and keep original coords
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            valid_polys = [np.array(p, np.int32).reshape((-1, 1, 2)) for p in seg if len(p) >= 6]
            if not valid_polys: continue
            
            cv2.fillPoly(mask, valid_polys, 1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in contours:
                area = cv2.contourArea(c)
                if area > min_area:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    new_coco["annotations"].append({
                        "id": out_ann_id_counter,
                        "image_id": out_img_id_counter,
                        "category_id": 1,
                        "bbox": [float(bx), float(by), float(bw), float(bh)],
                        "area": int(area),
                        "segmentation": [c.flatten().tolist()],
                        "iscrowd": 0
                    })
                    out_ann_id_counter += 1
                    total_full_anns += 1

        parent_img_id = out_img_id_counter
        out_img_id_counter += 1

        # ==========================================
        # PASS 2: CROPS (e.g. 1000x1000)
        # ==========================================
        # Skip cropping if original image is smaller than crop_size (already covered by 1x1)
        if w_img > crop_size or h_img > crop_size:
            crop_boxes = get_crop_bboxes(w_img, h_img, crop_size=crop_size, overlap=overlap)
            
            for row, (x1, y1, x2, y2) in enumerate(crop_boxes):
                crop_w = x2 - x1
                crop_h = y2 - y1
                
                crop_img = image[y1:y2, x1:x2]
                crop_filename = f"{batch_name}_{orig_name}_crop_{row}.jpg"
                cv2.imwrite(str(img_out_dir / crop_filename), crop_img)
                
                new_coco["images"].append({
                    "id": out_img_id_counter,
                    "file_name": f"images/{crop_filename}",
                    "width": crop_w,
                    "height": crop_h
                })
                
                for ann in anns:
                    seg = ann.get('segmentation', [])
                    if not seg or not isinstance(seg, list): continue

                    tile_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
                    valid_polys = []

                    # Shift polygons to crop's local coordinate system
                    for poly in seg:
                        if len(poly) >= 6:
                            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                            pts[:, 0, 0] -= x1
                            pts[:, 0, 1] -= y1
                            valid_polys.append(pts)

                    if not valid_polys: continue

                    cv2.fillPoly(tile_mask, valid_polys, 1)
                    contours, _ = cv2.findContours(tile_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for c in contours:
                        area = cv2.contourArea(c)
                        if area > min_area:
                            bx, by, bw, bh = cv2.boundingRect(c)
                            new_coco["annotations"].append({
                                "id": out_ann_id_counter,
                                "image_id": out_img_id_counter,
                                "category_id": 1,
                                "bbox": [float(bx), float(by), float(bw), float(bh)],
                                "area": int(area),
                                "segmentation": [c.flatten().tolist()],
                                "iscrowd": 0
                            })
                            out_ann_id_counter += 1
                            total_crop_anns += 1
                            
                out_img_id_counter += 1

    with open(out_dir / "_annotations.coco.json", 'w') as f:
        json.dump(new_coco, f)

    print(f"[Success] {split_name}: Generated {out_img_id_counter - 1} images.")
    print(f"          - Full-res annotations: {total_full_anns}")
    print(f"          - Cropped annotations:  {total_crop_anns}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Mixed-Scale COCO dataset (Full Res + Crops)")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw directory")
    parser.add_argument("--crop-size", type=int, default=1000, help="Size of the square crops")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio between crops (0.0 to 1.0)")
    parser.add_argument("--min-area", type=int, default=500, help="Minimum area threshold for objects")
    args = parser.parse_args()

    splits, img_id_to_anns, base_coco = merge_and_split_data(args.raw_dir)

    if splits is not None:
        for split_name in ["train", "valid", "test"]:
            process_split(
                split_name=split_name,
                images_list=splits[split_name],
                img_id_to_anns=img_id_to_anns,
                base_coco=base_coco,
                min_area=args.min_area,
                crop_size=args.crop_size,
                overlap=args.overlap
            )