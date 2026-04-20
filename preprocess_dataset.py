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
    
    # Robust JSON search: find any .json with 'annotation', 'coco', or 'cleaned' in its name
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

            # Alert if image is physically missing (e.g. case sensitivity in extensions .JPG vs .jpg)
            if not source_path.exists():
                print(f"    [!] Missing image: {img['file_name']} (Check extension case)")
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

    # 2. Dataset Split
    print(f"\n--- Step 2: Splitting Data (Total images: {len(all_images)}) ---")
    random.seed(42) # Fix seed for reproducible splits
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
        "categories": coco_categories
    }

    return splits, img_id_to_anns, base_coco


def process_split(split_name, images_list, img_id_to_anns, base_coco, min_area=200, do_tiling=True):
    out_dir = Path(f"data/{split_name}")

    if out_dir.exists():
        print(f"[!] Deleting old folder {out_dir}...")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Step 3: Processing {split_name} (Slicing: {'ON' if do_tiling else 'OFF'}) ---")

    new_coco = {
        "info": base_coco.get("info", {}),
        "categories": [{"id": 1, "name": "leaf"}],
        "images": [],
        "annotations": []
    }

    out_img_id_counter = 1
    out_ann_id_counter = 1
    original_ann_count = 0

    for img_info in tqdm(images_list, desc=f"Rendering {split_name}"):
        source_path = img_info["source_path"]
        batch_name = img_info["batch_name"]
        
        image = cv2.imread(source_path)
        if image is None:
            print(f"\n[-] Failed to read: {source_path}")
            continue

        h, w = image.shape[:2]

        if do_tiling:
            rows, cols = 2, 2
            tile_w, tile_h = w // 2, h // 2
        else:
            rows, cols = 1, 1
            tile_w, tile_h = w, h

        unified_img_id = img_info["id"]
        anns = img_id_to_anns.get(unified_img_id, [])
        original_ann_count += len(anns)
        
        orig_name = Path(img_info['file_name']).stem

        for row in range(rows):
            for col in range(cols):
                x_off = col * tile_w
                y_off = row * tile_h

                tile_img = image[y_off:y_off+tile_h, x_off:x_off+tile_w]

                if do_tiling:
                    tile_filename = f"{batch_name}_{orig_name}_tile_{row}_{col}.jpg"
                else:
                    tile_filename = f"{batch_name}_{orig_name}.jpg"

                cv2.imwrite(str(out_dir / tile_filename), tile_img)

                new_img_id = out_img_id_counter
                new_coco["images"].append({
                    "id": new_img_id,
                    "file_name": tile_filename,
                    "width": tile_w,
                    "height": tile_h
                })
                out_img_id_counter += 1

                for ann in anns:
                    seg = ann.get('segmentation', [])
                    if not seg or not isinstance(seg, list):
                        continue

                    tile_mask = np.zeros((tile_h, tile_w), dtype=np.uint8)
                    valid_polys = []

                    for poly in seg:
                        if len(poly) >= 6:
                            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                            pts[:, 0, 0] -= x_off
                            pts[:, 0, 1] -= y_off
                            valid_polys.append(pts)

                    if not valid_polys:
                        continue

                    cv2.fillPoly(tile_mask, valid_polys, 1)

                    contours, _ = cv2.findContours(tile_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    new_segments = []
                    valid_contours = []
                    total_valid_area = 0

                    for c in contours:
                        island_area = cv2.contourArea(c)
                        if island_area > min_area:
                            valid_contours.append(c)
                            new_segments.append(c.flatten().tolist())
                            total_valid_area += island_area

                    if not valid_contours or total_valid_area <= min_area:
                        continue

                    all_points = np.concatenate(valid_contours)
                    bx, by, bw, bh = cv2.boundingRect(all_points)

                    new_coco["annotations"].append({
                        "id": out_ann_id_counter,
                        "image_id": new_img_id,
                        "category_id": 1,
                        "bbox": [float(bx), float(by), float(bw), float(bh)],
                        "area": int(total_valid_area),
                        "segmentation": new_segments,
                        "iscrowd": 0
                    })
                    out_ann_id_counter += 1

    with open(out_dir / "_annotations.coco.json", 'w') as f:
        json.dump(new_coco, f)

    kept_count = len(new_coco['annotations'])
    removed_count = original_ann_count - kept_count
    print(f"[Success] {split_name}: Saved {kept_count} out of {original_ann_count} annotations (Removed small islands: {removed_count})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate, split and clean COCO dataset")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw directory containing batches")
    parser.add_argument("--no-tiling", action="store_true", help="Disable 2x2 image slicing")
    parser.add_argument("--min-area", type=int, default=200, help="Minimum area threshold for objects")
    args = parser.parse_args()

    do_tiling = not args.no_tiling

    splits, img_id_to_anns, base_coco = merge_and_split_data(args.raw_dir)

    if splits is not None:
        for split_name in ["train", "valid", "test"]:
            process_split(
                split_name=split_name,
                images_list=splits[split_name],
                img_id_to_anns=img_id_to_anns,
                base_coco=base_coco,
                min_area=args.min_area,
                do_tiling=do_tiling
            )