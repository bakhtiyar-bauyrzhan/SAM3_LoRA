import json
import cv2
import numpy as np
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def process_split(split_name, min_area=200, do_tiling=True):
    raw_dir = Path(f"data/{split_name}_raw")
    out_dir = Path(f"data/{split_name}")
    
    ann_path = raw_dir / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"[-] Skip: {ann_path} not found.")
        return

    # ЖЕСТКАЯ ОЧИСТКА: Удаляем старую папку с результатами
    if out_dir.exists():
        print(f"[!] Delete old folder {out_dir}...")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Processing {split_name} (Slicing: {'ON' if do_tiling else 'OFF'}) ---")
    
    with open(ann_path, 'r') as f:
        coco = json.load(f)
    
    original_ann_count = len(coco.get('annotations', []))

    new_coco = {
        "info": coco.get("info", {}),
        "categories": coco.get("categories", []),
        "images": [],
        "annotations": []
    }

    img_id_counter = 1
    ann_id_counter = 1

    img_to_anns = {}
    for ann in coco.get('annotations', []):
        iid = ann['image_id']
        if iid not in img_to_anns: 
            img_to_anns[iid] = []
        img_to_anns[iid].append(ann)

    for img_info in tqdm(coco.get('images', []), desc=f"Processing {split_name}"):
        img_path = raw_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        if image is None: continue
        
        h, w = image.shape[:2]
        
        # if slicing off, then grid 1x1, else 2x2 (one 4k img to four fullhd imgs)
        if do_tiling:
            rows, cols = 2, 2
            tile_w, tile_h = w // 2, h // 2
        else:
            rows, cols = 1, 1
            tile_w, tile_h = w, h
        
        for row in range(rows):
            for col in range(cols):
                x_off = col * tile_w
                y_off = row * tile_h
                
                tile_img = image[y_off:y_off+tile_h, x_off:x_off+tile_w]
                
                # file name depends if we sliced or not
                if do_tiling:
                    tile_filename = f"{Path(img_info['file_name']).stem}_tile_{row}_{col}.jpg"
                else:
                    tile_filename = img_info['file_name']
                    
                cv2.imwrite(str(out_dir / tile_filename), tile_img)
                
                new_img_id = img_id_counter
                new_coco["images"].append({
                    "id": new_img_id,
                    "file_name": tile_filename,
                    "width": tile_w,
                    "height": tile_h
                })
                img_id_counter += 1
                
                anns = img_to_anns.get(img_info['id'], [])
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
                    
                    # small islands filtration
                    contours, _ = cv2.findContours(tile_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    new_segments = []
                    valid_contours = []
                    total_valid_area = 0
                    
                    for c in contours:
                        island_area = cv2.contourArea(c)
                        # take only regions that are > min_area
                        if island_area > min_area:
                            valid_contours.append(c)
                            new_segments.append(c.flatten().tolist())
                            total_valid_area += island_area
                            
                    if not valid_contours or total_valid_area <= min_area:
                        continue
                        
                    all_points = np.concatenate(valid_contours)
                    bx, by, bw, bh = cv2.boundingRect(all_points)
                    
                    new_coco["annotations"].append({
                        "id": ann_id_counter,
                        "image_id": new_img_id,
                        "category_id": ann['category_id'],
                        "bbox": [float(bx), float(by), float(bw), float(bh)],
                        "area": int(total_valid_area),
                        "segmentation": new_segments,
                        "iscrowd": 0
                    })
                    ann_id_counter += 1

    with open(out_dir / "_annotations.coco.json", 'w') as f:
        json.dump(new_coco, f)
        
    kept_count = len(new_coco['annotations'])
    removed_count = original_ann_count - kept_count
    print(f"[Success] Saved clean regions: {kept_count} out of {original_ann_count} (Removed: {removed_count})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleaning and slicing COCO dataset")
    parser.add_argument("--no-tiling", action="store_true", help="Turn off slicing, keep original resolution")
    parser.add_argument("--min-area", type=int, default=200, help="min area filtration threshold, default")
    args = parser.parse_args()

    # If --no-tiling included, then do_tiling == False
    do_tiling = not args.no_tiling

    for split in ["train", "valid", "test"]:
        process_split(split, min_area=args.min_area, do_tiling=do_tiling)