import json
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

def process_split(split_name, min_area=200):
    raw_dir = Path(f"data/{split_name}_raw")
    out_dir = Path(f"data/{split_name}")
    
    ann_path = raw_dir / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"[-] Skip: {ann_path} not found.")
        return

    if out_dir.exists():
        print(f"[!] Delete old folder {out_dir}...")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Processing {split_name} ---")
    
    with open(ann_path, 'r') as f:
        coco = json.load(f)

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

    for img_info in tqdm(coco.get('images', []), desc=f"Slicing {split_name}"):
        img_path = raw_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        if image is None: continue
        
        h, w = image.shape[:2]
        tile_w, tile_h = w // 2, h // 2
        
        for row in range(2):
            for col in range(2):
                x_off = col * tile_w
                y_off = row * tile_h
                
                tile_img = image[y_off:y_off+tile_h, x_off:x_off+tile_w]
                tile_filename = f"{Path(img_info['file_name']).stem}_tile_{row}_{col}.jpg"
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
                    
                    area = int(cv2.countNonZero(tile_mask))
                    
                    if area <= min_area:
                        continue
                        
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
                        "id": ann_id_counter,
                        "image_id": new_img_id,
                        "category_id": ann['category_id'],
                        "bbox": [float(bx), float(by), float(bw), float(bh)],
                        "area": area,
                        "segmentation": new_segments,
                        "iscrowd": 0
                    })
                    ann_id_counter += 1

    with open(out_dir / "_annotations.coco.json", 'w') as f:
        json.dump(new_coco, f)
        
    print(f"[Success] Clear objects count: {len(new_coco['annotations'])}")

if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        process_split(split, min_area=200)