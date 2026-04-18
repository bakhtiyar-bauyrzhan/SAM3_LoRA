import json
from pathlib import Path

def check_coco_format(json_path):
    path = Path(json_path)
    if not path.exists():
        print(f"File not found: {json_path}")
        return

    print(f"File analysis: {path.name} ...")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    images = {img['id']: img['file_name'] for img in data.get('images', [])}
    
    error_count = 0
    
    for ann in data.get('annotations', []):
        ann_id = ann.get('id', 'Unknown')
        img_id = ann.get('image_id')
        img_name = images.get(img_id, f"ID={img_id}")
        
        has_error = False
        if 'segmentation' not in ann:
            has_error = True
            reason = "No key 'segmentation' (only Bounding Box)"
        elif not ann['segmentation']:
            has_error = True
            reason = "Key exists, bus array is empty (No polygon)"
        elif isinstance(ann['segmentation'], list) and len(ann['segmentation'][0]) < 6:
            has_error = True
            reason = "Polygon has less than 3 points (Cannot build a figure)"
            
        if has_error:
            print(f"[-] Error! Picture: {img_name} | Annotation ID: {ann_id-1} | Reason: {reason}")
            error_count += 1
            
    total_anns = len(data.get('annotations', []))
    print(f"\nSummary: Found {error_count} bad annotations out of {total_anns}.\n")

check_coco_format("../data/train/_annotations.coco.json")
check_coco_format("../data/valid/_annotations.coco.json")