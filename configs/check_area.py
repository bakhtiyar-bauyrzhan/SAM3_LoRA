import json
from pathlib import Path

def count_small_objects(json_path, threshold=200):
    path = Path(json_path)
    if not path.exists():
        print(f"[-] File not found: {json_path}")
        return

    print(f"\nFile analysis: {path.parent.name}/{path.name}")
    print(f"Searching for object with area <= {threshold} pixels...")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    annotations = data.get('annotations', [])
    total_anns = len(annotations)
    
    if total_anns == 0:
        print("The file has 0 annotations.")
        return

    small_objects_count = 0
    zero_area_count = 0

    for ann in annotations:
        area = ann.get('area', 0)
        
        if area == 0:
            zero_area_count += 1
        elif area <= threshold:
            small_objects_count += 1

    total_small = small_objects_count + zero_area_count
    percent = (total_small / total_anns) * 100

    print(f"  Total objects: {total_anns}")
    print(f"  Objects with area = 0 (broken): {zero_area_count}")
    print(f"  Objects с 0 < area <= {threshold}: {small_objects_count}")
    print(f"  Total: {total_small} out {total_anns} ({percent:.2f}%)")

count_small_objects("../data/train/_annotations.coco.json", threshold=500)
count_small_objects("../data/valid/_annotations.coco.json", threshold=500)