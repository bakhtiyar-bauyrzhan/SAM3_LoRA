import json
import argparse

def analyze_coco_masks(json_path):
    """
    Parses a COCO JSON file to count masks and find the minimum area.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[-] Error: File not found at {json_path}")
        return
    except json.JSONDecodeError:
        print(f"[-] Error: Invalid JSON format in {json_path}")
        return
        
    annotations = data.get("annotations", [])
    total_masks = len(annotations)
    
    if total_masks == 0:
        print("[!] No annotations found in the file.")
        return

    # Track minimum area and associated annotation IDs
    min_area = float('inf')
    min_area_ids = []

    for ann in annotations:
        area = ann.get("area")
        
        # Skip if area is missing
        if area is None:
            continue
            
        # Update minimum area and reset ID list if a new minimum is found
        if area < min_area:
            min_area = area
            min_area_ids = [ann["id"]]
        # Append ID if it matches the current minimum area
        elif area == min_area:
            min_area_ids.append(ann["id"])

    # Output results
    print(f"--- COCO Dataset Analysis ---")
    print(f"File: {json_path}")
    print(f"Total masks (annotations): {total_masks}")
    
    if min_area != float('inf'):
        print(f"Minimum area found: {min_area}")
        print(f"Annotation ID(s) with minimum area: {min_area_ids}")
    else:
        print("[-] Could not find valid 'area' fields in annotations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze COCO JSON annotations for mask counts and min area.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to _annotations.coco.json")
    
    args = parser.parse_args()
    analyze_coco_masks(args.input)