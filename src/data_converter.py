import os
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple

CLASS_MAPPING = {
    "Cap1": "Cap1",
    "Cap2": "Cap2",
    "Cap3": "Cap3",
    "Cap4": "Cap4",
    "MOSFET": "MOSFET",
    "Mov": "MOV",
    "MOV": "MOV",
    "Resistor": "Resistor",
    "Resestor": "Resistor",  # Fix typo in raw dataset
    "Transformer": "Transformer"
}

CANONICAL_CLASSES = [
    "Cap1",
    "Cap2",
    "Cap3",
    "Cap4",
    "MOSFET",
    "MOV",
    "Resistor",
    "Transformer"
]

CLASS_TO_ID = {cls_name: idx for idx, cls_name in enumerate(CANONICAL_CLASSES)}


def convert_and_split_dataset(
    raw_dataset_dir: str,
    output_dir: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42
) -> Dict[str, any]:
    """
    Collects all gold-annotated images from train and validation folders (1299 total),
    creates a reproducible train/val/test split (80/10/10), converts annotations
    to standard YOLO/RT-DETR format, and creates dataset YAML.
    """
    random.seed(seed)
    raw_path = Path(raw_dataset_dir)
    out_path = Path(output_dir)
    
    # Collect all annotated samples from train and validation
    all_samples = []
    source_splits = ["train", "validation"]
    
    for split in source_splits:
        split_img_in = raw_path / split / "img"
        split_ann_in = raw_path / split / "ann"
        
        if not split_ann_in.exists() or not split_img_in.exists():
            continue
            
        for ann_file in split_ann_in.glob("*.json"):
            with open(ann_file, "r", encoding="utf-8") as f:
                ann_data = json.load(f)
                
            objects = ann_data.get("objects", [])
            if not objects:
                continue
                
            img_width = ann_data.get("size", {}).get("width")
            img_height = ann_data.get("size", {}).get("height")
            if not img_width or not img_height:
                continue
                
            base_img_name = ann_file.stem
            if base_img_name.endswith(".jpg"):
                img_name = base_img_name
            else:
                img_name = base_img_name + ".jpg"
                
            src_img_path = split_img_in / img_name
            if not src_img_path.exists():
                found = list(split_img_in.glob(f"{ann_file.stem}*"))
                if found:
                    src_img_path = found[0]
                else:
                    continue
                    
            all_samples.append({
                "img_path": src_img_path,
                "ann_data": ann_data,
                "name": src_img_path.name
            })
            
    print(f"Total valid annotated samples collected: {len(all_samples)}")
    
    # Shuffle and partition
    random.shuffle(all_samples)
    total = len(all_samples)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    
    splits = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train:n_train + n_val],
        "test": all_samples[n_train + n_val:]
    }
    
    # Clean previous yolo directory if exists
    if out_path.exists():
        shutil.rmtree(out_path)
        
    stats = {
        "classes": CANONICAL_CLASSES,
        "class_to_id": CLASS_TO_ID,
        "splits": {}
    }
    
    for split_name, samples in splits.items():
        split_img_out = out_path / "images" / split_name
        split_lbl_out = out_path / "labels" / split_name
        split_img_out.mkdir(parents=True, exist_ok=True)
        split_lbl_out.mkdir(parents=True, exist_ok=True)
        
        split_stats = {"images": 0, "annotations": 0, "class_counts": {cls: 0 for cls in CANONICAL_CLASSES}}
        
        for sample in samples:
            src_img = sample["img_path"]
            ann_data = sample["ann_data"]
            
            dst_img = split_img_out / src_img.name
            shutil.copy2(src_img, dst_img)
            
            img_w = ann_data["size"]["width"]
            img_h = ann_data["size"]["height"]
            
            yolo_lines = []
            for obj in ann_data.get("objects", []):
                raw_cls = obj.get("classTitle", "")
                norm_cls = CLASS_MAPPING.get(raw_cls)
                if not norm_cls or norm_cls not in CLASS_TO_ID:
                    continue
                    
                cls_id = CLASS_TO_ID[norm_cls]
                points = obj.get("points", {}).get("exterior", [])
                if len(points) != 2:
                    continue
                    
                p1, p2 = points[0], points[1]
                x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                
                x1 = max(0.0, min(float(x1), float(img_w)))
                x2 = max(0.0, min(float(x2), float(img_w)))
                y1 = max(0.0, min(float(y1), float(img_h)))
                y2 = max(0.0, min(float(y2), float(img_h)))
                
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue
                    
                x_center = x1 + (w / 2.0)
                y_center = y1 + (h / 2.0)
                
                norm_x = x_center / img_w
                norm_y = y_center / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                yolo_lines.append(f"{cls_id} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}")
                split_stats["class_counts"][norm_cls] += 1
                split_stats["annotations"] += 1
                
            dst_lbl = split_lbl_out / f"{dst_img.stem}.txt"
            with open(dst_lbl, "w", encoding="utf-8") as lf:
                lf.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
                
            split_stats["images"] += 1
            
        stats["splits"][split_name] = split_stats
        print(f"Split [{split_name}]: {split_stats['images']} images, {split_stats['annotations']} annotations.")
        
    dataset_yaml_content = f"""# PCB Component Detection Dataset for RT-DETR
path: {out_path.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
"""
    for idx, cls_name in enumerate(CANONICAL_CLASSES):
        dataset_yaml_content += f"  {idx}: {cls_name}\n"
        
    yaml_path = out_path.parent / "pcb_data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(dataset_yaml_content)
        
    print(f"\nGenerated YAML config at: {yaml_path}")
    return stats


if __name__ == "__main__":
    raw_dir = r"c:\Users\devas\projects\kiran\dataset"
    out_dir = r"c:\Users\devas\projects\kiran\data\pcb_yolo"
    print("Partitioning and converting PCB dataset...")
    stats = convert_and_split_dataset(raw_dir, out_dir)
    print("Dataset preparation complete!")
