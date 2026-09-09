import argparse
import os
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from ultralytics import RTDETR

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

CLASS_TO_ID = {cls: idx for idx, cls in enumerate(CANONICAL_CLASSES)}
ID_TO_CLASS = {idx: cls for idx, cls in enumerate(CANONICAL_CLASSES)}


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculates Intersection over Union for [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter_area <= 0:
        return 0.0
        
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def evaluate_model(
    model_path: str = "runs/detect/pcb_rtdetr/weights/best.pt",
    data_yaml: str = "data/pcb_data.yaml",
    output_dir: str = "runs/evaluation",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 640
) -> Dict[str, any]:
    """
    Evaluates fine-tuned RT-DETR on the test split, computes comprehensive metrics,
    and analyzes failure cases.
    """
    # Auto-resolve path
    resolved_model_path = model_path
    if not Path(resolved_model_path).exists():
        nested_alt = Path("runs/detect/runs/detect/pcb_rtdetr/weights/best.pt")
        if nested_alt.exists():
            resolved_model_path = str(nested_alt)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = out_dir / "failure_cases"
    failures_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  SMART PCB INSPECTION — QUANTITATIVE EVALUATION & FAILURE ANALYSIS")
    print("=" * 70)
    print(f"Model Path: {resolved_model_path}")
    print(f"Data YAML: {data_yaml}")
    print(f"Confidence Threshold: {conf_threshold} | IoU Threshold: {iou_threshold}")
    
    # 1. Run Ultralytics validation on test split
    model = RTDETR(resolved_model_path)
    val_results = model.val(
        data=data_yaml,
        split="test",
        imgsz=imgsz,
        conf=conf_threshold,
        iou=iou_threshold,
        project=str(out_dir),
        name="test_metrics",
        exist_ok=True,
        plots=True
    )
    
    # Extract overall metrics
    metrics_summary = {
        "mAP50": float(val_results.box.map50),
        "mAP50_95": float(val_results.box.map),
        "precision": float(val_results.box.mp),
        "recall": float(val_results.box.mr),
        "f1_score": float(2 * (val_results.box.mp * val_results.box.mr) / (val_results.box.mp + val_results.box.mr + 1e-6)),
        "class_metrics": {}
    }
    
    # Extract per-class metrics
    for idx, cls_name in enumerate(CANONICAL_CLASSES):
        if idx < len(val_results.box.p):
            p = float(val_results.box.p[idx])
            r = float(val_results.box.r[idx])
            ap50 = float(val_results.box.ap50[idx])
            ap = float(val_results.box.ap[idx])
            f1 = float(2 * (p * r) / (p + r + 1e-6))
            metrics_summary["class_metrics"][cls_name] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1_score": round(f1, 4),
                "mAP50": round(ap50, 4),
                "mAP50_95": round(ap, 4)
            }
            
    print("\nQuantitative Metrics on Test Split:")
    print(f"  mAP@0.50:      {metrics_summary['mAP50']:.4f}")
    print(f"  mAP@0.50:0.95: {metrics_summary['mAP50_95']:.4f}")
    print(f"  Precision:     {metrics_summary['precision']:.4f}")
    print(f"  Recall:        {metrics_summary['recall']:.4f}")
    print(f"  F1-Score:      {metrics_summary['f1_score']:.4f}")
    print("\nPer-Class Breakdown:")
    for cls_name, cm in metrics_summary["class_metrics"].items():
        print(f"  - {cls_name:12s} | mAP50: {cm['mAP50']:.4f} | Prec: {cm['precision']:.4f} | Rec: {cm['recall']:.4f} | F1: {cm['f1_score']:.4f}")
        
    # 2. Detailed Failure Case Analysis on Test Set
    test_img_dir = Path("data/pcb_yolo/images/test")
    test_lbl_dir = Path("data/pcb_yolo/labels/test")
    
    failure_candidates = []
    
    for img_file in test_img_dir.glob("*.jpg"):
        lbl_file = test_lbl_dir / f"{img_file.stem}.txt"
        if not lbl_file.exists():
            continue
            
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        h_orig, w_orig = img.shape[:2]
        
        # Load Ground Truth boxes
        gt_boxes = []
        with open(lbl_file, "r") as lf:
            for line in lf:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    x1 = (cx - bw / 2.0) * w_orig
                    y1 = (cy - bh / 2.0) * h_orig
                    x2 = (cx + bw / 2.0) * w_orig
                    y2 = (cy + bh / 2.0) * h_orig
                    gt_boxes.append({
                        "class_id": cls_id,
                        "class_name": ID_TO_CLASS[cls_id],
                        "box": [x1, y1, x2, y2]
                    })
                    
        # Predict with RT-DETR
        preds = model.predict(source=str(img_file), conf=conf_threshold, verbose=False)[0]
        pred_boxes = []
        for box in preds.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            pred_boxes.append({
                "class_id": cls_id,
                "class_name": ID_TO_CLASS.get(cls_id, f"Class_{cls_id}"),
                "conf": conf,
                "box": xyxy
            })
            
        # Match GT and Preds to identify errors
        matched_gt = set()
        matched_pred = set()
        errors = []
        
        # Check for False Positives / Class Confusions
        for p_idx, pb in enumerate(pred_boxes):
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, gb in enumerate(gt_boxes):
                iou = calculate_iou(pb["box"], gb["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx
                    
            if best_iou >= 0.5:
                matched_pred.add(p_idx)
                matched_gt.add(best_g_idx)
                gb = gt_boxes[best_g_idx]
                if pb["class_id"] != gb["class_id"]:
                    errors.append({
                        "type": "CLASS_CONFUSION",
                        "description": f"Predicted '{pb['class_name']}' (conf: {pb['conf']:.2f}) but Ground Truth is '{gb['class_name']}' (IoU: {best_iou:.2f})",
                        "pred": pb,
                        "gt": gb
                    })
            elif best_iou < 0.2:
                errors.append({
                    "type": "FALSE_POSITIVE",
                    "description": f"Spurious '{pb['class_name']}' detection (conf: {pb['conf']:.2f}) with no matching GT object",
                    "pred": pb,
                    "gt": None
                })
                
        # Check for False Negatives (Missed Objects)
        for g_idx, gb in enumerate(gt_boxes):
            if g_idx not in matched_gt:
                errors.append({
                    "type": "FALSE_NEGATIVE_MISS",
                    "description": f"Missed ground truth object '{gb['class_name']}'",
                    "pred": None,
                    "gt": gb
                })
                
        if errors:
            failure_candidates.append({
                "image_name": img_file.name,
                "image_path": str(img_file),
                "gt_count": len(gt_boxes),
                "pred_count": len(pred_boxes),
                "error_count": len(errors),
                "errors": errors,
                "gt_boxes": gt_boxes,
                "pred_boxes": pred_boxes
            })
            
    # Sort failure cases by severity / interesting diversity
    failure_candidates.sort(key=lambda x: x["error_count"], reverse=True)
    
    # Categorize top 5 distinct failure cases
    categorized_failures = []
    category_types = [
        ("CLASS_CONFUSION", "Capacitor Inter-Class Confusion (Cap1 vs Cap2/Cap3)"),
        ("FALSE_NEGATIVE_MISS", "Dense Cluster / Small SMD Resistor Miss"),
        ("FALSE_POSITIVE", "Edge Reflection / Background Artifact False Positive"),
        ("LOW_CONFIDENCE", "Extreme Perspective / Lighting Glare Ambiguity"),
        ("OCCLUSION_OVERLAP", "Adjacent Component Boundary Occlusion")
    ]
    
    top_5_cases = []
    for f in failure_candidates[:15]:
        if len(top_5_cases) >= 5:
            break
        # Render visual diagnostic image
        img = cv2.imread(f["image_path"])
        vis_img = img.copy()
        
        # Draw GT in Green
        for gb in f["gt_boxes"]:
            b = [int(v) for v in gb["box"]]
            cv2.rectangle(vis_img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.putText(vis_img, f"GT: {gb['class_name']}", (b[0], max(15, b[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Draw Preds in Red/Orange
        for pb in f["pred_boxes"]:
            b = [int(v) for v in pb["box"]]
            cv2.rectangle(vis_img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cv2.putText(vis_img, f"Pred: {pb['class_name']} ({pb['conf']:.2f})", (b[0], min(vis_img.shape[0] - 5, b[3] + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        save_path = failures_dir / f"failure_case_{len(top_5_cases)+1}_{f['image_name']}"
        cv2.imwrite(str(save_path), vis_img)
        
        top_5_cases.append({
            "case_id": len(top_5_cases) + 1,
            "image": f["image_name"],
            "diagnostic_image": str(save_path),
            "gt_count": f["gt_count"],
            "pred_count": f["pred_count"],
            "primary_error_type": f["errors"][0]["type"],
            "error_details": [e["description"] for e in f["errors"][:3]]
        })
        
    metrics_summary["top_5_failure_cases"] = top_5_cases
    
    # Save complete evaluation output
    eval_json_path = out_dir / "evaluation_report.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
        
    print(f"\nSaved evaluation report to: {eval_json_path}")
    print(f"Generated {len(top_5_cases)} diagnostic failure case visualizations in {failures_dir}")
    print("=" * 70)
    return metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RT-DETR and Extract Failure Cases")
    parser.add_argument("--model", type=str, default="runs/detect/pcb_rtdetr/weights/best.pt", help="Path to trained weights")
    parser.add_argument("--data", type=str, default="data/pcb_data.yaml", help="Path to data YAML")
    parser.add_argument("--out", type=str, default="runs/evaluation", help="Output directory")
    args = parser.parse_args()
    
    evaluate_model(model_path=args.model, data_yaml=args.data, output_dir=args.out)
