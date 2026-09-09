import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
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

ID_TO_CLASS = {idx: cls for idx, cls in enumerate(CANONICAL_CLASSES)}
CLASS_TO_ID = {cls: idx for idx, cls in enumerate(CANONICAL_CLASSES)}


class PCBDetector:
    """
    Production-grade inference and spatial geometry extractor for fine-tuned RT-DETR.
    """
    def __init__(
        self,
        model_path: str = "runs/detect/pcb_rtdetr/weights/best.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        # Auto-resolve nested path if default is not directly found
        resolved_path = model_path
        if not Path(resolved_path).exists():
            nested_alt = Path("runs/detect/runs/detect/pcb_rtdetr/weights/best.pt")
            if nested_alt.exists():
                resolved_path = str(nested_alt)
                
        self.model_path = resolved_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        if device is None:
            self.device = "0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
        self._load_model()
        
    def _load_model(self):
        if not Path(self.model_path).exists():
            nested_alt = Path("runs/detect/runs/detect/pcb_rtdetr/weights/best.pt")
            if nested_alt.exists():
                self.model_path = str(nested_alt)
            else:
                print(f"Warning: Model checkpoint '{self.model_path}' not found yet. Using base or waiting for training.")
                return
        self.model = RTDETR(self.model_path)
        
    def detect(
        self,
        image_input: Union[str, np.ndarray, bytes],
        conf: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Runs object detection on an image and returns structured predictions,
        normalized coordinates, centroid geometry, and aggregate statistics.
        """
        if self.model is None:
            self._load_model()
            if self.model is None:
                raise RuntimeError(f"Model weights not available at {self.model_path}")
                
        confidence = conf if conf is not None else self.conf_threshold
        
        # Handle bytes or path
        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, str):
            img = cv2.imread(image_input)
        else:
            img = image_input
            
        if img is None:
            raise ValueError("Failed to decode or read image input.")
            
        h_orig, w_orig = img.shape[:2]
        
        # Inference
        results = self.model.predict(
            source=img,
            conf=confidence,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        detections = []
        class_counts = {cls_name: 0 for cls_name in CANONICAL_CLASSES}
        confidences = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = self.model.names.get(cls_id, ID_TO_CLASS.get(cls_id, f"Class_{cls_id}"))
            conf_val = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            
            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = x2 - x1
            bh = y2 - y1
            
            det_item = {
                "id": len(detections) + 1,
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf_val, 4),
                "box_xyxy": [round(v, 2) for v in xyxy],
                "box_normalized": [
                    round(x1 / w_orig, 4),
                    round(y1 / h_orig, 4),
                    round(x2 / w_orig, 4),
                    round(y2 / h_orig, 4)
                ],
                "centroid": [round(cx, 2), round(cy, 2)],
                "centroid_normalized": [round(cx / w_orig, 4), round(cy / h_orig, 4)],
                "dimensions": [round(bw, 2), round(bh, 2)],
                "area_pixels": round(bw * bh, 2)
            }
            detections.append(det_item)
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            confidences.append(conf_val)
            
        # Overall quality and summary metrics
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        min_conf = float(np.min(confidences)) if confidences else 0.0
        
        return {
            "image_dimensions": {"width": w_orig, "height": h_orig},
            "total_detections": len(detections),
            "detections": detections,
            "class_counts": class_counts,
            "quality_metrics": {
                "mean_confidence": round(mean_conf, 4),
                "min_confidence": round(min_conf, 4),
                "detection_density": round(len(detections) / ((w_orig * h_orig) / 1e6), 2)
            }
        }
