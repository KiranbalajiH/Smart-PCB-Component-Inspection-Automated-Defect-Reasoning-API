import argparse
import os
import sys
import time
import json
import torch
from pathlib import Path
from ultralytics import RTDETR

def train_model(
    data_yaml: str = "data/pcb_data.yaml",
    model_name: str = "rtdetr-l.pt",
    epochs: int = 25,
    imgsz: int = 640,
    batch_size: int = 4,
    workers: int = 2,
    device: str = "0" if torch.cuda.is_available() else "cpu",
    project: str = "runs/detect",
    name: str = "pcb_rtdetr",
    seed: int = 42,
    learning_rate: float = 0.0001,
):
    """
    Trains RT-DETR on the PCB dataset with reproducible hyperparameters and structured logging.
    """
    start_time = time.time()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    print("=" * 70)
    print("  SMART PCB INSPECTION — RT-DETR TRAINING PIPELINE")
    print("=" * 70)
    print(f"Hardware: {device_name}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Model Architecture: {model_name}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Image Size: {imgsz}")
    print(f"Seed: {seed} | Device: {device} | Base LR: {learning_rate}")
    print(f"Dataset YAML: {data_yaml}")
    print("=" * 70)
    
    # Save training configuration metadata for reproducibility
    config_metadata = {
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "seed": seed,
        "device": device,
        "hardware": device_name,
        "pytorch_version": torch.__version__,
        "dataset_config": str(Path(data_yaml).resolve()),
        "base_lr": learning_rate,
        "optimizer": "AdamW",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Initialize RT-DETR model
    model = RTDETR(model_name)
    
    # Train RT-DETR
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=workers,
        device=device,
        project=project,
        name=name,
        seed=seed,
        lr0=learning_rate,
        optimizer="AdamW",
        amp=True,
        save=True,
        save_period=5,
        exist_ok=True,
        val=True,
        plots=True,
        verbose=True
    )
    
    elapsed_minutes = (time.time() - start_time) / 60.0
    config_metadata["training_duration_minutes"] = round(elapsed_minutes, 2)
    config_metadata["best_weight_path"] = str(Path(project) / name / "weights" / "best.pt")
    
    meta_path = Path(project) / name / "training_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(config_metadata, f, indent=4)
        
    print("=" * 70)
    print(f"Training completed successfully in {elapsed_minutes:.2f} minutes.")
    print(f"Best model weights saved at: {config_metadata['best_weight_path']}")
    print(f"Metadata recorded at: {meta_path}")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune RT-DETR on PCB Component Dataset")
    parser.add_argument("--data", type=str, default="data/pcb_data.yaml", help="Path to data YAML")
    parser.add_argument("--model", type=str, default="rtdetr-l.pt", help="RT-DETR base checkpoint")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="CUDA device or cpu")
    parser.add_argument("--name", type=str, default="pcb_rtdetr", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        seed=args.seed
    )
