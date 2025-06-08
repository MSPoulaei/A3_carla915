#!/usr/bin/env python3
"""
YOLOv11 Training Script for CARLA Dataset
Generated automatically by YOLOv11DataPreprocessor
"""

from ultralytics import YOLO
import yaml
from pathlib import Path

def main():
    # Dataset configuration
    dataset_path = Path("./yolo_dataset")
    config_file = dataset_path / "dataset.yaml"
    
    # Load dataset configuration
    with open(config_file, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    print(f"Training YOLOv11 on dataset: {dataset_config['path']}")
    print(f"Classes: {dataset_config['names']}")
    print(f"Number of classes: {dataset_config['nc']}")
    
    # Initialize YOLOv11 model
    # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
    model = YOLO('yolo11n-seg.pt')  # nano model for segmentation
    
    # Training parameters
    training_args = {
        'data': str(config_file),
        'epochs': 50,
        'imgsz': 512,
        'batch': 32,
        'workers': 8,
        'device': '0',  # Use GPU if available
        'project': 'carla_yolo_training_result',
        'name': 'carla_segmentation_nano',
        'save_period': 10,
        'val': True,
        'plots': True,
        'verbose': True,
        
        # Optimization parameters
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Augmentation parameters
        # 'hsv_h': 0.015,
        # 'hsv_s': 0.7,
        # 'hsv_v': 0.4,
        # 'degrees': 0.0,
        # 'translate': 0.1,
        # 'scale': 0.5,
        # 'shear': 0.0,
        # 'perspective': 0.0,
        # 'flipud': 0.0,
        # 'fliplr': 0.5,
        # 'mosaic': 1.0,
        # 'mixup': 0.0,
        # 'copy_paste': 0.0,

        'hsv_h': 0.0,
        'hsv_s': 0.0,
        'hsv_v': 0.0,
        'degrees': 0.0,
        'translate': 0.0,
        'scale': 0.0,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.0,
        'mosaic': 0.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        
        # Loss parameters
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 2.0,
        'label_smoothing': 0.0,
        
        # Validation parameters
        #'val_period': 1,
        'save_json': True,
        'save_hybrid': False,
        'conf': 0.001,
        'iou': 0.6,
        'max_det': 300,
        'half': False,
        'dnn': False,
        'plots': True
    }
    
    # Start training
    print("Starting training...")
    results = model.train(**training_args)
    print("---------------------------------------------------------------------------------------")
    # Validation
    print("Running validation...")
    val_results = model.val()
    print("---------------------------------------------------------------------------------------")
    
    # Export model
    print("Exporting model...")
    model.export(format='onnx')  # Export to ONNX format
    print("---------------------------------------------------------------------------------------")
    
    print("Training completed!")
    print(f"Best model saved at: {model.trainer.best}")
    print(f"Results: {results}")

if __name__ == "__main__":
    main()
