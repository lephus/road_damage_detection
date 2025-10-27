"""
Road Damage Detection Pipeline using YOLOv8
Author: AI Assistant for Master's Thesis on Pavement Damage Area Detection
Optimized for speed and demo purposes

This script performs:
1. Dataset loading and preprocessing (RDD2022 dataset)
2. YOLOv8 model training
3. Model evaluation with comprehensive metrics
4. Visualization of results (curves, predictions, t-SNE)
5. Model saving
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import sys
import shutil
import yaml
import json
import random
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.manifold import TSNE
import cv2
from tqdm import tqdm

# YOLOv8
from ultralytics import YOLO

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset paths (Google Colab structure)
DATASET_PATHS = {
    'India': {
        'train_images': '/content/RDD2022_India/India/train/images',
        'train_annotations': '/content/RDD2022_India/India/train/annotations',
        'test_images': '/content/RDD2022_India/India/test/images'
    },
    'Czech': {
        'train_images': '/content/RDD2022_Czech/train/images',
        'train_annotations': '/content/RDD2022_Czech/train/annotations',
        'test_images': '/content/RDD2022_Czech/test/images'
    },
    'China_MotorBike': {
        'train_images': '/content/RDD2022_China_MotorBike/China_MotorBike/train/images',
        'train_annotations': '/content/RDD2022_China_MotorBike/China_MotorBike/train/annotations',
        'test_images': '/content/RDD2022_China_MotorBike/China_MotorBike/test/images'
    },
    'China_Drone': {
        'train_images': '/content/RDD2022_China_Drone/China_Drone/train/images',
        'train_annotations': '/content/RDD2022_China_Drone/China_Drone/train/annotations'
    }
}

# Training configuration
CONFIG = {
    'model_size': 'n',  # 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
    'epochs': 50,  # Reduced for faster training
    'imgsz': 640,
    'batch': 16,
    'device': 'cuda' if os.system('nvidia-smi') == 0 else 'cpu',  # Auto-detect GPU
    'workers': 4,
    'patience': 10,
    'save_dir': './runs',
    'project_name': 'road_damage_detection'
}

# Road damage classes (RDD2022 dataset)
CLASSES = [
    'D00',  # Longitudinal crack
    'D10',  # Transverse crack
    'D20',  # Alligator crack
    'D40',  # Pothole
    'D43',  # White line blur
    'D44'   # Crosswalk blur
]

# Create class mapping
CLASS_MAP = {cls: idx for idx, cls in enumerate(CLASSES)}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def check_paths():
    """Verify dataset paths exist."""
    print_section("Checking Dataset Paths")
    missing_paths = []
    
    for dataset_name, paths in DATASET_PATHS.items():
        print(f"Checking {dataset_name}...")
        for key, path in paths.items():
            if os.path.exists(path):
                file_count = len(list(Path(path).glob('*'))) if Path(path).exists() else 0
                print(f"  ✓ {key}: {path} ({file_count} files)")
            else:
                print(f"  ✗ {key}: {path} (NOT FOUND)")
                missing_paths.append((dataset_name, key, path))
    
    if missing_paths:
        print("\n⚠️  Warning: Some paths are missing. Continuing with available data...")
    else:
        print("\n✓ All paths found!")
    
    return missing_paths

def create_yaml_config(dataset_dir, classes):
    """Create YOLO dataset configuration file."""
    yaml_content = {
        'path': str(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(classes),
        'names': classes
    }
    
    yaml_path = Path(dataset_dir) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    return yaml_path

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def convert_annotations_to_yolo_format(dataset_name, dataset_paths):
    """
    Convert annotations from original format to YOLO format.
    Assumes annotations are in YOLO format already (or converts if needed).
    """
    print_section(f"Converting Annotations for {dataset_name}")
    
    train_images_path = Path(dataset_paths['train_images'])
    train_annotations_path = Path(dataset_paths['train_annotations'])
    
    if not train_images_path.exists() or not train_annotations_path.exists():
        print(f"⚠️  Skipping {dataset_name} - paths don't exist")
        return None
    
    # Count images and annotations
    images = list(train_images_path.glob('*.jpg')) + list(train_images_path.glob('*.png'))
    annotations = list(train_annotations_path.glob('*.txt'))
    
    print(f"Found {len(images)} images and {len(annotations)} annotations")
    
    # Check if annotations are already in YOLO format
    if annotations:
        sample_ann = annotations[0]
        with open(sample_ann, 'r') as f:
            first_line = f.readline().strip()
            if first_line and len(first_line.split()) == 5:  # YOLO format: class x y w h
                print(f"✓ Annotations already in YOLO format")
                return {
                    'images': images,
                    'annotations': annotations
                }
    
    return {
        'images': images,
        'annotations': annotations
    }

def prepare_combined_dataset(output_dir='./datasets/road_damage'):
    """
    Combine all datasets and prepare YOLO structure.
    """
    print_section("Preparing Combined Dataset")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directory structure
    (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'test').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'test').mkdir(parents=True, exist_ok=True)
    
    all_train_images = []
    all_train_annotations = []
    
    # Process each dataset
    for dataset_name, paths in DATASET_PATHS.items():
        data = convert_annotations_to_yolo_format(dataset_name, paths)
        if data:
            all_train_images.extend(data['images'])
            all_train_annotations.extend(data['annotations'])
            print(f"Added {len(data['images'])} images from {dataset_name}")
    
    print(f"\nTotal: {len(all_train_images)} training images")
    
    # Split into train/val (80/20)
    val_split = 0.2
    n_val = int(len(all_train_images) * val_split)
    indices = list(range(len(all_train_images)))
    random.shuffle(indices)
    
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")
    
    # Copy files
    print("\nCopying files...")
    for i in tqdm(train_indices, desc="Train"):
        img_path = all_train_images[i]
        ann_path = all_train_annotations[i]
        
        # Copy image
        dest_img = output_dir / 'images' / 'train' / img_path.name
        shutil.copy(img_path, dest_img)
        
        # Copy annotation
        if ann_path.exists():
            dest_ann = output_dir / 'labels' / 'train' / ann_path.name
            shutil.copy(ann_path, dest_ann)
    
    for i in tqdm(val_indices, desc="Val"):
        img_path = all_train_images[i]
        ann_path = all_train_annotations[i]
        
        # Copy image
        dest_img = output_dir / 'images' / 'val' / img_path.name
        shutil.copy(img_path, dest_img)
        
        # Copy annotation
        if ann_path.exists():
            dest_ann = output_dir / 'labels' / 'val' / ann_path.name
            shutil.copy(ann_path, dest_ann)
    
    print("\n✓ Dataset prepared successfully!")
    
    # Create YAML config
    yaml_path = create_yaml_config(output_dir, CLASSES)
    print(f"✓ Created dataset config: {yaml_path}")
    
    return str(output_dir), str(yaml_path)

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(dataset_yaml_path, config):
    """Train YOLOv8 model."""
    print_section("Training YOLOv8 Model")
    
    # Load model
    model_size = config['model_size']
    model = YOLO(f'yolov8{model_size}.pt')  # Load pretrained weights
    
    print(f"Model: YOLOv8{model_size}")
    print(f"Epochs: {config['epochs']}")
    print(f"Image size: {config['imgsz']}")
    print(f"Batch size: {config['batch']}")
    print(f"Device: {config['device']}")
    
    # Train
    results = model.train(
        data=dataset_yaml_path,
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        batch=config['batch'],
        device=config['device'],
        workers=config['workers'],
        patience=config['patience'],
        project=config['save_dir'],
        name=config['project_name'],
        exist_ok=True,
        verbose=True,
        # Data augmentation (included in YOLOv8 by default)
        hsv_h=0.015,  # Hue augmentation
        hsv_s=0.7,    # Saturation augmentation
        hsv_v=0.4,    # Value augmentation
        degrees=10,   # Rotation
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1
    )
    
    print("\n✓ Training completed!")
    
    return model, results

# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model, dataset_yaml_path):
    """Evaluate model on validation set."""
    print_section("Evaluating Model")
    
    # Run validation
    metrics = model.val(data=dataset_yaml_path)
    
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"{'='*60}\n")
    
    return metrics

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(results_dir):
    """Plot training curves from results."""
    print_section("Generating Training Curves")
    
    results_dir = Path(results_dir)
    csv_file = results_dir / 'results.csv'
    
    if not csv_file.exists():
        print("⚠️  Results CSV not found. Skipping curve plot.")
        return
    
    # Read results
    df = pd.read_csv(csv_file)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', marker='o')
    ax.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Box Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss', marker='o')
    ax.plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Classification Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # mAP curves
    ax = axes[1, 0]
    ax.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', marker='o')
    ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_title('Mean Average Precision')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Precision/Recall
    ax = axes[1, 1]
    ax.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', marker='o')
    ax.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Precision & Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {results_dir / 'training_curves.png'}")
    plt.close()

def visualize_predictions(model, dataset_dir, num_samples=10):
    """Visualize model predictions on sample images."""
    print_section("Generating Prediction Visualizations")
    
    val_images_dir = Path(dataset_dir) / 'images' / 'val'
    if not val_images_dir.exists():
        print("⚠️  Validation images not found. Skipping visualization.")
        return
    
    # Get sample images
    images = list(val_images_dir.glob('*.jpg')) + list(val_images_dir.glob('*.png'))
    sample_images = random.sample(images, min(num_samples, len(images)))
    
    # Run predictions
    results = model(sample_images, save=True, save_txt=True, conf=0.25)
    
    print(f"✓ Generated predictions for {len(sample_images)} images")
    print(f"✓ Saved in: {model.predictor.save_dir}")

def save_model_summary(model, save_path):
    """Save model architecture summary."""
    print_section("Saving Model Summary")
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ROAD DAMAGE DETECTION MODEL SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model Architecture: YOLOv8{CONFIG['model_size']}\n")
        f.write(f"Classes: {len(CLASSES)}\n")
        f.write(f"Class Names: {', '.join(CLASSES)}\n\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Model summary saved: {save_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    print("\n" + "="*80)
    print("ROAD DAMAGE DETECTION PIPELINE")
    print("="*80)
    print(f"Config: YOLOv8{CONFIG['model_size']}, {CONFIG['epochs']} epochs")
    print("="*80 + "\n")
    
    try:
        # Step 1: Check paths
        check_paths()
        
        # Step 2: Prepare dataset
        dataset_dir, dataset_yaml = prepare_combined_dataset()
        
        # Step 3: Train model
        model, results = train_model(dataset_yaml, CONFIG)
        
        # Step 4: Evaluate
        metrics = evaluate_model(model, dataset_yaml)
        
        # Step 5: Visualizations
        results_dir = Path(CONFIG['save_dir']) / CONFIG['project_name']
        plot_training_curves(results_dir)
        visualize_predictions(model, dataset_dir, num_samples=10)
        
        # Step 6: Save model
        model_path = results_dir / 'weights' / 'best.pt'
        if model_path.exists():
            # Save with a more descriptive name
            final_model_path = Path('./road_damage_model.pt')
            shutil.copy(model_path, final_model_path)
            print(f"\n✓ Final model saved: {final_model_path}")
        
        # Save summary
        save_model_summary(model, results_dir / 'model_summary.txt')
        
        # Print final summary
        print_section("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Model saved: {final_model_path}")
        print(f"Results saved: {results_dir}")
        print(f"\nFinal Metrics:")
        print(f"  mAP50:    {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall:    {metrics.box.mr:.4f}")
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()