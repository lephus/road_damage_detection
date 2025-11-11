"""
Road Damage Detection Training Script
Based on research papers and RDD2022 dataset
Supports YOLOv8 object detection with comprehensive metrics and visualization
"""

import os
import sys
import time
import json
import random
import shutil
import argparse
import gc
from pathlib import Path
from datetime import datetime
from collections import Counter
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm

# Machine Learning libraries
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_curve, auc, confusion_matrix, classification_report
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

# YOLOv8
from ultralytics import YOLO

# Weights & Biases for experiment tracking
import wandb

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class RoadDamageTrainer:
    """Main trainer class for road damage detection"""
    
    def __init__(self, dataset_root, output_dir, epochs=100, batch_size=16, img_size=640, wandb_api_key=None, dataset_limit=None, model_name='yolov8s.pt'):
        """
        Initialize the trainer
        
        Args:
            dataset_root: Root directory containing datasets
            output_dir: Directory to save outputs
            epochs: Number of training epochs (minimum 100)
            batch_size: Batch size for training
            img_size: Image size for training (640x640)
            wandb_api_key: Weights & Biases API key for tracking
            dataset_limit: Maximum number of images to use from dataset (None = use all)
            model_name: YOLOv8 model variant (yolov8n/s/m/l/x.pt), default: yolov8s.pt
        """
        self.dataset_root = dataset_root
        self.output_dir = output_dir
        self.epochs = max(epochs, 100)  # Ensure minimum 100 epochs
        self.img_size = img_size
        self.dataset_limit = dataset_limit
        self.model_name = model_name
        
        # Auto-adjust batch size based on available memory and device
        # Detect device first
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.batch_size = batch_size
        elif torch.backends.mps.is_available():
            self.device = 'mps'
            # MPS (Apple Silicon) has limited memory - reduce batch size
            self.batch_size = min(batch_size, 8)
            print(f"⚠️  MPS device detected - reducing batch size from {batch_size} to {self.batch_size} to prevent OOM")
        else:
            self.device = 'cpu'
            self.batch_size = min(batch_size, 4)
            print(f"⚠️  CPU device detected - reducing batch size from {batch_size} to {self.batch_size}")
        
        # Valid damage classes
        self.VALID_CLASSES = ['D00', 'D10', 'D20', 'D40', 'D43', 'D44']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.VALID_CLASSES)}
        self.num_classes = len(self.VALID_CLASSES)
        
        # Initialize Weights & Biases
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        
        # Initialize wandb run
        model_short_name = self.model_name.replace('.pt', '')
        run_name = f"road_damage_{model_short_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project="road-damage-detection",
            name=run_name,
            config={
                "model": self.model_name,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "img_size": self.img_size,
                "num_classes": self.num_classes,
                "classes": self.VALID_CLASSES,
                "dataset_root": str(dataset_root),
                "output_dir": str(output_dir),
                "dataset_limit": self.dataset_limit,
                "device": self.device,  # Device detected before wandb init
            }
        )
        
        # Dataset paths configuration
        self.dataset_paths = {
            'India': {
                'train_images': os.path.join(dataset_root, 'India/train/images'),
                'train_annotations': os.path.join(dataset_root, 'India/train/annotations'),
                'test_images': os.path.join(dataset_root, 'India/test/images')
            },
            'Czech': {
                'train_images': os.path.join(dataset_root, 'Czech/train/images'),
                'train_annotations': os.path.join(dataset_root, 'Czech/train/annotations'),
                'test_images': os.path.join(dataset_root, 'Czech/test/images')
            },
            'China_MotorBike': {
                'train_images': os.path.join(dataset_root, 'China_MotorBike/train/images'),
                'train_annotations': os.path.join(dataset_root, 'China_MotorBike/train/annotations'),
                'test_images': os.path.join(dataset_root, 'China_MotorBike/test/images')
            },
            'China_Drone': {
                'train_images': os.path.join(dataset_root, 'China_Drone/train/images'),
                'train_annotations': os.path.join(dataset_root, 'China_Drone/train/annotations')
            },
            'Japan': {
                'train_images': os.path.join(dataset_root, 'Japan/train/images'),
                'train_annotations': os.path.join(dataset_root, 'Japan/train/annotations'),
                'test_images': os.path.join(dataset_root, 'Japan/test/images')
            },
        }
        
        # Create output directory structure
        self.setup_directories()
        
        # Training metrics storage
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'mAP50': [],
            'mAP50_95': [],
            'precision': [],
            'recall': [],
            'epochs': []
        }
        
        # Print configuration
        print(f"🖥️  Using device: {self.device}")
        print(f"🤖 Model: {self.model_name}")
        print(f"📊 Training for {self.epochs} epochs")
        print(f"🎯 Target accuracy: >= 85%")
        if self.dataset_limit is not None:
            print(f"📉 Dataset limit: {self.dataset_limit} images")
        else:
            print(f"📉 Dataset limit: None (using all available images)")
        print(f"🔗 W&B Run: {wandb.run.url if wandb.run else 'Not initialized'}")
        
    def setup_directories(self):
        """Setup output directory structure"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # YOLO format directories
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.output_dir, f'{split}/images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, f'{split}/labels'), exist_ok=True)
        
        # Results directories
        os.makedirs(os.path.join(self.output_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)
        
        print(f"✅ Output directories created at: {self.output_dir}")
    
    def parse_xml_to_yolo(self, xml_path, img_width, img_height):
        """
        Parse XML annotations and convert to YOLO format
        
        Args:
            xml_path: Path to XML annotation file
            img_width: Image width
            img_height: Image height
            
        Returns:
            boxes: List of bounding boxes in YOLO format
            labels: List of class labels
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in self.VALID_CLASSES:
                continue
            cls_idx = self.class_to_idx[cls_name]
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Skip invalid bounding boxes
            if xmax <= xmin or ymax <= ymin:
                continue
            
            # Convert to YOLO format (center_x, center_y, width, height) normalized
            center_x = (xmin + xmax) / 2 / img_width
            center_y = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            boxes.append([center_x, center_y, width, height])
            labels.append(cls_idx)
        
        return boxes, labels
    
    def load_dataset(self):
        """
        Load and prepare dataset from multiple sources
        
        Returns:
            all_images: List of image paths
            all_labels: List of (boxes, labels, xml_path) tuples
        """
        print("\n📂 Loading dataset...")
        all_images = []
        all_labels = []
        dataset_stats = {}
        
        for dataset_name, paths in self.dataset_paths.items():
            img_dir = paths['train_images']
            ann_dir = paths['train_annotations']
            
            if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
                print(f"⚠️  Skipping {dataset_name}: Directory not found")
                continue
            
            dataset_count = 0
            for img_file in tqdm(os.listdir(img_dir), desc=f"Loading {dataset_name}"):
                if not img_file.endswith(('.jpg', '.png', '.jpeg')):
                    continue
                
                img_path = os.path.join(img_dir, img_file)
                xml_path = os.path.join(ann_dir, 'xmls', f"{img_file.split('.')[0]}.xml")
                
                if not os.path.exists(xml_path):
                    continue
                
                # Read image to get dimensions
                img = cv2.imread(img_path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                
                # Parse annotations
                try:
                    boxes, labels = self.parse_xml_to_yolo(xml_path, w, h)
                    if len(boxes) > 0:
                        all_images.append(img_path)
                        all_labels.append((boxes, labels, xml_path))
                        dataset_count += 1
                except Exception as e:
                    continue
            
            dataset_stats[dataset_name] = dataset_count
            print(f"  ✓ {dataset_name}: {dataset_count} images")
        
        print(f"\n✅ Total dataset: {len(all_images)} valid image-annotation pairs")
        
        # Apply dataset limit if specified
        original_count = len(all_images)
        if self.dataset_limit is not None and self.dataset_limit > 0:
            if len(all_images) > self.dataset_limit:
                print(f"\n📊 Limiting dataset from {len(all_images)} to {self.dataset_limit} images...")
                # Randomly sample to maintain diversity
                indices = random.sample(range(len(all_images)), self.dataset_limit)
                all_images = [all_images[i] for i in indices]
                all_labels = [all_labels[i] for i in indices]
                print(f"✅ Dataset limited to {len(all_images)} images")
            else:
                print(f"ℹ️  Dataset limit ({self.dataset_limit}) is greater than available images ({len(all_images)}), using all images")
        
        # Save dataset statistics
        stats = {
            **dataset_stats,
            "original_total": original_count,
            "final_total": len(all_images),
            "limit_applied": self.dataset_limit if self.dataset_limit is not None else None
        }
        with open(os.path.join(self.output_dir, 'metrics', 'dataset_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Log dataset statistics to wandb
        wandb.log({
            "dataset/original_total_images": original_count,
            "dataset/total_images": len(all_images),
            "dataset/limit_applied": self.dataset_limit if self.dataset_limit is not None else original_count
        })
        for dataset_name, count in dataset_stats.items():
            wandb.log({f"dataset/{dataset_name}": count})
        
        # Clear memory after dataset loading
        gc.collect()
        
        return all_images, all_labels
    
    def preprocess_and_save(self, images, labels, split='train'):
        """
        Preprocess and save images/labels in YOLO format
        
        Args:
            images: List of image paths
            labels: List of (boxes, labels, xml_path) tuples
            split: Dataset split ('train', 'val', 'test')
        """
        img_out_dir = os.path.join(self.output_dir, f'{split}/images')
        lbl_out_dir = os.path.join(self.output_dir, f'{split}/labels')
        
        for img_path, (boxes, lbls, xml_path) in tqdm(zip(images, labels), 
                                                        desc=f"Processing {split}",
                                                        total=len(images)):
            # Copy and resize image
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (self.img_size, self.img_size))
            img_name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(img_out_dir, img_name), img)
            
            # Free memory
            del img
            
            # Save YOLO annotations
            lbl_path = os.path.join(lbl_out_dir, f"{img_name.split('.')[0]}.txt")
            with open(lbl_path, 'w') as f:
                for box, lbl in zip(boxes, lbls):
                    f.write(f"{lbl} {' '.join(map(str, box))}\n")
    
    def split_dataset(self, images, labels):
        """
        Split dataset into train/val/test sets
        
        Args:
            images: List of image paths
            labels: List of labels
            
        Returns:
            Tuple of (n_train, n_val, n_test)
        """
        print("\n📊 Splitting dataset...")
        
        # 80% train, 10% val, 10% test
        train_imgs, test_imgs, train_lbls, test_lbls = train_test_split(
            images, labels, test_size=0.1, random_state=42
        )
        train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
            train_imgs, train_lbls, test_size=0.1111, random_state=42  # ≈10% of total
        )
        
        self.preprocess_and_save(train_imgs, train_lbls, 'train')
        self.preprocess_and_save(val_imgs, val_lbls, 'val')
        self.preprocess_and_save(test_imgs, test_lbls, 'test')
        
        # Clear memory after preprocessing
        gc.collect()
        
        print(f"  Train: {len(train_imgs)} images")
        print(f"  Val: {len(val_imgs)} images")
        print(f"  Test: {len(test_imgs)} images")
        
        # Log split sizes to wandb
        wandb.log({
            "dataset/train_size": len(train_imgs),
            "dataset/val_size": len(val_imgs),
            "dataset/test_size": len(test_imgs)
        })
        
        return len(train_imgs), len(val_imgs), len(test_imgs)
    
    def create_yolo_config(self):
        """Create YOLO configuration file"""
        config = f"""path: {self.output_dir}
train: train/images
val: val/images
test: test/images

nc: {self.num_classes}
names: {self.VALID_CLASSES}
"""
        config_path = os.path.join(self.output_dir, 'data.yaml')
        with open(config_path, 'w') as f:
            f.write(config)
        
        print(f"✅ YOLO config created: {config_path}")
        return config_path
    
    def train(self):
        """
        Train the YOLO model
        
        Returns:
            model: Trained model
            results: Training results
            training_time: Total training time in seconds
        """
        print(f"\n🚀 Starting training with {self.model_name}...")
        print(f"⏱️  Epochs: {self.epochs}")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"🖼️  Image size: {self.img_size}x{self.img_size}")
        print(f"🖥️  Device: {self.device}")
        
        # Clear memory before training
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        elif self.device == 'mps':
            torch.mps.empty_cache()
        
        # Initialize model
        model = YOLO(self.model_name)
        
        # Start timing
        start_time = time.time()
        
        # Configure training parameters based on device
        # For MPS, disable mixed precision (AMP) as it can cause memory issues
        use_amp = True if self.device == 'cuda' else False
        
        # Train the model (YOLOv8 will automatically log to wandb if initialized)
        results = model.train(
            data=os.path.join(self.output_dir, 'data.yaml'),
            epochs=self.epochs,
            imgsz=self.img_size,
            batch=self.batch_size,
            device=self.device,
            augment=True,
            lr0=0.001,
            cos_lr=True,
            patience=20,
            amp=use_amp,  # Mixed precision training (disabled for MPS)
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            project=os.path.join(self.output_dir, 'runs'),
            name='road_damage_detection',
            exist_ok=True,
            verbose=True,
            plots=True,
            workers=2,  # Reduce workers to save memory
            cache=False  # Don't cache images in RAM
        )
        
        # End timing
        training_time = time.time() - start_time
        
        print(f"\n✅ Training completed!")
        print(f"⏱️  Total training time: {training_time/3600:.2f} hours ({training_time:.2f} seconds)")
        
        # Log training time to wandb
        wandb.log({
            "training/total_time_seconds": training_time,
            "training/total_time_minutes": training_time / 60,
            "training/total_time_hours": training_time / 3600
        })
        
        # Save training time
        with open(os.path.join(self.output_dir, 'metrics', 'training_time.txt'), 'w') as f:
            f.write(f"Training Time: {training_time:.2f} seconds\n")
            f.write(f"Training Time: {training_time/60:.2f} minutes\n")
            f.write(f"Training Time: {training_time/3600:.2f} hours\n")
        
        # Save best model
        best_model_path = os.path.join(self.output_dir, 'runs', 'road_damage_detection', 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            shutil.copy(best_model_path, os.path.join(self.output_dir, 'models', 'best_model.pt'))
            print(f"💾 Best model saved: {os.path.join(self.output_dir, 'models', 'best_model.pt')}")
            # Log model artifact to wandb
            artifact = wandb.Artifact('best_model', type='model')
            artifact.add_file(os.path.join(self.output_dir, 'models', 'best_model.pt'))
            wandb.log_artifact(artifact)
        
        return model, results, training_time
    
    def extract_features(self, model, images, max_samples=500):
        """
        Extract features from the model for t-SNE visualization
        
        Args:
            model: Trained YOLO model
            images: List of image paths
            max_samples: Maximum number of samples to process
            
        Returns:
            features: Extracted features
            labels: Corresponding labels
        """
        print(f"\n🔍 Extracting features for t-SNE visualization...")
        features = []
        labels = []
        
        # Limit number of samples for computational efficiency
        sample_images = random.sample(images, min(len(images), max_samples))
        
        model.eval()
        for img_path in tqdm(sample_images, desc="Extracting features"):
            try:
                # Read and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img = cv2.resize(img, (self.img_size, self.img_size))
                
                # Get label
                lbl_path = img_path.replace('/images/', '/labels/').replace('.jpg', '.txt').replace('.png', '.txt')
                if os.path.exists(lbl_path):
                    with open(lbl_path, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            lbl = int(lines[0].split()[0])
                            labels.append(lbl)
                        else:
                            labels.append(-1)  # No label
                else:
                    labels.append(-1)
                
                # Extract features using model prediction
                results = model.predict(img, verbose=False)
                if len(results) > 0 and len(results[0].boxes) > 0:
                    # Use average of all box features
                    box_features = []
                    for box in results[0].boxes:
                        feat = box.xywh[0].cpu().numpy()  # Use box coordinates as features
                        box_features.append(feat)
                    features.append(np.mean(box_features, axis=0))
                else:
                    # Use zeros for images with no detections
                    features.append(np.zeros(4))
                    
            except Exception as e:
                continue
        
        return np.array(features), np.array(labels)
    
    def visualize_tsne(self, model, split='test'):
        """
        Visualize t-SNE of feature embeddings
        
        Args:
            model: Trained model
            split: Dataset split to use ('train', 'val', 'test')
        """
        print(f"\n📊 Generating t-SNE visualization for {split} set...")
        
        # Get images
        img_dir = os.path.join(self.output_dir, f'{split}/images')
        images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                 if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Extract features
        features, labels = self.extract_features(model, images)
        
        # Apply t-SNE
        print("  Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        embeddings = tsne.fit_transform(features)
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        # Create color map
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes + 1))
        
        for i in range(self.num_classes):
            mask = labels == i
            plt.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                       c=[colors[i]], label=self.VALID_CLASSES[i], 
                       alpha=0.6, s=50)
        
        # Plot no-label points
        mask = labels == -1
        if np.any(mask):
            plt.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                       c='gray', label='No Detection', 
                       alpha=0.3, s=30, marker='x')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f't-SNE Visualization of Feature Embeddings ({split.upper()} set)', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.tight_layout()
        
        # Save
        tsne_path = os.path.join(self.output_dir, 'visualizations', f'tsne_{split}.png')
        plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log t-SNE visualization to wandb
        wandb.log({f"{split}/tsne": wandb.Image(tsne_path)})
        
        print(f"  ✓ t-SNE plot saved: {tsne_path}")
    
    def evaluate_model(self, model, split='test'):
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            split: Dataset split to evaluate ('test' or 'val')
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        print(f"\n📊 Evaluating model on {split} set...")
        
        # Clear memory before evaluation
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        elif self.device == 'mps':
            torch.mps.empty_cache()
        
        # Get test images
        img_dir = os.path.join(self.output_dir, f'{split}/images')
        test_images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        y_true = []
        y_pred = []
        y_scores = []
        
        # Perform predictions
        for idx, img_path in enumerate(tqdm(test_images, desc="Evaluating")):
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            results = model.predict(img, conf=0.25, verbose=False)
            
            # Free image memory
            del img
            
            # Get ground truth
            lbl_path = img_path.replace('/images/', '/labels/').replace('.jpg', '.txt').replace('.png', '.txt')
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        lbl = int(line.split()[0])
                        y_true.append(lbl)
            
            # Get predictions
            if len(results) > 0 and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    y_pred.append(int(box.cls))
                    y_scores.append(float(box.conf))
            
            # Periodic memory cleanup every 100 images
            if idx > 0 and idx % 100 == 0:
                gc.collect()
                if self.device == 'mps':
                    torch.mps.empty_cache()
        
        # Ensure we have predictions
        if len(y_true) == 0 or len(y_pred) == 0:
            print("⚠️  No predictions found!")
            return {}
        
        # Pad to same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        y_scores = y_scores[:min_len]
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\n📈 Evaluation Metrics:")
        print(f"  Accuracy:  {accuracy*100:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        if accuracy >= 0.85:
            print(f"  🎯 Target accuracy achieved! ({accuracy*100:.2f}% >= 85%)")
        else:
            print(f"  ⚠️  Target accuracy not reached ({accuracy*100:.2f}% < 85%)")
        
        # Save metrics
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        # Log metrics to wandb
        wandb.log({
            f"{split}/accuracy": accuracy,
            f"{split}/precision": precision,
            f"{split}/recall": recall,
            f"{split}/f1_score": f1
        })
        
        with open(os.path.join(self.output_dir, 'metrics', f'metrics_{split}.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Classification report
        report = classification_report(y_true, y_pred, 
                                       target_names=self.VALID_CLASSES,
                                       zero_division=0)
        print(f"\n📊 Classification Report:\n{report}")
        
        with open(os.path.join(self.output_dir, 'metrics', f'classification_report_{split}.txt'), 'w') as f:
            f.write(report)
        
        # Confusion matrix
        self.plot_confusion_matrix(y_true, y_pred, split)
        
        # ROC curve
        self.plot_roc_curves(y_true, y_scores, split)
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, split='test'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.VALID_CLASSES,
                   yticklabels=self.VALID_CLASSES)
        plt.title(f'Confusion Matrix ({split.upper()} set)', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_path = os.path.join(self.output_dir, 'visualizations', f'confusion_matrix_{split}.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log confusion matrix to wandb
        wandb.log({f"{split}/confusion_matrix": wandb.Image(cm_path)})
        
        print(f"  ✓ Confusion matrix saved: {cm_path}")
    
    def plot_roc_curves(self, y_true, y_scores, split='test'):
        """Plot ROC curves and calculate AUC"""
        # Binarize labels for multiclass ROC
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        
        # For simplicity, plot binary ROC (damage vs no damage)
        y_true_binary = [1 if y >= 0 else 0 for y in y_true]
        y_scores_array = np.array(y_scores)
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_scores_array)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({split.upper()} set)', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        roc_path = os.path.join(self.output_dir, 'visualizations', f'roc_curve_{split}.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log ROC curve and AUC to wandb
        wandb.log({
            f"{split}/roc_curve": wandb.Image(roc_path),
            f"{split}/auc": roc_auc
        })
        
        print(f"  ✓ ROC curve saved: {roc_path}")
        print(f"  📊 AUC: {roc_auc:.4f}")
        
        # Save AUC
        with open(os.path.join(self.output_dir, 'metrics', f'auc_{split}.txt'), 'w') as f:
            f.write(f"AUC: {roc_auc:.4f}\n")
        
        return roc_auc
    
    def plot_training_history(self):
        """Plot training history from YOLO results"""
        print("\n📊 Plotting training history...")
        
        # Read results from YOLO training
        results_path = os.path.join(self.output_dir, 'runs', 'road_damage_detection', 'results.csv')
        
        if not os.path.exists(results_path):
            print("⚠️  Training results not found!")
            return
        
        # Load results
        df = pd.read_csv(results_path)
        df.columns = df.columns.str.strip()  # Remove whitespace from column names
        
        # Log training metrics to wandb
        for _, row in df.iterrows():
            metrics_to_log = {}
            for col in df.columns:
                if col != 'epoch' and pd.notna(row[col]):
                    try:
                        metrics_to_log[f"training/{col}"] = float(row[col])
                    except (ValueError, TypeError):
                        continue
            if metrics_to_log:
                wandb.log(metrics_to_log, step=int(row['epoch']) if 'epoch' in df.columns else None)
        
        # Create comprehensive training plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Loss plots
        if 'train/box_loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='blue')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Box Loss')
            axes[0, 0].set_title('Box Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
        
        if 'train/cls_loss' in df.columns:
            axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss', color='green')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Class Loss')
            axes[0, 1].set_title('Classification Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
        
        # mAP plots
        if 'metrics/mAP50(B)' in df.columns:
            axes[0, 2].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='orange')
            if 'metrics/mAP50-95(B)' in df.columns:
                axes[0, 2].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='red')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('mAP')
            axes[0, 2].set_title('Mean Average Precision')
            axes[0, 2].legend()
            axes[0, 2].grid(alpha=0.3)
        
        # Precision and Recall
        if 'metrics/precision(B)' in df.columns:
            axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='purple')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_title('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
        
        if 'metrics/recall(B)' in df.columns:
            axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='brown')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].set_title('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
        
        # Learning rate
        if 'lr/pg0' in df.columns:
            axes[1, 2].plot(df['epoch'], df['lr/pg0'], label='Learning Rate', color='black')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].set_title('Learning Rate Schedule')
            axes[1, 2].legend()
            axes[1, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        
        history_path = os.path.join(self.output_dir, 'visualizations', 'training_history.png')
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log training history to wandb
        wandb.log({"training/history": wandb.Image(history_path)})
        
        print(f"  ✓ Training history saved: {history_path}")
    
    def visualize_predictions(self, model, split='test', n_samples=10):
        """
        Visualize sample predictions
        
        Args:
            model: Trained model
            split: Dataset split ('test' or 'val')
            n_samples: Number of samples to visualize
        """
        print(f"\n🖼️  Visualizing predictions on {split} set...")
        
        # Get images
        img_dir = os.path.join(self.output_dir, f'{split}/images')
        images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                 if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Sample random images
        samples = random.sample(images, min(n_samples, len(images)))
        
        # Create grid
        n_cols = 5
        n_rows = (len(samples) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, img_path in enumerate(samples):
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Predict
            results = model.predict(img, conf=0.25, verbose=False)
            
            # Draw predictions
            if len(results) > 0 and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls)
                    conf = float(box.conf)
                    
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label = f"{self.VALID_CLASSES[cls]} {conf:.2f}"
                    cv2.putText(img_rgb, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            axes[idx].imshow(img_rgb)
            axes[idx].axis('off')
            axes[idx].set_title(os.path.basename(img_path), fontsize=8)
        
        # Hide empty subplots
        for idx in range(len(samples), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        pred_path = os.path.join(self.output_dir, 'visualizations', f'predictions_{split}.png')
        plt.savefig(pred_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log predictions visualization to wandb
        wandb.log({f"{split}/sample_predictions": wandb.Image(pred_path)})
        
        print(f"  ✓ Predictions visualization saved: {pred_path}")
    
    def run(self):
        """Run the complete training pipeline"""
        print("\n" + "="*70)
        print("🚀 ROAD DAMAGE DETECTION TRAINING PIPELINE")
        print("="*70)
        
        # Load dataset
        images, labels = self.load_dataset()
        
        # Split dataset
        n_train, n_val, n_test = self.split_dataset(images, labels)
        
        # Create YOLO config
        self.create_yolo_config()
        
        # Train model
        model, results, training_time = self.train()
        
        # Load best model for evaluation
        best_model_path = os.path.join(self.output_dir, 'models', 'best_model.pt')
        if os.path.exists(best_model_path):
            model = YOLO(best_model_path)
        
        # Plot training history
        self.plot_training_history()
        
        # Evaluate on test set
        test_metrics = self.evaluate_model(model, split='test')
        
        # Evaluate on validation set
        val_metrics = self.evaluate_model(model, split='val')
        
        # Visualize predictions
        self.visualize_predictions(model, split='test', n_samples=10)
        self.visualize_predictions(model, split='val', n_samples=10)
        
        # t-SNE visualization
        self.visualize_tsne(model, split='test')
        self.visualize_tsne(model, split='train')
        
        # Final summary
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"📁 Output directory: {self.output_dir}")
        print(f"💾 Best model: {best_model_path}")
        print(f"⏱️  Training time: {training_time/3600:.2f} hours")
        if test_metrics:
            print(f"🎯 Test Accuracy: {test_metrics.get('accuracy', 0)*100:.2f}%")
            print(f"📊 Test F1-Score: {test_metrics.get('f1_score', 0):.4f}")
        print(f"🔗 W&B Run: {wandb.run.url if wandb.run else 'Not initialized'}")
        print("="*70)
        
        # Finalize wandb run
        wandb.finish()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Road Damage Detection Training')
    
    # Get the workspace root (parent of the 'local' directory)
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent
    default_dataset_root = workspace_root / 'datasets'
    
    parser.add_argument('--dataset_root', type=str, default=str(default_dataset_root),
                       help='Root directory containing datasets')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (minimum 100)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training (default: 8, auto-adjusted based on device)')
    parser.add_argument('--img_size', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--wandb_api_key', type=str, 
                       default='405f59a92c4a17c4554bfa1af4a298b4ec94495c',
                       help='Weights & Biases API key for tracking')
    parser.add_argument('--dataset_limit', type=int, default=None,
                       help='Maximum number of images to use from dataset (None = use all)')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                       help='YOLOv8 model variant (default: yolov8s.pt - recommended for Mac M4)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = RoadDamageTrainer(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        wandb_api_key=args.wandb_api_key,
        dataset_limit=args.dataset_limit,
        model_name=args.model
    )
    
    # Run training
    trainer.run()


if __name__ == '__main__':
    main()

