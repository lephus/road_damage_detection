"""
Road Damage Detection Training Script (Kaggle-Optimized)
Fixed: wandb, NumPy, paths, FileNotFoundError, auto-create folders
"""

import os
import sys
import time
import json
import random
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm

# Machine Learning
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

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class RoadDamageTrainer:
    def __init__(self, dataset_root, output_dir, epochs=100, batch_size=16, img_size=640, max_images_per_dataset=None):
        self.dataset_root = dataset_root
        self.output_dir = output_dir
        self.epochs = max(epochs, 100)
        self.batch_size = batch_size
        self.img_size = img_size
        self.max_images_per_dataset = max_images_per_dataset
        
        self.VALID_CLASSES = ['D00', 'D10', 'D20', 'D40', 'D43', 'D44']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.VALID_CLASSES)}
        self.num_classes = len(self.VALID_CLASSES)
        
        self.dataset_paths = {
            'India': {
                'train_images': '/kaggle/input/dut-rdd/RDD2022_India/India/test/images',
                'train_annotations': '/kaggle/input/dut-rdd/RDD2022_India/India/test/annotations',
            },
            'Czech': {
                'train_images': '/kaggle/input/rdd2022-more/RDD2022_Czech/Czech/train/images',
                'train_annotations': '/kaggle/input/dut-rdd/RDD2022_Czech/train/annotations',
            },
            'China_MotorBike': {
                'train_images': '/kaggle/input/dut-rdd/RDD2022_China_MotorBike/China_MotorBike/train/images',
                'train_annotations': '/kaggle/input/dut-rdd/RDD2022_China_MotorBike/China_MotorBike/train/annotations',
            },
            'China_Drone': {
                'train_images': '/kaggle/input/dut-rdd/RDD2022_China_Drone/China_Drone/train/images',
                'train_annotations': '/kaggle/input/dut-rdd/RDD2022_China_Drone/China_Drone/train/annotations',
            },
            'Japan': {
                'train_images': '/kaggle/input/rdd2022-more/RDD2022_Japan/Japan/train/images',
                'train_annotations': '/kaggle/input/rdd2022-more/RDD2022_Japan/Japan/train/annotations',
            },
        }
        
        self.setup_directories()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        print(f"Training for {self.epochs} epochs")
        if self.max_images_per_dataset:
            print(f"Limited mode: Max {self.max_images_per_dataset} images per dataset")
        else:
            print(f"Loading all available images")

    def setup_directories(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.output_dir, f'{split}/images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, f'{split}/labels'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)
        print(f"Output directories ready: {self.output_dir}")

    def parse_xml_to_yolo(self, xml_path, img_width, img_height):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes, labels = [], []
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
            if xmax <= xmin or ymax <= ymin:
                continue
            center_x = (xmin + xmax) / 2 / img_width
            center_y = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            boxes.append([center_x, center_y, width, height])
            labels.append(cls_idx)
        return boxes, labels

    def load_dataset(self):
        print("\nLoading dataset...")
        all_images, all_labels, dataset_stats = [], [], {}
        
        for name, paths in self.dataset_paths.items():
            img_dir = paths['train_images']
            ann_dir = paths['train_annotations']
            if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
                print(f"Skipping {name}: Not found")
                continue
            
            count = 0
            img_files = os.listdir(img_dir)
            if self.max_images_per_dataset:
                img_files = img_files[:self.max_images_per_dataset]
            
            for img_file in tqdm(img_files, desc=f"Loading {name}"):
                if self.max_images_per_dataset and count >= self.max_images_per_dataset:
                    break
                if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                
                img_path = os.path.join(img_dir, img_file)
                xml_path = os.path.join(ann_dir, 'xmls', Path(img_file).stem + '.xml')
                if not os.path.exists(xml_path):
                    continue
                
                img = cv2.imread(img_path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                
                try:
                    boxes, labels = self.parse_xml_to_yolo(xml_path, w, h)
                    if boxes:
                        all_images.append(img_path)
                        all_labels.append((boxes, labels, xml_path))
                        count += 1
                except:
                    continue
            
            dataset_stats[name] = count
            print(f"  {name}: {count} images")
        
        print(f"\nTotal valid pairs: {len(all_images)}")
        with open(os.path.join(self.output_dir, 'metrics', 'dataset_stats.json'), 'w') as f:
            json.dump(dataset_stats, f, indent=2)
        
        return all_images, all_labels

    def preprocess_and_save(self, images, labels, split='train'):
        img_out = os.path.join(self.output_dir, f'{split}/images')
        lbl_out = os.path.join(self.output_dir, f'{split}/labels')
        for img_path, (boxes, lbls, _) in tqdm(zip(images, labels), desc=f"Processing {split}", total=len(images)):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.img_size, self.img_size))
            name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(img_out, name), img)
            with open(os.path.join(lbl_out, Path(name).stem + '.txt'), 'w') as f:
                for box, lbl in zip(boxes, lbls):
                    f.write(f"{lbl} {' '.join(map(str, box))}\n")

    def split_dataset(self, images, labels):
        print("\nSplitting dataset...")
        train_imgs, test_imgs, train_lbls, test_lbls = train_test_split(images, labels, test_size=0.1, random_state=42)
        train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(train_imgs, train_lbls, test_size=0.1111, random_state=42)
        self.preprocess_and_save(train_imgs, train_lbls, 'train')
        self.preprocess_and_save(val_imgs, val_lbls, 'val')
        self.preprocess_and_save(test_imgs, test_lbls, 'test')
        print(f"  Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")
        return len(train_imgs), len(val_imgs), len(test_imgs)

    def create_yolo_config(self):
        config = f"""path: {self.output_dir}
train: train/images
val: val/images
test: test/images

nc: {self.num_classes}
names: {self.VALID_CLASSES}
"""
        path = os.path.join(self.output_dir, 'data.yaml')
        with open(path, 'w') as f:
            f.write(config)
        print(f"YOLO config: {path}")

    def train(self, model_name='yolov8s.pt'):
        print(f"\nStarting training with {model_name}...")
        os.environ['WANDB_DISABLED'] = 'true'  # Tắt wandb
        
        model = YOLO(model_name)
        start = time.time()
        
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
            amp=True,
            save=True,
            save_period=10,
            project=self.output_dir,   # Lưu vào output_dir/train/
            name='train',
            exist_ok=True,
            verbose=True,
            plots=True
        )
        
        training_time = time.time() - start
        print(f"Training completed in {training_time/60:.1f} min")
        
        with open(os.path.join(self.output_dir, 'metrics', 'training_time.txt'), 'w') as f:
            f.write(f"{training_time:.2f}\n")
        
        return model, results, training_time

    def extract_features(self, model, images, max_samples=2000):
        print("\nExtracting features for t-SNE...")
        features, labels = [], []
        samples = random.sample(images, min(len(images), max_samples))
        model.eval()
        for path in tqdm(samples, desc="Extracting"):
            try:
                img = cv2.resize(cv2.imread(path), (self.img_size, self.img_size))
                lbl_path = path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
                lbl = -1
                if os.path.exists(lbl_path):
                    with open(lbl_path) as f:
                        line = f.readline().strip()
                        if line:
                            lbl = int(line.split()[0])
                labels.append(lbl)
                
                res = model.predict(img, verbose=False)
                if res and len(res[0].boxes) > 0:
                    feats = [box.xywh[0].cpu().numpy() for box in res[0].boxes]
                    features.append(np.mean(feats, axis=0))
                else:
                    features.append(np.zeros(4))
            except:
                continue
        return np.array(features), np.array(labels)

    def visualize_tsne(self, model, split='test'):
        print(f"\nt-SNE for {split}...")
        img_dir = os.path.join(self.output_dir, f'{split}/images')
        images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not images:
            print("No images for t-SNE")
            return
        feats, lbls = self.extract_features(model, images)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feats)-1), n_iter=1000)
        emb = tsne.fit_transform(feats)
        
        plt.figure(figsize=(12, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes + 1))
        for i in range(self.num_classes):
            mask = lbls == i
            plt.scatter(emb[mask, 0], emb[mask, 1], c=[colors[i]], label=self.VALID_CLASSES[i], alpha=0.6, s=50)
        if np.any(lbls == -1):
            plt.scatter(emb[lbls == -1, 0], emb[lbls == -1, 1], c='gray', label='No Detection', alpha=0.3, s=30, marker='x')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f't-SNE ({split.upper()})')
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'visualizations', f'tsne_{split}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    def evaluate_model(self, model, split='test'):
        print(f"\nEvaluating {split}...")
        img_dir = os.path.join(self.output_dir, f'{split}/images')
        images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        y_true, y_pred, y_scores = [], [], []
        
        for path in tqdm(images, desc="Predicting"):
            img = cv2.imread(path)
            if img is None:
                continue
            res = model.predict(img, conf=0.25, verbose=False)
            
            lbl_path = path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
            if os.path.exists(lbl_path):
                with open(lbl_path) as f:
                    for line in f:
                        y_true.append(int(line.split()[0]))
            
            if res and len(res[0].boxes) > 0:
                for box in res[0].boxes:
                    y_pred.append(int(box.cls))
                    y_scores.append(float(box.conf))
        
        if not y_true or not y_pred:
            print("No valid predictions!")
            return {}
        
        min_len = min(len(y_true), len(y_pred))
        y_true, y_pred, y_scores = y_true[:min_len], y_pred[:min_len], y_scores[:min_len]
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"\nMetrics: Acc: {acc*100:.2f}% | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
        metrics = {'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1_score': float(f1)}
        with open(os.path.join(self.output_dir, 'metrics', f'metrics_{split}.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        report = classification_report(y_true, y_pred, target_names=self.VALID_CLASSES, zero_division=0)
        print(f"\nClassification Report:\n{report}")
        with open(os.path.join(self.output_dir, 'metrics', f'report_{split}.txt'), 'w') as f:
            f.write(report)
        
        self.plot_confusion_matrix(y_true, y_pred, split)
        self.plot_roc_curves(y_true, y_scores, split)
        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, split):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.VALID_CLASSES, yticklabels=self.VALID_CLASSES)
        plt.title(f'Confusion Matrix ({split.upper()})')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'visualizations', f'cm_{split}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  CM saved: {path}")

    def plot_roc_curves(self, y_true, y_scores, split):
        y_true_bin = [1 if y >= 0 else 0 for y in y_true]
        y_scores = np.array(y_scores)
        fpr, tpr, _ = roc_curve(y_true_bin, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0,1],[0,1], 'k--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'ROC ({split.upper()})')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'visualizations', f'roc_{split}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ROC saved: {path} (AUC: {roc_auc:.4f})")

    def plot_training_history(self):
        print("\nPlotting training history...")
        csv_path = os.path.join(self.output_dir, 'train', 'results.csv')
        if not os.path.exists(csv_path):
            print("No results.csv found")
            return
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        plots = [
            ('train/box_loss', 'Box Loss'), ('train/cls_loss', 'Class Loss'), ('metrics/mAP50(B)', 'mAP@0.5'),
            ('metrics/precision(B)', 'Precision'), ('metrics/recall(B)', 'Recall'), ('lr/pg0', 'LR')
        ]
        for ax, (col, title) in zip(axes.flat, plots):
            if col in df.columns:
                ax.plot(df['epoch'], df[col])
                ax.set_title(title)
                ax.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'visualizations', 'history.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  History saved: {path}")

    def visualize_predictions(self, model, split='test', n_samples=10):
        print(f"\nVisualizing {split} predictions...")
        img_dir = os.path.join(self.output_dir, f'{split}/images')
        images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        samples = random.sample(images, min(n_samples, len(images)))
        
        cols = 5
        rows = (len(samples) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
        axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
        
        for idx, path in enumerate(samples):
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            res = model.predict(img, conf=0.25, verbose=False)
            if res and len(res[0].boxes) > 0:
                for box in res[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls)
                    conf = float(box.conf)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img, f"{self.VALID_CLASSES[cls]} {conf:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(os.path.basename(path), fontsize=8)
        for idx in range(len(samples), len(axes)):
            axes[idx].axis('off')
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'visualizations', f'pred_{split}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Predictions saved: {path}")

    def run(self):
        print("\n" + "="*70)
        print("ROAD DAMAGE DETECTION PIPELINE")
        print("="*70)
        
        images, labels = self.load_dataset()
        if len(images) == 0:
            print("ERROR: Không có ảnh nào được tải. Kiểm tra dataset.")
            return
        
        self.split_dataset(images, labels)
        self.create_yolo_config()
        model, _, training_time = self.train('yolov8s.pt')
        
        # TỰ ĐỘNG TẠO THƯ MỤC + SAO CHÉP MODEL
        weights_dir = os.path.join(self.output_dir, 'train', 'weights')
        best_pt = os.path.join(weights_dir, 'best.pt')
        last_pt = os.path.join(weights_dir, 'last.pt')
        final_model = os.path.join(self.output_dir, 'models', 'best_model.pt')
        
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(os.path.dirname(final_model), exist_ok=True)
        
        if os.path.exists(best_pt):
            shutil.copy(best_pt, final_model)
            print(f"Best model copied: {final_model}")
        elif os.path.exists(last_pt):
            shutil.copy(last_pt, final_model)
            print(f"Using last.pt: {final_model}")
        else:
            try:
                model.save(final_model)
                print(f"Model saved manually: {final_model}")
            except:
                print("Cannot save model!")
                final_model = None
        
        if final_model and os.path.exists(final_model):
            model = YOLO(final_model)
            print(f"Loaded best model: {final_model}")
        else:
            print("Using in-memory model")
        
        self.plot_training_history()
        self.evaluate_model(model, 'test')
        self.evaluate_model(model, 'val')
        self.visualize_predictions(model, split='test', n_samples=10)
        self.visualize_predictions(model, split='val', n_samples=10)
        self.visualize_tsne(model, 'test')
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print(f"Output: {self.output_dir}")
        print(f"Best model: {final_model if final_model else 'Not saved'}")
        print(f"Time: {training_time/60:.1f} min")
        print("="*70)


def main():
    DATASET_ROOT = '/kaggle/input'
    OUTPUT_DIR = '/kaggle/working/outputs'
    EPOCHS = 200
    BATCH_SIZE = 16
    IMG_SIZE = 640
    MAX_IMAGES = 2000  # None = full

    print("\n" + "="*70)
    print("CONFIG")
    print("="*70)
    print(f"Root: {DATASET_ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Size: {IMG_SIZE}")
    print(f"Max images: {MAX_IMAGES}")
    print("="*70 + "\n")
    
    if not os.path.exists(DATASET_ROOT):
        print(f"ERROR: {DATASET_ROOT} not found!")
        return
    
    trainer = RoadDamageTrainer(
        dataset_root=DATASET_ROOT,
        output_dir=OUTPUT_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        max_images_per_dataset=MAX_IMAGES
    )
    trainer.run()


if __name__ == '__main__':
    main()