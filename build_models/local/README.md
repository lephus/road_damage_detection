# Road Damage Detection Training - Local Environment

Hệ thống training model phát hiện hư hỏng đường bộ sử dụng YOLOv8, được tối ưu hóa cho Mac M4 (Apple Silicon).

## 📋 Mục lục

- [Tính năng](#tính-năng)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
- [Sử dụng](#sử-dụng)
- [Kết quả](#kết-quả)
- [Tùy chỉnh](#tùy-chỉnh)

## ✨ Tính năng

### Yêu cầu từ bài toán:
- ✅ **Log lại history train và plot trực quan hóa**: Lưu đầy đủ training history và tạo các biểu đồ loss, mAP, precision, recall
- ✅ **Lưu best model**: Lưu model tốt nhất dưới dạng `.pt` (PyTorch)
- ✅ **t-SNE visualization**: Hiển thị t-SNE ở layer cuối cùng cho cả tập train và test
- ✅ **Metrics đầy đủ**: F1-score, Recall, Precision, Accuracy
- ✅ **ROC & AUC**: Vẽ đường cong ROC và tính AUC
- ✅ **Time tracking**: Ghi lại thời gian training
- ✅ **Minimum 100 epochs**: Mặc định 100 epochs, có thể tùy chỉnh
- ✅ **Target accuracy ≥ 85%**: Hiển thị thông báo khi đạt mục tiêu

### Tính năng kỹ thuật:
- 🚀 Tối ưu hóa cho Apple Silicon M4 (MPS acceleration)
- 📊 Comprehensive metrics và visualization
- 💾 Tự động lưu best model và checkpoints
- 🎯 Multi-class object detection với 6 loại hư hỏng
- 🌍 Hỗ trợ multi-dataset (India, Czech, China, Japan)
- 📈 Real-time training progress tracking

## 🖥️ Yêu cầu hệ thống

### Phần cứng:
- **Mac M4** hoặc các chip Apple Silicon khác (M1, M2, M3)
- RAM: Tối thiểu 16GB (khuyến nghị 32GB+)
- Ổ cứng: Tối thiểu 50GB trống cho dataset và outputs

### Phần mềm:
- **macOS**: Big Sur (11.0) hoặc mới hơn
- **Python**: 3.8 - 3.11
- **Homebrew** (khuyến nghị để cài đặt Python)

## 🔧 Cài đặt

### Bước 1: Cài đặt Python (nếu chưa có)

```bash
# Sử dụng Homebrew
brew install python@3.10

# Hoặc tải từ python.org
# https://www.python.org/downloads/
```

### Bước 2: Clone hoặc tải project

```bash
cd /Users/lehuuphu/Downloads/DUT-ths/ComputerVision/road_damage_detection/build_models/local
```

### Bước 3: Chuẩn bị dataset (xem phần [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu))

## 📂 Chuẩn bị dữ liệu

### Cấu trúc thư mục dataset:

```
/path/to/your/dataset/
├── India/
│   ├── train/
│   │   ├── images/
│   │   └── annotations/
│   │       └── xmls/
│   └── test/
│       └── images/
├── Czech/
│   ├── train/
│   │   ├── images/
│   │   └── annotations/
│   │       └── xmls/
│   └── test/
│       └── images/
├── China_MotorBike/
│   ├── train/
│   │   ├── images/
│   │   └── annotations/
│   │       └── xmls/
│   └── test/
│       └── images/
├── China_Drone/
│   └── train/
│       ├── images/
│       └── annotations/
│           └── xmls/
└── Japan/
    ├── train/
    │   ├── images/
    │   └── annotations/
    │       └── xmls/
    └── test/
        └── images/
```

### Các lớp hư hỏng (RDD2022):
- **D00**: Lateral crack (vết nứt dọc)
- **D10**: Longitudinal crack (vết nứt ngang)
- **D20**: Alligator crack (vết nứt hình da cá sấu)
- **D40**: Pothole (ổ gà)
- **D43**: Cross walk blur (vạch sang đường mờ)
- **D44**: Whiteline blur (vạch kẻ trắng mờ)

## 🚀 Sử dụng

### Cách 1: Sử dụng script tự động (Khuyến nghị)

```bash
# Đặt đường dẫn dataset
export DATASET_ROOT=/path/to/your/RDD2022/dataset

# Chạy script
./setup_and_train.sh
```

Script sẽ tự động:
1. ✅ Kiểm tra Python và dependencies
2. ✅ Tạo virtual environment
3. ✅ Cài đặt tất cả packages cần thiết
4. ✅ Kiểm tra dataset
5. ✅ Chạy training
6. ✅ Tạo visualizations và metrics

### Cách 2: Chạy thủ công

```bash
# 1. Tạo virtual environment
python3 -m venv road_damage_env
source road_damage_env/bin/activate

# 2. Cài đặt dependencies
pip install --upgrade pip
pip install torch torchvision
pip install ultralytics opencv-python-headless numpy pandas matplotlib seaborn scikit-learn tqdm PyYAML lxml

# 3. Chạy training
python train_road_damage.py \
    --dataset_root /path/to/your/dataset \
    --output_dir ./outputs \
    --epochs 100 \
    --batch_size 16 \
    --model yolov8m.pt
```

### Tùy chỉnh tham số:

```bash
# Sử dụng model lớn hơn (chính xác hơn nhưng chậm hơn)
export MODEL=yolov8l.pt

# Tăng số epochs
export EPOCHS=150

# Điều chỉnh batch size (giảm nếu hết RAM)
export BATCH_SIZE=8

# Sau đó chạy
./setup_and_train.sh
```

### Tham số dòng lệnh:

```bash
python train_road_damage.py \
    --dataset_root /path/to/dataset \
    --output_dir ./outputs \
    --epochs 100 \              # Số epochs (tối thiểu 100)
    --batch_size 16 \           # Batch size
    --img_size 640 \            # Kích thước ảnh
    --model yolov8m.pt          # Model variant
```

### Model variants:
- `yolov8n.pt`: Nano - Nhanh nhất, nhẹ nhất
- `yolov8s.pt`: Small - Cân bằng tốc độ/độ chính xác
- `yolov8m.pt`: Medium - **Khuyến nghị** (mặc định)
- `yolov8l.pt`: Large - Chính xác cao
- `yolov8x.pt`: Extra Large - Chính xác nhất nhưng chậm nhất

## 📊 Kết quả

Sau khi training hoàn thành, kết quả được lưu trong thư mục `outputs/`:

```
outputs/
├── models/
│   └── best_model.pt              # ⭐ Best model (sử dụng cho inference)
├── metrics/
│   ├── dataset_stats.json         # Thống kê dataset
│   ├── metrics_test.json          # Metrics trên tập test
│   ├── metrics_val.json           # Metrics trên tập validation
│   ├── classification_report_test.txt  # Chi tiết classification
│   ├── auc_test.txt               # AUC score
│   └── training_time.txt          # Thời gian training
├── visualizations/
│   ├── training_history.png       # 📈 Biểu đồ training (loss, mAP, precision, recall)
│   ├── confusion_matrix_test.png  # 📊 Confusion matrix
│   ├── roc_curve_test.png         # 📉 ROC curve
│   ├── tsne_test.png              # 🎨 t-SNE visualization (test set)
│   ├── tsne_train.png             # 🎨 t-SNE visualization (train set)
│   ├── predictions_test.png       # 🖼️ Sample predictions (test)
│   └── predictions_val.png        # 🖼️ Sample predictions (validation)
├── runs/
│   └── road_damage_detection/
│       ├── weights/
│       │   ├── best.pt            # Best weights
│       │   └── last.pt            # Last weights
│       ├── results.csv            # Training results (mỗi epoch)
│       └── [other YOLO outputs]
├── train/                         # Processed training data
├── val/                           # Processed validation data
└── test/                          # Processed test data
```

### Metrics được tính toán:

1. **Accuracy**: Độ chính xác tổng thể
2. **Precision**: Độ chính xác của predictions
3. **Recall**: Khả năng phát hiện (sensitivity)
4. **F1-Score**: Trung bình điều hòa của Precision và Recall
5. **ROC-AUC**: Area Under the ROC Curve
6. **mAP@0.5**: Mean Average Precision ở IoU threshold 0.5
7. **mAP@0.5:0.95**: Mean Average Precision ở các IoU thresholds

### Visualizations:

1. **Training History**: Loss curves, mAP curves, Precision/Recall curves
2. **Confusion Matrix**: Hiển thị phân loại đúng/sai cho từng class
3. **ROC Curve**: Đánh giá performance ở các thresholds
4. **t-SNE**: Visualize feature space (phân tách các classes)
5. **Sample Predictions**: Kết quả dự đoán trên ảnh thực tế

## 🎯 Mục tiêu và đánh giá

### Mục tiêu:
- ✅ **Accuracy ≥ 85%** trên tập test
- ✅ **F1-Score cao** (cân bằng precision và recall)
- ✅ **AUC gần 1.0** (khả năng phân biệt tốt)

### Thời gian training (ước tính trên Mac M4):
- **100 epochs với YOLOv8m**: ~4-8 giờ (tùy vào số lượng ảnh)
- **150 epochs với YOLOv8l**: ~6-12 giờ

## ⚙️ Tùy chỉnh

### Điều chỉnh hyperparameters:

Sửa file `train_road_damage.py` tại hàm `train()`:

```python
results = model.train(
    data=...,
    epochs=self.epochs,
    imgsz=self.img_size,
    batch=self.batch_size,
    lr0=0.001,              # Learning rate ban đầu
    cos_lr=True,            # Cosine learning rate scheduler
    patience=20,            # Early stopping patience
    augment=True,           # Data augmentation
    amp=True,               # Mixed precision
    # ... các tham số khác
)
```

### Thêm/bớt datasets:

Sửa file `train_road_damage.py` tại `self.dataset_paths`:

```python
self.dataset_paths = {
    'India': { ... },
    'Czech': { ... },
    # Thêm dataset mới
    'NewCountry': {
        'train_images': os.path.join(dataset_root, 'NewCountry/train/images'),
        'train_annotations': os.path.join(dataset_root, 'NewCountry/train/annotations'),
        'test_images': os.path.join(dataset_root, 'NewCountry/test/images')
    }
}
```

### Thay đổi số lượng classes:

Sửa `self.VALID_CLASSES` trong `train_road_damage.py`:

```python
self.VALID_CLASSES = ['D00', 'D10', 'D20', 'D40', 'D43', 'D44']  # Thêm/bớt classes
```

## 🐛 Troubleshooting

### Lỗi: "MPS backend not available"
- **Giải pháp**: Cập nhật PyTorch lên phiên bản mới nhất hỗ trợ MPS
```bash
pip install --upgrade torch torchvision
```

### Lỗi: "Out of memory"
- **Giải pháp**: Giảm batch size
```bash
export BATCH_SIZE=8  # hoặc 4
./setup_and_train.sh
```

### Lỗi: "Dataset not found"
- **Giải pháp**: Kiểm tra lại đường dẫn dataset
```bash
# Kiểm tra cấu trúc
ls -la /path/to/your/dataset/India/train/images
ls -la /path/to/your/dataset/India/train/annotations/xmls
```

### Training quá chậm:
- **Giải pháp 1**: Sử dụng model nhỏ hơn (`yolov8s.pt` hoặc `yolov8n.pt`)
- **Giải pháp 2**: Giảm image size
```bash
python train_road_damage.py --img_size 416 ...
```
- **Giải pháp 3**: Giảm số epochs (không khuyến nghị nếu muốn accuracy cao)

### Accuracy thấp (< 85%):
- **Giải pháp 1**: Tăng số epochs (150-200)
- **Giải pháp 2**: Sử dụng model lớn hơn (`yolov8l.pt` hoặc `yolov8x.pt`)
- **Giải pháp 3**: Thêm data augmentation
- **Giải pháp 4**: Điều chỉnh learning rate

## 📚 Tài liệu tham khảo

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [RDD2022 Dataset](https://github.com/sekilab/RoadDamageDetector)

## 📝 Notes

- Model được tối ưu hóa cho Apple Silicon (MPS acceleration)
- Hỗ trợ multi-class object detection
- Tự động data preprocessing và augmentation
- Comprehensive evaluation metrics
- Real-time progress tracking

## 🤝 Hỗ trợ

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra phần [Troubleshooting](#-troubleshooting)
2. Xem logs trong quá trình chạy
3. Kiểm tra file `outputs/runs/road_damage_detection/` để xem chi tiết lỗi

---

**Chúc bạn training thành công! 🚀**

