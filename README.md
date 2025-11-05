# 🛣️ Hệ thống Phát hiện Hư hỏng Đường bộ sử dụng YOLOv8

## 📋 Thông tin Sinh viên

- **Họ và tên**: LÊ HỮU PHÚ
- **MSSV**: 102250404
- **Lớp**: K50.KMT_UD
- **Email**: phule9225@gmail.com

---

## 📖 Tổng quan dự án

Dự án này xây dựng hệ thống phát hiện hư hỏng đường bộ tự động sử dụng mô hình YOLOv8 (You Only Look Once version 8) - một kiến trúc deep learning hiện đại cho bài toán object detection. Hệ thống có khả năng phát hiện và phân loại 6 loại hư hỏng đường bộ phổ biến từ hình ảnh.

### 🎯 Mục tiêu

- Phát hiện và phân loại tự động các loại hư hỏng đường bộ
- Đạt độ chính xác cao trên bộ dữ liệu đa quốc gia
- Cung cấp các metrics và visualizations đầy đủ để đánh giá mô hình
- Tối ưu hóa cho môi trường Kaggle và local development

### ✨ Tính năng chính

- ✅ **Object Detection**: Phát hiện và định vị hư hỏng với bounding boxes
- ✅ **Multi-class Classification**: Phân loại 6 loại hư hỏng khác nhau
- ✅ **Comprehensive Metrics**: Đánh giá đầy đủ với Precision, Recall, F1-Score, mAP, AUC
- ✅ **Visualization Tools**: Training curves, confusion matrix, ROC curves, t-SNE plots
- ✅ **Multi-dataset Support**: Hỗ trợ training trên nhiều dataset từ các quốc gia khác nhau
- ✅ **Model Export**: Lưu best model để sử dụng cho inference

---

## 🔍 Các loại hư hỏng được phát hiện

Hệ thống phát hiện 6 loại hư hỏng đường bộ theo chuẩn RDD2022:

| Class | Mô tả | Tên tiếng Anh |
|-------|-------|---------------|
| **D00** | Vết nứt dọc | Lateral crack |
| **D10** | Vết nứt ngang | Longitudinal crack |
| **D20** | Vết nứt hình da cá sấu | Alligator crack |
| **D40** | Ổ gà | Pothole |
| **D43** | Vạch sang đường mờ | Cross walk blur |
| **D44** | Vạch kẻ trắng mờ | White line blur |

---

## 📊 Kết quả Training

### Thông số Training

- **Model**: YOLOv8n (Nano)
- **Epochs**: 50
- **Batch Size**: 16
- **Image Size**: 640x640
- **Dataset**: RDD2022 (India, Czech, China, Japan)

### Metrics cuối cùng (Epoch 50)

| Metric | Giá trị |
|--------|---------|
| **Precision** | 0.6267 (62.67%) |
| **Recall** | 0.5064 (50.64%) |
| **mAP@0.5** | 0.5432 (54.32%) |
| **mAP@0.5:0.95** | 0.2805 (28.05%) |
| **Train Box Loss** | 1.4459 |
| **Train Cls Loss** | 1.4011 |
| **Train DFL Loss** | 1.4472 |

### Training Progress

Mô hình đã được training trong 50 epochs với các metrics được cải thiện liên tục:

- **Loss giảm dần**: Từ ~2.1 xuống ~1.4 (giảm ~33%)
- **Precision tăng**: Từ ~30% lên ~63% (tăng hơn 2 lần)
- **Recall tăng**: Từ ~20% lên ~51% (tăng hơn 2.5 lần)
- **mAP@0.5 tăng**: Từ ~7% lên ~54% (tăng hơn 7 lần)

### Visualizations

Các biểu đồ và visualizations được lưu trong thư mục `outputs/`:

- 📈 **Training Curves**: Loss, mAP, Precision, Recall qua các epochs
- 📊 **Confusion Matrix**: Ma trận nhầm lẫn cho từng class
- 📉 **ROC Curve**: Đường cong ROC và AUC score
- 📈 **PR Curve**: Precision-Recall curve
- 🎨 **t-SNE Visualization**: Trực quan hóa feature space
- 🖼️ **Sample Predictions**: Kết quả dự đoán trên ảnh test

---

## 🏗️ Cấu trúc dự án

```
road_damage_detection/
├── build_models/              # Thư mục chứa code training
│   ├── local/                 # Code cho môi trường local
│   │   ├── train_road_damage_kaggle.py  # Script training chính
│   │   ├── inference.py       # Script inference
│   │   ├── config.yaml        # File cấu hình
│   │   └── README.md          # Tài liệu chi tiết
│   ├── road_damage_detection_v1.ipynb  # Notebook v1
│   └── road_damage_detection_v2.ipynb  # Notebook v2
├── datasets/                  # Thư mục dataset
│   ├── DATASET.md            # Thông tin dataset
│   └── tests/                # Ảnh test
├── outputs/                   # Kết quả training
│   ├── weights/              # Model weights
│   │   ├── best.pt           # Best model
│   │   └── last.pt           # Last checkpoint
│   ├── results.csv           # Training metrics
│   ├── confusion_matrix.png  # Confusion matrix
│   ├── PR_curve.png          # PR curve
│   ├── ROC_curve.png         # ROC curve
│   └── results.png           # Training curves
├── demo/                     # Demo application
│   └── demo.py               # Demo script
├── papers-research/          # Papers và nghiên cứu
└── README.md                 # File này
```

---

## 🚀 Cài đặt và Sử dụng

### Yêu cầu hệ thống

- **Python**: 3.8 - 3.11
- **PyTorch**: >= 1.13.0
- **CUDA**: Không bắt buộc (có thể chạy trên CPU)
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB+)
- **Disk Space**: Tối thiểu 10GB

### Cài đặt dependencies

```bash
# Cài đặt PyTorch (chọn version phù hợp với hệ thống)
pip install torch torchvision torchaudio

# Cài đặt các thư viện khác
pip install ultralytics opencv-python numpy pandas matplotlib seaborn scikit-learn tqdm PyYAML lxml
```

### Training

#### 1. Chuẩn bị Dataset

Dataset RDD2022 cần được tổ chức theo cấu trúc:

```
dataset/
├── India/
│   ├── train/
│   │   ├── images/
│   │   └── annotations/
│   └── test/
│       └── images/
├── Czech/
├── China_MotorBike/
├── China_Drone/
└── Japan/
```

#### 2. Chạy Training

```bash
cd build_models/local

python train_road_damage_kaggle.py
```

Hoặc với các tham số tùy chỉnh:

```python
# Trong file train_road_damage_kaggle.py, chỉnh sửa:
DATASET_ROOT = '/path/to/your/dataset'
EPOCHS = 50
BATCH_SIZE = 16
MODEL_NAME = 'yolov8n.pt'  # hoặc yolov8s, yolov8m, yolov8l, yolov8x
```

### Inference

Sử dụng model đã train để dự đoán trên ảnh mới:

```bash
python inference.py \
    --model outputs/weights/best.pt \
    --image path/to/image.jpg \
    --output_dir ./results
```

---

## 📈 Kết quả chi tiết

### Training History

Quá trình training trong 50 epochs cho thấy:

- **Loss Convergence**: Model đã hội tụ tốt với loss giảm ổn định
- **mAP Improvement**: mAP@0.5 tăng từ 7.45% (epoch 1) lên 54.32% (epoch 50)
- **Stable Training**: Validation loss không tăng, cho thấy không có overfitting nghiêm trọng

### Performance Analysis

1. **Precision (0.6267)**: Mô hình có độ chính xác trung bình, khoảng 63% predictions là đúng
2. **Recall (0.5064)**: Mô hình phát hiện được khoảng 51% tổng số hư hỏng thực tế
3. **mAP@0.5 (0.5432)**: Mean Average Precision ở IoU threshold 0.5 đạt 54.32%
4. **mAP@0.5:0.95 (0.2805)**: mAP trung bình ở nhiều IoU thresholds

### Cải thiện có thể

Để nâng cao performance, có thể:

- ✅ Tăng số epochs (100-150 epochs)
- ✅ Sử dụng model lớn hơn (yolov8m, yolov8l)
- ✅ Tăng batch size (nếu có đủ GPU memory)
- ✅ Data augmentation mạnh hơn
- ✅ Fine-tuning hyperparameters (learning rate, optimizer)
- ✅ Sử dụng thêm dataset hoặc data augmentation

---

## 🔬 Đánh giá và Metrics

### Metrics được tính toán

1. **Object Detection Metrics**:
   - mAP@0.5: Mean Average Precision tại IoU=0.5
   - mAP@0.5:0.95: mAP trung bình tại nhiều IoU thresholds (0.5 đến 0.95)

2. **Classification Metrics**:
   - Precision: Độ chính xác của predictions
   - Recall: Khả năng phát hiện (sensitivity)
   - F1-Score: Trung bình điều hòa của Precision và Recall
   - Accuracy: Độ chính xác tổng thể

3. **Visualization Metrics**:
   - ROC Curve: Receiver Operating Characteristic curve
   - AUC: Area Under the ROC Curve
   - Confusion Matrix: Ma trận nhầm lẫn cho từng class
   - t-SNE: Feature space visualization

### Output Files

Sau khi training, các file kết quả được lưu trong `outputs/`:

```
outputs/
├── weights/
│   ├── best.pt              # Model tốt nhất
│   └── last.pt              # Checkpoint cuối cùng
├── results.csv              # Metrics chi tiết từng epoch
├── confusion_matrix.png     # Confusion matrix
├── confusion_matrix_normalized.png
├── PR_curve.png             # Precision-Recall curve
├── ROC_curve.png            # ROC curve
├── F1_curve.png             # F1-Score curve
├── R_curve.png              # Recall curve
├── P_curve.png              # Precision curve
├── results.png              # Training curves tổng hợp
└── train_batch*.jpg         # Sample training images
```

---

## 🛠️ Công nghệ sử dụng

- **Framework**: YOLOv8 (Ultralytics)
- **Deep Learning**: PyTorch
- **Computer Vision**: OpenCV
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Metrics**: scikit-learn
- **Data Format**: YOLO format (YAML + TXT annotations)

---

## 📚 Tài liệu tham khảo

1. **Dataset**: 
   - RDD2022: [https://rdd2022.sekilab.global/](https://rdd2022.sekilab.global/)
   - Road Damage Dataset 2022 với ~26,000 ảnh từ nhiều quốc gia

2. **YOLOv8 Documentation**:
   - [Ultralytics YOLOv8](https://docs.ultralytics.com/)
   - [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)

3. **Papers**:
   - Các paper nghiên cứu về road damage detection trong thư mục `papers-research/`

---

## 🎓 Ứng dụng thực tế

Hệ thống này có thể được ứng dụng trong:

- 🏗️ **Bảo trì hạ tầng**: Tự động phát hiện hư hỏng để lập kế hoạch sửa chữa
- 🚗 **Smart City**: Tích hợp vào hệ thống quản lý giao thông thông minh
- 📱 **Mobile Apps**: Ứng dụng di động để người dùng báo cáo hư hỏng
- 🚁 **Drone Inspection**: Phát hiện hư hỏng từ ảnh chụp bằng drone
- 📊 **Data Analysis**: Phân tích xu hướng hư hỏng theo thời gian và địa điểm

---

## 📝 License

Dự án này được phát triển cho mục đích học tập và nghiên cứu.

---

## 👨‍💻 Tác giả

**LÊ HỮU PHÚ**
- MSSV: 102250404
- Lớp: K50.KMT_UD
- Email: phule9225@gmail.com

---

## 🙏 Lời cảm ơn

- Cảm ơn Ultralytics team cho YOLOv8 framework
- Cảm ơn RDD2022 team cho bộ dataset chất lượng
- Cảm ơn Kaggle platform cho môi trường training

---

**Cập nhật lần cuối**: 2025

**Phiên bản**: 1.0
