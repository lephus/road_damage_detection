# 🚀 Quick Start Guide

Hướng dẫn nhanh để bắt đầu training model phát hiện hư hỏng đường bộ trên Mac M4.

## ⚡ Cài đặt nhanh (5 phút)

### Bước 1: Chuẩn bị dataset

1. Download RDD2022 dataset từ [https://github.com/sekilab/RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector)

2. Giải nén vào một thư mục, ví dụ: `/Users/yourname/datasets/RDD2022/`

3. Kiểm tra cấu trúc thư mục:
```bash
ls -la /Users/yourname/datasets/RDD2022/
# Bạn sẽ thấy: India/, Czech/, China_MotorBike/, China_Drone/, Japan/
```

### Bước 2: Set dataset path

```bash
export DATASET_ROOT=/Users/yourname/datasets/RDD2022
```

### Bước 3: Chạy training

```bash
cd /Users/lehuuphu/Downloads/DUT-ths/ComputerVision/road_damage_detection/build_models/local
./setup_and_train.sh
```

**Xong!** Script sẽ tự động:
- ✅ Cài đặt tất cả dependencies
- ✅ Tạo virtual environment
- ✅ Chạy training
- ✅ Tạo metrics và visualizations

## 📊 Theo dõi tiến trình

Training sẽ hiển thị:
```
Epoch 1/100: 100%|██████████| 1234/1234 [12:34<00:00, 1.64it/s]
  Loss: 0.245
  mAP@0.5: 0.678
  Precision: 0.734
  Recall: 0.689
```

## 🎯 Kết quả

Sau khi hoàn thành, kiểm tra:

```bash
# Best model
ls -lh outputs/models/best_model.pt

# Metrics
cat outputs/metrics/metrics_test.json

# Visualizations
open outputs/visualizations/training_history.png
open outputs/visualizations/tsne_test.png
```

## 🔍 Sử dụng model để inference

```bash
# Activate virtual environment (nếu chưa activate)
source road_damage_env/bin/activate

# Detect trên 1 ảnh
python inference.py \
    --model outputs/models/best_model.pt \
    --image /path/to/road_image.jpg \
    --output_dir ./results \
    --show

# Detect trên nhiều ảnh
python inference.py \
    --model outputs/models/best_model.pt \
    --image_dir /path/to/images/ \
    --output_dir ./results
```

## ⚙️ Tùy chỉnh nhanh

### Training lâu quá?

```bash
# Sử dụng model nhỏ hơn
export MODEL=yolov8s.pt
./setup_and_train.sh
```

### Out of memory?

```bash
# Giảm batch size
export BATCH_SIZE=8
./setup_and_train.sh
```

### Muốn accuracy cao hơn?

```bash
# Tăng epochs và sử dụng model lớn
export EPOCHS=150
export MODEL=yolov8l.pt
./setup_and_train.sh
```

## 📈 Mục tiêu

- ✅ **Accuracy ≥ 85%**
- ✅ **F1-Score ≥ 0.80**
- ✅ **Training hoàn thành trong 4-8 giờ** (Mac M4 với YOLOv8m)

## 🆘 Gặp lỗi?

### Lỗi: "command not found: python3"
```bash
# Cài Python qua Homebrew
brew install python@3.10
```

### Lỗi: "Dataset not found"
```bash
# Kiểm tra lại path
export DATASET_ROOT=/correct/path/to/RDD2022
./setup_and_train.sh
```

### Lỗi: "Out of memory"
```bash
# Giảm batch size
export BATCH_SIZE=4
./setup_and_train.sh
```

## 📚 Chi tiết

Xem file [README.md](README.md) để biết thêm chi tiết về:
- Cấu hình nâng cao
- Hyperparameter tuning
- Troubleshooting
- Metrics và evaluation

## 💡 Tips

1. **Sử dụng terminal với quyền administrator** để tránh lỗi permission

2. **Đảm bảo có đủ dung lượng ổ cứng** (tối thiểu 50GB)

3. **Đóng các ứng dụng khác** khi training để tối ưu performance

4. **Kiểm tra nhiệt độ Mac** - nếu quá nóng, hãy nghỉ giữa chừng

5. **Backup kết quả** sau mỗi lần training thành công

---

**Happy Training! 🎉**

Nếu thành công, bạn sẽ có một model phát hiện hư hỏng đường bộ với độ chính xác ≥85%!

