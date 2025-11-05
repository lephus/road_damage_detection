# 📑 Project File Index

Quick reference guide to all files in this project.

## 🎯 Start Here

| File | Purpose | When to Use |
|------|---------|-------------|
| **QUICKSTART.md** | 5-minute quick start guide | First time setup |
| **README.md** | Complete documentation | Detailed information |
| **PROJECT_SUMMARY.md** | Project overview | Understanding the system |

## 🚀 Main Scripts

### Training

| File | Type | Purpose | Usage |
|------|------|---------|-------|
| `train_road_damage.py` | Python | Main training script | `python train_road_damage.py --help` |
| `setup_and_train.sh` | Bash | Automated setup + training | `./setup_and_train.sh` |

### Inference

| File | Type | Purpose | Usage |
|------|------|---------|-------|
| `inference.py` | Python | Detect damage in images | `python inference.py --help` |
| `example_inference.sh` | Bash | Inference examples | `./example_inference.sh` |

### Testing

| File | Type | Purpose | Usage |
|------|------|---------|-------|
| `test_environment.py` | Python | Verify environment setup | `python test_environment.py` |

## 📋 Configuration Files

| File | Purpose | Format |
|------|---------|--------|
| `config.yaml` | Training configuration template | YAML |
| `.gitignore` | Git ignore rules | Text |

## 📚 Documentation

| File | Content | Audience |
|------|---------|----------|
| `README.md` | Full documentation with all details | Everyone |
| `QUICKSTART.md` | Fast 5-minute start guide | Beginners |
| `PROJECT_SUMMARY.md` | Project overview and requirements | Reviewers, researchers |
| `INDEX.md` | This file - quick reference | Everyone |

## 🗂️ File Organization

```
build_models/local/
│
├── 📖 Documentation
│   ├── README.md                  ⭐ Start here for complete docs
│   ├── QUICKSTART.md              ⭐ 5-minute quick start
│   ├── PROJECT_SUMMARY.md         Project overview
│   └── INDEX.md                   This file
│
├── 🐍 Python Scripts
│   ├── train_road_damage.py       ⭐ Main training script
│   ├── inference.py               ⭐ Inference script
│   └── test_environment.py        Environment testing
│
├── 🔧 Shell Scripts
│   ├── setup_and_train.sh         ⭐ Automated setup + training
│   └── example_inference.sh       Inference examples
│
├── ⚙️ Configuration
│   ├── config.yaml                Configuration template
│   └── .gitignore                 Git ignore rules
│
└── 📦 Auto-generated (by setup script)
    └── requirements.txt           Python dependencies
```

## 🎓 Learning Path

### Beginner
1. Read `QUICKSTART.md`
2. Run `./setup_and_train.sh`
3. Check results in `outputs/`

### Intermediate
1. Read `README.md`
2. Customize `config.yaml`
3. Run `train_road_damage.py` with custom args
4. Experiment with `inference.py`

### Advanced
1. Read `PROJECT_SUMMARY.md`
2. Modify `train_road_damage.py` for custom needs
3. Implement custom callbacks
4. Fine-tune hyperparameters

## 📝 File Descriptions

### train_road_damage.py
**Complete training pipeline with:**
- Multi-dataset loading (India, Czech, China, Japan)
- Data preprocessing and augmentation
- YOLOv8 training with Apple Silicon optimization
- Comprehensive metrics (Accuracy, Precision, Recall, F1, AUC)
- Visualizations (training curves, t-SNE, ROC, confusion matrix)
- Auto model saving and checkpointing

**Key class**: `RoadDamageTrainer`

**Main functions**:
- `load_dataset()`: Load and validate data
- `train()`: Train the model
- `evaluate_model()`: Compute metrics
- `visualize_tsne()`: t-SNE visualization
- `plot_training_history()`: Training curves

### inference.py
**Inference script for trained models:**
- Single image detection
- Batch processing
- Confidence threshold tuning
- Annotated image output
- Detection summary reports

**Key class**: `RoadDamageDetector`

**Main functions**:
- `detect_image()`: Single image inference
- `detect_batch()`: Batch inference
- `_draw_detections()`: Visualize results

### setup_and_train.sh
**Automated setup and training:**
- Environment checking
- Virtual environment creation
- Dependency installation
- Dataset validation
- Training execution
- Results summary

**Steps**:
1. Check Python
2. Create venv
3. Install packages
4. Verify dataset
5. Run training
6. Show results

### test_environment.py
**Environment verification:**
- Python version check
- Package imports test
- PyTorch device detection (MPS/CUDA/CPU)
- YOLO model loading test
- t-SNE functionality test
- Visualization test

**Output**: Pass/Fail for each component

### config.yaml
**Configuration template with:**
- Dataset paths
- Training parameters
- Model selection
- Augmentation settings
- Output configuration
- Evaluation settings

**Customizable**: All training hyperparameters

### README.md
**Complete documentation including:**
- Installation instructions
- Usage examples
- Configuration guide
- Troubleshooting
- API reference

**Length**: Comprehensive (~500 lines)

### QUICKSTART.md
**Quick start guide with:**
- 3-step installation
- Basic usage
- Common issues
- Quick tips

**Length**: Concise (~100 lines)

### PROJECT_SUMMARY.md
**Project overview with:**
- Requirements checklist
- Technical details
- Expected results
- Implementation notes

**Purpose**: Project review and understanding

## 🔧 Executable Files

Files that need execute permission (already set):
```bash
chmod +x setup_and_train.sh
chmod +x example_inference.sh
chmod +x test_environment.py
```

## 📊 Output Structure

After training, you'll have:

```
outputs/
├── models/
│   └── best_model.pt              # Use this for inference
├── metrics/
│   ├── metrics_test.json          # Test metrics
│   ├── classification_report_*.txt
│   ├── auc_test.txt
│   └── training_time.txt
├── visualizations/
│   ├── training_history.png       # Training curves
│   ├── tsne_*.png                 # t-SNE plots
│   ├── roc_curve_*.png            # ROC curves
│   ├── confusion_matrix_*.png     # Confusion matrices
│   └── predictions_*.png          # Sample predictions
└── runs/
    └── road_damage_detection/
        ├── weights/
        └── results.csv
```

## 🎯 Quick Commands Reference

```bash
# Test environment
python test_environment.py

# Train with defaults (100 epochs, YOLOv8m, batch=16)
./setup_and_train.sh

# Train with custom settings
export EPOCHS=150
export BATCH_SIZE=32
export MODEL=yolov8l.pt
./setup_and_train.sh

# Manual training
python train_road_damage.py \
    --dataset_root /path/to/dataset \
    --epochs 100 \
    --batch_size 16 \
    --model yolov8m.pt

# Inference on single image
python inference.py \
    --model outputs/models/best_model.pt \
    --image /path/to/image.jpg \
    --show

# Batch inference
python inference.py \
    --model outputs/models/best_model.pt \
    --image_dir /path/to/images/ \
    --output_dir ./results

# Run inference examples
./example_inference.sh
```

## 📦 Dependencies

Core packages (auto-installed by setup script):
- PyTorch (with MPS support for Apple Silicon)
- Ultralytics YOLOv8
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- tqdm, PyYAML

## 🔗 External Resources

- YOLOv8 Docs: https://docs.ultralytics.com/
- PyTorch Docs: https://pytorch.org/docs/
- RDD2022 Dataset: https://github.com/sekilab/RoadDamageDetector

## 📞 Getting Help

1. **Quick issues**: Check QUICKSTART.md
2. **Detailed issues**: Check README.md → Troubleshooting
3. **Environment issues**: Run `test_environment.py`
4. **Configuration issues**: Check `config.yaml` template
5. **Understanding code**: Read PROJECT_SUMMARY.md

## ✅ Checklist for First-Time Users

- [ ] Read QUICKSTART.md
- [ ] Download RDD2022 dataset
- [ ] Run `test_environment.py`
- [ ] Set DATASET_ROOT environment variable
- [ ] Run `./setup_and_train.sh`
- [ ] Check `outputs/metrics/metrics_test.json`
- [ ] Verify accuracy ≥ 85%
- [ ] Try `inference.py` on test images
- [ ] Backup `best_model.pt`

## 🎯 File Usage Priority

### Must Read (Before Starting)
1. ⭐⭐⭐ QUICKSTART.md
2. ⭐⭐⭐ README.md (sections relevant to you)

### Must Use (For Training)
1. ⭐⭐⭐ setup_and_train.sh OR
2. ⭐⭐⭐ train_road_damage.py

### Must Use (For Inference)
1. ⭐⭐⭐ inference.py

### Optional but Recommended
1. ⭐⭐ test_environment.py (before first run)
2. ⭐⭐ config.yaml (for customization)
3. ⭐⭐ PROJECT_SUMMARY.md (for understanding)

### Reference Only
1. ⭐ INDEX.md (this file)
2. ⭐ example_inference.sh (for learning)

---

**Last Updated**: November 1, 2025

**Total Files**: 10 main files + auto-generated outputs

**Project Status**: ✅ Ready for use

