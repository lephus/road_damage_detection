#!/bin/bash

###############################################################################
# Road Damage Detection - Setup and Training Script for Mac M4
# This script sets up the environment and runs training on Apple Silicon
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print header
echo "============================================================================"
echo "🚀 Road Damage Detection - Setup and Training (Mac M4 Optimized)"
echo "============================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

print_info "Script directory: $SCRIPT_DIR"
print_info "Project root: $PROJECT_ROOT"

# Configuration
VENV_NAME="road_damage_env"
VENV_DIR="$SCRIPT_DIR/$VENV_NAME"
DATASET_ROOT="${DATASET_ROOT:-/path/to/your/dataset}"  # Override with environment variable
OUTPUT_DIR="$SCRIPT_DIR/outputs"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MODEL="${MODEL:-yolov8m.pt}"

print_info "Configuration:"
echo "  Virtual Environment: $VENV_DIR"
echo "  Dataset Root: $DATASET_ROOT"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Model: $MODEL"
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_warning "This script is optimized for macOS (Mac M4). You are running on: $OSTYPE"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for Apple Silicon
if [[ $(uname -m) == "arm64" ]]; then
    print_success "Detected Apple Silicon (M-series chip)"
    APPLE_SILICON=true
else
    print_warning "Not running on Apple Silicon. PyTorch MPS acceleration may not be available."
    APPLE_SILICON=false
fi

# Step 1: Check Python installation
print_info "Step 1: Checking Python installation..."

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed!"
    print_info "Please install Python 3.10 or later:"
    print_info "  brew install python@3.10"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
print_success "Python version: $PYTHON_VERSION"

# Check minimum Python version (3.8)
if [[ $(echo "$PYTHON_VERSION >= 3.8" | bc -l) -eq 0 ]]; then
    print_error "Python 3.8 or later is required. Found: $PYTHON_VERSION"
    exit 1
fi

# Step 2: Create virtual environment
print_info "Step 2: Setting up virtual environment..."

if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists at: $VENV_DIR"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        print_info "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
print_success "Virtual environment activated"

# Step 3: Upgrade pip
print_info "Step 3: Upgrading pip..."
pip install --upgrade pip setuptools wheel
print_success "pip upgraded"

# Step 4: Install dependencies
print_info "Step 4: Installing dependencies..."

# Create requirements.txt
cat > "$SCRIPT_DIR/requirements.txt" << EOF
# Core ML libraries
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Deep Learning - PyTorch for Apple Silicon
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.0.0
torchvision>=0.15.0

# Computer Vision
opencv-python-headless>=4.8.0
Pillow>=10.0.0

# YOLOv8
ultralytics>=8.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
tqdm>=4.65.0
PyYAML>=6.0
lxml>=4.9.0

# t-SNE
scikit-learn>=1.3.0
EOF

print_info "Installing Python packages..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# Install PyTorch with MPS support for Apple Silicon
if [ "$APPLE_SILICON" = true ]; then
    print_info "Installing PyTorch with MPS (Metal Performance Shaders) support..."
    pip install --upgrade torch torchvision
    print_success "PyTorch with MPS support installed"
fi

print_success "All dependencies installed"

# Step 5: Verify installation
print_info "Step 5: Verifying installation..."

python3 << EOF
import sys
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib
import seaborn
from ultralytics import YOLO
from sklearn.manifold import TSNE

print("✅ All packages imported successfully!")
print(f"  Python: {sys.version}")
print(f"  PyTorch: {torch.__version__}")
print(f"  OpenCV: {cv2.__version__}")
print(f"  NumPy: {np.__version__}")
print(f"  Pandas: {pd.__version__}")

# Check for MPS (Apple Silicon GPU)
if torch.backends.mps.is_available():
    print("  🚀 MPS (Apple Silicon GPU) is AVAILABLE!")
    print("  Training will use GPU acceleration")
elif torch.cuda.is_available():
    print("  🚀 CUDA is AVAILABLE!")
else:
    print("  ⚠️  No GPU acceleration available (will use CPU)")
EOF

print_success "Installation verified"

# Step 6: Check dataset
print_info "Step 6: Checking dataset..."

if [ "$DATASET_ROOT" = "/path/to/your/dataset" ]; then
    print_error "Dataset root not set!"
    print_info "Please set the DATASET_ROOT environment variable:"
    print_info "  export DATASET_ROOT=/path/to/your/RDD2022/dataset"
    print_info "Or edit this script to set the correct path."
    print_info ""
    print_info "Expected structure:"
    print_info "  \$DATASET_ROOT/"
    print_info "    India/"
    print_info "      train/images/"
    print_info "      train/annotations/"
    print_info "      test/images/"
    print_info "    Czech/"
    print_info "      train/images/"
    print_info "      ..."
    print_info "    (and so on for other countries)"
    echo ""
    read -p "Enter dataset root path: " DATASET_ROOT
fi

if [ ! -d "$DATASET_ROOT" ]; then
    print_error "Dataset directory not found: $DATASET_ROOT"
    print_info "Please download the RDD2022 dataset first."
    exit 1
fi

print_success "Dataset found at: $DATASET_ROOT"

# List available datasets
print_info "Available datasets:"
for country in India Czech China_MotorBike China_Drone Japan; do
    if [ -d "$DATASET_ROOT/$country" ]; then
        echo "  ✓ $country"
    else
        echo "  ✗ $country (not found)"
    fi
done
echo ""

# Step 7: Create output directory
print_info "Step 7: Creating output directory..."
mkdir -p "$OUTPUT_DIR"
print_success "Output directory created: $OUTPUT_DIR"

# Step 8: Run training
print_info "Step 8: Starting training..."
echo "============================================================================"
echo "🚀 TRAINING STARTED"
echo "============================================================================"
echo ""
print_info "This may take several hours depending on your hardware..."
print_info "Training progress will be displayed below."
echo ""

# Run training script
python3 "$SCRIPT_DIR/train_road_damage.py" \
    --dataset_root "$DATASET_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --model "$MODEL"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    print_success "TRAINING COMPLETED SUCCESSFULLY!"
    echo "============================================================================"
    echo ""
    print_info "Results saved to: $OUTPUT_DIR"
    print_info ""
    print_info "Generated files:"
    print_info "  📊 Metrics: $OUTPUT_DIR/metrics/"
    print_info "  📈 Visualizations: $OUTPUT_DIR/visualizations/"
    print_info "  💾 Models: $OUTPUT_DIR/models/"
    print_info "  📝 Training logs: $OUTPUT_DIR/runs/"
    echo ""
    
    # Show key metrics if available
    if [ -f "$OUTPUT_DIR/metrics/metrics_test.json" ]; then
        print_info "Test Set Metrics:"
        cat "$OUTPUT_DIR/metrics/metrics_test.json"
        echo ""
    fi
    
    if [ -f "$OUTPUT_DIR/metrics/training_time.txt" ]; then
        print_info "Training Time:"
        cat "$OUTPUT_DIR/metrics/training_time.txt"
        echo ""
    fi
    
    print_success "You can now use the trained model for inference!"
    print_info "Best model location: $OUTPUT_DIR/models/best_model.pt"
    
else
    echo ""
    echo "============================================================================"
    print_error "TRAINING FAILED!"
    echo "============================================================================"
    echo ""
    print_info "Please check the error messages above."
    print_info "Common issues:"
    print_info "  - Insufficient memory (try reducing batch size)"
    print_info "  - Dataset path incorrect"
    print_info "  - Missing dependencies"
    exit 1
fi

# Step 9: Deactivate virtual environment (optional, commented out to keep it active)
# deactivate
# print_info "Virtual environment deactivated"

echo ""
print_success "All done! 🎉"
echo ""

