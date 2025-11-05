#!/usr/bin/env python3
"""
Test Environment Setup
Quick script to verify all dependencies are installed correctly
"""

import sys
import platform

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_success(text):
    """Print success message"""
    print(f"✅ {text}")

def print_error(text):
    """Print error message"""
    print(f"❌ {text}")

def print_warning(text):
    """Print warning message"""
    print(f"⚠️  {text}")

def test_python_version():
    """Test Python version"""
    print_header("Testing Python Version")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"Python version: {version_str}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version_str} is compatible")
        return True
    else:
        print_error(f"Python 3.8+ required, found {version_str}")
        return False

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    if package_name is None:
        package_name = module_name
    
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print_success(f"{package_name}: {version}")
        return True, version
    except ImportError as e:
        print_error(f"{package_name}: NOT INSTALLED")
        return False, None

def test_pytorch():
    """Test PyTorch and device availability"""
    print_header("Testing PyTorch")
    
    success, version = test_import('torch', 'PyTorch')
    if not success:
        return False
    
    import torch
    
    # Test device availability
    print("\nDevice Availability:")
    
    # CPU (always available)
    print_success("CPU: Available")
    
    # CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        print_success(f"CUDA: Available (Device: {torch.cuda.get_device_name(0)})")
    else:
        print_warning("CUDA: Not available")
    
    # MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        print_success("MPS (Apple Silicon GPU): Available ⭐")
    else:
        print_warning("MPS: Not available")
    
    # Recommended device
    if torch.backends.mps.is_available():
        recommended = "mps"
    elif torch.cuda.is_available():
        recommended = "cuda"
    else:
        recommended = "cpu"
    
    print(f"\n🎯 Recommended device: {recommended}")
    
    return True

def test_core_packages():
    """Test core packages"""
    print_header("Testing Core Packages")
    
    packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('sklearn', 'scikit-learn'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'tqdm'),
    ]
    
    all_success = True
    for module, name in packages:
        success, version = test_import(module, name)
        if not success:
            all_success = False
    
    return all_success

def test_ultralytics():
    """Test Ultralytics YOLO"""
    print_header("Testing Ultralytics YOLOv8")
    
    success, version = test_import('ultralytics', 'Ultralytics')
    if not success:
        return False
    
    try:
        from ultralytics import YOLO
        print_success("YOLO model class imported successfully")
        
        # Try to load a model (will download if not exists)
        print("\nTesting model loading (this may take a moment)...")
        model = YOLO('yolov8n.pt')
        print_success("YOLOv8n model loaded successfully")
        
        return True
    except Exception as e:
        print_error(f"Failed to load YOLO model: {str(e)}")
        return False

def test_tsne():
    """Test t-SNE functionality"""
    print_header("Testing t-SNE")
    
    try:
        from sklearn.manifold import TSNE
        import numpy as np
        
        # Quick test
        X = np.random.rand(50, 10)
        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(X)
        
        print_success("t-SNE working correctly")
        return True
    except Exception as e:
        print_error(f"t-SNE test failed: {str(e)}")
        return False

def test_visualization():
    """Test visualization libraries"""
    print_header("Testing Visualization")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set backend to non-interactive for testing
        plt.switch_backend('Agg')
        
        # Quick plot test
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        plt.close(fig)
        
        print_success("Matplotlib working correctly")
        print_success("Seaborn working correctly")
        
        return True
    except Exception as e:
        print_error(f"Visualization test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("\n" + "=" * 70)
    print("  ROAD DAMAGE DETECTION - ENVIRONMENT TEST")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("Python Version", test_python_version()))
    results.append(("Core Packages", test_core_packages()))
    results.append(("PyTorch", test_pytorch()))
    results.append(("Ultralytics YOLOv8", test_ultralytics()))
    results.append(("t-SNE", test_tsne()))
    results.append(("Visualization", test_visualization()))
    
    # Summary
    print_header("Test Summary")
    
    all_passed = True
    for test_name, passed in results:
        if passed:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")
            all_passed = False
    
    print("\n" + "=" * 70)
    
    if all_passed:
        print_success("ALL TESTS PASSED! ✨")
        print("\n🚀 Your environment is ready for training!")
        print("\nNext steps:")
        print("  1. Prepare your dataset")
        print("  2. Set DATASET_ROOT environment variable")
        print("  3. Run: ./setup_and_train.sh")
        return 0
    else:
        print_error("SOME TESTS FAILED!")
        print("\n📝 Please install missing packages:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

