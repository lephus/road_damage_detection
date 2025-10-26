import os
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# Define dataset paths based on provided structure
dataset_paths = {
    'India': {
        'train_images': '/kaggle/input/dut-rdd/RDD2022_India/India/train/images',
        'train_annotations': '/kaggle/input/dut-rdd/RDD2022_India/India/train/annotations',
        'test_images': '/kaggle/input/dut-rdd/RDD2022_India/India/test/images'
    },
    'China_MotorBike': {
        'train_images': '/kaggle/input/dut-rdd/RDD2022_China_MotorBike/China_MotorBike/train/images',
        'train_annotations': '/kaggle/input/dut-rdd/RDD2022_China_MotorBike/China_MotorBike/train/annotations',
        'test_images': '/kaggle/input/dut-rdd/RDD2022_China_MotorBike/China_MotorBike/test/images'
    },
    'China_Drone': {
        'train_images': '/kaggle/input/dut-rdd/RDD2022_China_Drone/China_Drone/train/images',
        'train_annotations': '/kaggle/input/dut-rdd/RDD2022_China_Drone/China_Drone/train/annotations'
    }
}

# Class mapping for RDD2022
class_names = {
    'D00': 'Longitudinal Crack',
    'D10': 'Transverse Crack',
    'D20': 'Alligator Crack',
    'D40': 'Pothole'
}

def analyze_dataset_structure():
    report = {}
    
    for dataset_name, paths in dataset_paths.items():
        print(f"\nAnalyzing {dataset_name}...")
        report[dataset_name] = {}
        
        # Analyze training data
        if 'train_images' in paths and 'train_annotations' in paths:
            train_images = [f for f in os.listdir(paths['train_images']) if f.endswith(('.jpg', '.png'))]
            train_annotations = [f for f in os.listdir(paths['train_annotations']) if f.endswith('.xml')]
            
            # Count files
            report[dataset_name]['train'] = {
                'num_images': len(train_images),
                'num_annotations': len(train_annotations)
            }
            
            # Check for missing pairs
            img_stems = {os.path.splitext(f)[0] for f in train_images}
            ann_stems = {os.path.splitext(f)[0] for f in train_annotations}
            missing_annotations = img_stems - ann_stems
            missing_images = ann_stems - img_stems
            report[dataset_name]['train']['missing_annotations'] = len(missing_annotations)
            report[dataset_name]['train']['missing_images'] = len(missing_images)
            
            # Analyze class distribution
            class_counts = Counter()
            for ann_file in train_annotations:
                try:
                    tree = ET.parse(os.path.join(paths['train_annotations'], ann_file))
                    root = tree.getroot()
                    for obj in root.findall('object'):
                        class_name = obj.find('name').text
                        if class_name in class_names:
                            class_counts[class_name] += 1
                except Exception as e:
                    print(f"Error parsing {ann_file}: {e}")
            report[dataset_name]['train']['class_distribution'] = dict(class_counts)
        
        # Analyze test data (if available)
        if 'test_images' in paths:
            test_images = [f for f in os.listdir(paths['test_images']) if f.endswith(('.jpg', '.png'))]
            report[dataset_name]['test'] = {
                'num_images': len(test_images),
                'num_annotations': 0  # Test sets may not have annotations
            }
    
    return report

def print_report(report):
    print("\n=== Dataset Analysis Report ===")
    for dataset_name, data in report.items():
        print(f"\n{dataset_name}:")
        if 'train' in data:
            print("  Training Data:")
            print(f"    Images: {data['train']['num_images']}")
            print(f"    Annotations: {data['train']['num_annotations']}")
            print(f"    Missing Annotations: {data['train']['missing_annotations']}")
            print(f"    Missing Images: {data['train']['missing_images']}")
            print("    Class Distribution:")
            for class_id, count in data['train']['class_distribution'].items():
                print(f"      {class_names[class_id]}: {count}")
        if 'test' in data:
            print("  Test Data:")
            print(f"    Images: {data['test']['num_images']}")
            print(f"    Annotations: {data['test']['num_annotations']}")

# Run analysis
report = analyze_dataset_structure()
print_report(report)

# Save report to file
with open('/kaggle/working/dataset_analysis_report.txt', 'w') as f:
    f.write("=== Dataset Analysis Report ===\n")
    for dataset_name, data in report.items():
        f.write(f"\n{dataset_name}:\n")
        if 'train' in data:
            f.write("  Training Data:\n")
            f.write(f"    Images: {data['train']['num_images']}\n")
            f.write(f"    Annotations: {data['train']['num_annotations']}\n")
            f.write(f"    Missing Annotations: {data['train']['missing_annotations']}\n")
            f.write(f"    Missing Images: {data['train']['missing_images']}\n")
            f.write("    Class Distribution:\n")
            for class_id, count in data['train']['class_distribution'].items():
                f.write(f"      {class_names[class_id]}: {count}\n")
        if 'test' in data:
            f.write("  Test Data:\n")
            f.write(f"    Images: {data['test']['num_images']}\n")
            f.write(f"    Annotations: {data['test']['num_annotations']}\n")

print("\nReport saved to /kaggle/working/dataset_analysis_report.txt")