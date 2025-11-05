"""
Road Damage Detection - Inference Script
Use trained model to detect road damage in images
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt


class RoadDamageDetector:
    """Road damage detector using trained YOLO model"""
    
    def __init__(self, model_path):
        """
        Initialize detector
        
        Args:
            model_path: Path to trained model (.pt file)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Class names
        self.class_names = ['D00', 'D10', 'D20', 'D40', 'D43', 'D44']
        self.class_descriptions = {
            'D00': 'Lateral crack',
            'D10': 'Longitudinal crack',
            'D20': 'Alligator crack',
            'D40': 'Pothole',
            'D43': 'Cross walk blur',
            'D44': 'Whiteline blur'
        }
        
        print("✅ Model loaded successfully!")
        
    def detect_image(self, image_path, conf_threshold=0.25, save_path=None):
        """
        Detect road damage in a single image
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            save_path: Path to save output image (optional)
            
        Returns:
            results: Detection results
            annotated_image: Image with detections drawn
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # Detect
        results = self.model.predict(image, conf=conf_threshold, verbose=False)
        
        # Annotate image
        annotated_image = self._draw_detections(image.copy(), results[0])
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, annotated_image)
            print(f"✓ Saved result to: {save_path}")
        
        return results[0], annotated_image
    
    def detect_batch(self, image_dir, output_dir, conf_threshold=0.25):
        """
        Detect road damage in a batch of images
        
        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save output images
            conf_threshold: Confidence threshold
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(Path(image_dir).glob(ext))
        
        if len(image_files) == 0:
            print(f"⚠️  No images found in: {image_dir}")
            return
        
        print(f"Found {len(image_files)} images")
        
        # Process each image
        results_summary = []
        for img_path in image_files:
            print(f"\nProcessing: {img_path.name}")
            
            try:
                # Detect
                results, annotated_img = self.detect_image(
                    str(img_path),
                    conf_threshold=conf_threshold,
                    save_path=os.path.join(output_dir, img_path.name)
                )
                
                # Count detections
                detections = self._count_detections(results)
                results_summary.append({
                    'image': img_path.name,
                    'detections': detections,
                    'total': sum(detections.values())
                })
                
                # Print summary
                if sum(detections.values()) > 0:
                    print(f"  Detected {sum(detections.values())} damage(s):")
                    for cls, count in detections.items():
                        if count > 0:
                            print(f"    {cls} ({self.class_descriptions[cls]}): {count}")
                else:
                    print("  No damage detected")
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                continue
        
        # Save summary
        self._save_summary(results_summary, output_dir)
        
        print(f"\n✅ Batch processing complete!")
        print(f"📁 Results saved to: {output_dir}")
    
    def _draw_detections(self, image, results):
        """Draw bounding boxes and labels on image"""
        if len(results.boxes) == 0:
            return image
        
        # Define colors for each class
        colors = {
            0: (255, 0, 0),      # D00 - Red
            1: (0, 255, 0),      # D10 - Green
            2: (0, 0, 255),      # D20 - Blue
            3: (255, 255, 0),    # D40 - Yellow
            4: (255, 0, 255),    # D43 - Magenta
            5: (0, 255, 255),    # D44 - Cyan
        }
        
        for box in results.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls)
            conf = float(box.conf)
            
            # Get color
            color = colors.get(cls, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{self.class_names[cls]} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Background for text
            cv2.rectangle(image, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)
            
            # Text
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image
    
    def _count_detections(self, results):
        """Count detections by class"""
        detections = {cls: 0 for cls in self.class_names}
        
        for box in results.boxes:
            cls = int(box.cls)
            detections[self.class_names[cls]] += 1
        
        return detections
    
    def _save_summary(self, results_summary, output_dir):
        """Save detection summary to file"""
        summary_path = os.path.join(output_dir, 'detection_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("ROAD DAMAGE DETECTION SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            for result in results_summary:
                f.write(f"Image: {result['image']}\n")
                f.write(f"Total detections: {result['total']}\n")
                
                if result['total'] > 0:
                    for cls, count in result['detections'].items():
                        if count > 0:
                            f.write(f"  {cls} ({self.class_descriptions[cls]}): {count}\n")
                
                f.write("\n")
            
            # Overall statistics
            f.write("=" * 70 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("=" * 70 + "\n")
            
            total_images = len(results_summary)
            images_with_damage = sum(1 for r in results_summary if r['total'] > 0)
            total_detections = sum(r['total'] for r in results_summary)
            
            f.write(f"Total images processed: {total_images}\n")
            f.write(f"Images with damage: {images_with_damage}\n")
            f.write(f"Total detections: {total_detections}\n")
            
            # Count by class
            class_counts = {cls: 0 for cls in self.class_names}
            for result in results_summary:
                for cls, count in result['detections'].items():
                    class_counts[cls] += count
            
            f.write("\nDetections by class:\n")
            for cls, count in class_counts.items():
                if count > 0:
                    f.write(f"  {cls} ({self.class_descriptions[cls]}): {count}\n")
        
        print(f"✓ Summary saved to: {summary_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Road Damage Detection - Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--image', type=str,
                       help='Path to input image (for single image inference)')
    parser.add_argument('--image_dir', type=str,
                       help='Path to directory containing images (for batch inference)')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--show', action='store_true',
                       help='Display results (for single image only)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image_dir must be specified")
    
    # Initialize detector
    detector = RoadDamageDetector(args.model)
    
    # Single image inference
    if args.image:
        print(f"\n🔍 Processing single image: {args.image}")
        
        output_path = os.path.join(args.output_dir, f"detected_{os.path.basename(args.image)}")
        os.makedirs(args.output_dir, exist_ok=True)
        
        results, annotated_img = detector.detect_image(
            args.image,
            conf_threshold=args.conf,
            save_path=output_path
        )
        
        # Print results
        print("\n📊 Detection Results:")
        if len(results.boxes) > 0:
            detections = detector._count_detections(results)
            print(f"Total detections: {sum(detections.values())}")
            for cls, count in detections.items():
                if count > 0:
                    print(f"  {cls} ({detector.class_descriptions[cls]}): {count}")
        else:
            print("No damage detected")
        
        # Show image if requested
        if args.show:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
            plt.title("Detection Results")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    
    # Batch inference
    elif args.image_dir:
        print(f"\n🔍 Processing batch of images from: {args.image_dir}")
        detector.detect_batch(
            args.image_dir,
            args.output_dir,
            conf_threshold=args.conf
        )


if __name__ == '__main__':
    main()

