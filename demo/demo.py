# Tạo môi trường ảo
# python3 -m venv yolov8-env

# Kích hoạt môi trường
# source yolov8-env/bin/activate

import os
import cv2
from ultralytics import YOLO

# Paths
# model_path = "../model/best_rdd2022.pt"  # Update with the path to your trained model
model_path = "../outputs/weights/best.pt"
test_images_dir = "../datasets/tests/images"
output_dir = "../datasets/tests/outputs"  # Folder to save annotated images
os.makedirs(output_dir, exist_ok=True)

# Load the YOLOv8 model
model = YOLO(model_path)

# Class names from RDD2022
class_names = ['Longitudinal Crack', 'Transverse Crack', 'Alligator Crack', 'Pothole']

# Counters for classification
damaged_roads = 0
good_roads = 0
results_list = []

# Process each image
for img_file in os.listdir(test_images_dir):
    if img_file.endswith(('.jpg', '.png')):
        img_path = os.path.join(test_images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load {img_file}")
            continue

        # Run inference
        results = model.predict(img_path, conf=0.5, save=False)

        # Check for damage
        has_damage = False
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls)
            label = class_names[class_id]
            confidence = box.conf.item()
            has_damage = True
            detections.append({'class': label, 'confidence': confidence, 'bbox': (x1, y1, x2, y2)})
            
            # Draw bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Classify and count
        if has_damage:
            damaged_roads += 1
            status = "Đường hỏng"
        else:
            good_roads += 1
            status = "Đường tốt"
        
        results_list.append({'image': img_file, 'status': status, 'detections': detections})
        
        # Save annotated image
        cv2.imwrite(os.path.join(output_dir, img_file), img)

# Print summary
print("\n=== Kết quả phân loại ===")
print(f"Tổng số ảnh: {damaged_roads + good_roads}")
print(f"Đường hỏng (có hư hại): {damaged_roads}")
print(f"Đường tốt (không hư hại): {good_roads}")
print(f"Ảnh đã xử lý được lưu tại: {output_dir}")

# Save detailed results to a text file
with open(os.path.join(output_dir, 'classification_results.txt'), 'w') as f:
    f.write("=== Kết quả phân loại ===\n")
    f.write(f"Tổng số ảnh: {damaged_roads + good_roads}\n")
    f.write(f"Đường hỏng (có hư hại): {damaged_roads}\n")
    f.write(f"Đường tốt (không hư hại): {good_roads}\n\n")
    f.write("Chi tiết từng ảnh:\n")
    for result in results_list:
        f.write(f"\nẢnh: {result['image']}\n")
        f.write(f"Trạng thái: {result['status']}\n")
        if result['detections']:
            f.write("Hư hại phát hiện:\n")
            for det in result['detections']:
                f.write(f"  - {det['class']} (Confidence: {det['confidence']:.2f})\n")

print(f"Kết quả chi tiết được lưu tại: {os.path.join(output_dir, 'classification_results.txt')}")