---

# YOLOv9 for American Sign Language Detection

This project leverages the YOLOv9 model for real-time American Sign Language (ASL) detection. The primary goal is to build a robust system that can detect and recognize Sign language gestures using a webcam input, perform image augmentation, and handle model training, testing, and predictions.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Sources and Collection](#data-sources-and-collection)
3. [Preprocessing](#preprocessing)
4. [Model Implementation](#model-implementation)
5. [Training the Model](#training-the-model)
6. [Testing the Model](#testing-the-model)
7. [Augmentation](#augmentation)
8. [Code Implementation](#code-implementation)
9. [Dependencies](#dependencies)
10. [Conclusion](#conclusion)

---

## Project Overview

The project utilizes a YOLOv9-based model to detect ASL gestures in real-time. It involves collecting ASL images, processing them with image augmentations, training a YOLOv9 model, and implementing webcam-based real-time inference.

---

## Data Sources and Collection

The dataset is collected from an existing ASL dataset containing images of different ASL signs. The images are grouped into categories where each class corresponds to a letter or word in the ASL alphabet. The dataset is organized by individual directories for each sign, where the image files are stored.

**Preprocessing steps:**

- **Image resizing**: Each image is resized to 640x640 to match the input size expected by YOLO.
- **Normalization**: The pixel values are normalized to a [0,1] range.
- **Bounding Box Annotation**: Using the MediaPipe library, hand landmarks are identified, and bounding boxes are created around the hands for object detection tasks. These bounding boxes are saved in YOLO format, which includes normalized coordinates for the bounding box (center_x, center_y, width, height).

---

## Preprocessing

The preprocessing steps ensure the dataset is in the correct format for training. Here's the code to preprocess the dataset:

```python
def preprocess_and_save_images(input_dir, output_dir, size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)

    for entry in os.listdir(input_dir):
        entry_path = os.path.join(input_dir, entry)

        if os.path.isfile(entry_path):
            # Single image, not a folder
            try:
                img = cv2.imread(entry_path)
                img = cv2.resize(img, size)
                img = img / 255.0  # Normalize
                output_path = os.path.join(output_dir, entry)
                cv2.imwrite(output_path, (img * 255).astype(np.uint8))
            except Exception as e:
                print(f"Skipping {entry_path}: {e}")

        elif os.path.isdir(entry_path):
            # If it's a folder, process all images inside
            output_class_path = os.path.join(output_dir, entry)
            os.makedirs(output_class_path, exist_ok=True)

            for img_file in tqdm(os.listdir(entry_path), desc=f"Processing {entry}"):
                img_path = os.path.join(entry_path, img_file)
                output_path = os.path.join(output_class_path, img_file)

                try:
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, size)
                    img = img / 255.0  # Normalize
                    cv2.imwrite(output_path, (img * 255).astype(np.uint8))
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")
```

---

## Model Implementation

The YOLOv9 model is used for detecting ASL signs. YOLO (You Only Look Once) is a popular object detection algorithm known for its real-time performance. The algorithm is based on convolutional neural networks (CNNs) to extract spatial features and perform object localization and classification in a single pass.

We have implemented the YOLOv9 architecture, which consists of convolutional layers, bounding box prediction, and classification. The architecture is trained using the images and corresponding bounding box annotations from the ASL dataset.

---

## Training the Model

To train the model, the following steps were followed:

1. **Load Pretrained YOLOv9 Model**: The pretrained YOLOv9 model is fine-tuned on the ASL dataset.

2. **Model Configuration**: We specify the dataset, the number of epochs, batch size, and image size.

Here’s the training code:

```python
from ultralytics import YOLO

# Load pretrained YOLOv9n model
model = YOLO('/path/to/yolov9s.pt')

# Train on your ASL dataset
results = model.train(
    data="/path/to/data.yaml",
    epochs=10,
    imgsz=640,
    batch=16,
    project="asl_yolo9_training",
    name="yolov9n_asl",
    pretrained=True
)
```

---

## Testing the Model

After training, the model can be tested with the following code. This will run the model on test images and save the predictions.

```python
results = model.predict(source='/Users/jonathanodonkor/Desktop/Work/4th Year/Semester 2/Intro to Ai/final_project/American Sign Language Letters.v1-v1.yolov9/test/images', save=True)
```

---

## Augmentation

Image augmentation helps improve model generalization. We applied random rotations, flips, and resizing to the training data. Below is the image augmentation code:

```python
import cv2
import numpy as np
import os

def augment_image(image):
    # Random rotation
    angle = np.random.randint(-30, 30)
    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Random horizontal flip
    if np.random.rand() > 0.5:
        rotated = cv2.flip(rotated, 1)

    # Random scaling
    scale = np.random.uniform(0.8, 1.2)
    resized = cv2.resize(rotated, None, fx=scale, fy=scale)

    # Cropping or padding to original size
    if scale < 1.0:
        delta_w = image.shape[1] - resized.shape[1]
        delta_h = image.shape[0] - resized.shape[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        augmented = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        augmented = cv2.resize(resized, (image.shape[1], image.shape[0]))

    final_img = cv2.resize(augmented, (640, 640))
    return final_img

# Paths
words_dir = '/Users/jonathanodonkor/Desktop/Work/4th Year/Semester 2/Intro to Ai/final_project/Dataset'
output_base_path = '/Users/jonathanodonkor/Desktop/Work/4th Year/Semester 2/Intro to Ai/final_project/augmented_dataset/images'

# Ensure output directory exists
os.makedirs(output_base_path, exist_ok=True)

# Process each image directly
for idx, image_file in enumerate(os.listdir(words_dir)):
    image_path = os.path.join(words_dir, image_file)

    if not os.path.isfile(image_path) or not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    print(f'Processing: {image_path}')

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image {image_path}")
        continue

    # Save the original
    base_name = os.path.splitext(image_file)[0]
    original_save_path = os.path.join(output_base_path, f"{base_name}_original.jpg")
    cv2.imwrite(original_save_path, cv2.resize(image, (640, 640)))

    # Generate and save 3 augmentations
    for i in range(3):
        aug_img = augment_image(image)
        aug_img_name = os.path.join(output_base_path, f"{base_name}_aug{i}.jpg")
        cv2.imwrite(aug_img_name, aug_img)
        print(f'Saved: {aug_img_name}')

print("Image augmentation complete.")

```

---

## Code Implementation

The app uses several key components:

- **MediaPipe** for detecting hands and generating bounding box annotations.
- **YOLOv9** model for training and inference.
- **OpenCV** for webcam capture and image processing.

Here's how the webcam-based real-time inference is implemented:

```python
import cv2
import time
from ultralytics import YOLO

# Load your trained model
model = YOLO('/Users/jonathanodonkor/Desktop/Work/4th Year/Semester 2/Intro to Ai/final_project/asl_yolo9_training/yolov9n_asl2/weights/best.pt')

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Failed to open webcam.")
else:
    print("✅ Webcam initialized!")
    time.sleep(2)  # Allow camera to warm up

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        # Run YOLOv9 inference
        results = model(frame, imgsz=640, conf=0.1)[0]  # Lower conf if needed

        # Visualize detections on the frame
        annotated_frame = results.plot()

        # Show result
        cv2.imshow('ASL Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```

---

## Dependencies

- **Ultralytics YOLO**: For object detection using the YOLOv9 architecture.
- **OpenCV**: For image capture, processing, and display.
- **MediaPipe**: For hand landmark detection.
- **Matplotlib & Pandas**: For visualizing model metrics.

Install the dependencies via pip:

```bash
pip install ultralytics opencv-python mediapipe matplotlib pandas
```

---

## Conclusion

This project demonstrates how YOLOv9 can be applied to detect sign-language signs in real-time using a webcam. The model leverages deep learning techniques like CNNs and advanced data augmentation strategies to achieve accuracy and performance.
