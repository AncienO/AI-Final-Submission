import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load the trained YOLOv9 model
# -------------------------------
MODEL_PATH = '/Users/jonathanodonkor/Desktop/Work/4th Year/Semester 2/Intro to Ai/final_project/asl_yolo9_training/yolov9n_asl2/weights/best.pt'
model = YOLO(MODEL_PATH)
print(f"✅ Loaded model from {MODEL_PATH}")


# --------------------------------------------------
# 2. Function: Predict and display on test images
# --------------------------------------------------
def predict_test_images(test_images_dir):
    print(f"🔎 Running prediction on images in {test_images_dir}...")
    results = model.predict(source=test_images_dir, save=True)

    for idx, r in enumerate(results):
        img = r.plot()  # Draw detections on image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8, 8))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f'Prediction {idx+1}')
        plt.show()

        # Only show first 5 images for quick check
        if idx == 4:
            break


# --------------------------------------------------
# 3. Function: Real-time webcam detection
# --------------------------------------------------
def predict_live_webcam():
    print("🎥 Starting webcam for live ASL detection...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        # Run YOLO prediction on frame
        results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
        
        # Draw detection results
        annotated_frame = results[0].plot()

        # Show the frame
        cv2.imshow('Live ASL Detection', annotated_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting live detection...")
            break

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------
# 4. Main app controller
# --------------------------------------------------
if __name__ == "__main__":
    print("🖐 Welcome to ASL Detection App")
    print("1️⃣ Test on static images")
    print("2️⃣ Live test with webcam")
    choice = input("Select an option (1/2): ")

    if choice == '1':
        test_images_dir = '/Users/jonathanodonkor/Desktop/Work/4th Year/Semester 2/Intro to Ai/final_project/American Sign Language Letters.v1-v1.yolov9/test/images'
        predict_test_images(test_images_dir)
    elif choice == '2':
        predict_live_webcam()
    else:
        print("❌ Invalid choice. Exiting.")
