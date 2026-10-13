import cv2
import numpy as np
import os

def collect_face_data() -> None:
    # Ask the user for their name to label the dataset folder
    name: str = input("Enter your name (e.g., lalit): ").strip().lower()
    if not name:
        print("❌ Name cannot be empty.")
        return

    # Create a dataset directory structure
    dataset_dir: str = "dataset"
    user_dir: str = os.path.join(dataset_dir, name)
    os.makedirs(user_dir, exist_ok=True)

    # Load local Haar cascade
    cascade_filename: str = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_filename):
        print(f"❌ Error: {cascade_filename} not found. Run face_detection.py first to download it.")
        return

    face_cascade: cv2.CascadeClassifier = cv2.CascadeClassifier(cascade_filename)
    cap: cv2.VideoCapture = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print(f"\n📷 Initializing data collection for: {name.capitalize()}")
    print("👉 Look at the camera and slowly move your head around (up, down, side to side).")
    print("🎯 We will capture 30 sample images. Press 'q' to abort early.\n")

    count: int = 0
    target_samples: int = 30

    while count < target_samples:
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            count += 1
            # Extract and resize the face region to a standard size (200x200)
            face_roi: np.ndarray = gray[y:y+h, x:x+w]
            face_resized: np.ndarray = cv2.resize(face_roi, (200, 200))

            # Save the face image to the user's directory
            file_path: str = os.path.join(user_dir, f"{name}_{count}.jpg")
            cv2.imwrite(file_path, face_resized)

            # Draw visual feedback on the live screen
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}/{target_samples}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show the video feed
        cv2.imshow("Data Collector - Look at Camera", frame)

        # Break if 'q' is pressed or 30 samples reached
        if cv2.waitKey(100) & 0xFF == ord('q'):
            print("⚠️ Data collection aborted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if count >= target_samples:
        print(f"\n✅ Success! Captured {count} training images for '{name.capitalize()}' in '{user_dir}/'.")
    else:
        print(f"\n⚠️ Collection ended with {count} images.")

if __name__ == "__main__":
    collect_face_data()