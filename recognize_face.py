import cv2
import numpy as np
import os
import pickle

def recognize_faces() -> None:
    trainer_path: str = "trainer/trainer.yml"
    labels_path: str = "trainer/labels.pickle"
    cascade_filename: str = "haarcascade_frontalface_default.xml"

    # Verify required model assets exist
    if not os.path.exists(trainer_path) or not os.path.exists(labels_path):
        print("❌ Error: Trained model or labels not found. Please run train_model.py first!")
        return

    # Load label mapping dictionary
    with open(labels_path, "rb") as f:
        label_ids = pickle.load(f)

    # Initialize recognizer and load trained weights
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_path)

    face_cascade = cv2.CascadeClassifier(cascade_filename)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("\n📷 Starting real-time facial recognition...")
    print("🖥️ A live video window will open. Look into the camera.")
    print("⌨️ Press the 'q' key inside the video window to quit.\n")

    while True:
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()

        if not ret:
            print("❌ Failed to grab frame from camera.")
            break

        gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            roi_gray: np.ndarray = gray[y:y+h, x:x+w]
            roi_resized: np.ndarray = cv2.resize(roi_gray, (200, 200))

            # Predict identity using the trained LBPH model
            # Returns a tuple: (predicted_label_id, confidence_distance)
            # Note: In LBPH, lower confidence distance means a better match (0 is perfect)
            id_, confidence = recognizer.predict(roi_resized)

            # Define threshold for recognition (lower number = stricter matching)
            if confidence < 75:
                name: str = label_ids.get(id_, "Unknown")
                # Convert distance into a rough percentage score
                confidence_score: int = max(0, min(100, int(100 - confidence)))
                display_label = f"{name.capitalize()} ({confidence_score}%)"
                box_color = (0, 255, 0)  # Green for recognized
            else:
                display_label = "Unknown"
                box_color = (0, 0, 255)  # Red for unknown

            # Draw bounding box and label on the live video frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(
                img=frame,
                text=display_label,
                org=(x, y - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=box_color,
                thickness=2
            )

        # Show live video window
        cv2.imshow("Real-Time Facial Recognition - Press 'q' to Exit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Recognition session closed. Program terminated cleanly.")

if __name__ == "__main__":
    recognize_faces()