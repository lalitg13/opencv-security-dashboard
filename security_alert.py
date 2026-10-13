import cv2
import numpy as np
import os
import pickle
import time
import subprocess
from datetime import datetime

def security_monitor() -> None:
    trainer_path: str = "trainer/trainer.yml"
    labels_path: str = "trainer/labels.pickle"
    cascade_filename: str = "haarcascade_frontalface_default.xml"
    alerts_dir: str = "unknown_alerts"

    # Verify model assets exist
    if not os.path.exists(trainer_path) or not os.path.exists(labels_path):
        print("❌ Error: Trained model or labels not found. Run train_model.py first!")
        return

    # Ensure alerts directory exists
    os.makedirs(alerts_dir, exist_ok=True)

    # Load labels and trained model
    with open(labels_path, "rb") as f:
        label_ids = pickle.load(f)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_path)

    face_cascade = cv2.CascadeClassifier(cascade_filename)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("\n🚨 Starting AI Security Monitor...")
    print("🖥️ Monitoring for unknown faces. Snapshots and macOS notifications will trigger on intrusion.")
    print("⌨️ Press 'q' inside the video window to stop.\n")

    last_alert_time: float = 0.0
    cooldown_seconds: float = 5.0  # Prevents alert spamming (waits 5 seconds between alerts)

    while True:
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()

        if not ret:
            print("❌ Failed to grab frame from camera.")
            break

        gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        current_time: float = time.time()
        unknown_detected_in_frame: bool = False

        for (x, y, w, h) in faces:
            roi_gray: np.ndarray = gray[y:y+h, x:x+w]
            roi_resized: np.ndarray = cv2.resize(roi_gray, (200, 200))

            id_, confidence = recognizer.predict(roi_resized)

            if confidence < 75:
                name: str = label_ids.get(id_, "Unknown")
                confidence_score: int = max(0, min(100, int(100 - confidence)))
                display_label = f"{name.capitalize()} ({confidence_score}%)"
                box_color = (0, 255, 0)  # Green for recognized
            else:
                display_label = "Unknown"
                box_color = (0, 0, 255)  # Red for unknown
                unknown_detected_in_frame = True

            # Draw bounding box and label
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

        # Trigger alert if an unknown face is present and cooldown period has passed
        if unknown_detected_in_frame and (current_time - last_alert_time > cooldown_seconds):
            last_alert_time = current_time
            timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            snapshot_path: str = os.path.join(alerts_dir, f"unknown_{timestamp}.jpg")

            # Save the video frame to disk
            cv2.imwrite(snapshot_path, frame)
            print(f"🚨 [SECURITY ALERT] Unknown face detected! Snapshot saved to '{snapshot_path}'")

            # Trigger native macOS desktop notification using AppleScript (osascript)
            try:
                applescript_command: str = (
                    f'display notification "An unrecognized face was detected on your camera!" '
                    f'with title "Security Alert 🚨" sound name "Sosumi"'
                )
                subprocess.run(["osascript", "-e", applescript_command], check=False)
            except Exception as e:
                print(f"⚠️ Could not trigger macOS notification: {e}")

        # Display live security feed
        cv2.imshow("AI Security Monitor - Press 'q' to Exit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Security monitor terminated cleanly.")

if __name__ == "__main__":
    security_monitor()