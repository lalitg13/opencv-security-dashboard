import cv2
import numpy as np
import os
import urllib.request

def detect_faces() -> None:
    cascade_filename: str = "haarcascade_frontalface_default.xml"
    
    # Automatically download the Haar cascade file if it's missing from the folder
    if not os.path.exists(cascade_filename):
        print("📥 Haar cascade file not found locally. Downloading official file...")
        url: str = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try:
            urllib.request.urlretrieve(url, cascade_filename)
            print("✅ Successfully downloaded Haar cascade file!")
        except Exception as e:
            print(f"❌ Failed to download file automatically: {e}")
            return

    # 1. Load the Haar Cascade classifier from the local file
    face_cascade: cv2.CascadeClassifier = cv2.CascadeClassifier(cascade_filename)

    if face_cascade.empty():
        print(f"❌ Error: Could not load Haar cascade from {cascade_filename}. The file may be corrupted.")
        return

    # 2. Open the default web camera (Mac FaceTime HD camera)
    cap: cv2.VideoCapture = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam. Ensure no other application is using it and camera permissions are granted.")
        return

    print("📷 Webcam opened successfully!")
    print("🖥️ A live video window will open. Position your face in front of the camera.")
    print("⌨️ Press the 'q' key inside the video window to quit.")

    while True:
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()

        if not ret:
            print("❌ Error: Failed to grab frame from camera.")
            break

        # 3. Convert frame to grayscale for the cascade detector
        gray_frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 4. Detect faces
        faces = face_cascade.detectMultiScale(
            image=gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # 5. Draw bounding boxes around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(
                img=frame,
                pt1=(x, y),
                pt2=(x + w, y + h),
                color=(0, 255, 0),
                thickness=2
            )
            cv2.putText(
                img=frame,
                text="Face Detected",
                org=(x, y - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0, 255, 0),
                thickness=2
            )

        # 6. Display live video feed
        cv2.imshow("Live Face Detection - Press 'q' to Exit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7. Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Webcam closed. Program terminated cleanly.")

if __name__ == "__main__":
    detect_faces()