import cv2
from flask import Flask, render_template, Response
import numpy as np
import os

app = Flask(__name__)

# Initialize the default web camera (Mac FaceTime HD camera)
camera = cv2.VideoCapture(0)

# Load Haar cascade locally for live face detection on the web stream
cascade_filename: str = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_filename)

def generate_frames():
    """Generator function that yields MJPEG video frames from the webcam."""
    while True:
        success: bool
        frame: np.ndarray
        success, frame = camera.read()
        
        if not success:
            break
        
        # Optional: Apply real-time face detection on the streaming frame
        if not face_cascade.empty():
            gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Web Stream Active", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Encode the OpenCV frame as a JPEG image
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in multipart HTTP response format (MJPEG)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Renders the main dashboard webpage."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route that serves the streaming video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the Flask development server on localhost port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)