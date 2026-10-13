# OpenCV Real-Time Facial Recognition and Security Dashboard

A computer vision and security system built using Python, OpenCV, Machine Learning (LBPH), and Flask. The system performs real-time face detection and recognition, triggers automated intrusion alerts with desktop notifications, and streams live feeds to a local web dashboard.

## Features

* **Face Detection**: Identifies human faces in live webcam streams using OpenCV Haar Cascades.
* **Facial Recognition**: Matches identities against trained face datasets using the Local Binary Patterns Histograms (LBPH) algorithm.
* **Intrusion Monitoring**: Enforces cooldown periods, captures timestamped snapshots of unrecognized individuals, and issues native macOS desktop notifications.
* **Web Dashboard**: Broadcasts an MJPEG video stream via a Flask backend server to a responsive HTML5 interface.

## File Structure

* `app.py`: Flask backend server and streaming interface.
* `collect_data.py`: Gathers and normalizes training image samples from the webcam.
* `train_model.py`: Trains the LBPH recognition model and serializes output weights.
* `recognize_face.py`: Real-time identification script with confidence thresholding.
* `security_alert.py`: Security monitor for intrusion logging and system notifications.
* `templates/index.html`: Frontend dashboard interface.

## Technology Stack

* **Language**: Python
* **Libraries**: OpenCV (`cv2`), NumPy, Flask
* **Platform**: macOS