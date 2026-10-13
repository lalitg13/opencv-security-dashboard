import cv2
import numpy as np
import os
import pickle

def train_recognizer() -> None:
    dataset_dir: str = "dataset"
    if not os.path.exists(dataset_dir):
        print("❌ Error: 'dataset' folder not found. Run collect_data.py first.")
        return

    # Initialize the LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    cascade_filename: str = "haarcascade_frontalface_default.xml"
    
    if not os.path.exists(cascade_filename):
        print(f"❌ Error: {cascade_filename} not found.")
        return

    face_cascade = cv2.CascadeClassifier(cascade_filename)

    faces = []
    labels = []
    label_ids = {}
    current_id = 0

    print("🔄 Scanning dataset folder and preparing training data...")

    # Walk through the dataset directory structure
    for root, dirs, files in os.walk(dataset_dir):
        for dir_name in dirs:
            label_ids[current_id] = dir_name
            subject_dir = os.path.join(root, dir_name)
            
            for file in os.listdir(subject_dir):
                if file.endswith(".jpg") or file.endswith(".png"):
                    path = os.path.join(subject_dir, file)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        faces.append(img)
                        labels.append(current_id)
            current_id += 1

    if not faces:
        print("❌ Error: No training images found in the dataset folder.")
        return

    # Train the recognizer model with our collected faces and integer labels
    recognizer.train(faces, np.array(labels))

    # Save the trained model and label dictionary to disk
    os.makedirs("trainer", exist_ok=True)
    trainer_path: str = "trainer/trainer.yml"
    labels_path: str = "trainer/labels.pickle"
    
    recognizer.save(trainer_path)
    with open(labels_path, "wb") as f:
        pickle.dump(label_ids, f)

    print(f"\n✅ Success! Model trained with {len(faces)} images across {len(label_ids)} subject(s).")
    print(f"💾 Saved trained model to '{trainer_path}' and label map to '{labels_path}'.")

if __name__ == "__main__":
    train_recognizer()