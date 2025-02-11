import cv2
import torch
from ultralytics import YOLO
from fer import FER

emotions1 = []  # Stores detected emotions

# Load YOLOv8 model for face detection
yolo_model = YOLO("LateFusion/yolov8n-face.pt")  # Make sure this file exists in your project

# Load FER for emotion recognition
emotion_detector = FER(mtcnn=True)

# Function to detect face using YOLOv8 and classify emotions (No Bounding Box)
def detect_emotion_from_frame(frame):
    results = yolo_model(frame)  # Run YOLO face detection
    emotion_labels = []  # Store detected emotions for this frame

    h, w, _ = frame.shape  # Get frame dimensions

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get face bounding box
            confidence = float(box.conf[0])  # Confidence score

            # 🔹 Adjust confidence threshold (detects smaller & lower faces)
            if confidence > 0.3:  
                # 🔹 Ensure bounding box is inside image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face_roi = frame[y1:y2, x1:x2]  # Crop the detected face

                if face_roi.size == 0:  # 🔹 Avoid processing empty crops
                    continue  

                # Perform emotion detection on the cropped face
                emotion_results = emotion_detector.detect_emotions(face_roi)

                if emotion_results:
                    emotions = emotion_results[0]["emotions"]
                    
                    # Only keep Happy, Sad, and Neutral (Relax)
                    happy = emotions.get("happy", 0)
                    sad = emotions.get("sad", 0)
                    neutral = emotions.get("neutral", 0)  # Using neutral as "Relax"

                    # Determine dominant emotion
                    if happy > max(sad, neutral):
                        emotion_label = "Happy"
                    elif sad > max(happy, neutral):
                        emotion_label = "Sad"
                    else:
                        emotion_label = "Relax"

                    emotion_labels.append(emotion_label)
    
    return emotion_labels
