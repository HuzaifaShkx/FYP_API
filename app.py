import os
import cv2
import numpy as np
import pickle
import torch
from torch import nn
from scipy.signal import butter, lfilter
import pyodbc
from flask import Flask, jsonify,request,send_file
import json
from datetime import datetime
import pandas as pd
from scipy.signal import butter, filtfilt
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import load_model
import torch
from ultralytics import YOLO
from fer import FER
from collections import Counter




app=Flask(__name__)

UPLOAD_FOLDER_Images = 'FYPAPIs/Uploads/Images/'
UPLOAD_FOLDER_EEG = 'FYPAPIs/Uploads/eeg/'
UPLOAD_FOLDER_recorded = 'FYPAPIs/Uploads/RecordedVideos/'
ALLOWED_EXTENSIONS_img = {'png', 'jpg', 'jpeg'}
ALLOWED_EXTENSIONS_eeg = {'dat','edf','cnt','csv'}
ALLOWED_EXTENSIONS_video = {'avi','mp4'}
UPLOAD_FOLDER_watchedVideo = 'FYPAPIs/Uploads/watchedvideos/'

conn = pyodbc.connect('DRIVER={SQL Server};SERVER=DESKTOP-8BL3MIG;DATABASE=EEG_FYP;Trusted_Connection=yes;')
#cursor=conn.cursor()
# Load YOLOv8 model for face detection
yolo_model = YOLO("FYPAPIs\yolov8n-face.pt")  # Make sure this file exists in your project

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
def predict_eeg_fuse(model, csv_file, lowcut=1.0, highcut=50.0, sampling_rate=256, chunk_size=1536):
    """
    Predict the class for a full EEG CSV file using the trained model.

    Parameters:
    - model: Trained EEGNet model.
    - csv_file: Path to the EEG CSV file.
    - lowcut, highcut: Band-pass filter cutoff frequencies.
    - sampling_rate: EEG sampling rate in Hz.
    - chunk_size: Number of samples per chunk.

    Returns:
    - predictions: Array of predicted class probabilities for each chunk.
    - predicted_classes: Array of predicted class labels for each chunk.
    """
    # Load the EEG data
    data = pd.read_csv(csv_file)
    columns_to_drop = ['timestamps']  # Drop unnecessary columns if present
    data = data.drop(columns=columns_to_drop, errors='ignore')
    eeg_array = data.to_numpy()

    # Apply band-pass filter
    eeg_filtered = bandpass_filter(eeg_array, lowcut, highcut, sampling_rate)

    # Chunk the filtered data
    num_chunks = len(eeg_filtered) // chunk_size
    if num_chunks == 0:
        raise ValueError("EEG data is too short for the specified chunk size.")

    eeg_chunks = eeg_filtered[:num_chunks * chunk_size].reshape(num_chunks, chunk_size, eeg_array.shape[1])
    eeg_chunks = np.transpose(eeg_chunks, (0, 2, 1))  # Shape: (num_chunks, channels, samples)
    eeg_chunks = eeg_chunks[..., np.newaxis]  # Add extra dimension for model input: (num_chunks, channels, samples, 1)

    # Predict using the model
    predictions = model.predict(eeg_chunks)  # Shape: (num_chunks, num_classes)
    predicted_classes = np.argmax(predictions, axis=1)  # Get the class labels (0, 1, 2)

    return predictions, predicted_classes

def load_single_trial(file_path, trial_index):
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
        eeg_data = data['data'][trial_index, :32, :]  # Load all 32 channels for the specified trial
        labels = data['labels'][trial_index]  # Load the corresponding label for the trial
    return eeg_data, labels

# Bandpass filter for EEG data (1-50Hz)
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=1.0, highcut=50.0, fs=128.0, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=-1)
    return y

# Read the EEG CSV file
file_path = "../EEG_Related_Data/EEG's/2021-Arid-0194-Happy.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Define frequency bands
sampling_rate = 256  # Adjust based on the data sampling rate (samples per second)

# Define filter functions
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Apply filters for each band on the channels
channels = ['TP9', 'AF7', 'AF8', 'TP10']
bands = ['Delta', 'Theta', 'Alpha', 'Beta']
filtered_data = {}

for channel in channels:
    filtered_data[channel] = {}
    for band in bands:
        if band == 'Delta':
            lowcut, highcut = 0.5, 4
        elif band == 'Theta':
            lowcut, highcut = 4, 8
        elif band == 'Alpha':
            lowcut, highcut = 8, 12
        elif band == 'Beta':
            lowcut, highcut = 12, 30

        filtered_data[channel][band] = bandpass_filter(data[channel], lowcut, highcut, sampling_rate)


# # Preprocessing EEG signals
# def preprocess_eeg(eeg_data):
#     eeg_data = bandpass_filter(eeg_data)
#     eeg_data = (eeg_data - np.mean(eeg_data, axis=0, keepdims=True)) / np.std(eeg_data, axis=0, keepdims=True)
#     return eeg_data

# # Feature extraction - reshape for CNN input
# def prepare_data_for_cnn(eeg_data):
#     num_channels, num_timepoints = eeg_data.shape
#     eeg_data = eeg_data.reshape(1, 1, num_channels, num_timepoints)  # [batch_size, channels, height, width]
#     return eeg_data


# # Map DEAP values to emotions based on predefined thresholds
# def map_to_emotion(valence, arousal, dominance, liking):
#     if valence > 5 and arousal > 5:
#         return 'Happy'
#     elif valence < 5 and arousal > 5:
#         return 'Angry'
#     elif valence < 5 and arousal < 5:
#         return 'Sad'
#     elif valence > 5 and arousal < 5:
#         return 'Surprise'
#     elif valence <= 5 and arousal <= 5:
#         return 'Neutral'
#     elif valence > 5 and arousal <= 5:
#         return 'Relaxed'
#     else:
#         return 'unknown'

# class EEGNet(nn.Module):
#     def __init__(self, num_classes):
#         super(EEGNet, self).__init__()

#         self.block1 = nn.Sequential(
#             nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),  # Same padding
#             nn.BatchNorm2d(16),
#             nn.Conv2d(16, 32, (32, 1), groups=16, bias=False),  # Depthwise convolution
#             nn.BatchNorm2d(32),
#             nn.ELU(),
#             nn.AvgPool2d((1, 4)),
#             nn.Dropout(0.5)
#         )

#         self.block2 = nn.Sequential(
#             nn.Conv2d(32, 16, (1, 16), bias=False),
#             nn.BatchNorm2d(16),
#             nn.ELU(),
#             nn.AvgPool2d((1, 8)),
#             nn.Dropout(0.5)
#         )

#         # Automatically calculate the size of the flatten layer by passing dummy input
#         self.flatten_size = self._get_flatten_size()
#         self.fc = nn.Linear(self.flatten_size, num_classes)

#     def _get_flatten_size(self):
#         with torch.no_grad():
#             x = torch.randn(1, 1, 32, 8064)  # Use the actual input shape here
#             x = self.block1(x)
#             x = self.block2(x)
#             flatten_size = x.view(1, -1).size(1)
#         return flatten_size

#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = x.view(x.size(0), -1)  # Flatten
#         x = self.fc(x)
#         return x

# # Predict the emotion from a single trial using a trained model
# def predict_emotion(model, eeg_data, device):
#     # Preprocess the data
#     eeg_data = preprocess_eeg(eeg_data)
#     X = prepare_data_for_cnn(eeg_data)

#     # Convert data to PyTorch tensor and move it to the specified device (CPU/GPU)
#     X = torch.tensor(X, dtype=torch.float32).to(device)  # No need for unsqueeze(0), already has batch dimension

#     # Set the model to evaluation mode
#     model.eval()

#     # Make prediction
#     with torch.no_grad():
#         predictions = model(X)
    
#     predicted_emotion = torch.argmax(predictions, dim=1).item()
#     confidence_score = torch.max(predictions).item()

#     return predicted_emotion, confidence_score

# Band-pass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# Prediction function
def predict_eeg(model, csv_file, lowcut=1.0, highcut=50.0, sampling_rate=256, chunk_size=1536):
    """
    Predict the class for a full EEG CSV file using the trained model.

    Parameters:
    - model: Trained EEGNet model.
    - csv_file: Path to the EEG CSV file.
    - lowcut, highcut: Band-pass filter cutoff frequencies.
    - sampling_rate: EEG sampling rate in Hz.
    - chunk_size: Number of samples per chunk.

    Returns:
    - predictions: Array of predicted class probabilities for each chunk.
    - predicted_classes: Array of predicted class labels for each chunk.
    """
    # Load the EEG data
    data = pd.read_csv(csv_file)
    columns_to_drop = ['timestamps']  # Drop unnecessary columns if present
    data = data.drop(columns=columns_to_drop, errors='ignore')
    eeg_array = data.to_numpy()

    # Apply band-pass filter
    eeg_filtered = bandpass_filter(eeg_array, lowcut, highcut, sampling_rate)

    # Chunk the filtered data
    num_chunks = len(eeg_filtered) // chunk_size
    if num_chunks == 0:
        raise ValueError("EEG data is too short for the specified chunk size.")

    eeg_chunks = eeg_filtered[:num_chunks * chunk_size].reshape(num_chunks, chunk_size, eeg_array.shape[1])
    eeg_chunks = np.transpose(eeg_chunks, (0, 2, 1))  # Shape: (num_chunks, channels, samples)
    eeg_chunks = eeg_chunks[..., np.newaxis]  # Add extra dimension for model input: (num_chunks, channels, samples, 1)

    # Predict using the model
    predictions = model.predict(eeg_chunks)  # Shape: (num_chunks, num_classes)
    predicted_classes = np.argmax(predictions, axis=1)  # Get the class labels (0, 1, 2)

    return predictions, predicted_classes

def allowed_file_img(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_img

def allowed_file_eeg(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_eeg

def allowed_file_video(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_video


# Define the API route
@app.route('/eeg_bands', methods=['GET'])
def get_eeg_bands():
    for channel in filtered_data:
        for band in filtered_data[channel]:
            if isinstance(filtered_data[channel][band], np.ndarray):
                filtered_data[channel][band] = filtered_data[channel][band].tolist()

    # Prepare the response with filtered data
   
    # Return the data as a JSON response
    return jsonify(filtered_data)

@app.route('/eeg_bands_file/<string:filepath>', methods=['GET'])
def get_eeg_bands_file(filepath):
    try:
        data = pd.read_csv(f"C:/FYP_Code/FYPAPIs/Uploads/eeg/{filepath}")
        # data = pd.read_csv(f"{filepath}")
        # Define frequency bands
        sampling_rate = 256  # Adjust based on the data sampling rate (samples per second)


        # Apply filters for each band on the channels
        channels = ['TP9', 'AF7', 'AF8', 'TP10']
        bands = ['Delta', 'Theta', 'Alpha', 'Beta']
        filtered_data = {}

        for channel in channels:
            filtered_data[channel] = {}
            for band in bands:
                if band == 'Delta':
                    lowcut, highcut = 0.5, 4
                elif band == 'Theta':
                    lowcut, highcut = 4, 8
                elif band == 'Alpha':
                    lowcut, highcut = 8, 12
                elif band == 'Beta':
                    lowcut, highcut = 12, 30

                filtered_data[channel][band] = bandpass_filter(data[channel], lowcut, highcut, sampling_rate)


        for channel in filtered_data:
            for band in filtered_data[channel]:
                if isinstance(filtered_data[channel][band], np.ndarray):
                    filtered_data[channel][band] = filtered_data[channel][band].tolist()

        # Prepare the response with filtered data
    
        # Return the data as a JSON response
        return jsonify(filtered_data),200
    except Exception:
        return 500

@app.route('/DoctorSignup', methods=['POST'])
def add_doctor():
    try:
        filepath=''
        user = request.form.get('user')
        file=request.files.get("image")
        if not user:
            return 'No user data found'

        # Convert user data JSON to Python dictionary
        user_dict = json.loads(user)

        required_fields = ['email', 'name', 'password', 'gender', 'dob', 'role','contact']
        for field in required_fields:
            if field not in user_dict:
                return f'Missing required field: {field}'

    # Check if the user with the same email already exists
        cursor=conn.cursor()
        cursor.execute('SELECT * FROM [User] WHERE ID=?', (user_dict['email'],))
        existing_user = cursor.fetchone()
        if existing_user:
            return jsonify({"status": "User with the same email already exists"}), 409
        # Get image file from the request
        if not file:
            filepath=""
        elif file and allowed_file_img(file.filename):    
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            filepath = os.path.join(UPLOAD_FOLDER_Images, f"{timestamp}{file.filename}")
            file.save(filepath)
            
       
        cursor.execute('insert into [user] values(?,?,?,?,?,?,?,?)',(user_dict['email'],user_dict['name'],user_dict['password'],user_dict['gender'],user_dict['dob'],user_dict['role'],filepath,user_dict['role']))
        cursor.execute('select * from [User] where name=? and ID=?',user_dict["name"],user_dict["email"])
        result=cursor.fetchone()
        if result:
            
            cursor.execute('insert into doctor values(?,?)',user_dict['email'],user_dict['contact'])
            conn.commit()
            cursor.close()

            return jsonify({"status":"SignUp"}),200
        else:
            conn.rollback()
            cursor.close()
            
        return jsonify({"status":"SignUp Failed"}),201
    except Exception as e:
        #cursor.close()
        return jsonify({"Exception":str(e)}),500

@app.route('/PatientSignup', methods=['POST'])
def add_patient():
    try:
        filepath=''
        user = request.form.get('user')
        file=request.files.get("image")
        if not user:
            return 'No user data found'

        # Convert user data JSON to Python dictionary
        user_dict = json.loads(user)

        required_fields = ['email', 'name', 'password', 'gender', 'dob', 'role','weight','height','contact']
        for field in required_fields:
            if field not in user_dict:
                return f'Missing required field: {field}',400

    # Check if the user with the same email already exists
        cursor=conn.cursor()
        cursor.execute('SELECT * FROM [User] WHERE ID=?', (user_dict['email'],))
        existing_user = cursor.fetchone()
        if existing_user:
            return jsonify({"status": "User with the same email already exists"}), 409
        # Get image file from the request
        if not file:
            filepath=""    
        elif file and allowed_file_img(file.filename):    
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            filepath = os.path.join(UPLOAD_FOLDER_Images, f"{timestamp}{file.filename}")
            file.save(filepath)
        
        cursor.execute('insert into [user] values(?,?,?,?,?,?,?,?)',(user_dict['email'],user_dict['name'],user_dict['password'],user_dict['gender'],user_dict['dob'],user_dict['role'],filepath,user_dict['role']))
        cursor.execute('select * from [User] where ID=?',user_dict["email"])
        result=cursor.fetchone()
        if result:
            cursor.execute('insert into patient values(?,?,?,?)',(user_dict["weight"],user_dict["height"],user_dict['email'],user_dict['contact']))
            conn.commit()
            cursor.close()
        
            return jsonify({"status":"SignUp"}),200
        else:
            conn.rollback()
            cursor.close()
        return jsonify({"status":"SignUp Failed"}),201
    except Exception as e:
       # cursor.close()
        return jsonify({"Exception":str(e)}),500

@app.route('/SupervisorSignup', methods=['POST'])
def add_supervisor():
    try:
        filepath=''
        user = request.form.get('user')
        file=request.files.get("image")
        if not user:
            return 'No user data found'

        # Convert user data JSON to Python dictionary
        user_dict = json.loads(user)

        required_fields = ['email', 'name', 'password', 'gender', 'dob', 'role']
        for field in required_fields:
            if field not in user_dict:
                return f'Missing required field: {field}'

    # Check if the user with the same email already exists
        cursor=conn.cursor()
        cursor.execute('SELECT * FROM [User] WHERE ID=?', (user_dict['email']))
        existing_user = cursor.fetchone()
        if existing_user:
            return jsonify({"status": "User with the same email already exists"}), 409
        # Get image file from the request
        if not file:
            filepath=""
        
        elif file and allowed_file_img(file.filename):    
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            filepath = os.path.join(UPLOAD_FOLDER_Images, f"{timestamp}{file.filename}")
            file.save(filepath)
        
        cursor.execute('insert into [user] values(?,?,?,?,?,?,?,?)',(user_dict['email'],user_dict['name'],user_dict['password'],user_dict['gender'],user_dict['dob'],user_dict['role'],filepath,user_dict['role']))
        cursor.execute('select * from [User] where ID=?',user_dict["email"])
        result=cursor.fetchone()
        if result:
            
            cursor.execute('insert into Supervisor values(?)',user_dict['email'])
            # Retrieve the last inserted ID
            cursor.execute('SELECT id from Supervisor where UserId=?',user_dict['email'])
            supervisor_id = cursor.fetchone()[0]
            cursor.execute('insert into DoctorSupervisorRelationship values(?,?)',user_dict['doctor_id'],supervisor_id)
            conn.commit()
            cursor.close()

            return jsonify({"status":"SignUp"}),200
        else:
            conn.rollback()
            cursor.close()
        
        return jsonify({"status":"SignUp Failed"}),201
    except Exception as e:    
        #cursor.close()
        return jsonify({"Exception":str(e)}),500

@app.route('/login',methods=['POST'])
def login():
    try:
        cursor=conn.cursor()
        cursor.execute('select * from [User] where password=? and ID=?',request.json["password"],request.json["id"])
        user=cursor.fetchone()
        if user:
            cursor.close()
        
            return jsonify({"status":"Login","role":user[7]}),200
        else:
            cursor.close()
            return jsonify({"status":"Incorrect ID or Password"}),401
    except Exception as e:
        return jsonify({"Exception":str(e)}),500


@app.route('/getSupervisorDoctor/<int:sup_id>')
def getSupDoctor(sup_id):
    try:
        cursor=conn.cursor()
        cursor.execute('select doctor_id from DoctorSupervisorRelationship where supervisor_id=?',(sup_id))
        doctor_id=cursor.fetchone()
        if doctor_id:
            d_id={"doctor_id":doctor_id[0]}
            cursor.close()

            return jsonify(d_id),200
        else:
            cursor.close()
    
            return jsonify({"status":"User with this id does not exist"}),405
    except Exception as e:
        return jsonify({"Exception":str(e)}),500
    

@app.route('/getAllPatient')
def getAllPatient():
    try:
        cursor = conn.cursor()
        patientlist=[]
        cursor.execute('select p.Id,u.name,u.gender,u.dob,p.height,p.weight,p.contact,u.imgPath from Patient [p] join [User] [u] on u.Id=p.UserId')
        patient=cursor.fetchall()
        if patient==None:
            cursor.close()
        
            return jsonify({"status":"No Patient Found"}),401
        for i in patient:
            patient = {
            "id":i[0],
            "name": i[1],
            "gender": i[2],
            "dob": i[3],
            "height": i[4],
            "weight": i[5],
            "contact":i[6],
            "imgPath":i[7]
            }
            patientlist.append(patient)
        cursor.close()
    
        return jsonify(patientlist),200
    except Exception as e:
        return jsonify({"Exception":str(e)}),500

@app.route('/getAllDoctors')
def getAllDoctors():
    doctorlist=[]
    try:
        cursor=conn.cursor()
        cursor.execute('select u.Id,u.name,u.gender,u.dob,d.contact,u.ImgPath from Doctor [d] join [User] [u] on u.Id=d.UserId')
        doctors=cursor.fetchall()
        if doctors==None:
            cursor.close()
            return jsonify({"status":"No Doctor Found"}),401
        for i in doctors:
            doctor= {
            "email":i[0],
            "name": i[1],
            "gender": i[2],
            "dob": i[3],
            "contact": i[4],
            "imgpath": i[5],
            }
            doctorlist.append(doctor)
        cursor.close()
    
        return jsonify(doctorlist),200
    except Exception as e:
        return jsonify({"Exception":str(e)}),500

    
@app.route('/getPatientById/<int:patient_id>')
def getPatient(patient_id):
    try:
        cursor=conn.cursor()
        cursor.execute('select p.id,u.name,u.gender,u.dob,p.height,p.weight,p.contact,u.imgPath from Patient [p] join [User] [u] on u.Id=p.UserId where p.id=?',(patient_id))
        patient=cursor.fetchone()
        if patient:
            patient_dict = {
            "id":patient[0],
            "name": patient[1],
            "gender": patient[2],
            "dob": patient[3],
            "height": patient[4],
            "weight": patient[5],
            "contact":patient[6],
            "imgpath":patient[7]
            }
            cursor.close()

            return jsonify(patient_dict),200
        else:
            cursor.close()
    
            return jsonify({"status":"User with this id does not exist"}),405
    except Exception as e:
        return jsonify({"Exception":str(e)}),500
    
@app.route('/getPatientByEmail/<string:patient_email>')
def getPatientByEmail(patient_email):
    try:
        cursor=conn.cursor()
        cursor.execute('select p.id,u.name,u.gender,u.dob,p.height,p.weight,p.contact,u.imgPath from Patient [p] join [User] [u] on u.Id=p.UserId where u.id=?',(patient_email))
        patient=cursor.fetchone()
        if patient:
            patient_dict = {
            "id":patient[0],
            "name": patient[1],
            "gender": patient[2],
            "dob": patient[3],
            "height": patient[4],
            "weight": patient[5],
            "contact":patient[6],
            "imgpath":patient[7]
            }
            cursor.close()

            return jsonify(patient_dict),200
        else:
            cursor.close()
    
            return jsonify({"status":"User with this email does not exist"}),405
    except Exception as e:
        return jsonify({"Exception":str(e)}),500
    
@app.route('/getDoctorByEmail/<string:doctor_email>')
def getDoctorByEmail(doctor_email):
    try:
        cursor=conn.cursor()
        cursor.execute('select d.id,u.name,u.gender,u.dob,d.contact,u.imgPath from Doctor [d] join [User] [u] on u.Id=d.UserId where u.id=?',(doctor_email))
        patient=cursor.fetchone()
        if patient:
            patient_dict = {
            "id":patient[0],
            "name": patient[1],
            "gender": patient[2],
            "dob": patient[3],
            "contact":patient[4],
            "imgpath":patient[5]
            }
            cursor.close()

            return jsonify(patient_dict),200
        else:
            cursor.close()
    
            return jsonify({"status":"User with this email does not exist"}),405
    except Exception as e:
        return jsonify({"Exception":str(e)}),500
    
@app.route('/getDoctorById/<int:doctor_id>')
def getDoctorById(doctor_id):
    try:
        cursor=conn.cursor()
        cursor.execute('select d.id,u.name,u.gender,u.dob,d.contact,u.imgPath from Doctor [d] join [User] [u] on u.Id=d.UserId where d.id=?',(doctor_id))
        patient=cursor.fetchone()
        if patient:
            patient_dict = {
            "id":patient[0],
            "name": patient[1],
            "gender": patient[2],
            "dob": patient[3],
            "contact":patient[4],
            "imgpath":patient[5]
            }
            cursor.close()

            return jsonify(patient_dict),200
        else:
            cursor.close()
    
            return jsonify({"status":"User with this email does not exist"}),405
    except Exception as e:
        return jsonify({"Exception":str(e)}),500
    
@app.route('/getSupervisorById/<int:supervisor_id>')
def getSupervisorById(supervisor_id):
    try:
        cursor=conn.cursor()
        cursor.execute('select s.id,u.id,u.name,u.gender,u.dob,u.imgPath from Supervisor [s] join [User] [u] on u.Id=s.UserId where s.Id=?',(supervisor_id))
        supervisor=cursor.fetchone()
        if supervisor:
            supervisor_dict = {
            "id":supervisor[0],
            "email": supervisor[1],
            "name": supervisor[2],
            "gender": supervisor[3],
            "dob": supervisor[4],
            "imgpath":supervisor[5]
            }
            cursor.close()

            return jsonify(supervisor_dict),200
        else:
            cursor.close()
    
            return jsonify({"status":"User with this id does not exist"}),405
    except Exception as e:
        return jsonify({"Exception":str(e)}),500
    
@app.route('/getSupervisorByEmail/<supervisor_email>')
def getSupervisorByEmail(supervisor_email):
    try:
        cursor=conn.cursor()
        cursor.execute('select s.id,u.id,u.name,u.gender,u.dob,u.imgPath from Supervisor [s] join [User] [u] on u.Id=s.UserId where u.Id=?',(supervisor_email))
        supervisor=cursor.fetchone()
        if supervisor:
            supervisor_dict = {
            "id":supervisor[0],
            "email": supervisor[1],
            "name": supervisor[2],
            "gender": supervisor[3],
            "dob": supervisor[4],
            "imgpath":supervisor[5]
            }
            cursor.close()

            return jsonify(supervisor_dict),200
        else:
            cursor.close()
    
            return jsonify({"status":"User with this email does not exist"}),405
    except Exception as e:
        return jsonify({"Exception":str(e)}),500
#code for getting registered patients
@app.route('/getRegisteredPatient/<int:doctor_id>')
def getRegisteredPatient(doctor_id):
    patientlist=[]
    try:
        cursor=conn.cursor()
        cursor.execute("SELECT p.id,u.name,u.gender,u.dob,p.height,p.weight,p.contact,u.ImgPath FROM Patient p JOIN [User] u ON p.UserId = u.Id JOIN Appointment a ON p.Id = a.patient_id WHERE a.doctor_id = ? and a.appStatus='true';",doctor_id)
        patient=cursor.fetchall()
        if patient:
            for i in patient:
                patient = {
                "id":i[0],
                "name": i[1],
                "gender": i[2],
                "dob": i[3],
                "height": i[4],
                "weight": i[5],
                "contact":i[6],
                "imgpath":i[7]
                }
                patientlist.append(patient)
            cursor.close()
        
            return jsonify(patientlist),200
        else:
            cursor.close()
            
            return jsonify({"message":"Not Found"}),405
    except Exception as e:
        return jsonify({"Exception":str(e)}),500
    
@app.route('/getNewPatient/<int:doctor_id>')
def getNewPatient(doctor_id):
    patientlist=[]
    try:
        cursor=conn.cursor()
        cursor.execute("SELECT p.id,u.name,u.gender,u.dob,p.height,p.weight,p.contact,u.ImgPath,a.AppId FROM Patient p JOIN [User] u ON p.UserId = u.Id JOIN Appointment a ON p.Id = a.patient_id WHERE a.doctor_id = ? and a.appStatus='false';",doctor_id)
        patient=cursor.fetchall()
        if patient:
            for i in patient:
                patient = {
                "id":i[0],
                "name": i[1],
                "gender": i[2],
                "dob": i[3],
                "height": i[4],
                "weight": i[5],
                "contact":i[6],
                "imgpath":i[7],
                "appid":i[8]
                }
                patientlist.append(patient)
            cursor.close()
        
            return jsonify(patientlist),200
        else:
            cursor.close()
        
            return jsonify({"message":"Not Found"}),405
    except Exception as e:
        return jsonify({"Exception":str(e)}),500
from datetime import datetime
@app.route('/getTodaysAppointments/<int:doctor_id>')
def getTodaysAppointment(doctor_id):
    patientlist=[]
    current_datetime = datetime.now()

# Get just the current date
    current_date = current_datetime.strftime('%Y-%m-%d')
    try:
        cursor=conn.cursor()
        cursor.execute("SELECT p.id,u.name,u.gender,u.dob,p.height,p.weight,p.contact,u.ImgPath,a.AppId FROM Patient p JOIN [User] u ON p.UserId = u.Id JOIN Appointment a ON p.Id = a.patient_id WHERE a.doctor_id = ? and a.appStatus='false' and a.dates = ?;",(doctor_id,current_date))
        patient=cursor.fetchall()
        if patient:
            for i in patient:
                patient = {
                "id":i[0],
                "name": i[1],
                "gender": i[2],
                "dob": i[3],
                "height": i[4],
                "weight": i[5],
                "contact":i[6],
                "imgpath":i[7],
                "appid":i[8]
                }
                patientlist.append(patient)
            cursor.close()
        
            return jsonify(patientlist),200
        else:
            cursor.close()
        
            return jsonify({"message":"Not Found"}),405
    except Exception as e:
        return jsonify({"Exception":str(e)}),500
    
@app.route('/getNewPatientAppointmentDate/<int:doctor_id>/<int:patient_id>')
def getNewPatientAppointmentDate(doctor_id, patient_id):
    
    try:
        cursor=conn.cursor()
        cursor.execute("SELECT a.dates,a.times FROM Patient p JOIN [User] u ON p.UserId = u.Id JOIN Appointment a ON p.Id = a.patient_id WHERE a.doctor_id = ? and a.patient_id=? and a.appStatus='false' ORDER BY a.dates DESC;",doctor_id,patient_id)
        p=cursor.fetchall()[0]
        if p:
            patient = {
                "time":p[1],
                "date": p[0],
                }
               
            cursor.close()
        
            return jsonify(patient),200
        else:
            cursor.close()
        
            return jsonify({"message":"Not Found"}),405
    except Exception as e:
        return jsonify({"Exception":str(e)}),500


#add prescribtion to appointement
@app.route('/prescribe',methods=['POST'])
def addPrescribe():
    try:
        prescribe = request.get_json()
        cursor=conn.cursor()
        cursor.execute('Update Appointment set prescribtion=?,appStatus=? where AppId=?',prescribe["prescribtion"],prescribe["appStatus"],prescribe["appId"])
        rowcount=cursor.rowcount
        conn.commit()
        cursor.close()
        if rowcount>=1:
            return jsonify({"status":"Prescribtion added"}),200
    except Exception as e:
        return jsonify({"Exception":str(e)}),500
    
@app.route('/getPatientPrescription/<int:patient_id>')
def getPatientPrescription(patient_id):
    try:
        cursor=conn.cursor()
        cursor.execute("select u.name,u.ImgPath,app.prescribtion from Appointment app join Doctor d on app.doctor_id=d.Id join [User] u  on d.UserId=u.Id where app.patient_id=? and app.appStatus='true'",patient_id)
        pres=cursor.fetchall()
        preslist=[]
        if pres:
            for i in pres:
                prescription = {
                "name":i[0],
                "imgpath": i[1],
                "prescribtion": i[2]
                }
                preslist.append(prescription)
            cursor.close()
            return jsonify(preslist),200
        else:
            cursor.close()
            return jsonify({"message":"Not Found"}),405

    except Exception as e:
        cursor.close()
        return jsonify({"Exception":str(e)}),500


@app.route('/getTimeSlots', methods=['POST'])
def getTimeSlots():
    try:
        dat = request.form.get('date')
        did=request.form.get('doctor_id')
        cursor=conn.cursor()
        cursor.execute("select times,dates from Appointment where dates=? and doctor_id=?",(dat,did))
        slots=cursor.fetchall()
        slotslist=[]
        if slots:
            for i in slots:
                slot = {
                "time":i[0],
                "date": i[1],
               
                }
                slotslist.append(slot)
            cursor.close()
            return jsonify(slotslist),200
        else:
            cursor.close()
            return jsonify({"message":"Not Found"}),404

    except Exception as e:
        cursor.close()
        print(e)
        return jsonify({"Exception":str(e)}),500

@app.route('/addAppointment',methods=['POST'])
def addAppointment():
    try:
        appointment = request.get_json()
        cursor=conn.cursor()
        cursor.execute('insert into Appointment(dates,times,appStatus,prescribtion,doctor_id,patient_id) values (?,?,?,Null,?,?)',appointment['date'],appointment['time'],appointment['status'],appointment['doctorid'],appointment['patientid'])
        row=cursor.rowcount
        if row>=1:
            conn.commit()
            cursor.close()
            return jsonify({"status":"Appointment added"})
    except Exception as e:
        cursor.close()
        return jsonify({"Exception":str(e)}),500

@app.route('/addEEG',methods=['POST'])
def addEEG():
    try:
        
        file = request.files['eeg']
        channel=4
        if not file:
            return jsonify({"status":"No File selected"}),404
        elif file and allowed_file_eeg(file.filename):    
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            filepath = os.path.join(UPLOAD_FOLDER_EEG, f"{timestamp}{file.filename}")
            file.save(filepath)
        cursor=conn.cursor()
        cursor.execute('insert into EEG values (?,?)',channel,filepath)
        row=cursor.rowcount
        if row>=1:
            conn.commit()
            cursor.close()
            return jsonify({"status":"EEG Uploaded"}),200
        else:
            conn.rollback()
            cursor.close()
            return jsonify({"status":"EEG Not Uploaded"}),400
    except Exception as e:
        cursor.close()
        return jsonify({"Exception":str(e)}),500


@app.route('/addCaptureVideo',methods=['POST'])
def addCaptureVideo():
    try:
        filepath=""
        file = request.files['file']
        starttime=request.form.get('starttime')
        endtime=request.form.get('endtime')
        if not file:
            filepath=""
        elif file and allowed_file_video(file.filename):    
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            filepath = os.path.join(UPLOAD_FOLDER_recorded, f"{timestamp}{file.filename}")
            file.save(filepath)
        cursor=conn.cursor()
        cursor.execute('insert into CaptureVideo values(?,?,?)',(filepath,starttime,endtime))
        row=cursor.rowcount
        if row>=1:
            conn.commit()
            cursor.close()
            return jsonify({"status":"Video Uploaded"})
        else:
            conn.rollback()
            cursor.close()
            return jsonify({"status":"Video Not Uploaded"})
    except Exception as e:
        return jsonify({"Exception":str(e)}),500
    
@app.route('/uploadFusionPrediction',methods=['POST'])
def uploadFusionPrediction():
    eegfile = request.files['eeg']
    videofile = request.files['file']
    sessionid=request.form.get('sessionid')
    if not eegfile and not videofile:
        return jsonify({"status":"No eeg or video File selected"}),404
    elif eegfile and allowed_file_eeg(eegfile.filename):    
        eegfilepath = os.path.join(UPLOAD_FOLDER_EEG, f"{eegfile.filename}")
        eegfile.save(eegfilepath)
        videofilepath = os.path.join(UPLOAD_FOLDER_recorded, f"{videofile.filename}")
        videofile.save(videofilepath)
    cursor=conn.cursor()
    cursor.execute('insert into ExperimentFusion(EEGPath,VideoPath,sessionId) OUTPUT INSERTED.id values(?,?,?)',(eegfilepath,videofilepath,sessionid))
    result = cursor.fetchone()
    inserted_id = result[0] if result else None

    conn.commit()
    cursor.close()    
    return jsonify({"status":"Fusion Prediction files Uploaded",'id':inserted_id}),200



@app.route('/addWatchedVideo',methods=['POST'])
def addWatchedVideo():
    try:
        filepath=""
        file = request.files['file']
        category=request.form.get('category')
        if not file:
            filepath=""
        elif file and allowed_file_video(file.filename):    
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            filepath = os.path.join(UPLOAD_FOLDER_watchedVideo, f"{timestamp}{file.filename}")
            file.save(filepath)
        cursor=conn.cursor()
        cursor.execute('insert into WatchedVideo values(?,?)',filepath,category)
        row=cursor.rowcount
        if row>=1:
            conn.commit()
            cursor.close()
            return jsonify({"status":"Video Uploaded"})
        else:
            conn.rollback()
            cursor.close()
            return jsonify({"status":"Video Not Uploaded"})
    except Exception as e:
        conn.rollback()
        cursor.close()
        return jsonify({"Exception":str(e)}),500
    

@app.route('/emotionResult/<int:expeiment_id>')
def get_emotionResult(expeiment_id):
    try:
        cursor=conn.cursor()
        cursor.execute("select u.name,e.emotionType from Experiment ex join ExperimentEmotion exe on ex.ExperimentId=exe.experimentId join Emotion e on exe.emotionID=e.EmotionID join Patient p on ex.patientId=p.Id join [User] u on u.Id=p.UserId where ex.ExperimentId=?",expeiment_id)
        emotion=cursor.fetchall()
        if emotion:
            cursor.close()
            return jsonify({"patientName":emotion[0],"emotion":emotion[1]})
        else:
            cursor.close()
            return jsonify({"message":"No emotion Found"}),405

    except Exception as e:
        cursor.close()
        return jsonify({"Exception":str(e)}),500


@app.route('/DoctorUpdate', methods=['POST'])
def update_doctor():
    try:
        filepath=''
        user = request.form.get('user')
        file=request.files.get("image")
        if not user:
            return 'No user data found'

        user_dict = json.loads(user)

        required_fields = ['email', 'name', 'password', 'gender', 'dob', 'role','contact']
        for field in required_fields:
            if field not in user_dict:
                return f'Missing required field: {field}'

        cursor=conn.cursor()
        if not file:
            filepath=""
        elif file and allowed_file_img(file.filename):    
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            filepath = os.path.join(UPLOAD_FOLDER_Images, f"{timestamp}{file.filename}")
            file.save(filepath)
        cursor.execute('update [user] set name=?,password=?,gender=?,dob=?,defaultRole=?,ImgPath=?,currentrole=? where id=?',(user_dict['name'],user_dict['password'],user_dict['gender'],user_dict['dob'],user_dict['role'],filepath,user_dict['role'],user_dict['email']))
        row1=cursor.rowcount
        cursor.execute('update Doctor set contact=? where UserId=?',user_dict['contact'],user_dict['email'])
        row2=cursor.rowcount
        if row1==1 and row2==1:
            conn.commit()
            cursor.close()
            return jsonify({"status":"Updated"}),200
        else:
            conn.rollback()
            cursor.close()
            return jsonify({"status":"Update failed"}),201
    except Exception as e:
        cursor.close()
        return jsonify({"Exception":str(e)}),500


@app.route('/PatientUpdate', methods=['POST'])
def update_patient():
    try:
        filepath=''
        user = request.form.get('user')
        file=request.files.get("image")
        if not user:
            return 'No user data found'

        # Convert user data JSON to Python dictionary
        user_dict = json.loads(user)

        required_fields = ['email', 'name', 'password', 'gender', 'dob', 'role','weight','height','contact']
        for field in required_fields:
            if field not in user_dict:
                return f'Missing required field: {field}'

    # Check if the user with the same email already exists
        cursor=conn.cursor()
        
        if not file:
            filepath=""    
        elif file and allowed_file_img(file.filename):    
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            filepath = os.path.join(UPLOAD_FOLDER_Images, f"{timestamp}{file.filename}")
            file.save(filepath)
       
        cursor.execute('update [user] set name=?,password=?,gender=?,dob=?,defaultRole=?,ImgPath=?,currentrole=? where id=?',(user_dict['name'],user_dict['password'],user_dict['gender'],user_dict['dob'],user_dict['role'],filepath,user_dict['role'],user_dict['email']))
        row1=cursor.rowcount
        cursor.execute('update patient set weight=?,height=?,contact=? where UserId=?',user_dict["weight"],user_dict["height"],user_dict['contact'],user_dict['email'])
        row2=cursor.rowcount
        if row1==1 and row2==1:
            
            conn.commit()
            cursor.close()
            return jsonify({"status":"Updated"}),200
        else:
            conn.rollback()
            cursor.close()
            return jsonify({"status":"Update failed"}),201
    except Exception as e:
        return jsonify({"Exception":str(e)}),500

@app.route('/SupervisorUpdate', methods=['POST'])
def update_supervisor():
    try:
        filepath=''
        user = request.form.get('user')
        file=request.files.get("image")
        if not user:
            return 'No user data found'

        user_dict = json.loads(user)

        required_fields = ['email', 'name', 'password', 'gender', 'dob', 'role']
        for field in required_fields:
            if field not in user_dict:
                return f'Missing required field: {field}'

        cursor=conn.cursor()
        if not file:
            filepath=""
        elif file and allowed_file_img(file.filename):    
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            filepath = os.path.join(UPLOAD_FOLDER_Images, f"{timestamp}{file.filename}")
            file.save(filepath)
        cursor.execute('update [user] set name=?,password=?,gender=?,dob=?,defaultRole=?,ImgPath=?,currentrole=? where id=?',(user_dict['name'],user_dict['password'],user_dict['gender'],user_dict['dob'],user_dict['role'],filepath,user_dict['role'],user_dict['email']))
        row1=cursor.rowcount
        if row1==1:
            
            conn.commit()
            cursor.close()
            return jsonify({"status":"Updated"}),200
        else:
            conn.rollback()
            cursor.close()
            return jsonify({"status":"Update failed"}),201
    except Exception as e:
        cursor.close()
        return jsonify({"Exception":str(e)}),500

@app.route('/image/<path:filename>')  # Accept full paths, not just filenames
def serve_image(filename):
    # Construct the full path to the file, assuming base directory is C:/FYP_Code/
    base_directory = 'C:/FYP_Code/'
    
    # Join the base directory with the requested file path
    full_path = os.path.join(base_directory, filename)
    
    # Check if the file exists, if not, return a 404
    if not os.path.exists(full_path):
        return "File not found", 404

    # Serve the file
    return send_file(full_path, mimetype='image/jpeg')


# @app.route('/predictEEGEmotion',methods=["POST"])
# def predictEEGEmotion():
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     tn=int(request.form.get('trialNo'))
#     sbNo=int(request.form.get('subjectNo'))
#     file_path = fr'C:\Project Dataset\data_preprocessed_python\data_preprocessed_python\s{sbNo}.dat'
#     #trial_index = int(input("Enter trial no 1-40: "))  # Index of the trial you want to predict
#     trial_index = tn # Index of the trial you want to predict
#     # Load a single trial's EEG data
#     eeg_data, true_label = load_single_trial(file_path, trial_index)
#     model=EEGNet(num_classes=7)
#     # Load the pre-trained model (.pth)
#     state_dict = torch.load(r'D:/Trained-Models/eegnet_eeg_emotion_recognition6.pth')
#     model.load_state_dict(state_dict)
#     model = model.to(device)

#     # Predict the emotion
#     predicted_emotion, score = predict_emotion(model, eeg_data, device)

#     # Convert the prediction to the emotion label using the previously saved LabelBinarizer classes
#     le_classes = np.load(r'D:/Trained-Models/label_classes6.npy')
#     emotion_label = le_classes[predicted_emotion]

#     print(f"Predicted Emotion: {emotion_label}")
#     print(f"True Label: {map_to_emotion(*true_label)}")
#     print(f"Predicted Score: {score}")
#     return jsonify({"predicted emotion":emotion_label, "score":score,"tn":tn})

@app.route('/predictEEGEmotion/<string:filen>')
def predictEEGEmotion(filen):
    # Load the trained model
    model_path = r"C:\FYP_Code\Serious_Work\E\Trained_Model/Model9_dropout-256.h5"  # Update with your model path
    model = load_model(model_path)

    # File path to the new EEG CSV file
    eeg_file = f"C:/FYP_Code/FYPAPIs/Uploads/eeg/{filen}"  # Update with your EEG file path

    # Map emotions to labels
    label_map = {"Happy": 0, "Sad": 1, "Relax": 2, "Stress 1":3,"Stress 2":4,"Stress 3":5}  # Map "Relax" to Neutral

    # Make predictions
    try:
        predictions, predicted_classes = predict_eeg(model, eeg_file)

        # Map predicted classes to emotion labels
        mapped_emotions = [list(label_map.keys())[list(label_map.values()).index(cls)] for cls in predicted_classes]

        # Display the results
        print("Predictions (class probabilities):")
        print(predictions)
        print("\nPredicted Classes:")
        print(predicted_classes)
        print("\nMapped Emotions:")
        print(mapped_emotions)
        return jsonify(mapped_emotions),200
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred",500
    
@app.route("/AllEEGFilesName")
def getEEGFileName():
    # Path to the folder
    folder_path = "FYPAPIs/Uploads/eeg"

    # Get the list of file names
    file_names = os.listdir(folder_path)
    return file_names
# @app.route("/recordStream")
# def EEGRecord():
#      # Note: an existing Muse LSL stream is required
#     record(60,filename=f"C:/FYP_Code/FYPAPIs/Uploads/eeg/s1")

#     # Note: Recording is synchronous, so code here will not execute until the stream has been closed
#     print('Recording has ended')
@app.route("/AddSession", methods=['POST'])
def addSession():
    try:
        session = request.get_json()
        cursor = conn.cursor()

        # Use the correct primary key column name
        cursor.execute(
            'INSERT INTO Session(SupervisorId, AppointmentId) OUTPUT INSERTED.ID VALUES (?, ?)',
            (session['supervisorid'], session['appointmentid'])
        )

        result = cursor.fetchone()
        inserted_id = result[0] if result else None

        conn.commit()
        cursor.close()

        return jsonify({"status": "Experiment added", "id": inserted_id})

    except Exception as e:
        conn.rollback()
        cursor.close()
        return jsonify({"Exception": str(e)}), 500
    
@app.route("/AddExperiment", methods=['POST'])
def addExperiment():
    try:
        experiment = request.get_json()
        cursor = conn.cursor()

        # Use the correct primary key column name (e.g., "id" or actual column name)
        cursor.execute(
            'INSERT INTO Experiment (EEGPath, result, sessionid) OUTPUT INSERTED.id VALUES (?, ?, ?)',
            (experiment['EEGPath'], experiment['result'], experiment['sessionid'])
        )

        inserted_id = cursor.fetchone()[0]  # Fetch the inserted ID
        conn.commit()
        cursor.close()

        return jsonify({"status": "Experiment added", "id": inserted_id})

    except Exception as e:
        conn.rollback()
        cursor.close()
        return jsonify({"Exception": str(e)}), 500

@app.route("/GetSessions/<int:doctorid>/<int:patientid>", methods=['GET'])
def getSession(doctorid, patientid):
    try:
        cursor = conn.cursor()
        cursor.execute('select s.id,a.AppId from Appointment a join Session s on a.AppId=s.appointmentId where a.doctor_id=? and a.patient_id=?',(doctorid,patientid))
        result = cursor.fetchall()
        sessions = []

        for row in result:
            s = {
                'sessionid': row[0],
                'appid': row[1],
                
            }
            sessions.append(s)
        
        cursor.close()
        return jsonify(sessions),200
    except Exception as e:
        return jsonify({'status':"Sessions not found"}),500
    
@app.route("/GetExperiments/<int:sessionid>", methods=['GET'])
def getExperiments(sessionid):
    try:
        cursor = conn.cursor()
        cursor.execute('select * from Experiment e where e.sessionid=?',(sessionid))
        result = cursor.fetchall()
        experiments = []

        for row in result:
            ex = {
                'id': row[0],
                'EEGPath': row[1],
                'Result': row[2],
                'SessionID': row[3]
            }
            experiments.append(ex)
        cursor.close()
        return jsonify(experiments),200 
    except Exception as e:
        return jsonify({'status':"Experiments not found"}),500
    
@app.route("/GetSessionsForPatient/<int:patientid>", methods=['GET'])
def getSessionForPatient( patientid):
    try:
        cursor = conn.cursor()
        cursor.execute('select a.AppId,a.doctor_id,s.id as [SessionId],s.supervisorId from Appointment a join Session s on a.AppId=s.appointmentId where a.patient_id=?',(patientid))
        result = cursor.fetchall()
        sessions = []

        for row in result:
            ex = {
                'appid': row[0],
                'doctorid': row[1],
                'sessionid': row[2],
                'supervisorid': row[3]
            }
            sessions.append(ex)
        cursor.close()
        return jsonify(sessions),200 
    except Exception as e:
        return jsonify({'status':"Sessions not found"}),500
    

@app.route('/emotionfromframe')    
def EmotionFromFrame():
    emotions1=[]
    # Load the video
    video_path = r'E:/DataSet Emotions Only/S49/Videos/S49-Happy.mp4'  # Update with your correct video path
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames // fps  # Total video duration in seconds

    # Extract frames at 6-second intervals (10 chunks in the video)
    num_frames = 10  # Total frames we want
    frame_times = [i * (video_duration // num_frames) for i in range(num_frames)]  # Select evenly spaced frames
    frame_indices = [t * fps for t in frame_times]  # Convert seconds to frame indices

    all_emotions = []  # Store all detected emotions

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # Jump to the frame at the given second
        ret, frame = cap.read()

        if ret:
            # Detect emotion in the current frame
            emotion_labels = detect_emotion_from_frame(frame)
            all_emotions.append((i // fps, emotion_labels))  # Store frame time & detected emotions

    cap.release()
    cv2.destroyAllWindows()

    # Print detected emotions at the end
    print("\n🔹 Detected Emotions for Selected Frames:")
    for frame_time, emotions in all_emotions:
        if emotions:  # Store detected emotions in list
            emotions1.append(emotions[0])
        print(f"At {frame_time} seconds: {', '.join(emotions) if emotions else 'No face detected'}")

    print("\n📌 All Detected Emotions:", emotions1)
    # return jsonify(emotions1),200
    #  Load the trained model
    model_path = r"C:\FYP_Code\Serious_Work\E\Trained_Model/Model9_dropout-256.h5"  # Update with your model path
    model = load_model(model_path)

    # File path to the new EEG CSV file
    vn="S9-Relax"
    eeg_file = f"E:/DataSet Emotions Only/S49/EEG/S49-Happy.csv" # Update with your EEG file path

    # Map emotions to labels
    label_map = {"Happy": 0, "Sad": 1, "Relax": 2, "Stress 1":3,"Stress 2":4,"Stress 3":5}  # Map "Relax" to Neutral

    # Make predictions

    predictions, predicted_classes = predict_eeg_fuse(model, eeg_file)

        # Map predicted classes to emotion labels
    mapped_emotions = [list(label_map.keys())[list(label_map.values()).index(cls)] for cls in predicted_classes]

        # Display the results
    print("Predictions (class probabilities):")
    print(predictions)
    print("\nPredicted Classes:")
    print(predicted_classes)
    print("\nMapped Emotions:")
    print(mapped_emotions)
    # Perform Late Fusion using Majority Voting
    fused_emotions = []
    for eeg, video in zip(mapped_emotions, emotions1):
        combined = [eeg, video]  # Combine predictions from both modalities
        fused_emotion = Counter(combined).most_common(1)[0][0]  # Get most common emotion
        fused_emotions.append(fused_emotion)
    return jsonify(fused_emotions),200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True) 



