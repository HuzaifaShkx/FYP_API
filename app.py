import os
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
app=Flask(__name__)

UPLOAD_FOLDER_Images = 'FYPAPIs/Uploads/Images/'
UPLOAD_FOLDER_EEG = 'FYPAPIs/Uploads/eeg/'
UPLOAD_FOLDER_recorded = 'FYPAPIs/Uploads/RecordedVideos/'
ALLOWED_EXTENSIONS_img = {'png', 'jpg', 'jpeg'}
ALLOWED_EXTENSIONS_eeg = {'dat','edf','cnt'}
ALLOWED_EXTENSIONS_video = {'avi','mp4'}
UPLOAD_FOLDER_watchedVideo = 'FYPAPIs/Uploads/watchedvideos/'

conn = pyodbc.connect('DRIVER={SQL Server};SERVER=DESKTOP-8BL3MIG;DATABASE=EEG_FYP;Trusted_Connection=yes;')
#cursor=conn.cursor()
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


# Preprocessing EEG signals
def preprocess_eeg(eeg_data):
    eeg_data = bandpass_filter(eeg_data)
    eeg_data = (eeg_data - np.mean(eeg_data, axis=0, keepdims=True)) / np.std(eeg_data, axis=0, keepdims=True)
    return eeg_data

# Feature extraction - reshape for CNN input
def prepare_data_for_cnn(eeg_data):
    num_channels, num_timepoints = eeg_data.shape
    eeg_data = eeg_data.reshape(1, 1, num_channels, num_timepoints)  # [batch_size, channels, height, width]
    return eeg_data


# Map DEAP values to emotions based on predefined thresholds
def map_to_emotion(valence, arousal, dominance, liking):
    if valence > 5 and arousal > 5:
        return 'Happy'
    elif valence < 5 and arousal > 5:
        return 'Angry'
    elif valence < 5 and arousal < 5:
        return 'Sad'
    elif valence > 5 and arousal < 5:
        return 'Surprise'
    elif valence <= 5 and arousal <= 5:
        return 'Neutral'
    elif valence > 5 and arousal <= 5:
        return 'Relaxed'
    else:
        return 'unknown'

class EEGNet(nn.Module):
    def __init__(self, num_classes):
        super(EEGNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),  # Same padding
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (32, 1), groups=16, bias=False),  # Depthwise convolution
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 16, (1, 16), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )

        # Automatically calculate the size of the flatten layer by passing dummy input
        self.flatten_size = self._get_flatten_size()
        self.fc = nn.Linear(self.flatten_size, num_classes)

    def _get_flatten_size(self):
        with torch.no_grad():
            x = torch.randn(1, 1, 32, 8064)  # Use the actual input shape here
            x = self.block1(x)
            x = self.block2(x)
            flatten_size = x.view(1, -1).size(1)
        return flatten_size

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Predict the emotion from a single trial using a trained model
def predict_emotion(model, eeg_data, device):
    # Preprocess the data
    eeg_data = preprocess_eeg(eeg_data)
    X = prepare_data_for_cnn(eeg_data)

    # Convert data to PyTorch tensor and move it to the specified device (CPU/GPU)
    X = torch.tensor(X, dtype=torch.float32).to(device)  # No need for unsqueeze(0), already has batch dimension

    # Set the model to evaluation mode
    model.eval()

    # Make prediction
    with torch.no_grad():
        predictions = model(X)
    
    predicted_emotion = torch.argmax(predictions, dim=1).item()
    confidence_score = torch.max(predictions).item()

    return predicted_emotion, confidence_score


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
            "imgPath":patient[7]
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
            "imgPath":patient[7]
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
                "imgPath":i[7]
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
        cursor.execute("SELECT p.id,u.name,u.gender,u.dob,p.height,p.weight,p.contact,u.ImgPath FROM Patient p JOIN [User] u ON p.UserId = u.Id JOIN Appointment a ON p.Id = a.patient_id WHERE a.doctor_id = ? and a.appStatus='false';",doctor_id)
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


@app.route('/getTimeSlots')
def getTimeSlots(patient_id):
    try:
        data = request.form.get('slots')
        cursor=conn.cursor()
        cursor.execute("select times,dates from Appointment where dates=? and doctor_id=?",data['date'],data['doctor_id'])
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
            return jsonify({"message":"Not Found"}),405

    except Exception as e:
        cursor.close()
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
        
        file = request.files['file']
        channel=request.form.get('channel')
        if not file:
            return jsonify({"status":"No File selected"}),301
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
            return jsonify({"status":"EEG Uploaded"}),201
        else:
            conn.rollback()
            cursor.close()
            return jsonify({"status":"EEG Not Uploaded"}),405
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
    
@app.route('/addExperiment',methods=['POST'])
def addExperiment():
    try:
        experiment = request.get_json()
        cursor=conn.cursor()
        cursor.execute('insert into Experiment(patientId,Supervisor_id,watchedVideo_id,eegRecorded_id,captureVideo_id,appointment_id) values (?,?,?,?,?,?)',experiment['patientid'],experiment['supervisorid'],experiment['watchedvideoid'],experiment['eegrecordedid'],experiment['capturevideoid'],experiment['appointmentid'])
        row=cursor.rowcount
        if row>=1:
            conn.commit()
            cursor.close()
            return jsonify({"status":"Experiment added"})
        else:
            conn.rollback()
            cursor.close()
            return jsonify({"status":"Experiment Not added"})
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


@app.route('/predictEEGEmotion',methods=["POST"])
def predictEEGEmotion():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tn=int(request.form.get('trialNo'))
    sbNo=int(request.form.get('subjectNo'))
    file_path = fr'C:\Project Dataset\data_preprocessed_python\data_preprocessed_python\s{sbNo}.dat'
    #trial_index = int(input("Enter trial no 1-40: "))  # Index of the trial you want to predict
    trial_index = tn # Index of the trial you want to predict
    # Load a single trial's EEG data
    eeg_data, true_label = load_single_trial(file_path, trial_index)
    model=EEGNet(num_classes=7)
    # Load the pre-trained model (.pth)
    state_dict = torch.load(r'D:/Trained-Models/eegnet_eeg_emotion_recognition6.pth')
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Predict the emotion
    predicted_emotion, score = predict_emotion(model, eeg_data, device)

    # Convert the prediction to the emotion label using the previously saved LabelBinarizer classes
    le_classes = np.load(r'D:/Trained-Models/label_classes6.npy')
    emotion_label = le_classes[predicted_emotion]

    print(f"Predicted Emotion: {emotion_label}")
    print(f"True Label: {map_to_emotion(*true_label)}")
    print(f"Predicted Score: {score}")
    return jsonify({"predicted emotion":emotion_label, "score":score,"tn":tn})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True) 



