import tensorflow
import librosa
import wave
import pyaudio
import os
import speech_recognition as sr
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tflite_runtime.interpreter import Interpreter
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
import numpy as np
from tensorflow import keras
import smtplib
from email.message import EmailMessage
from sklearn import preprocessing

def sound_Rec():
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 10
    WAVE_OUTPUT_FILENAME = "file.wav"

    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

def mailmessage():
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("domesticviolencedemo@gmail.com", "domesticviolence@123")
    server.sendmail("domesticviolencedemo@gmail.com","abhimanyu9551@gmail.com",'Domestic Violence Threat, Help neeeded immediately' )
    print('mail sent')

sound_Rec()
filename = "file.wav"
model = keras.models.load_model(r"C:\Users\hpmsi\Downloads\output-20220411T164542Z-001\model.h5")

list = ['angry','crying', 'disgust', 'fear','sad','screaming','shouting']
#list_df = pd.DataFrame(list,columns=['list_types'])
le = preprocessing.LabelEncoder()
lists=le.fit_transform(list)
#function part
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

#print(mfccs_scaled_features)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
print(mfccs_scaled_features)
#print(mfccs_scaled_features.shape)
predicted_label=model.predict_classes(mfccs_scaled_features)
print(predicted_label)
prediction_model= le.inverse_transform(predicted_label)
print(prediction_model)
for i in range(0,6):
        if (predicted_label==i):
            mailmessage()
            print('mail sent')
            break

