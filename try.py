import tensorflow as tf
import numpy as np
import librosa
import os
import speech_recognition as sr
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tflite_runtime.interpreter import Interpreter

# Parameters
# debug_time = 0
# debug_acc = 0
list = ["screaming", "crying", "shouting", "anger", "sad", "disgust", "fear"]
rec_duration = 0.30
model_path = "C:/Users/hpmsi/Downloads/output-20220411T164542Z-001model.tflite"

# Load model (interpreter)
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)

# label encoder

labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))


def mailmessage():
    from_email_addr = 'domesticviolencedemo@gmail.com'
    from_email_password = 'domesticviolence@123'
    to_email_addr = 'abhimanyu9551@gmail.com@gmail.com'
    msg['Subject'] = 'DOMESTIC VIOLENCE DETECTED..!!'
    msg['From'] = from_email_addr
    msg['To'] = to_email_addr
    return True


# Compute features
#def computefeat():
filename = "speech.wav"
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs = np.mean(mfccs_features.T, axis=0)
    # predicted_label = model.predict_classes(mfccs_scaled_features)
    # #in_tensor =mfccs_scaled_features.reshape(1,-1)
    # interpreter.set_tensor(input_details[0]['index'],in_tensor)
    # interpreter.invoke()
    # output_data = interpreter.get_tensor(output_details[0]['index'])
    # prediction_class = labelencoder.inverse_transform(output_data)
#return mfccs

in_tensor = np.float32(mfccs.reshape(1, -1))
interpreter.set_tensor(input_details[0]['index'], in_tensor)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
predict = labelencoder.inverse_transform(output_data)
for i in enumerate(list):
    if (predict == i):
        mailmessage()




# Start streaming from microphone
#def sound_Rec():
r = sr.Recognizer()


with sr.Microphone() as source:
    audio = r.listen(source)

    with open('speech.wav', 'wb')as f:
        f.write(audio.get_wav_data())
# email


#label encoder

labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))
    
   
# Compute features
def computefeat():
    filename="speech.wav"
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs= np.mean(mfccs_features.T,axis=0)
    #predicted_label = model.predict_classes(mfccs_scaled_features)
    # #in_tensor =mfccs_scaled_features.reshape(1,-1)
    # interpreter.set_tensor(input_details[0]['index'],in_tensor)
    # interpreter.invoke()
    # output_data = interpreter.get_tensor(output_details[0]['index'])
    # prediction_class = labelencoder.inverse_transform(output_data)

# Load model (interpreter)
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#print(input_details)
in_tensor = mfccs.reshape(1,-1)
rp =interpreter.set_tensor(input_details[0]['index'], in_tensor)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
#predict = labelencoder.inverse_transform(output_data)
#print(output_data)
#outputdata=labelencoder.inverse_transform(rp)
#outputdata
#for i in enumerate (list):
        #if (predict==i):
            #mailmessage()
            #break;
val = output_data[0][0]
print(val)
# Start streaming from microphone
#def sound_Rec():
r=sr.Recognizer()

with sr.Microphone() as source:
	audio=r.listen(source)

	with open('speech.wav','wb')as f:
	    f.write(audio.get_wav_data())
#email
def mailmessage():
    from_email_addr = 'domesticviolencedemo@gmail.com'
    from_email_password = 'domesticviolence@123'
    to_email_addr = 'abhimanyu9551@gmail.com@gmail.com'
    msg[ 'Subject'] = 'DOMESTIC VIOLENCE DETECTED..!!'
    msg['From'] = from_email_addr
    msg['To'] = to_email_addr
    return True
