🚨 Domestic Violence Audio Detection System (AI + IoT)
📌 Overview

This project is an AI-powered domestic violence detection system that analyzes real-time audio to detect:

Aggression (angry shouting)

Distress (crying, screaming, fear)

Impact sounds (object hits, gunshot-like sounds)

Normal environmental audio

The system is designed to be deployed on a Raspberry Pi as a smart home safety device, similar to a fire alarm — but for domestic distress detection.

🎯 Problem Statement

Domestic violence often goes undetected due to:

Social stigma

Lack of immediate reporting

Fear or inability to call for help

This project aims to create a passive AI-based monitoring system that can:

Detect high-risk audio patterns

Trigger alerts

Enable faster intervention

🏗 System Architecture

Audio Input (Microphone / IoT Device)

Preprocessing (Resampling to 16kHz, 3-second window)

Feature Extraction (Mel Spectrogram)

CNN-based Multi-Class Classifier

Alert Trigger (Future Deployment on Raspberry Pi)

📊 Dataset Engineering

Multiple public datasets were combined and standardized:

UrbanSound8K → Normal & Impact baseline

ESC-50 → Environmental + Distress signals

FSC22 → Impact + Environmental robustness

Screaming dataset → Distress class

CREMA-D → Aggression + Distress emotional speech

All audio was:

Converted to 16kHz

Trimmed/Padded to 3 seconds

Converted to mono

Balanced across classes

📁 Final Balanced Dataset
Class	Samples
Normal	4000
Distress	3000
Aggression	1200
Impact	750

Dataset split:

70% Train

15% Validation

15% Test

🧠 Model Architecture

Input: 128 Mel Spectrogram

Conv2D (32 filters)

MaxPooling

Conv2D (64 filters)

MaxPooling

Dense (128)

Softmax Output (4 classes)

Loss Function: Sparse Categorical Crossentropy
Optimizer: Adam

📈 Model Performance

Test Accuracy: 77%

Classification Report:

Aggression F1: 0.78

Distress F1: 0.72

Impact F1: 0.76

Normal F1: 0.80

The model demonstrates strong baseline performance for multi-class emotional audio classification.

🔧 Technologies Used

Python

TensorFlow / Keras

Librosa

NumPy

Scikit-learn

Ubuntu (WSL2)

Git & GitHub

🚀 Future Improvements

Add real-world domestic argument audio

Apply data augmentation (noise, pitch shift)

Improve distress recall

Convert model to TensorFlow Lite

Deploy on Raspberry Pi with real-time streaming

Build dashboard + alert system

🔒 Ethical Considerations

Privacy-sensitive system

Intended for opt-in use only

Designed as assistive safety technology

Not a surveillance tool
