import os
import pandas as pd
import librosa
import numpy as np
import soundfile as sf

# ---- PATHS ----
BASE_PATH = "/mnt/c/Users/Jaid Mulani/OneDrive/Documents/DATASET/ESC-50-master/ESC-50-master"
AUDIO_PATH = os.path.join(BASE_PATH, "audio")
METADATA_PATH = os.path.join(BASE_PATH, "meta/esc50.csv")

OUTPUT_PATH = "/home/abhimanyu/dv_project/processed_dataset"

TARGET_SR = 16000
DURATION = 3
SAMPLES = TARGET_SR * DURATION

# ---- CLASS MAPPING ----
label_map = {
    "crying_baby": "distress",
    "glass_breaking": "impact",
    "door_slam": "impact",
    "thunderstorm": "normal",
    "dog": "normal",
    "rain": "normal",
    "coughing": "normal",
    "sneezing": "normal"
}

df = pd.read_csv(METADATA_PATH)

for index, row in df.iterrows():
    filename = row['filename']
    label = row['category']

    if label not in label_map:
        continue

    input_file = os.path.join(AUDIO_PATH, filename)
    output_class = label_map[label]
    output_file = os.path.join(OUTPUT_PATH, output_class, "esc_" + filename)

    try:
        y, sr = librosa.load(input_file, sr=TARGET_SR, mono=True)

        if len(y) > SAMPLES:
            y = y[:SAMPLES]
        else:
            y = np.pad(y, (0, SAMPLES - len(y)))

        sf.write(output_file, y, TARGET_SR)
        print(f"Processed: {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("ESC-50 Processing complete!")
