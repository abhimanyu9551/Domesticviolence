import os
import pandas as pd
import librosa
import numpy as np
import soundfile as sf

# ---- PATHS ----
AUDIO_PATH = "/mnt/c/Users/Jaid Mulani/OneDrive/Documents/DATASET/archive (31)/Audio Wise V1.0-20220916T202003Z-001/Audio Wise V1.0"
METADATA_PATH = "/mnt/c/Users/Jaid Mulani/OneDrive/Documents/DATASET/archive (31)/Metadata-20220916T202011Z-001/Metadata/Metadata V1.0 FSC22.csv"

OUTPUT_PATH = "/home/abhimanyu/dv_project/processed_dataset"

TARGET_SR = 16000
DURATION = 3
SAMPLES = TARGET_SR * DURATION

# ---- CLASS MAPPING ----
label_map = {
    "Gunshot": "impact",
    "Axe": "impact",
    "WoodChop": "impact",
    "Firework": "impact",
    "Chainsaw": "impact",

    "VehicleEngine": "normal",
    "Speaking": "normal",
    "Silence": "normal",
    "Rain": "normal",
    "Wind": "normal",
    "Thunderstorm": "normal"
}

df = pd.read_csv(METADATA_PATH)

for index, row in df.iterrows():
    filename = row['Dataset File Name']
    label = row['Class Name']

    if label not in label_map:
        continue

    input_file = os.path.join(AUDIO_PATH, filename)
    output_class = label_map[label]
    output_file = os.path.join(OUTPUT_PATH, output_class, "fsc_" + filename)

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

print("FSC22 Processing complete!")
