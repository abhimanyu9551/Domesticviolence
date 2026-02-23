import os
import pandas as pd
import librosa
import numpy as np
import soundfile as sf

# ---- PATHS ----
BASE_PATH = "/mnt/c/Users/Jaid Mulani/OneDrive/Documents/DATASET/archive (30)/UrbanSound8K/UrbanSound8K"
AUDIO_PATH = os.path.join(BASE_PATH, "audio")
METADATA_PATH = os.path.join(BASE_PATH, "metadata/UrbanSound8K.csv")

OUTPUT_PATH = "/home/abhimanyu/dv_project/processed_dataset"

TARGET_SR = 16000
DURATION = 3
SAMPLES = TARGET_SR * DURATION

# ---- CLASS MAPPING ----
label_map = {
    "air_conditioner": "normal",
    "children_playing": "normal",
    "dog_bark": "normal",
    "engine_idling": "normal",
    "street_music": "normal",
    "drilling": "normal",
    "jackhammer": "normal",
    "siren": "normal",
    "car_horn": "normal",
    "gun_shot": "impact"
}

# ---- LOAD METADATA ----
df = pd.read_csv(METADATA_PATH)

for index, row in df.iterrows():
    filename = row['slice_file_name']
    fold = f"fold{row['fold']}"
    label = row['class']

    if label not in label_map:
        continue

    input_file = os.path.join(AUDIO_PATH, fold, filename)
    output_class = label_map[label]
    output_file = os.path.join(OUTPUT_PATH, output_class, filename)

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

print("Processing complete!")
