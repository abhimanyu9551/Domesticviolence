import os
import librosa
import numpy as np
import soundfile as sf

# ---- PATHS ----
CREMA_PATH = "/mnt/c/Users/Jaid Mulani/OneDrive/Documents/DATASET/Crema"
OUTPUT_PATH = "/home/abhimanyu/dv_project/processed_dataset"

TARGET_SR = 16000
DURATION = 3
SAMPLES = TARGET_SR * DURATION

def get_emotion_from_filename(filename):
    parts = filename.split("_")
    emotion = parts[2]
    return emotion

label_map = {
    "ANG": "aggression",
    "SAD": "distress",
    "FEA": "distress",
    "NEU": "normal"
}

files = os.listdir(CREMA_PATH)

for file in files:
    emotion_code = get_emotion_from_filename(file)

    if emotion_code not in label_map:
        continue

    input_file = os.path.join(CREMA_PATH, file)
    output_class = label_map[emotion_code]
    output_file = os.path.join(
        OUTPUT_PATH,
        output_class,
        "crema_" + file
    )

    try:
        y, sr = librosa.load(input_file, sr=TARGET_SR, mono=True)

        if len(y) > SAMPLES:
            y = y[:SAMPLES]
        else:
            y = np.pad(y, (0, SAMPLES - len(y)))

        sf.write(output_file, y, TARGET_SR)
        print(f"Processed: {file}")

    except Exception as e:
        print(f"Error processing {file}: {e}")

print("CREMA processing complete!")
