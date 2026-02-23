import os
import librosa
import numpy as np
import soundfile as sf

# ---- PATHS ----
SCREAM_PATH = "/mnt/c/Users/Jaid Mulani/OneDrive/Documents/DATASET/archive (32)/Screaming"
NOT_SCREAM_PATH = "/mnt/c/Users/Jaid Mulani/OneDrive/Documents/DATASET/archive (32)/NotScreaming"

OUTPUT_PATH = "/home/abhimanyu/dv_project/processed_dataset"

TARGET_SR = 16000
DURATION = 3
SAMPLES = TARGET_SR * DURATION


def process_folder(input_folder, output_class, prefix):
    files = os.listdir(input_folder)

    for file in files:
        input_file = os.path.join(input_folder, file)
        output_file = os.path.join(
            OUTPUT_PATH,
            output_class,
            prefix + "_" + file
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


# Process screaming → distress
process_folder(SCREAM_PATH, "distress", "scr")

# Process not screaming → normal
process_folder(NOT_SCREAM_PATH, "normal", "noscr")

print("Scream dataset processing complete!")
