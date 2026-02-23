import os
import random
import shutil

BASE_PATH = "balanced_dataset"
OUTPUT_BASE = "dataset_split"

splits = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

classes = ["normal", "distress", "aggression", "impact"]

# Create folder structure
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_BASE, split, cls), exist_ok=True)

for cls in classes:
    files = os.listdir(os.path.join(BASE_PATH, cls))
    random.shuffle(files)

    total = len(files)
    train_end = int(total * splits["train"])
    val_end = train_end + int(total * splits["val"])

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    for f in train_files:
        shutil.copy(
            os.path.join(BASE_PATH, cls, f),
            os.path.join(OUTPUT_BASE, "train", cls, f)
        )

    for f in val_files:
        shutil.copy(
            os.path.join(BASE_PATH, cls, f),
            os.path.join(OUTPUT_BASE, "val", cls, f)
        )

    for f in test_files:
        shutil.copy(
            os.path.join(BASE_PATH, cls, f),
            os.path.join(OUTPUT_BASE, "test", cls, f)
        )

    print(f"{cls} split done.")

print("Dataset splitting complete.")
