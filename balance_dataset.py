import os
import random
import shutil

BASE_PATH = "processed_dataset"
OUTPUT_PATH = "balanced_dataset"

target_counts = {
    "normal": 4000,
    "distress": 3000,
    "aggression": 1200,
    "impact": 750
}

for class_name, target in target_counts.items():
    source_folder = os.path.join(BASE_PATH, class_name)
    dest_folder = os.path.join(OUTPUT_PATH, class_name)

    files = os.listdir(source_folder)
    random.shuffle(files)

    selected_files = files[:target]

    for file in selected_files:
        shutil.copy(
            os.path.join(source_folder, file),
            os.path.join(dest_folder, file)
        )

    print(f"{class_name} -> Copied {len(selected_files)} files")

print("Balancing complete!")
