import random
import shutil
from pathlib import Path


# Project root = folder where this file is located
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset folders inside the repository
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "dataset"

# Create YOLO folder structure
for split in ["train", "val"]:
    (OUTPUT_PATH / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / "labels" / split).mkdir(parents=True, exist_ok=True)

print("Created output folders.")

# Match image files with corresponding label files
pairs = []

for label_file in PROCESSED_PATH.rglob("*_bbox.txt"):
    stem = label_file.stem.replace("_bbox", "")

    for ext in [".jpg", ".jpeg", ".png"]:
        img_files = list(RAW_PATH.rglob(stem + ext))
        if img_files:
            pairs.append((img_files[0], label_file))
            break

print(f"Found {len(pairs)} image-label pairs.")

# Split into train and validation sets
random.seed(42)
random.shuffle(pairs)

split_idx = int(len(pairs) * 0.8)
train_pairs = pairs[:split_idx]
val_pairs = pairs[split_idx:]

print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")


def copy_pairs(pair_list, split):
    for i, (img, lbl) in enumerate(pair_list):
        new_name = f"sperm_{split}_{i:04d}"

        shutil.copy(img, OUTPUT_PATH / "images" / split / (new_name + img.suffix.lower()))
        shutil.copy(lbl, OUTPUT_PATH / "labels" / split / (new_name + ".txt"))


copy_pairs(train_pairs, "train")
copy_pairs(val_pairs, "val")

print("Copied all image and label files.")

# Create dataset.yaml for YOLO
yaml_content = """path: dataset
train: images/train
val: images/val

nc: 6
names:
  - sperm
  - leucocyte
  - agglutination
  - red blood cell
  - urethral epithelial cells
  - crystals
"""

with open(OUTPUT_PATH / "dataset.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_content)

print("Created dataset.yaml.")