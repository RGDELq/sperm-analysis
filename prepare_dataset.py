import shutil
from pathlib import Path
from collections import defaultdict


# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "dataset"


# =====================================================
# CLEAN OLD YOLO DATASET
# =====================================================

if OUTPUT_PATH.exists():
    shutil.rmtree(OUTPUT_PATH)


# =====================================================
# CREATE YOLO FOLDER STRUCTURE
# =====================================================

for split in ["train", "val"]:
    (OUTPUT_PATH / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / "labels" / split).mkdir(parents=True, exist_ok=True)

print("Created output folders.")


# =====================================================
# MATCH IMAGE-LABEL PAIRS AND GROUP BY PATIENT
# =====================================================

patient_groups = defaultdict(list)
missing_images = []

label_files = sorted(PROCESSED_PATH.rglob("*_bbox.txt"))

for label_file in label_files:
    stem = label_file.stem.replace("_bbox", "")

    img_file = None

    for ext in [".jpg", ".jpeg", ".png"]:
        matches = sorted(RAW_PATH.rglob(stem + ext))
        if matches:
            img_file = matches[0]
            break

    if img_file is None:
        missing_images.append(label_file.name)
        continue

    relative_path = img_file.relative_to(RAW_PATH)
    patient_id = relative_path.parts[0]

    patient_groups[patient_id].append((img_file, label_file))


# =====================================================
# DETERMINISTIC PATIENT SPLIT
# =====================================================

patients = sorted(patient_groups.keys())

split_idx = int(len(patients) * 0.8)

train_patients = patients[:split_idx]
val_patients = patients[split_idx:]


train_pairs = []
val_pairs = []

for patient_id in train_patients:
    train_pairs.extend(patient_groups[patient_id])

for patient_id in val_patients:
    val_pairs.extend(patient_groups[patient_id])


# =====================================================
# PRINT SPLIT INFO
# =====================================================

print(f"Found image-label pairs: {sum(len(v) for v in patient_groups.values())}")
print(f"Found patients:          {len(patients)}")

print(f"Train patients:          {len(train_patients)}")
print(f"Val patients:            {len(val_patients)}")

print(f"Train images:            {len(train_pairs)}")
print(f"Val images:              {len(val_pairs)}")

if missing_images:
    print(f"Missing images:          {len(missing_images)}")
    print("First missing examples:")
    for name in missing_images[:10]:
        print(f"  {name}")


print("\nTRAIN PATIENTS:")
for p in train_patients:
    print(p)

print("\nVAL PATIENTS:")
for p in val_patients:
    print(p)


# =====================================================
# COPY FILES
# =====================================================

def copy_pairs(pair_list, split):
    pair_list = sorted(pair_list, key=lambda x: str(x[0]))

    for i, (img, lbl) in enumerate(pair_list):
        new_name = f"sperm_{split}_{i:04d}"

        shutil.copy(
            img,
            OUTPUT_PATH / "images" / split / (new_name + img.suffix.lower())
        )

        shutil.copy(
            lbl,
            OUTPUT_PATH / "labels" / split / (new_name + ".txt")
        )


copy_pairs(train_pairs, "train")
copy_pairs(val_pairs, "val")

print("\nCopied all image and label files.")


# =====================================================
# CREATE dataset.yaml
# =====================================================

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