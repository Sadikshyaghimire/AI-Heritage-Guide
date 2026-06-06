import os
import random
import shutil
from pathlib import Path

# -----------------------------
# SETTINGS
# -----------------------------

random.seed(42)

RAW_DIR = Path("dataset/raw")
AI_DIR = Path("dataset/ai_generated")
OUTPUT_DIR = Path("dataset/final_dataset")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

classes = [
    "swayambhunath",
    "bagh_bhairab",
    "boudhanath",
    "bhaktapur_durbar",
    "maitidevi",
    "pashupatinath",
    "patan_durbar",
    "kathmandu_durbar",
    "dharahara",
    "budhanilkantha"
]

# -----------------------------
# CLEAR OLD FINAL DATASET
# -----------------------------

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

for split in ["train", "val", "test"]:
    for cls in classes:
        (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

# -----------------------------
# HELPER FUNCTION
# -----------------------------

def collect_images(folder_path, source_type):
    collected = []

    if folder_path.exists():
        for img in folder_path.iterdir():
            if img.suffix.lower() in IMAGE_EXTENSIONS:
                collected.append({
                    "path": img,
                    "source": source_type
                })

    return collected

# -----------------------------
# PROCESS EACH CLASS
# -----------------------------

for cls in classes:
    print(f"\nProcessing: {cls}")

    raw_images = collect_images(RAW_DIR / cls, "raw")
    ai_images = collect_images(AI_DIR / cls, "ai")

    images = raw_images + ai_images
    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_imgs = images[:train_count]
    val_imgs = images[train_count:train_count + val_count]
    test_imgs = images[train_count + val_count:]

    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }

    for split_name, split_imgs in splits.items():
        for idx, item in enumerate(split_imgs, start=1):
            img_path = item["path"]
            source_type = item["source"]
            extension = img_path.suffix.lower()

            new_filename = f"{cls}_{source_type}_{split_name}_{idx:04d}{extension}"

            destination = OUTPUT_DIR / split_name / cls / new_filename

            shutil.copy2(img_path, destination)

    print(f"Total Images: {total}")
    print(f"Train: {len(train_imgs)}")
    print(f"Val: {len(val_imgs)}")
    print(f"Test: {len(test_imgs)}")

print("\nDataset splitting complete.")