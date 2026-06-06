import os
import shutil
import random
from pathlib import Path

CLASS_NAME = "bagh_bhairab"

RAW_DIR = Path(f"dataset/raw/{CLASS_NAME}")
AI_DIR = Path(f"dataset/ai_generated/{CLASS_NAME}")

TRAIN_DIR = Path(f"dataset/final_dataset/train/{CLASS_NAME}")
VAL_DIR = Path(f"dataset/final_dataset/val/{CLASS_NAME}")
TEST_DIR = Path(f"dataset/final_dataset/test/{CLASS_NAME}")

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

image_extensions = [".jpg", ".jpeg", ".png", ".webp"]

raw_images = [
    p for p in RAW_DIR.iterdir()
    if p.suffix.lower() in image_extensions
]

ai_images = [
    p for p in AI_DIR.iterdir()
    if p.suffix.lower() in image_extensions
]

random.seed(42)
random.shuffle(raw_images)
random.shuffle(ai_images)

test_raw = raw_images[:5]
val_raw = raw_images[5:11]
train_raw = raw_images[11:]

train_ai = ai_images

def copy_images(images, destination, prefix):
    for idx, img_path in enumerate(images, start=1):
        new_name = f"{prefix}_{idx:03d}{img_path.suffix.lower()}"
        shutil.copy2(img_path, destination / new_name)

copy_images(train_raw, TRAIN_DIR, "raw_train")
copy_images(train_ai, TRAIN_DIR, "ai_train")
copy_images(val_raw, VAL_DIR, "raw_val")
copy_images(test_raw, TEST_DIR, "raw_test")

print("Split complete.")
print(f"Train raw: {len(train_raw)}")
print(f"Train AI: {len(train_ai)}")
print(f"Validation raw: {len(val_raw)}")
print(f"Test raw: {len(test_raw)}")