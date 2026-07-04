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

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

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

    random.shuffle(raw_images)
    random.shuffle(ai_images)

    def split_images(image_list):
        total = len(image_list)

        train = int(total * TRAIN_RATIO)
        val = int(total * VAL_RATIO)

        train_imgs = image_list[:train]
        val_imgs = image_list[train:train + val]
        test_imgs = image_list[train + val:]

        return train_imgs, val_imgs, test_imgs

    raw_train, raw_val, raw_test = split_images(raw_images)
    ai_train, ai_val, ai_test = split_images(ai_images)

    train_imgs = raw_train + ai_train
    val_imgs = raw_val + ai_val
    test_imgs = raw_test + ai_test

    random.shuffle(train_imgs)
    random.shuffle(val_imgs)
    random.shuffle(test_imgs)

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

            new_filename = (
                f"{cls}_{source_type}_{split_name}_{idx:04d}{extension}"
            )

            destination = (
                OUTPUT_DIR /
                split_name /
                cls /
                new_filename
            )

            shutil.copy2(img_path, destination)

    print(f"Raw Images : {len(raw_images)}")
    print(f"AI Images  : {len(ai_images)}")
    print(f"Total      : {len(raw_images) + len(ai_images)}")
    print(f"Train      : {len(train_imgs)}")
    print(f"Validation : {len(val_imgs)}")
    print(f"Test       : {len(test_imgs)}")