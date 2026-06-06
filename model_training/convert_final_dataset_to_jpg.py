from pathlib import Path
from PIL import Image

DATASET_DIR = Path("dataset/final_dataset")
VALID_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

converted = 0
skipped = 0

for img_path in DATASET_DIR.rglob("*"):
    if img_path.suffix.lower() not in VALID_EXTENSIONS:
        continue

    try:
        img = Image.open(img_path).convert("RGB")

        new_path = img_path.with_suffix(".jpg")
        img.save(new_path, "JPEG", quality=95)

        if img_path != new_path:
            img_path.unlink()

        converted += 1

    except Exception as e:
        skipped += 1
        print(f"Skipped {img_path}: {e}")

print(f"Converted images: {converted}")
print(f"Skipped images: {skipped}")