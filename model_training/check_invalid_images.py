from pathlib import Path
from PIL import Image

DATASET = Path("dataset/final_dataset")

bad_files = []

count = 0

for img_path in DATASET.rglob("*"):

    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]:
        continue

    try:
        with Image.open(img_path) as img:
            img.verify()

        with Image.open(img_path) as img:
            img.load()

        count += 1

    except Exception as e:
        bad_files.append((img_path, str(e)))

print("=" * 60)
print("Images checked:", count)
print("Invalid images:", len(bad_files))
print("=" * 60)

for file, err in bad_files:
    print(file)
    print(err)
    print("-" * 40)