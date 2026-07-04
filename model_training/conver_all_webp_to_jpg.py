from pathlib import Path
from PIL import Image

ROOT = Path("dataset/final_dataset")

count = 0

for file in ROOT.rglob("*.webp"):
    try:
        img = Image.open(file).convert("RGB")

        new_path = file.with_suffix(".jpg")

        img.save(new_path, "JPEG", quality=95)

        file.unlink()

        count += 1
        print("Converted:", file.name)

    except Exception as e:
        print("FAILED:", file)
        print(e)

print("\nFinished.")
print("Converted", count, "images.")