import os
from pathlib import Path
from PIL import Image

# ==========================================================
# Convert every image in dataset/final_dataset
# into a real JPEG image.
#
# - Reads every supported image
# - Converts to RGB
# - Saves as genuine JPEG
# - Deletes the old file if necessary
# - Skips corrupted files
# ==========================================================

ROOT_DIR = Path("dataset/final_dataset")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp"
}

converted = 0
already_jpg = 0
failed = 0

print("=" * 70)
print("Converting images to TRUE JPEG format...")
print("=" * 70)

for image_path in ROOT_DIR.rglob("*"):

    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        continue

    try:

        with Image.open(image_path) as img:

            img = img.convert("RGB")

            new_path = image_path.with_suffix(".jpg")

            img.save(
                new_path,
                "JPEG",
                quality=95
            )

        # Delete original if extension changed
        if image_path != new_path:
            image_path.unlink()

        # If same filename (.jpg), overwrite succeeded
        if image_path.suffix.lower() == ".jpg":
            already_jpg += 1
        else:
            converted += 1
            print(f"Converted: {image_path}")

    except Exception as e:

        failed += 1

        print(f"\nFAILED : {image_path}")
        print(e)
        print("-" * 60)

print("\n" + "=" * 70)
print("Conversion Finished")
print("=" * 70)

print(f"Already JPG : {already_jpg}")
print(f"Converted   : {converted}")
print(f"Failed      : {failed}")

print("=" * 70)