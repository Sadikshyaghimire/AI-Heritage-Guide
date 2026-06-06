from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random

CLASS_NAME = "bagh_bhairab"

TRAIN_DIR = Path(f"dataset/final_dataset/train/{CLASS_NAME}")
AUG_DIR = Path(f"dataset/augmented/{CLASS_NAME}")

AUG_DIR.mkdir(parents=True, exist_ok=True)

image_extensions = [".jpg", ".jpeg", ".png", ".webp"]

images = [
    p for p in TRAIN_DIR.iterdir()
    if p.suffix.lower() in image_extensions
]

random.seed(42)


def augment_image(img):

    img = img.convert("RGB")

    # Brightness
    if random.random() < 0.7:
        factor = random.uniform(0.75, 1.25)
        img = ImageEnhance.Brightness(img).enhance(factor)

    # Contrast
    if random.random() < 0.7:
        factor = random.uniform(0.8, 1.3)
        img = ImageEnhance.Contrast(img).enhance(factor)

    # Rotation
    if random.random() < 0.6:
        angle = random.uniform(-8, 8)
        img = img.rotate(
            angle,
            expand=True,
            fillcolor=(255, 255, 255)
        )

    # Blur
    if random.random() < 0.25:
        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(0.3, 1.0)
            )
        )

    img = img.resize((224, 224))

    return img


count = 0

for img_path in images:

    try:
        img = Image.open(img_path)

        for i in range(3):

            aug = augment_image(img)

            count += 1

            save_path = (
                AUG_DIR /
                f"{CLASS_NAME}_aug_{count:04d}.jpg"
            )

            aug.save(save_path, quality=95)

    except Exception as e:
        print(f"Skipped {img_path}: {e}")

print(f"\nCreated {count} augmented images.")
print(f"Saved in: {AUG_DIR}")