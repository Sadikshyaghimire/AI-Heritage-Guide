from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random

random.seed(42)

TRAIN_ROOT = Path("dataset/final_dataset/train")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# Target number of training images per class
TARGET_IMAGES = 180


def augment_image(img):
    img = img.convert("RGB")

    # Brightness
    if random.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(
            random.uniform(0.75, 1.25)
        )

    # Contrast
    if random.random() < 0.7:
        img = ImageEnhance.Contrast(img).enhance(
            random.uniform(0.8, 1.3)
        )

    # Small rotation
    if random.random() < 0.6:
        angle = random.uniform(-15, 15)
        img = img.rotate(
            angle,
            expand=True,
            fillcolor=(255, 255, 255)
        )

    # Slight blur
    if random.random() < 0.25:
        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(0.3, 1.0)
            )
        )

    # Resize back
    img = img.resize(
        (224, 224),
        Image.Resampling.LANCZOS
    )

    return img


total_created = 0

for class_dir in TRAIN_ROOT.iterdir():

    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    # ALL images currently present
    all_images = [
        p for p in class_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Only ORIGINAL images are used as augmentation source
    raw_images = [
        p for p in all_images
        if "_raw_" in p.name or "_ai_" in p.name
    ]

    current = len(all_images)

    if current >= TARGET_IMAGES:
        print(f"{class_name}: already has {current} images")
        continue

    needed = TARGET_IMAGES - current

    print(f"\n{class_name}")
    print(f"Current : {current}")
    print(f"Creating: {needed}")

    created = 0

    while created < needed:

        img_path = random.choice(raw_images)

        try:
            img = Image.open(img_path)

            aug = augment_image(img)

            filename = (
                class_dir /
                f"{class_name}_aug_{created+1:04d}.jpg"
            )

            aug.save(filename, quality=95)

            created += 1
            total_created += 1

        except Exception as e:
            print(e)

    print(f"Done ({created} created)")

print("\n--------------------------------")
print(f"Total augmented images: {total_created}")
print("--------------------------------")