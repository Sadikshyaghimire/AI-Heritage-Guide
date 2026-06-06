from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random

random.seed(42)

TRAIN_ROOT = Path("dataset/final_dataset/train")
AUG_ROOT = Path("dataset/augmented")

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

AUG_PER_IMAGE = 3

def augment_image(img):
    img = img.convert("RGB")

    if random.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))

    if random.random() < 0.7:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.3))

    if random.random() < 0.6:
        angle = random.uniform(-8, 8)
        img = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))

    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

    img = img.resize((224, 224))
    return img

total_created = 0

for class_dir in TRAIN_ROOT.iterdir():
    if not class_dir.is_dir():
        continue

    class_name = class_dir.name
    aug_class_dir = AUG_ROOT / class_name
    aug_class_dir.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in class_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    created = 0

    for img_path in images:
        try:
            img = Image.open(img_path)

            for i in range(AUG_PER_IMAGE):
                aug = augment_image(img)
                created += 1
                total_created += 1

                save_path = aug_class_dir / f"{class_name}_aug_{created:04d}.jpg"
                aug.save(save_path, quality=95)

        except Exception as e:
            print(f"Skipped {img_path}: {e}")

    print(f"{class_name}: created {created} augmented images")

print(f"\nTotal augmented images created: {total_created}")