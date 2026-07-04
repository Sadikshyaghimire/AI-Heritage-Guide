from pathlib import Path
import shutil

TRAIN_ROOT = Path("dataset/final_dataset/train")
AUG_ROOT = Path("dataset/augmented")

total = 0

for class_folder in AUG_ROOT.iterdir():

    if not class_folder.is_dir():
        continue

    train_folder = TRAIN_ROOT / class_folder.name

    if not train_folder.exists():
        print(f"Skipping {class_folder.name}")
        continue

    copied = 0

    for img in class_folder.iterdir():

        if img.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        destination = train_folder / img.name

        # Avoid overwriting if filename already exists
        if destination.exists():
            destination = train_folder / f"aug_{img.name}"

        shutil.copy2(img, destination)
        copied += 1
        total += 1

    print(f"{class_folder.name}: copied {copied} images")

print("-" * 40)
print(f"Total copied: {total}")
print("Done.")