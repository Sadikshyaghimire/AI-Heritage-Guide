import os
from PIL import Image

ROOT = "dataset/final_dataset"

bad_files = []

total = 0

for root, dirs, files in os.walk(ROOT):
    for file in files:

        path = os.path.join(root, file)

        total += 1

        try:
            with Image.open(path) as img:
                img.verify()

        except Exception as e:
            bad_files.append((path, str(e)))

print("=" * 60)
print(f"Checked {total} images")
print(f"Bad images: {len(bad_files)}")
print("=" * 60)

for path, err in bad_files:
    print(path)
    print(err)
    print("-" * 60)