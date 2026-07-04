import os
from PIL import Image
import imagehash
from tqdm import tqdm

# ----------------------------
# CONFIGURATION
# ----------------------------

RAW_DATASET = r"D:\FYP Artefacts\dataset\raw"

HERITAGES = [
    "swayambhunath"
]

HASH_SIZE = 16
HAMMING_THRESHOLD = 8

# ----------------------------

for heritage in HERITAGES:

    folder = os.path.join(RAW_DATASET, heritage)

    print(f"\nProcessing {heritage}")

    hashes = {}
    deleted = 0
    kept = 0

    files = sorted(os.listdir(folder))

    for filename in tqdm(files):

        path = os.path.join(folder, filename)

        try:
            img = Image.open(path).convert("RGB")
            current_hash = imagehash.phash(img, hash_size=HASH_SIZE)

            duplicate = False

            for existing_hash in hashes:

                distance = current_hash - existing_hash

                if distance <= HAMMING_THRESHOLD:
                    duplicate = True
                    break

            if duplicate:
                os.remove(path)
                deleted += 1
            else:
                hashes[current_hash] = filename
                kept += 1

        except Exception:
            continue

    print(f"Kept      : {kept}")
    print(f"Duplicates: {deleted}")

print("\nFinished.")