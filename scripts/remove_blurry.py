import cv2
import os

# ============================
# CONFIGURATION
# ============================

FOLDERS = [
    r"D:\FYP Artefacts\dataset\raw\pashupatinath",
    r"D:\FYP Artefacts\dataset\raw\swayambhunath"
]

BLUR_THRESHOLD = 120

# ============================

for folder in FOLDERS:

    removed = 0
    kept = 0

    print(f"\nProcessing: {folder}")

    for filename in os.listdir(folder):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(folder, filename)

        image = cv2.imread(path)

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        if variance < BLUR_THRESHOLD:
            os.remove(path)
            removed += 1
        else:
            kept += 1

    print(f"Kept    : {kept}")
    print(f"Removed : {removed}")

print("\nFinished.")