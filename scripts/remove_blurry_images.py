import os
import cv2

# ------------------------------------
# CHANGE ONLY IF NEEDED
# ------------------------------------

FOLDERS = [
    r"dataset/raw/kathmandu_durbar/video_frames",
    r"dataset/raw/bhaktapur_durbar/video_frames",
    r"dataset/raw/patan_durbar/video_frames"
]

THRESHOLD = 120

# ------------------------------------

total_removed = 0

for folder in FOLDERS:

    removed = 0

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        image = cv2.imread(path)

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if score < THRESHOLD:

            os.remove(path)

            removed += 1

    total_removed += removed

    print(f"{os.path.basename(os.path.dirname(folder))}: removed {removed} blurry images")

print(f"\nTotal blurry images removed: {total_removed}")