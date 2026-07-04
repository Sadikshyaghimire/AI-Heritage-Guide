import os
from PIL import Image
import imagehash

# ------------------------------------

FOLDERS = [
    r"dataset/raw/kathmandu_durbar/video_frames",
    r"dataset/raw/bhaktapur_durbar/video_frames",
    r"dataset/raw/patan_durbar/video_frames"
]

# ------------------------------------

total_removed = 0

for folder in FOLDERS:

    hashes = {}
    removed = 0

    for file in sorted(os.listdir(folder)):

        path = os.path.join(folder, file)

        try:
            h = imagehash.phash(Image.open(path))
        except:
            continue

        if h in hashes:
            os.remove(path)
            removed += 1
        else:
            hashes[h] = file

    total_removed += removed

    print(f"{os.path.basename(os.path.dirname(folder))}: removed {removed} duplicate images")

print(f"\nTotal duplicates removed: {total_removed}")