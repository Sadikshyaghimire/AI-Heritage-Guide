import os
import shutil

CLASSES = [
    "kathmandu_durbar",
    "bhaktapur_durbar",
    "patan_durbar"
]

for cls in CLASSES:

    source = os.path.join("dataset", "raw", cls, "video_frames")
    destination = os.path.join("dataset", "raw", cls)

    copied = 0

    for file in os.listdir(source):

        src = os.path.join(source, file)

        new_name = f"video_{file}"

        dst = os.path.join(destination, new_name)

        shutil.copy2(src, dst)

        copied += 1

    print(f"{cls}: copied {copied} images")

print("\nFinished merging frames into raw dataset.")