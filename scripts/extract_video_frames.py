import os
import cv2

# -----------------------------------
# VIDEO CONFIGURATION
# -----------------------------------

VIDEOS = [
    {
        "video": r"dataset/raw/kathmandu_durbar/Kathmandudurbarsquare.mp4",
        "output": r"dataset/raw/kathmandu_durbar/video_frames"
    },
    {
        "video": r"dataset/raw/bhaktapur_durbar/Bhaktapurdurbarsquare.mp4",
        "output": r"dataset/raw/bhaktapur_durbar/video_frames"
    },
    {
        "video": r"dataset/raw/patan_durbar/Patandurbarsquare.mp4",
        "output": r"dataset/raw/patan_durbar/video_frames"
    }
]

# Save every nth frame
FRAME_INTERVAL = 8

# -----------------------------------
# EXTRACT FRAMES
# -----------------------------------

for item in VIDEOS:

    video_path = item["video"]
    output_dir = item["output"]

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"\nCould not open {video_path}")
        continue

    frame_count = 0
    saved_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame_count % FRAME_INTERVAL == 0:

            filename = f"frame_{saved_count:05d}.jpg"

            cv2.imwrite(
                os.path.join(output_dir, filename),
                frame
            )

            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"\nFinished: {os.path.basename(video_path)}")
    print(f"Frames extracted: {saved_count}")

print("\nAll videos processed successfully.")