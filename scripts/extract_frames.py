import cv2
import os

# ==========================
# Configuration
# ==========================

VIDEO_FOLDER = r"D:\FYP Artefacts\dataset\videos"

OUTPUT_FOLDERS = {
    "pashupatinath.mp4": r"D:\FYP Artefacts\dataset\raw\pashupatinath",
    "swayambhu.mp4": r"D:\FYP Artefacts\dataset\raw\swayambhunath",
}

FRAME_INTERVAL = 1  # Save one frame every second

# ==========================
# Extraction Function
# ==========================

def extract_frames(video_path, output_folder, interval=1):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)

    frame_count = 0
    saved = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, f"{saved:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1

        frame_count += 1

    cap.release()

    print(f"{os.path.basename(video_path)}")
    print(f"Frames extracted: {saved}")
    print("-" * 30)

# ==========================
# Run
# ==========================

for video_name, output in OUTPUT_FOLDERS.items():
    video_path = os.path.join(VIDEO_FOLDER, video_name)
    extract_frames(video_path, output, FRAME_INTERVAL)

print("Frame extraction completed successfully.")