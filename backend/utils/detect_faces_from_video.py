# backend/utils/detect_faces_from_video.py

import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageOps
import imagehash

# Load YOLO model
MODEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "models", "yolov8m-face.pt"
))
print("[INFO] Using YOLO model:", MODEL_PATH)
yolo = YOLO(MODEL_PATH)


# ---------------------------------------------
# FIX 1 — EXIF Rotation Handler (MOST accurate)
# ---------------------------------------------
def fix_exif_rotation(frame):
    try:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_img = ImageOps.exif_transpose(pil_img)  # auto rotate
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except:
        return frame


# ---------------------------------------------
# FIX 2 — Shape-based correction
# ---------------------------------------------
def auto_rotate_by_shape(frame):
    h, w = frame.shape[:2]

    # If video is tall → rotate to landscape
    if h > w:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    return frame


# ---------------------------------------------
# Ensure uint8 BGR
# ---------------------------------------------
def _force_bgr_uint8(img):
    if img is None:
        return None
    img = np.asarray(img)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


# ---------------------------------------------
# Hash for duplicate detection
# ---------------------------------------------
def get_hash(img):
    return imagehash.phash(Image.fromarray(img), hash_size=8)


# ---------------------------------------------
# MAIN FACE DETECTOR (FINAL)
# ---------------------------------------------
def detect_faces_from_video(
        video_path,
        output_dir,
        max_unique_faces=8,
        frame_skip=2,
        resize_dim=(400, 400)
):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open video:", video_path)
        return []

    print("[INFO] Extracting faces...")

    unique_hashes = []
    results_list = []
    unique_count = 0
    frame_id = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % frame_skip != 0:
            continue

        frame = _force_bgr_uint8(frame)

        # Step 1: Fix EXIF rotation
        frame = fix_exif_rotation(frame)

        # Step 2: Rotate portrait → landscape
        frame = auto_rotate_by_shape(frame)

        # Run YOLO
        results = yolo(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        for (box, conf) in zip(boxes, confs):

            if conf < 0.55:
                continue

            x1, y1, x2, y2 = box

            pad = 0.20
            bw = x2 - x1
            bh = y2 - y1

            x1e = max(0, int(x1 - bw * pad))
            y1e = max(0, int(y1 - bh * pad))
            x2e = min(frame.shape[1], int(x2 + bw * pad))
            y2e = min(frame.shape[0], int(y2 + bh * pad))

            crop = frame[y1e:y2e, x1e:x2e]

            # Resize
            face_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, resize_dim, interpolation=cv2.INTER_AREA)

            # Duplicate filtering
            hsh = get_hash(face_rgb)
            if any(abs(hsh - u) < 12 for u in unique_hashes):
                continue

            unique_hashes.append(hsh)

            save_path = os.path.join(output_dir, f"face_{unique_count:04d}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))

            results_list.append(save_path)
            unique_count += 1
            print(f"[INFO] Saved clean face #{unique_count}")

            if unique_count >= max_unique_faces:
                break

        if unique_count >= max_unique_faces:
            break

    cap.release()
    print("[INFO] Total saved:", unique_count)
    return results_list
