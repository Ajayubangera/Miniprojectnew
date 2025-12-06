import os
from PIL import Image
import cv2
import numpy as np

BASE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "known_faces"
))

def force_clean(path):
    try:
        img = Image.open(path)
        img = img.convert("RGB")  # FORCE RGB 8-bit

        clean_path = path.replace(".jpg", "_clean.jpg")
        img.save(clean_path, "JPEG", quality=95)

        test = cv2.imread(clean_path)
        if test is None:
            print("❌ OpenCV still cannot read:", clean_path)
            return

        os.remove(path)
        os.rename(clean_path, path)

        print("✅ Fixed:", path)

    except Exception as e:
        print("❌ ERROR:", path, e)


def scan_and_fix():
    print("🔍 Fixing all known faces...")
    for root, _, files in os.walk(BASE):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                force_clean(os.path.join(root, f))

if __name__ == "__main__":
    scan_and_fix()
