import cv2
import numpy as np
import glob
import os

# === KONFIGURATION ===
kalibrierdatei = "E:\\HBK\\WiSe 24_25_Bachelor\\Python 30.05\\1_Kalibrierung\\calibration_data.npz"
input_folder = r"E:\HBK\WiSe 24_25_Bachelor\Python 30.05\2_Entzerrung\Bilder_2"
output_folder = r"E:\HBK\WiSe 24_25_Bachelor\Python 30.05\2_Entzerrung\undistorted_Bilder"
os.makedirs(output_folder, exist_ok=True)

# === Kalibrierdaten laden ===
if not os.path.exists(kalibrierdatei):
    print(f"❌ Kalibrierdaten nicht gefunden: {kalibrierdatei}")
    exit()

data = np.load(kalibrierdatei)
mtx = data["mtx"]
dist = data["dist"]
print("📂 Kalibrierdaten geladen.")

# === Eingabebilder laden ===
image_files = []
for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png"):
    image_files.extend(glob.glob(os.path.join(input_folder, ext)))

if not image_files:
    print(f"❌ Keine Bilder im Eingabeordner gefunden: {input_folder}")
    exit()

print("🔄 Entzerrung gestartet…")

for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Bild konnte nicht geladen werden: {img_path}")
        continue

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Optional: Zuschneiden auf gültigen Bereich
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]

    # Bild speichern
    filename = os.path.basename(img_path)
    save_path = os.path.join(output_folder, f"undistorted_{filename}")
    cv2.imwrite(save_path, undistorted)
    print(f"✅ Entzerrt gespeichert: {save_path}")

print("🎉 Alle Bilder wurden erfolgreich entzerrt und gespeichert.")
