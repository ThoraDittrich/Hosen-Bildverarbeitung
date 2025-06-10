import cv2
import numpy as np
import glob
import re
import os

# === KONFIGURATION ===
CHECKERBOARD = (10, 7)           # Anzahl der Innenkanten
SQUARE_SIZE = 30.0               # Quadratgröße in mm
MAX_IMAGES = 23

# === PFADANGABEN ===
calib_folder = r"E:\HBK\WiSe 24_25_Bachelor\Python 30.05\1_Kalibrierung\calibration_images_2"
preview_folder = r"E:\HBK\WiSe 24_25_Bachelor\Python 30.05\1_Kalibrierung\checkerboard_previews"
kalibrierdatei = "calibration_data.npz"

# === VORSCHAU-ORDNER ERSTELLEN ===
os.makedirs(preview_folder, exist_ok=True)

# === 3D-Punkte vorbereiten ===
objp_template = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp_template[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp_template *= SQUARE_SIZE

objpoints = []
imgpoints = []
gray_example = None

# === Kalibrierbilder laden ===
image_files = []
for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png"):
    image_files.extend(glob.glob(os.path.join(calib_folder, ext)))

# Doppelte Dateinamen entfernen
image_files = list(dict.fromkeys(image_files))

# Optional: Nach nummeriertem Dateinamen sortieren
image_files = sorted(
    image_files,
    key=lambda f: int(re.search(r'\((\d+)\)', f).group(1)) if re.search(r'\((\d+)\)', f) else 0
)

if not image_files:
    print("❌ Keine Kalibrierbilder gefunden!")
    exit()

print("📷 Gefundene Kalibrierbilder:")
for f in image_files:
    print(" -", os.path.basename(f))

# === Checkerboard erkennen ===
used = 0
for fname in image_files:
    if used >= MAX_IMAGES:
        break

    img = cv2.imread(fname)
    if img is None:
        print(f"{fname}: ❌ Bild konnte nicht geladen werden.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    print(f"{os.path.basename(fname)}: {'✅' if ret else '❌'}")

    if ret:
        objpoints.append(objp_template.copy())
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)
        used += 1
        if gray_example is None:
            gray_example = gray.copy()
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        # Vorschaubild speichern
        preview_filename = os.path.basename(fname)
        preview_path = os.path.join(preview_folder, preview_filename)
        cv2.imwrite(preview_path, img)

        cv2.imshow("Checkerboard", img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

if gray_example is None:
    print("❌ Keine gültigen Checkerboards erkannt.")
    exit()

# === Kalibrierung ===
image_size = gray_example.shape[::-1]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

print("\n🎯 Kameramatrix:\n", mtx)
print("🔧 Verzerrung:\n", dist.ravel())
print("📏 Reprojizierungsfehler (RMS):", ret)

# === Kalibrierdaten speichern ===
np.savez(kalibrierdatei, mtx=mtx, dist=dist)
print(f"💾 Kalibrierdaten gespeichert in '{kalibrierdatei}'")