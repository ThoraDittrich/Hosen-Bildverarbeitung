import cv2
import numpy as np
import os

# === KONFIGURATION ===
image_path = r"E:\\HBK\\WiSe 24_25_Bachelor\\Python 30.05\\3_Zuschnitt\\cropped_Bilder\\cropped_undistorted_reference.JPG"
square_size_mm = 30.0  # reale Größe eines Quadrats in mm
checkerboard_dims = (10, 7)  # Anzahl Innenkanten (nicht Felder!)
scale_file = r"E:\HBK\WiSe 24_25_Bachelor\Python 30.05\4_Maßstab\scale_data.npz"

# === Bild laden ===yy

img = cv2.imread(image_path)
if img is None:
    print("❌ Bild konnte nicht geladen werden.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# === Checkerboard finden ===
ret, corners = cv2.findChessboardCorners(gray, checkerboard_dims, None)

if not ret:
    print("❌ Checkerboard konnte nicht erkannt werden.")
    exit()

# === Genauere Eckenerkennung ===
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

# === Pixelabstand entlang X und Y bestimmen ===
p0 = corners_refined[0][0]
p1 = corners_refined[1][0]
p7 = corners_refined[checkerboard_dims[0]][0]

dx = np.linalg.norm(p1 - p0)
dy = np.linalg.norm(p7 - p0)

px_per_mm_x = dx / square_size_mm
px_per_mm_y = dy / square_size_mm
px_per_mm_avg = (px_per_mm_x + px_per_mm_y) / 2

# === Ausgabe ===
print("📐 Maßstab ermittelt:")
print(f" - Horizontal: {px_per_mm_x:.2f} px/mm")
print(f" - Vertikal:   {px_per_mm_y:.2f} px/mm")
print(f" - Mittelwert: {px_per_mm_avg:.2f} px/mm")

# === Maßstab speichern ===
np.savez(scale_file,
         px_per_mm_x=px_per_mm_x,
         px_per_mm_y=px_per_mm_y,
         px_per_mm_avg=px_per_mm_avg)

print(f"💾 Maßstab gespeichert in '{scale_file}'")

# === Optional: Checkerboard anzeigen ===
img_vis = cv2.drawChessboardCorners(img.copy(), checkerboard_dims, corners_refined, ret)
cv2.imshow("Checkerboard erkannt", img_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()