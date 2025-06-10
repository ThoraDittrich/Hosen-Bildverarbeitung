import cv2
import os
import glob

# === KONFIGURATION ===
input_folder = r"E:\HBK\WiSe 24_25_Bachelor\Python 30.05\2_Entzerrung\undistorted_Bilder"
output_folder = r"E:\HBK\WiSe 24_25_Bachelor\Python 30.05\3_Zuschnitt\cropped_Bilder"
dpi = 72  # Samsung NX3000 Standard-DPI

# cm → px Umrechnung
def cm_to_px(cm, dpi):
    return int((cm / 2.54) * dpi)

# Zuschneidewerte in cm
crop_left_right_cm = 3  # Seiten
crop_top_bottom_cm = 8  # Oben und unten

# Umrechnung in Pixel
crop_x = cm_to_px(crop_left_right_cm, dpi)
crop_y = cm_to_px(crop_top_bottom_cm, dpi)

# Ausgabeordner erstellen
os.makedirs(output_folder, exist_ok=True)

# Bilder laden
image_files = []
for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png"):
    image_files.extend(glob.glob(os.path.join(input_folder, ext)))

if not image_files:
    print(f"❌ Keine Bilder gefunden in: {input_folder}")
    exit()

print("✂️ Zuschneiden gestartet...")

for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Fehler beim Laden: {img_path}")
        continue

    h, w = img.shape[:2]

    # Prüfen, ob das Bild groß genug ist
    if w <= 2 * crop_x or h <= 2 * crop_y:
        print(f"⚠️ Bild zu klein zum Zuschneiden: {os.path.basename(img_path)}")
        continue

    # Zuschneiden: y = Höhe, x = Breite
    cropped = img[crop_y:h - crop_y, crop_x:w - crop_x]

    filename = os.path.basename(img_path)
    save_path = os.path.join(output_folder, f"cropped_{filename}")
    cv2.imwrite(save_path, cropped)
    print(f"✅ Gespeichert: {save_path}")

print("🎉 Alle Bilder wurden erfolgreich zugeschnitten.")