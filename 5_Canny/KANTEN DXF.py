import cv2
import numpy as np
import os
import glob
import ezdxf

# === EINSTELLUNGEN ===
input_folder = r"E:\HBK\WiSe 24_25_Bachelor\Python 30.05\3_Zuschnitt\cropped_Bilder"
reference_image_name = "cropped_undistorted_reference.JPG"
scale_file = r"E:\HBK\WiSe 24_25_Bachelor\Python 30.05\4_Maßstab\scale_data.npz"
output_folder = r"E:\HBK\WiSe 24_25_Bachelor\Python 30.05\5_Canny\dxf_output"
preview_out_dir = r"E:\HBK\WiSe 24_25_Bachelor\Python 30.05\5_Canny\preview_output"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(preview_out_dir, exist_ok=True)

# === Maßstab laden ===
scale_data = np.load(scale_file)
px_per_mm = float(scale_data["px_per_mm_avg"])
scale = 1 / px_per_mm
print(f"📏 Maßstab geladen: {scale:.4f} mm/px")

# === Bildliste vorbereiten ===
image_files = []
for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png"):
    image_files.extend(glob.glob(os.path.join(input_folder, ext)))
image_files = [f for f in image_files if os.path.basename(f) != reference_image_name]

if not image_files:
    print("❌ Keine verarbeitbaren Bilder gefunden.")
    exit()

# === Verarbeitung ===
for img_path in image_files:
    fname = os.path.basename(img_path)

    # === Skip wenn DXF schon existiert ===
    dxf_path = os.path.join(output_folder, os.path.splitext(fname)[0] + ".dxf")
    if os.path.exists(dxf_path):
        print(f"⏭️ Überspringe '{fname}' – DXF bereits vorhanden.")
        continue

    print(f"\n🔄 Verarbeite: {fname}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Bild nicht lesbar: {fname}")
        continue

    # === Kontur erkennen ===
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]
    blurred = cv2.GaussianBlur(l, (7, 7), 0)
    _, bin_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("⚠️ Keine Konturen erkannt.")
        continue

    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 3:
        print("⚠️ Kontur zu klein.")
        continue

    contour_points = [pt[0] for pt in largest]
    preview = img.copy()
    cv2.drawContours(preview, [largest], -1, (0, 255, 0), 2)

    # === Interaktive Ansicht ===
    zoom = 1.0
    offset_x, offset_y = 0, 0
    is_panning = False
    pan_start = (0, 0)
    click_positions = []

    def draw_preview():
        global preview_display
        resized = cv2.resize(preview, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
        h, w = resized.shape[:2]
        canvas = np.ones((800, 1200, 3), dtype=np.uint8) * 255

        y1 = max(0, offset_y)
        x1 = max(0, offset_x)
        y2 = min(800, offset_y + h)
        x2 = min(1200, offset_x + w)

        y1_img = max(0, -offset_y)
        x1_img = max(0, -offset_x)
        y2_img = y1_img + (y2 - y1)
        x2_img = x1_img + (x2 - x1)

        canvas[y1:y2, x1:x2] = resized[y1_img:y2_img, x1_img:x2_img]
        preview_display = canvas.copy()

        for pt in click_positions:
            x_disp = int(pt[0] * zoom + offset_x)
            y_disp = int(pt[1] * zoom + offset_y)
            cv2.circle(preview_display, (x_disp, y_disp), 6, (0, 0, 255), -1)

        cv2.putText(preview_display, fname, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    def mouse_callback(event, x, y, flags, param):
        global offset_x, offset_y, pan_start, is_panning, zoom

        if event == cv2.EVENT_LBUTTONDOWN:
            real_x = int((x - offset_x) / zoom)
            real_y = int((y - offset_y) / zoom)
            if 0 <= real_x < preview.shape[1] and 0 <= real_y < preview.shape[0]:
                if len(click_positions) < 2:
                    click_positions.append((real_x, real_y))
                    print(f"📍 Punkt {len(click_positions)} gesetzt: ({real_x}, {real_y})")
            draw_preview()

        elif event == cv2.EVENT_RBUTTONDOWN:
            is_panning = True
            pan_start = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and is_panning:
            dx = x - pan_start[0]
            dy = y - pan_start[1]
            offset_x += dx
            offset_y += dy
            pan_start = (x, y)
            draw_preview()

        elif event == cv2.EVENT_RBUTTONUP:
            is_panning = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            zoom_factor = 1.1 if flags > 0 else 0.9
            zoom_center_x = (x - offset_x) / zoom
            zoom_center_y = (y - offset_y) / zoom
            zoom *= zoom_factor
            offset_x = int(x - zoom_center_x * zoom)
            offset_y = int(y - zoom_center_y * zoom)
            draw_preview()

    cv2.namedWindow("Teilen: 2 Punkte klicken – ESC zum Abbrechen", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Teilen: 2 Punkte klicken – ESC zum Abbrechen", 1200, 800)
    cv2.setMouseCallback("Teilen: 2 Punkte klicken – ESC zum Abbrechen", mouse_callback)

    print("🖱️ Links: 2 Punkte klicken | Rechtsklick: verschieben | Mausrad: zoom | Taste 'y': bestätigen")
    draw_preview()
    while True:
        cv2.imshow("Teilen: 2 Punkte klicken – ESC zum Abbrechen", preview_display)
        key = cv2.waitKey(20)
        if key == ord('y') and len(click_positions) == 2:
            break
        elif key == 27:
            print("❌ Abgebrochen.")
            click_positions = []
            break
    cv2.destroyAllWindows()

    def find_closest_index(p):
        return min(range(len(contour_points)),
                   key=lambda i: np.hypot(p[0] - contour_points[i][0], p[1] - contour_points[i][1]))

    if len(click_positions) == 2:
        i1 = find_closest_index(click_positions[0])
        i2 = find_closest_index(click_positions[1])
        print(f"✂️ Schneide bei Index {i1} und {i2}")

        if i1 < i2:
            part_1 = contour_points[i1:i2 + 1]
            part_2 = contour_points[i2:] + contour_points[:i1 + 1]
        else:
            part_1 = contour_points[i1:] + contour_points[:i2 + 1]
            part_2 = contour_points[i2:i1 + 1]

        dxf = ezdxf.new(setup=True)
        msp = dxf.modelspace()

        def export_scaled(points):
            if len(points) >= 2:
                scaled = [(x * scale, y * scale) for x, y in points]
                msp.add_lwpolyline(scaled, close=False)

        export_scaled(part_1)
        export_scaled(part_2)
    else:
        dxf = ezdxf.new(setup=True)
        msp = dxf.modelspace()
        scaled = [(x * scale, y * scale) for x, y in contour_points]
        msp.add_lwpolyline(scaled, close=True)

    dxf.saveas(dxf_path)
    print(f"✅ DXF gespeichert: {dxf_path}")

    # === Vorschau-Bild mit nur Hauptkontur speichern ===
    preview_image = img.copy()
    cv2.drawContours(preview_image, [np.array(contour_points)], -1, (0, 255, 0), 2)
    out_jpg = os.path.join(preview_out_dir, os.path.splitext(fname)[0] + "_preview.jpg")
    cv2.imwrite(out_jpg, preview_image)
    print(f"🖼️ Vorschau gespeichert: {out_jpg}")
