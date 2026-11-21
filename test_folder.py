import os
import csv
import time
import cv2
import numpy as np
from ultralytics import YOLOWorld, FastSAM

# --- 1. PARAMETRY ROBOTA MOBILNEGO I KAMERY ---
OUTPUT_DIR = "wyniki"
CSV_FILE = os.path.join(OUTPUT_DIR, "wyniki_pomiarow.csv")

CAMERA_HEIGHT_M = 0.50   # Wysokość montażu kamery na robocie [m] (50 cm)
CAMERA_PITCH_DEG = 3.0   # Pochylenie kamery w dół [stopnie]
CAMERA_HFOV_DEG = 65.0   # Poziomy kąt widzenia kamery [stopnie]
MAX_ROBOT_RANGE_M = 8.0  # Maksymalny zasięg detekcji przeszkód przed robotem [m]

# Osobne progi pewności dla klas (kamienie w trybie zero-shot mają niższy wynik CLIP)
TREE_CONF = 0.035
STONE_CONF = 0.012

TREE_PROMPTS = ["tree trunk", "pine tree trunk", "vertical tree trunk", "tree stem"]
STONE_PROMPTS = ["boulder", "mossy rock", "rock", "stone", "grey rock"]
ALL_CLASSES = TREE_PROMPTS + STONE_PROMPTS

os.makedirs(OUTPUT_DIR, exist_ok=True)

valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
image_files = sorted([f for f in os.listdir(".") if f.lower().endswith(valid_ext)])

if not image_files:
    raise FileNotFoundError("Nie znaleziono żadnych zdjęć (.jpg, .png) w tym folderze!")

def preprocess_for_robot(bgr_img):
    """Filtr bilateralny tłumi szum trawy/kory, a łagodne CLAHE wyciąga detale z cienia."""
    smooth = cv2.bilateralFilter(bgr_img, d=5, sigmaColor=40, sigmaSpace=40)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

def filter_boxes(boxes, is_tree_flags, confs, nested_thresh=0.50, cross_iou_thresh=0.65, max_per_class=5):
    """Usuwa duplikaty wewnątrz tej samej klasy, zachowując kamienie stojące przed pniami drzew."""
    if len(boxes) == 0:
        return boxes, is_tree_flags, confs

    order = np.argsort(-confs)
    keep = []
    count_tree = 0
    count_stone = 0

    for idx in order:
        is_t = bool(is_tree_flags[idx])
        if is_t and count_tree >= max_per_class:
            continue
        if (not is_t) and count_stone >= max_per_class:
            continue

        b1 = boxes[idx]
        area1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        is_duplicate = False

        for k in keep:
            b2 = boxes[k]
            area2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
            ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0:
                continue

            if is_tree_flags[idx] == is_tree_flags[k]:
                # Ta sama klasa: usuwamy ramki zagnieżdżone w sobie
                if (inter / min(area1, area2)) > nested_thresh:
                    is_duplicate = True
                    break
            else:
                # Różne klasy: usuwamy tylko identyczne pokrycie (ten sam obiekt z dwiema etykietami)
                union = area1 + area2 - inter
                if (inter / max(1.0, union)) > cross_iou_thresh:
                    is_duplicate = True
                    break

        if not is_duplicate:
            keep.append(idx)
            if is_t:
                count_tree += 1
            else:
                count_stone += 1

    return boxes[keep], is_tree_flags[keep], confs[keep]

def fast_geometric_mask(h, w, x1, y1, x2, y2, is_tree):
    """Błyskawiczna awaryjna maska geometryczna (0.1 ms), gdy FastSAM nie wytnie obiektu."""
    mask = np.zeros((h, w), dtype=np.uint8)
    rw, rh = x2 - x1, y2 - y1
    if rw < 4 or rh < 4:
        return mask
    if is_tree:
        pad_x = int(0.12 * rw)
        mask[y1:y2, x1 + pad_x : x2 - pad_x] = 1
    else:
        cx_b, cy_b = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.ellipse(mask, (cx_b, cy_b), (max(1, int(rw * 0.44)), max(1, int(rh * 0.44))), 0, 0, 360, 1, -1)
    return mask

# --- 2. ŁADOWANIE MODELI Z GITHUBA ---
print("[1/2] Ładowanie detektora przeszkód (YOLO-World Large)...", flush=True)
detector = YOLOWorld("yolov8l-worldv2.pt")
detector.set_classes(ALL_CLASSES)

print("[2/2] Ładowanie modelu segmentacji konturów (FastSAM)...", flush=True)
segmentor = FastSAM("FastSAM-s.pt")

csv_rows = [["Plik", "Obiekt", "Pewnosc", "Odleglosc_Z_m", "Srednica_D_cm", "Stosunek_H_W", "Czas_ms"]]
close_kernel = np.ones((7, 7), np.uint8)

print("\n" + "=" * 82, flush=True)
print("ROZPOCZYNAM TEST WIZJI ROBOTA (ROZDZIELONE PROGI: DRZEWA + KAMIENIE)", flush=True)
print("=" * 82, flush=True)

# --- 3. PRZETWARZANIE ZDJĘĆ ---
for img_name in image_files:
    t_start = time.perf_counter()
    img_bgr = cv2.imread(img_name)
    if img_bgr is None:
        continue

    h_img, w_img = img_bgr.shape[:2]
    img_area = float(w_img * h_img)
    fx = w_img / (2.0 * np.tan(np.deg2rad(CAMERA_HFOV_DEG) / 2.0))
    fy = fx
    cy = h_img / 2.0

    img_proc = preprocess_for_robot(img_bgr)

    # Niski próg bazowy + większy limit max_det=35 + agnostic_nms=False (klasy nie blokują się wzajemnie)
    det_res = detector.predict(
        img_proc,
        imgsz=1024,
        conf=min(TREE_CONF, STONE_CONF),
        iou=0.50,
        max_det=35,
        agnostic_nms=False,
        verbose=False
    )[0]

    boxes = det_res.boxes.xyxy.cpu().numpy()
    cls_ids = det_res.boxes.cls.cpu().numpy().astype(int)
    confs = det_res.boxes.conf.cpu().numpy()
    is_tree_flags = np.array([ALL_CLASSES[c] in TREE_PROMPTS for c in cls_ids], dtype=bool)

    # Filtr strefy robota (ROI) + osobne progi pewności dla drzew i kamieni
    valid_idx = []
    for idx, (b, is_t, cf) in enumerate(zip(boxes, is_tree_flags, confs)):
        box_area = (b[2] - b[0]) * (b[3] - b[1])
        min_conf = TREE_CONF if is_t else STONE_CONF
        if cf >= min_conf and (0.002 * img_area <= box_area <= 0.55 * img_area) and (b[3] > 0.35 * h_img):
            valid_idx.append(idx)

    if len(valid_idx) > 0:
        boxes = boxes[valid_idx]
        is_tree_flags = is_tree_flags[valid_idx]
        confs = confs[valid_idx]
    else:
        boxes = np.empty((0, 4))
        is_tree_flags = np.empty((0,), dtype=bool)
        confs = np.empty((0,))

    boxes, is_tree_flags, confs = filter_boxes(boxes, is_tree_flags, confs, max_per_class=5)

    vis_img = img_bgr.copy()

    if len(boxes) == 0:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        print(f"Plik: {img_name:15s} -> Brak przeszkód w pasie ruchu robota ({elapsed_ms:.0f} ms)", flush=True)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"wynik_{img_name}"), vis_img)
        continue

    seg_res = segmentor(img_proc, bboxes=boxes.tolist(), verbose=False)[0]
    masks_raw = seg_res.masks.data.cpu().numpy() if seg_res.masks is not None else []

    font_scale = max(0.62, w_img / 1250.0)
    font_thick = max(2, int(w_img / 550))

    summary = []
    for idx, (box, is_tree_ai, conf) in enumerate(zip(boxes, is_tree_flags, confs)):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        box_area = max(1, (x2 - x1) * (y2 - y1))

        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        if idx < len(masks_raw):
            full_mask = cv2.resize(masks_raw[idx], (w_img, h_img), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            mask[y1:y2, x1:x2] = full_mask[y1:y2, x1:x2]
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        if np.sum(mask) < 0.20 * box_area:
            mask = fast_geometric_mask(h_img, w_img, x1, y1, x2, y2, bool(is_tree_ai))

        ys, xs = np.where(mask == 1)
        if len(xs) < 25:
            continue

        # --- 1. OBLICZENIE ODLEGŁOŚCI Z [m] Z PUNKTU STYKU Z PODŁOŻEM ---
        v_bottom = float(np.max(ys))
        alpha_v = np.arctan((v_bottom - cy) / fy)
        gamma = np.deg2rad(CAMERA_PITCH_DEG) + alpha_v
        z_dist = CAMERA_HEIGHT_M / np.tan(max(np.deg2rad(2.5), gamma))
        z_dist = float(np.clip(z_dist, 0.5, MAX_ROBOT_RANGE_M))

       # --- 2. KLASYFIKACJA KSZTAŁTU (H/W) I POMIAR ŚREDNICY D [cm] ---
        v_min, v_max = int(np.min(ys)), int(np.max(ys))
        mask_w = float(np.max(xs) - np.min(xs) + 1)
        mask_h = float(v_max - v_min + 1)

        # Szerokość mierzona w środkowym pasie (ignoruje szerokie korzenie u podstawy pnia)
        v_s = int(v_min + 0.35 * (v_max - v_min))
        v_e = int(v_min + 0.65 * (v_max - v_min))
        mid_widths = [np.max(np.where(mask[v, :] == 1)[0]) - np.min(np.where(mask[v, :] == 1)[0])
                      for v in range(v_s, v_e + 1) if np.sum(mask[v, :]) > 2]
        mid_w = float(np.median(mid_widths)) if mid_widths else mask_w

        aspect_ratio = mask_h / max(1.0, mid_w)
        touches_top = (v_min <= int(0.05 * h_img))

        if is_tree_ai:
            # Nie zamieniaj na kamień, jeśli pień wychodzi poza górę kadru lub YOLO ma dużą pewność
            if (not touches_top) and (aspect_ratio < 1.55) and (conf < 0.25):
                is_tree = False
            else:
                is_tree = True
        else:
            # Zamień na drzewo, jeśli obiekt jest smukły lub pionowo wychodzi poza górę kadru
            if aspect_ratio > 2.20 or (touches_top and aspect_ratio > 1.30):
                is_tree = True
            else:
                is_tree = False

        if is_tree:
            v_min, v_max = np.min(ys), np.max(ys)
            v_start = int(v_min + 0.35 * (v_max - v_min))
            v_end = int(v_min + 0.65 * (v_max - v_min))
            widths = [np.max(np.where(mask[v, :] == 1)[0]) - np.min(np.where(mask[v, :] == 1)[0])
                      for v in range(v_start, v_end + 1) if np.sum(mask[v, :]) > 2]
            du_px = float(np.median(widths)) if widths else mask_w

            alpha_h = np.arctan(du_px / (2.0 * fx))
            diameter_cm = ((2.0 * z_dist * np.sin(alpha_h)) / max(0.25, 1.0 - np.sin(alpha_h))) * 100.0

            name_pl = "Drzewo"
            color = (34, 180, 34)

            v_mid = (v_start + v_end) // 2
            mid_xs = np.where(mask[v_mid, :] == 1)[0]
            if len(mid_xs) > 1:
                cv2.line(vis_img, (int(mid_xs.min()), v_mid), (int(mid_xs.max()), v_mid), (0, 255, 255), font_thick + 1)
            label_y = v_mid - 10
        else:
            area_m2 = float(np.sum(mask == 1)) * (z_dist ** 2) / (fx * fy)
            diameter_cm = (2.0 * np.sqrt(area_m2 / np.pi)) * 100.0

            name_pl = "Kamien"
            color = (230, 120, 30)
            label_y = max(30, y1 - 8)

        summary.append((name_pl, conf, z_dist, diameter_cm, aspect_ratio))

        overlay = np.zeros_like(vis_img)
        overlay[mask == 1] = color
        vis_img = cv2.addWeighted(vis_img, 1.0, overlay, 0.42, 0)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, font_thick)

        txt = f"{name_pl} | Z:{z_dist:.2f}m | D:{diameter_cm:.1f}cm"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        tx = min(max(5, x1), max(5, w_img - tw - 10))
        ty = min(max(th + 10, label_y), h_img - 10)
        cv2.rectangle(vis_img, (tx - 3, ty - th - 5), (tx + tw + 5, ty + 4), (20, 20, 20), -1)
        cv2.putText(vis_img, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick)

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"wynik_{img_name}"), vis_img)

    print(f"\nPlik: {img_name} (czas: {elapsed_ms:.0f} ms)", flush=True)
    for name_pl, conf, z_dist, diameter_cm, ar in summary:
        print(f"   -> {name_pl:7s} | Pewność: {conf:.2f} | Z: {z_dist:.2f} m | D: {diameter_cm:.1f} cm | H/W: {ar:.2f}", flush=True)
        csv_rows.append([
            img_name,
            name_pl,
            f"{conf:.2f}".replace(".", ","),
            f"{z_dist:.2f}".replace(".", ","),
            f"{diameter_cm:.1f}".replace(".", ","),
            f"{ar:.2f}".replace(".", ","),
            f"{elapsed_ms:.0f}"
        ])

with open(CSV_FILE, mode="w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerows(csv_rows)

print("\n" + "=" * 82, flush=True)
print(f"GOTOWE! Pełne wyniki zapisano w '{OUTPUT_DIR}' i pliku Excela.", flush=True)