import cv2 as cv
import numpy as np
import os

# ===================== KONFIGURACJA =====================

PHOTOS_DIR = "photos"
RESULTS_DIR = "wyniki"
MAX_IMAGES_PER_VARIANT = 10
SCALE_DISPLAY = 0.8
IGNORE_BOTTOM_PERCENT = 0.1

wariant = {
    # "skaner": {"param1": 150, "param2": 40, "dp": 1.0},
    "biurko": {"param1": 206, "param2": 45, "dp": 1.0},
    # "flesz":  {"param1": 150, "param2": 60, "dp": 1.0}
}

coins = {
    "1 pln": 23.0,
    "2 pln": 21.5,
    "5 pln": 24.0
}

values = {
    "1 pln": 1.00,
    "2 pln": 2.00,
    "5 pln": 5.00
}

colors = {
    "1 pln": (255, 0, 255),
    "2 pln": (0, 100, 255),
    "5 pln": (0, 0, 255)
}

# ===================== CALLBACK SKALI =====================

def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN and param["points"] < 2:
        param["points"] += 1
        param["pts"].append((x, y))
        cv.circle(param["img"], (x, y), 8, (0, 0, 255), -1)

# ===================== PRZYGOTOWANIE KATALOGÓW =====================

os.makedirs(RESULTS_DIR, exist_ok=True)

# ===================== PĘTLA GŁÓWNA =====================

with open("wyniki.txt", "w", encoding="utf-8") as wyniki_txt:

    for nazwa_wariantu, hough_params in wariant.items():

        folder = os.path.join(PHOTOS_DIR, nazwa_wariantu)
        if not os.path.isdir(folder):
            continue

        variant_results_dir = os.path.join(RESULTS_DIR, nazwa_wariantu)
        os.makedirs(variant_results_dir, exist_ok=True)

        images = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
        images = images[:MAX_IMAGES_PER_VARIANT]

        print(f"\n>>> WARIANT: {nazwa_wariantu} | zdjęć: {len(images)}")

        for idx, img_name in enumerate(images, start=1):

            img_path = os.path.join(folder, img_name)
            img = cv.imread(img_path)
            if img is None:
                continue

            img = cv.resize(img, None, fx=SCALE_DISPLAY, fy=SCALE_DISPLAY)
            h_img, w_img = img.shape[:2]

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            gray_blur = cv.GaussianBlur(gray, (7, 7), 1.5)

            ignore_h = int(h_img * IGNORE_BOTTOM_PERCENT)
            gray_for_hough = gray_blur.copy()
            cv.rectangle(gray_for_hough, (0, h_img - ignore_h), (w_img, h_img), 0, -1)

            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]


            # ---------- KALIBRACJA SKALI ----------
            calib = {"points": 0, "pts": [], "img": img.copy()}
            win_name = f"Skala 1cm: {img_name}"

            cv.namedWindow(win_name, cv.WINDOW_NORMAL)
            # cv.imshow(win_name, calib["img"])
            cv.setMouseCallback(win_name, mouse_callback, calib)

            while calib["points"] < 2:
                cv.imshow(win_name, calib["img"])
                if cv.waitKey(1) & 0xFF == 27:
                    break

            cv.destroyAllWindows()
            if calib["points"] < 2:
                continue

            p1 = np.array(calib["pts"][0])
            p2 = np.array(calib["pts"][1])
            px_per_cm = np.linalg.norm(p1 - p2)

            # ---------- ZAKRES PROMIENI ----------
            min_d_mm = min(coins.values()) - 1.5
            max_d_mm = max(coins.values()) + 1.5

            minR = int((min_d_mm / 10 * px_per_cm) / 2)
            maxR = int((max_d_mm / 10 * px_per_cm) / 2)

            # ---------- HOUGH CIRCLES ----------
            circles = cv.HoughCircles(
                gray_for_hough,
                cv.HOUGH_GRADIENT,
                dp=hough_params["dp"],
                minDist=int(1.8 * px_per_cm),
                param1=hough_params["param1"],
                param2=hough_params["param2"],
                minRadius=minR,
                maxRadius=maxR
            )

            preview = img.copy()
            total_value = 0.0
            detected = 0

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for cx, cy, r in circles[0]:
                    diameter_mm = (2 * r) / px_per_cm * 10
                    diffs = {k: abs(v - diameter_mm) for k, v in coins.items()}
                    best = min(diffs, key=diffs.get)

                    tolerance = 0.8 + 0.04 * diameter_mm

                    mask_coin = np.zeros_like(gray)
                    cv.circle(mask_coin, (cx, cy), r, 255, -1)
                    mean_sat = cv.mean(sat, mask=mask_coin)[0]

                    if diffs[best] < tolerance:

                        if best == "1 pln" and mean_sat > 55:
                            best = "2 pln"
                        elif best in ["2 pln", "5 pln"] and mean_sat < 40:
                            best = "1 pln"


                        color = colors[best]
                        label = best
                        total_value += values[best]
                        detected += 1

                    cv.circle(preview, (cx, cy), r, color, 4)
                    cv.putText(
                        preview, label,
                        (cx - 40, cy - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 3
                    )

            cv.putText(
                preview,
                f"SUMA: {total_value:.2f} PLN",
                (30, 80),
                cv.FONT_HERSHEY_SIMPLEX, 2.0,
                (0, 0, 255), 4
            )

            out_name = f"wynik_{nazwa_wariantu}_{idx}.jpg"
            out_path = os.path.join(variant_results_dir, out_name)
            cv.imwrite(out_path, preview)

            wyniki_txt.write(
                f"wariant={nazwa_wariantu}, plik={img_name}, "
                f"suma={total_value:.2f}, monety={detected}\n"
            )

            print(f"  {img_name} -> {total_value:.2f} PLN")

            # ---------- WYŚWIETLENIE WYNIKU ----------
            show_name = f"Wynik: {img_name}"
            cv.namedWindow(show_name, cv.WINDOW_NORMAL)
            cv.imshow(show_name, preview)

            key = cv.waitKey(0)
            cv.destroyWindow(show_name)

            if key == 27:  # ESC
                print("Przerwano przez użytkownika.")
                exit()

print("\nZAKOŃCZONO PRZETWARZANIE")
