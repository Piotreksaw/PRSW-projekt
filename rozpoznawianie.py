import cv2 as cv
import numpy as np
import os

scale = 0.8

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

def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param["lista"].append((x, y))
        param["points"] += 1
        cv.circle(param["img"], (x, y), 10, (0, 0, 255), -1)

def nothing(x):
    pass


img_path = "photos/biurko_kalibracja.jpg"
img = cv.imread(img_path)

if img is None:
    print("Nie mozna otworzyc pliku", img_path)
    exit()

img = cv.resize(img, None, fx=scale, fy=scale)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)
blur = cv.GaussianBlur(gray, (7, 7), 1.5)

mask = cv.adaptiveThreshold(
    blur, 255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY_INV,
    51, 3
)
edges = cv.Canny(mask, 80, 160)

blur_bil = cv.bilateralFilter(gray, 9, 75, 75)

ignore_height_px = int(img.shape[0] * 0.15)
cv.rectangle(edges, (0, img.shape[0] - ignore_height_px), (img.shape[1], img.shape[0]), 0, -1)

# >>> HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
sat = hsv[:, :, 1]

param = {"points": 0, "lista": [], "img": img.copy()}

photo_ratio = 4 / 3
size_photo = 1000
top_text = "Zaznacz odcinek 1cm na obrazku"
cv.namedWindow(top_text, cv.WINDOW_NORMAL)
cv.resizeWindow(top_text, int(photo_ratio * size_photo), size_photo)

while True:
    cv.imshow(top_text, param["img"])
    if param["points"] == 2:
        cv.destroyAllWindows()
        break
    cv.setMouseCallback(top_text, mouse_callback, param)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

p1 = np.array(param["lista"][0])
p2 = np.array(param["lista"][1])
px_per_cm = np.linalg.norm(p1 - p2)

img_orig = img.copy()

min_d_mm = min(coins.values()) - 1.5
max_d_mm = max(coins.values()) + 1.5

minRadius_px = int((min_d_mm / 10.0 * px_per_cm) / 2.0)
maxRadius_px = int((max_d_mm / 10.0 * px_per_cm) / 2.0)

window_name = "Dostrajanie Detekcji"
cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.resizeWindow(window_name, int(photo_ratio * size_photo), size_photo)

cv.createTrackbar("Krawedzie (Param1)", window_name, 150, 400, nothing)
cv.createTrackbar("Srodki (Param2)", window_name, 45, 100, nothing)
cv.createTrackbar("Min Dystans (mm)", window_name, 18, 50, nothing)
cv.createTrackbar("dp x10", window_name, 10, 20, nothing)

while True:
    p1_val = max(cv.getTrackbarPos("Krawedzie (Param1)", window_name), 1)
    p2_val = max(cv.getTrackbarPos("Srodki (Param2)", window_name), 1)
    dist_mm = max(cv.getTrackbarPos("Min Dystans (mm)", window_name), 5)
    min_dist_px = int((dist_mm / 10.0) * px_per_cm)
    dp_val = max(cv.getTrackbarPos("dp x10", window_name), 10) / 10.0

    preview = img_orig.copy()
    total_value = 0.0

    circles = cv.HoughCircles(
        blur_bil,
        cv.HOUGH_GRADIENT,
        dp=dp_val,
        minDist=min_dist_px,
        param1=p1_val,
        param2=p2_val,
        minRadius=minRadius_px,
        maxRadius=maxRadius_px
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for cx, cy, r in circles[0]:

            diameter_mm = (2 * r) / px_per_cm * 10
            diffs = {k: abs(v - diameter_mm) for k, v in coins.items()}
            best = min(diffs, key=diffs.get)

            tolerance = 0.8 + 0.04 * diameter_mm

            # >>> HSV – maska monety
            mask_coin = np.zeros_like(gray)
            cv.circle(mask_coin, (cx, cy), r, 255, -1)
            mean_sat = cv.mean(sat, mask_coin)[0]

            label = "?"
            color = (0, 255, 255)

            if diffs[best] < tolerance:
                # korekta HSV
                if best == "1 pln" and mean_sat > 55:
                    best = "2 pln"
                elif best in ["2 pln", "5 pln"] and mean_sat < 40:
                    best = "1 pln"

                label = best
                color = colors[best]
                total_value += values[best]

            cv.circle(preview, (cx, cy), r, color, 6)
            cv.putText(
                preview, f"{label}",
                (cx - 50, cy - 15),
                cv.FONT_HERSHEY_SIMPLEX, 2.5 * scale, color, 8
            )

    cv.putText(
        preview, f"SUMA: {total_value:.2f} PLN",
        (40, 200),
        cv.FONT_HERSHEY_SIMPLEX,
        10 * scale, (0, 0, 255), int(15 * scale)
    )

    preview_final = preview.copy()
    cv.imshow(window_name, preview)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()

cv.imshow("Finalny Wynik (Zapisano)", preview_final)
cv.imwrite("wyniki/do_sprawozdania/kalibracja.png", preview_final)
print(f"\nZapisano wynik. Suma: {total_value:.2f} PLN")
cv.waitKey(0)
cv.destroyAllWindows()