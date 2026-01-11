import cv2 as cv
import numpy as np
import os

scale = 0.8

# ================== PARAMETRY MONET ==================
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
    "1 pln": (255, 0, 255),   # Magenta
    "2 pln": (0, 100, 255),   # Pomarańczowy
    "5 pln": (0, 0, 255)      # Czerwony
}

# ================== CALLBACKI ==================
def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param["lista"].append((x, y))
        param["points"] += 1
        cv.circle(param["img"], (x, y), 10, (0, 0, 255), -1)

def nothing(x):
    pass

# ================== WCZYTANIE ==================
img_path = "photos/monety_grosze.JPG"
img = cv.imread(img_path)

if img is None:
    print("Nie można otworzyć pliku", img_path)
    exit()

img = cv.resize(img, None, fx=scale, fy=scale)
img = cv.medianBlur(img, 7)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

param = {"points": 0, "lista": [], "img": img.copy()}

# ================== KALIBRACJA ==================
photo_ratio = 4/3
size_photo = 1000
top_text = "Zaznacz odcinek 1cm na obrazku"
cv.namedWindow(top_text, cv.WINDOW_NORMAL)
cv.resizeWindow(top_text, int(photo_ratio*size_photo), size_photo)

while True:
    cv.imshow(top_text, param["img"])
    if param["points"] == 2:
        cv.destroyAllWindows()
        break
    cv.setMouseCallback(top_text, mouse_callback, param)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Przeliczenie pikseli na cm
p1 = np.array(param["lista"][0])
p2 = np.array(param["lista"][1])
px_distance = np.linalg.norm(p1 - p2)
px_per_cm = px_distance / 1.0

# --- NANIESIENIE KALIBRACJI NA STAŁE ---
img_orig = img.copy()
cv.circle(img_orig, tuple(param["lista"][0]), 10, (0, 0, 255), -1)
cv.circle(img_orig, tuple(param["lista"][1]), 10, (0, 0, 255), -1)
cv.line(img_orig, tuple(param["lista"][0]), tuple(param["lista"][1]), (0, 255, 0), 5)
cv.putText(img_orig, "1cm", (param["lista"][0][0], param["lista"][0][1]-10),
           cv.FONT_HERSHEY_SIMPLEX, 1.8*scale, (0, 255, 0), 3)

# Dynamiczne promienie (uwzględniające 50gr)
min_d_mm = min(coins.values()) - 2.0
max_d_mm = max(coins.values()) + 2.0
minRadius_px = int((min_d_mm / 10.0 * px_per_cm) / 2.0)
maxRadius_px = int((max_d_mm / 10.0 * px_per_cm) / 2.0)

# ================== DOSTRAJANIE ==================
window_name = "Dostrajanie Detekcji"
cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.resizeWindow(window_name, int(photo_ratio*size_photo), size_photo)

cv.createTrackbar("Czułość (Param1)", window_name, 150, 400, nothing)
cv.createTrackbar("Rygor (Param2)", window_name, 45, 100, nothing)
cv.createTrackbar("Min Dystans (mm)", window_name, 18, 50, nothing)
cv.createTrackbar("dp x10", window_name, 10, 20, nothing)

print(f">>> Detekcja w zakresie: {min_d_mm}mm - {max_d_mm}mm")

while True:
    p1_val = max(cv.getTrackbarPos("Czułość (Param1)", window_name), 1)
    p2_val = max(cv.getTrackbarPos("Rygor (Param2)", window_name), 1)
    dist_mm = max(cv.getTrackbarPos("Min Dystans (mm)", window_name), 5)
    min_dist_px = int((dist_mm / 10.0) * px_per_cm)
    dp_val = max(cv.getTrackbarPos("dp x10", window_name), 10) / 10.0

    # gray_filtered = cv.medianBlur(gray, 7)
    preview = img_orig.copy()

    circles = cv.HoughCircles(
        gray,
        cv.HOUGH_GRADIENT,
        dp=dp_val,
        minDist=min_dist_px,
        param1=p1_val,
        param2=p2_val,
        minRadius=minRadius_px,
        maxRadius=maxRadius_px
    )

    total_value = 0.0

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for cx, cy, r in circles[0]:
            diameter_mm = (2 * r) / px_per_cm * 10
            diffs = {k: abs(v - diameter_mm) for k, v in coins.items()}
            best = min(diffs, key=diffs.get)

            if diffs[best] < 0.8:
                color = colors[best]
                total_value += values[best]
                label = best
            else:
                color = (0, 255, 255)
                label = "?"

            cv.circle(preview, (cx, cy), r, color, 6)
            cv.putText(preview, label, (cx - 50, cy - 15),
                       cv.FONT_HERSHEY_SIMPLEX, 2.5*scale, color, 8)

    cv.putText(preview, f"SUMA: {total_value:.2f} PLN",
               (40, 200), cv.FONT_HERSHEY_SIMPLEX, 10*scale, (0, 0, 255), int(15*scale))

    preview_final = preview.copy()
    cv.imshow(window_name, preview)

    if cv.waitKey(1) & 0xFF == ord('q'):
        # Wypisanie parametrów przy wyjściu
        print("Finalne parametry HoughCircles:")
        print(f"dp={dp_val}, param1={p1_val}, param2={p2_val}, minDist={min_dist_px} (px)")
        break

cv.destroyAllWindows()

# ================== FINALIZACJA ==================
cv.imshow('Finalny Wynik (Zapisano)', preview_final)
cv.imwrite('wynik_detekcji.png', preview_final)
print(f"\nZapisano wynik. Suma: {total_value:.2f} PLN")
cv.waitKey(0)
cv.destroyAllWindows()