import cv2 as cv
import numpy as np


scale = 0.5
img_path = "photos/IMG_20221011_172910.jpg"


coins_def = {
    "1 gr": 15.5, "2 gr": 17.5, "5 gr": 19.5,
    "10 gr": 16.5, "20 gr": 18.5, "50 gr": 20.5,
    "1 pln": 23.0, "2 pln": 21.5, "5 pln": 24.0
}
values = {
    "1 gr": 0.01, "2 gr": 0.02, "5 gr": 0.05, "10 gr": 0.10, "20 gr": 0.20,
    "50 gr": 0.50, "1 pln": 1.00, "2 pln": 2.00, "5 pln": 5.00
}


img = cv.imread(img_path)
if img is None:
    print(f"Błąd: Nie można otworzyć pliku {img_path}")
    exit()


img = cv.resize(img, None, fx=scale, fy=scale)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_blur = cv.medianBlur(gray, 5)


param = {"points": 0, "lista": [], "img": img.copy()}

def mouse_callback(event, x, y, flags, p):
    if event == cv.EVENT_LBUTTONDOWN:
        p["lista"].append((x, y))
        p["points"] += 1
        cv.circle(p["img"], (x, y), 5, (0, 0, 255), -1)
        cv.imshow("Kalibracja Skali", p["img"])


window_calib = "Kalibracja Skali"
cv.namedWindow(window_calib, cv.WINDOW_NORMAL)
cv.resizeWindow(window_calib, 800, 600)

cv.imshow(window_calib, param["img"])
cv.setMouseCallback(window_calib, mouse_callback, param)

print(">>> Zaznacz odcinek 5 cm na obrazku i naciśnij SPACJĘ <<<")
cv.waitKey(0)
cv.destroyWindow(window_calib)

if len(param["lista"]) < 2:
    print("Brak punktów! Używam skali domyślnej.")
    px_per_mm = 8.0 
else:
    p1, p2 = np.array(param["lista"][0]), np.array(param["lista"][1])
    px_dist = np.linalg.norm(p1 - p2)
    px_per_mm = (px_dist / 5.0) / 10.0
    print(f"Skala: {px_per_mm:.2f} px/mm")


def nothing(x):
    pass

window_name = "Dostrajanie Detekcji"
cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.resizeWindow(window_name, 1000, 700)

cv.createTrackbar("Czułość (Param1)", window_name, 100, 300, nothing)
cv.createTrackbar("Rygor (Param2)", window_name, 30, 100, nothing)
cv.createTrackbar("Min Dystans", window_name, 40, 150, nothing)

print(">>> Ustaw suwaki. Naciśnij 'q' aby zakończyć <<<")

final_circles = None

while True:
    p1_val = cv.getTrackbarPos("Czułość (Param1)", window_name)
    p2_val = cv.getTrackbarPos("Rygor (Param2)", window_name)
    min_dist = cv.getTrackbarPos("Min Dystans", window_name)

    p1_val = max(p1_val, 1)      
    p2_val = max(p2_val, 1)
    min_dist = max(min_dist, 10)

    preview = img.copy()

    try:
        circles = cv.HoughCircles(
            gray_blur, cv.HOUGH_GRADIENT, 1, 
            minDist=min_dist,
            param1=p1_val, 
            param2=p2_val,
            minRadius=15, maxRadius=100
        )
    except cv.error:
        circles = None

    if circles is not None:
        final_circles = circles
        circles_uint = np.uint16(np.around(circles))
        for i in circles_uint[0, :]:
            cv.circle(preview, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv.circle(preview, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv.imshow(window_name, preview)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()

print("\n--- WYNIK ---")
total_value = 0.0
result_img = img.copy()

if final_circles is not None:
    final_circles = np.uint16(np.around(final_circles))
    for i in final_circles[0, :]:
        cx, cy, r_px = i
        diameter_mm = (r_px * 2) / px_per_mm
        
        best_coin = min(coins_def.keys(), key=lambda c: abs(coins_def[c] - diameter_mm))
        diff = abs(coins_def[best_coin] - diameter_mm)
        
        if diff < 2.0:
            total_value += values[best_coin]
            color = (0, 255, 0)
            label = best_coin
        else:
            color = (0, 0, 255)
            label = "?"

        cv.circle(result_img, (cx, cy), r_px, color, 2)
        cv.putText(result_img, f"{label}", (cx - 20, cy), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        print(f"Moneta: {label: <6} | Zmierzono: {diameter_mm:.1f}mm")

print(f"\nŁĄCZNIE: {total_value:.2f} PLN")

window_result = "Final"
cv.namedWindow(window_result, cv.WINDOW_NORMAL)
cv.resizeWindow(window_result, 1000, 700)

cv.imshow(window_result, result_img)
cv.waitKey(0)
cv.destroyAllWindows()