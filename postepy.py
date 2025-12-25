
# Algorytm wyznaczenia skali obrazu

import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

scale = 0.3


#srednice
coins = {
    # "1 gr": 15.5,
    # "2 gr": 17.5,
    # "5 gr": 19.5,
    # "10 gr": 16.5,
    # "20 gr": 18.5,
    # "50 gr": 20.5,
    "1 pln": 23.0,
    "2 pln": 21.5,
    "5 pln": 24.0
}

# wartość nominalna w pln
values = {
    # "1 gr": 0.01,
    # "2 gr": 0.02,
    # "5 gr": 0.05,
    # "10 gr": 0.10,
    # "20 gr": 0.20,
    # "50 gr": 0.50,
    "1 pln": 1.00,
    "2 pln": 2.00,
    "5 pln": 5.00
}

# kolory (BGR)
colors = {
    # "1 gr":  (0, 255, 255),
    # "2 gr":  (0, 165, 255),
    # "5 gr":  (0, 140, 255),
    # "10 gr": (0, 255, 0),
    # "20 gr": (255, 255, 0),
    # "50 gr": (255, 0, 0),
    "1 pln":  (255, 0, 255),
    "2 pln":  (0, 100, 255),
    "5 pln":  (0, 0, 255)
}

# callback
def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param["lista"].append((x, y))
        param["points"] += 1
        cv.circle(param["img"], (x, y), 5, (0, 100, 0), -1)

# wczytywanie
img_path = ".//photos/drewno_cieple.jfif"
print(img_path)

img = cv.imread(img_path, 0)
# img = cv.resize(img, None, fx=scale, fy=scale)
img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
img = cv.medianBlur(img, 3)
cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
param = {"points": 0, "lista": [], "img": img}

if img is None:
    print("Nie udało się wczytać obrazu!")
    exit()


# zaznaczenie 5cm
while True:
    cv.imshow("Zaznacz odcinek 5cm na obrazku", img)
    if param["points"] == 2:
        cv.destroyAllWindows()
        break

    cv.setMouseCallback("Zaznacz odcinek 5cm na obrazku", mouse_callback, param)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        cv.destroyAllWindows()
        break

print("Wybrane punkty:", param["lista"])

# przeliczenie pikseli na cm
p1 = np.array(param["lista"][0])
p2 = np.array(param["lista"][1])

px_distance = np.linalg.norm(p1 - p2)
real_length_cm = 5.0
px_per_cm = px_distance / real_length_cm

print(f"Odległość w pikselach: {px_distance:.2f}")
print(f"Skala: {px_per_cm:.2f} px / cm")


# Detekcja monet

circles = cv.HoughCircles(
    img, cv.HOUGH_GRADIENT, 1, 200,
    param1=120, param2=100,
    minRadius=80, maxRadius=150
)

circles = np.uint16(np.around(circles))
print(f"Wykryte koła: {circles}\n")

total_value = 0.0

print("Wykryte monety")


#klasyfikacja + dodanie kolorowych obrysow
for i in circles[0, :]:
    cx, cy, radius_px = i

    diameter_px = radius_px * 2
    diameter_cm = diameter_px / px_per_cm
    diameter_mm = diameter_cm * 10.0

    # dopasowanie monety do tabeli rozmiarów coins
    diffs = {coin: abs(coins[coin] - diameter_mm) for coin in coins}

    # szukanie najlepszego dopasowania
    best_coin = min(diffs, key=diffs.get)
    best_diff = diffs[best_coin]

    # ograniczenie roznicy w oczekiwanej srednicy
    if best_diff > 1.50:
        best_coin = None
        color = (0, 0, 0)
    else:
        color = colors[best_coin]

    if best_coin is not None:
        print(f"{best_coin}: zmierzone {diameter_mm:.1f} mm")
        total_value += values[best_coin]
    else:
        print(f"Nierozpoznana moneta: średnica {diameter_mm:.1f} mm")

    # print(f"{best_coin}: zmierzone {diameter_mm:.1f} mm")

    # rysowanie okregow
    cv.circle(cimg, (cx, cy), radius_px, color, 3)
    cv.putText(cimg, best_coin, (cx - 40, cy - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


print(f"\nSUMA MONET: {total_value:.2f} pln\n")


# wyswietlanie i zapis
preview = cv.resize(cimg, None, fx=0.6, fy=0.6)
cv.imshow('detected circles', cimg)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('wynik_detekcji.png', cimg)


#####################################################################

# Implementacja kalibracji algorytmu detekcji

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
