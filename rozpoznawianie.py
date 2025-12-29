import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

scale = 1


#srednice
coins = {
    "1 pln": 23.0,
    "2 pln": 21.5,
    "5 pln": 24.0
}

# wartość
values = {
    "1 pln": 1.00,
    "2 pln": 2.00,
    "5 pln": 5.00
}

# przypisane kolory
colors = {
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
img = cv.resize(img, None, fx=scale, fy=scale)
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
if circles is None:
    print("nie wykryto monet")
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
