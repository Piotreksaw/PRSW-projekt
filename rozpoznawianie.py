import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

scale = 0.4


# -----------------------------------------
# Średnice monet w mm (prawidłowe wartości)
# -----------------------------------------
coins = {
    "1 gr": 15.5,
    "2 gr": 17.5,
    "5 gr": 19.5,
    "10 gr": 16.5,
    "20 gr": 18.5,
    "50 gr": 20.5,
    "1 pln": 23.0,
    "2 pln": 21.5,
    "5 pln": 24.0
}

# wartość nominalna w pln
values = {
    "1 gr": 0.01,
    "2 gr": 0.02,
    "5 gr": 0.05,
    "10 gr": 0.10,
    "20 gr": 0.20,
    "50 gr": 0.50,
    "1 pln": 1.00,
    "2 pln": 2.00,
    "5 pln": 5.00
}

# kolory (BGR)
colors = {
    "1 gr":  (0, 255, 255),
    "2 gr":  (0, 165, 255),
    "5 gr":  (0, 140, 255),
    "10 gr": (0, 255, 0),
    "20 gr": (255, 255, 0),
    "50 gr": (255, 0, 0),
    "1 pln":  (255, 0, 255),
    "2 pln":  (0, 100, 255),
    "5 pln":  (0, 0, 255)
}


# -----------------------------------------
# Callback myszy
# -----------------------------------------

def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param["lista"].append((x, y))
        param["points"] += 1
        cv.circle(param["img"], (x, y), 5, (0, 0, 255), -1)


# -----------------------------------------
# Wczytywanie zdjęcia
# -----------------------------------------

img_path = ".//photos/IMG_20221011_172837.jpg"
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


# -----------------------------------------
# Zaznaczenie odcinka 5 cm
# -----------------------------------------

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


# -----------------------------------------
# Detekcja monet (HoughCircles)
# -----------------------------------------

circles = cv.HoughCircles(
    img, cv.HOUGH_GRADIENT, 1, 200,
    param1=100, param2=130,
    minRadius=20, maxRadius=500
)

circles = np.uint16(np.around(circles))
print(f"Wykryte koła: {circles}\n")

total_value = 0.0

print("---- ROZPOZNANE MONETY ----")


# -----------------------------------------
# Klasyfikacja monet + kolorowanie
# -----------------------------------------

for i in circles[0, :]:
    cx, cy, radius_px = i

    diameter_px = radius_px * 2
    diameter_cm = diameter_px / px_per_cm
    diameter_mm = diameter_cm * 10.0

    # Wybór najbliższego nominału (minimalna różnica)
    best_coin = min(coins.keys(), key=lambda c: abs(coins[c] - diameter_mm))
    color = colors[best_coin]

    # dodawanie wartości
    total_value += values[best_coin]

    print(f"{best_coin}: zmierzone {diameter_mm:.1f} mm")

    # rysowanie okręgu i podpisu
    cv.circle(cimg, (cx, cy), radius_px, color, 3)
    cv.putText(cimg, best_coin, (cx - 40, cy - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


print(f"\nSUMA MONET: {total_value:.2f} pln\n")


# -----------------------------------------
# Wyświetlenie i zapis wyniku
# -----------------------------------------

cv.imshow('detected circles', cimg)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('wynik_detekcji.png', cimg)
