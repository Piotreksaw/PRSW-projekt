import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

scale = 0.4


#srednice
coins = {
    # "50 gr": 20.5,
    "1 pln": 23.0,
    "2 pln": 21.5,
    "5 pln": 24.0
}

# wartość
values = {
    # "50 gr": 0.50,
    "1 pln": 1.00,
    "2 pln": 2.00,
    "5 pln": 5.00
}

# przypisane kolory
colors = {
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
        cv.circle(param["img"], (x, y), 5, (0, 0, 255), -1)

def nothing(x):
    pass

# wczytywanie
img_path = "photos/monety_grosze_flesz.JPG"
img = cv.imread(img_path)

if img is None:
    print("Nie można otworzyć pliku", img_path)
    exit()

img = cv.resize(img, None, fx=scale, fy=scale)
img = cv.medianBlur(img, 1)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)



param = {"points": 0, "lista": [], "img": img.copy()}

photo_ratio = 4/3
size_photo = 1300
top_text = "Zaznacz odcinek 5cm na obrazku"
cv.namedWindow(top_text, cv.WINDOW_NORMAL)
cv.resizeWindow(top_text, int(photo_ratio*size_photo), size_photo)

# zaznaczenie 5cm
while True:
    cv.imshow(top_text, param["img"])
    # cv.imshow("Zaznacz odcinek 5cm na obrazku", img)
    if param["points"] == 2:
        cv.destroyAllWindows()
        break

    cv.setMouseCallback(top_text, mouse_callback, param)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        cv.destroyWindow(top_text)
        break

# przeliczenie pikseli na cm
p1 = np.array(param["lista"][0])
p2 = np.array(param["lista"][1])

px_distance = np.linalg.norm(p1 - p2)
real_length_cm = 1.0
px_per_cm = px_distance / real_length_cm


window_name = "Dostrajanie Detekcji"
cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.resizeWindow(window_name, int(photo_ratio*size_photo), size_photo)

cv.createTrackbar("Czułość (Param1)", window_name, 150, 350, nothing)
cv.createTrackbar("Rygor (Param2)", window_name, 100, 250, nothing)
cv.createTrackbar("Min Dystans", window_name, 260, 400, nothing)

print(">>> Ustaw suwaki. Naciśnij 'q' aby zakończyć <<<")

final_circles = None

img_orig = img.copy()

while True:
    p1_val = max(cv.getTrackbarPos("Czułość (Param1)", window_name), 1)
    p2_val = max(cv.getTrackbarPos("Rygor (Param2)", window_name), 1)
    min_dist = max(cv.getTrackbarPos("Min Dystans", window_name), 10)

    preview = img_orig.copy()

    circles = cv.HoughCircles(
        gray,
        cv.HOUGH_GRADIENT,
        1,
        minDist=min_dist,
        param1=p1_val,
        param2=p2_val,
        # minRadius=int(1.6 * px_per_cm),
        # maxRadius=int(2.6 * px_per_cm)
        minRadius=80,
        maxRadius=150
    )

    total_value = 0.0

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f"Wykryte koła: {circles}\n")


        for cx, cy, r in circles[0]:
            diameter_mm = (2 * r) / px_per_cm * 10

            diffs = {k: abs(v - diameter_mm) for k, v in coins.items()}
            best = min(diffs, key=diffs.get)

            if diffs[best] < 1.3:
                color = colors[best]
                total_value += values[best]
                label = best
            else:
                color = (0, 255, 255)
                label = "?"

            cv.circle(preview, (cx, cy), r, color, 3)
            cv.putText(preview, label, (cx - 30, cy - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv.putText(preview, f"SUMA: {total_value:.2f} PLN",
               (40, 60), cv.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,255), 3)

    cv.imshow(window_name, preview)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cv.destroyAllWindows()

print("paramtety:")
print(f"Minimalny dystans {min_dist}")
print(f"parametr 1: {p1_val}")
print(f"parametr 2: {p2_val} \n")

# Detekcja monet

# circles = cv.HoughCircles(
#     gray, cv.HOUGH_GRADIENT, 1, 200,
#     param1=120, param2=100,
#     minRadius=80, maxRadius=150
# )

# circles = np.uint16(np.around(circles))
# if circles is None:
#     print("nie wykryto monet")
# print(f"Wykryte koła: {circles}\n")
#

#
# print("Wykryte monety")



cv.putText(img, f"SUMA MONET: {total_value:.2f} pln", (50, 50),
               cv.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255) , 4)



print(f"\nSUMA MONET: {total_value:.2f} pln\n")

# wyswietlanie i zapis
preview = cv.resize(img, None, fx=scale, fy=scale)
cv.imshow('detected circles', img)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('wynik_detekcji.png', img)
