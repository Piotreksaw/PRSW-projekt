import cv2 as cv
import numpy as np
import os

# skala do zmniejszenia obrazu w przypadku mniejszego monitora
scale = 0.8

# paramtetry monet

# rozmiar
coins = {
    "1 pln": 23.0,
    "2 pln": 21.5,
    "5 pln": 24.0
}

# wartosci
values = {
    "1 pln": 1.00,
    "2 pln": 2.00,
    "5 pln": 5.00
}

# kolory
colors = {
    "1 pln": (255, 0, 255),   # Magenta
    "2 pln": (0, 100, 255),   # Pomaranczowy
    "5 pln": (0, 0, 255)      # Czerwony
}

# callback myszy
# uzytkownik wskazuje dwa punkty oddalone o 1 cm
def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param["lista"].append((x, y))
        param["points"] += 1
        cv.circle(param["img"], (x, y), 10, (0, 0, 255), -1)

# pusta funkcja wymagana przez createTrackbar
def nothing(x):
    pass

# wczytanie obrazu
img_path = "photos/monety_flesz.JPG"
img = cv.imread(img_path)

# zabezpieczenie przed bledna sciezka lub brakiem pliku
if img is None:
    print("Nie mozna otworzyc pliku", img_path)
    exit()

img = cv.resize(img, None, fx=scale, fy=scale)
img = cv.medianBlur(img, 7)  # poprawia stabilnosc HoughCircles
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# stworzenie listy paramterow rozpoznanych pieniedzy
param = {"points": 0, "lista": [], "img": img.copy()}

# poczatkowa kalibracja
photo_ratio = 4/3
size_photo = 1000
top_text = "Zaznacz odcinek 1cm na obrazku"
cv.namedWindow(top_text, cv.WINDOW_NORMAL)
cv.resizeWindow(top_text, int(photo_ratio * size_photo), size_photo)

while True:
    cv.imshow(top_text, param["img"])
    if param["points"] == 2:  # po zaznaczeniu dwoch punktow wyjscie z petli
        cv.destroyAllWindows()
        break
    cv.setMouseCallback(top_text, mouse_callback, param)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# przeliczenie pikseli na cm
p1 = np.array(param["lista"][0])
p2 = np.array(param["lista"][1])
px_distance = np.linalg.norm(p1 - p2)
px_per_cm = px_distance / 1.0

# dodanie zaznaczonego odcinka do zdjecia na stale, w celu lepszej analizy wynikow
img_orig = img.copy()
cv.circle(img_orig, tuple(param["lista"][0]), 10, (0, 0, 255), -1)
cv.circle(img_orig, tuple(param["lista"][1]), 10, (0, 0, 255), -1)
cv.line(img_orig, tuple(param["lista"][0]), tuple(param["lista"][1]), (0, 255, 0), 5)
cv.putText(
    img_orig, "1cm",
    (param["lista"][0][0], param["lista"][0][1] - 10),
    cv.FONT_HERSHEY_SIMPLEX, 1.8 * scale, (0, 255, 0), 3
)

# zakres srednic monet
min_d_mm = min(coins.values()) - 2.0
max_d_mm = max(coins.values()) + 2.0

# przeliczenie na piksele
minRadius_px = int((min_d_mm / 10.0 * px_per_cm) / 2.0)
maxRadius_px = int((max_d_mm / 10.0 * px_per_cm) / 2.0)

# dostrajanie detekcji
window_name = "Dostrajanie Detekcji"
cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.resizeWindow(window_name, int(photo_ratio * size_photo), size_photo)

# utworzenie suwakow do kalibracji wykrywania
cv.createTrackbar("Krawedzie (Param1)", window_name, 150, 400, nothing)
cv.createTrackbar("Srodki (Param2)", window_name, 45, 100, nothing)
cv.createTrackbar("Min Dystans (mm)", window_name, 18, 50, nothing)
cv.createTrackbar("dp x10", window_name, 10, 20, nothing)

while True:
    # zmiana wartosci suwakow
    p1_val = max(cv.getTrackbarPos("Krawedzie (Param1)", window_name), 1)
    p2_val = max(cv.getTrackbarPos("Srodki (Param2)", window_name), 1)
    dist_mm = max(cv.getTrackbarPos("Min Dystans (mm)", window_name), 5)
    min_dist_px = int((dist_mm / 10.0) * px_per_cm)
    dp_val = max(cv.getTrackbarPos("dp x10", window_name), 10) / 10.0

    preview = img_orig.copy()

    # algorytm Hough Circles
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

            # umieszczenie etykiet na monetach
            cv.circle(preview, (cx, cy), r, color, 6)
            cv.putText(
                preview, label,
                (cx - 50, cy - 15),
                cv.FONT_HERSHEY_SIMPLEX, 2.5 * scale, color, 8
            )

    # umieszczenie zliczonej sumy na zdjeciu
    cv.putText(
        preview, f"SUMA: {total_value:.2f} PLN",
        (40, 200), cv.FONT_HERSHEY_SIMPLEX,
        10 * scale, (0, 0, 255), int(15 * scale)
    )

    preview_final = preview.copy()
    cv.imshow(window_name, preview)

    if cv.waitKey(1) & 0xFF == ord('q'):
        # wypisanie parametrow przy wyjsciu, moze w przyszlosci znalezc zastosowanie przy uczeniu maszynowym
        print("Finalne parametry HoughCircles:")
        print(f"dp={dp_val}, param1={p1_val}, param2={p2_val}, minDist={min_dist_px} (px)")
        break

cv.destroyAllWindows()

# prezentacja finalnego obrazu i zapisanie go do zdjecia w formacie png
cv.imshow("Finalny Wynik (Zapisano)", preview_final)
cv.imwrite("wynik_detekcji.png", preview_final)
print(f"\nZapisano wynik. Suma: {total_value:.2f} PLN")
cv.waitKey(0)
cv.destroyAllWindows()


# w zjeciu z detekcją do pokazaniu suwaków zrobić nowy, który
# dodać opis preprocessingu 