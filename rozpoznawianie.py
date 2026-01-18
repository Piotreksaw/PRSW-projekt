import cv2 as cv
import numpy as np
import os

# skala do zmniejszenia obrazu w przypadku mniejszego monitora
scale = 0.8

# paramtetry monet

# rozmiar (srednice w mm)
coins = {
    "1 pln": 23.0,
    "2 pln": 21.5,
    "5 pln": 24.0
}

# wartosci nominalne
values = {
    "1 pln": 1.00,
    "2 pln": 2.00,
    "5 pln": 5.00
}

# kolory oznaczen (BGR)
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
img_path = "photos/flesz_kalibracja.JPG"
img = cv.imread(img_path)

# zabezpieczenie przed bledna sciezka lub brakiem pliku
if img is None:
    print("Nie mozna otworzyc pliku", img_path)
    exit()

# Preprocessing obrazu
img = cv.resize(img, None, fx=scale, fy=scale)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_original = gray.copy()

# Zastosowanie CLAHE i rozmycia Gaussa
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)
blur = cv.GaussianBlur(gray, (7, 7), 1.5)

# Opcjonalne maski pomocnicze
mask = cv.adaptiveThreshold(
    blur, 255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY_INV,
    51, 3
)
edges = cv.Canny(mask, 80, 160)

# Przygotowanie obrazów do mozaiki (muszą mieć 3 kanały BGR)
view1 = cv.cvtColor(gray_original, cv.COLOR_GRAY2BGR)
view2 = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
view3 = cv.cvtColor(blur, cv.COLOR_GRAY2BGR)
view4 = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

# Dodanie podpisów na podglądach
font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 8
thick = 6
cv.putText(view1, "1. Skala szarosci", (20, 200), font, font_scale, (0,255,0), thick)
cv.putText(view2, "2. CLAHE", (20, 200), font, font_scale, (0,255,0), thick)
cv.putText(view3, "3. Gaussian Blur", (20, 200), font, font_scale, (0,255,0), thick)
cv.putText(view4, "4. Krawedzie (Canny)", (20, 200), font, font_scale, (0,255,0), thick)

# Stworzenie podglądu
top_row = np.hstack((view1, view2))
bottom_row = np.hstack((view3, view4))
combined_preprocessing = np.vstack((top_row, bottom_row))

cv.namedWindow("Etapy Preprocessingu", cv.WINDOW_NORMAL)
cv.imshow("Etapy Preprocessingu", combined_preprocessing)
cv.imwrite("preprocessing.jpg", combined_preprocessing)

# Ignorowanie dolnej czesci zdjecia - na testowanej linijce byly narysowane okregi
ignore_height_px = int(img.shape[0] * 0.15)
cv.rectangle(edges, (0, img.shape[0] - ignore_height_px), (img.shape[1], img.shape[0]), 0, -1)

# Przygotowanie danych do analizy nasycenia (HSV)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
sat = hsv[:, :, 1]

# stworzenie listy paramterow rozpoznanych pieniedzy
param = {"points": 0, "lista": [], "img": img.copy()}

# poczatkowa kalibracja skali
photo_ratio = 4 / 3
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

# dodanie zaznaczonego odcinka do zdjecia na stale
img_orig = img.copy()
cv.circle(img_orig, tuple(param["lista"][0]), 10, (0, 0, 255), -1)
cv.circle(img_orig, tuple(param["lista"][1]), 10, (0, 0, 255), -1)
cv.line(img_orig, tuple(param["lista"][0]), tuple(param["lista"][1]), (0, 255, 0), 5)
cv.putText(
    img_orig, "1cm",
    (param["lista"][0][0], param["lista"][0][1] - 10),
    cv.FONT_HERSHEY_SIMPLEX, 1.2 * scale, (0, 255, 0), 3
)

# zakres srednic monet (z marginesem bledu)
min_d_mm = min(coins.values()) - 1.5
max_d_mm = max(coins.values()) + 1.5

# przeliczenie promieni na piksele dla HoughCircles
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
    # zmiana wartosci suwakow w czasie rzeczywistym
    p1_val = max(cv.getTrackbarPos("Krawedzie (Param1)", window_name), 1)
    p2_val = max(cv.getTrackbarPos("Srodki (Param2)", window_name), 1)
    dist_mm = max(cv.getTrackbarPos("Min Dystans (mm)", window_name), 5)
    min_dist_px = int((dist_mm / 10.0) * px_per_cm)
    dp_val = max(cv.getTrackbarPos("dp x10", window_name), 10) / 10.0

    preview = img_orig.copy()
    total_value = 0.0

    # Główny algorytm detekcji
    circles = cv.HoughCircles(
        blur,
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
            # Obliczenie rzeczywistej srednicy
            diameter_mm = (2 * r) / px_per_cm * 10
            diffs = {k: abs(v - diameter_mm) for k, v in coins.items()}
            best = min(diffs, key=diffs.get)

            # Tolerancja zalezna od wielkosci monety
            tolerance = 0.8 + 0.04 * diameter_mm

            #HSV – analiza nasycenia wewnatrz monety
            mask_coin = np.zeros_like(gray)
            cv.circle(mask_coin, (cx, cy), r, 255, -1)
            mean_sat = cv.mean(sat, mask_coin)[0]

            label = "?"
            color = (0, 255, 255)

            if diffs[best] < tolerance:
                # Korekta nominalu na podstawie nasycenia (odróżnienie 1 PLN od reszty)
                if best == "1 pln" and mean_sat > 55:
                    best = "2 pln"
                elif best in ["2 pln", "5 pln"] and mean_sat < 40:
                    best = "1 pln"

                label = best
                color = colors[best]
                total_value += values[best]

            # umieszczenie etykiet i okregów na podgladzie
            cv.circle(preview, (cx, cy), r, color, 6)
            cv.putText(
                preview, f"{label}",
                (cx - 50, cy - 15),
                cv.FONT_HERSHEY_SIMPLEX, 2.5 * scale, color, 8
            )

    # wyswietlenie zliczonej sumy
    cv.putText(
        preview, f"SUMA: {total_value:.2f} PLN",
        (40, 200),
        cv.FONT_HERSHEY_SIMPLEX,
        10 * scale, (0, 0, 255), int(15 * scale)
    )

    preview_final = preview.copy()
    cv.imshow(window_name, preview)

    if cv.waitKey(1) & 0xFF == ord('q'):
        # wypisanie parametrów do konsoli przy wyjściu
        print("Finalne parametry HoughCircles:")
        print(f"dp={dp_val}, param1={p1_val}, param2={p2_val}, minDist={min_dist_px} (px)")
        break

cv.destroyAllWindows()

# Prezentacja i zapis finalnego wyniku
cv.imshow("Finalny Wynik (Zapisano)", preview_final)
cv.imwrite("wyniki/do_sprawozdania/kalibracja.png", preview_final)
print(f"\nZapisano wynik. Suma: {total_value:.2f} PLN")
cv.waitKey(0)
cv.destroyAllWindows()