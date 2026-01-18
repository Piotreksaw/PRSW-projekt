import cv2 as cv
import numpy as np
import os

# konfiguracja

PHOTOS_DIR = "photos"
RESULTS_DIR = "wyniki"
MAX_IMAGES_PER_VARIANT = 10
SCALE_DISPLAY = 0.8  # skala do zmniejszenia obrazu w przypadku mniejszego monitora
IGNORE_BOTTOM_PERCENT = 0.1

wariant = {
    # "skaner": {"param1": 150, "param2": 40, "dp": 1.0},
    "biurko": {"param1": 180, "param2": 45, "dp": 1.0},
    # "flesz":  {"param1": 150, "param2": 45, "dp": 1.0}
}

# parametry monet
# rozmiar (w mm)
coins = {
    "1 pln": 23.0,
    "2 pln": 21.5,
    "5 pln": 24.0
}

# wartosci monet
values = {
    "1 pln": 1.00,
    "2 pln": 2.00,
    "5 pln": 5.00
}

# kolory dla etykiet
colors = {
    "1 pln": (255, 0, 255),  # Magenta
    "2 pln": (0, 100, 255),  # Pomaranczowy
    "5 pln": (0, 0, 255)  # Czerwony
}

# Słownik do przechowywania zbiorczych statystyk dla każdego wariantu
summary = {}


#callback myszy
# uzytkownik wskazuje dwa punkty oddalone o 1 cm
def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN and param["points"] < 2:
        param["points"] += 1
        param["pts"].append((x, y))
        cv.circle(param["img"], (x, y), 10, (0, 0, 255), -1)


# przygotowanie folderów
os.makedirs(RESULTS_DIR, exist_ok=True)

# główna pętla programu

with open("wyniki.txt", "w", encoding="utf-8") as wyniki_txt:
    for nazwa_wariantu, hough_params in wariant.items():
        # Inicjalizacja licznika dla danego wariantu
        summary[nazwa_wariantu] = {"count": 0, "total_cash": 0.0}

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

            # Poprawa stabilnosci HoughCircles
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            gray_blur = cv.GaussianBlur(gray, (7, 7), 1.5)

            # Wycięcie dolnego paska (np. jesli na dole jest linijka/tekst)
            ignore_h = int(h_img * IGNORE_BOTTOM_PERCENT)
            gray_for_hough = gray_blur.copy()
            cv.rectangle(gray_for_hough, (0, h_img - ignore_h), (w_img, h_img), 0, -1)

            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]

            # kalibracja skali
            # stworzenie listy paramterow rozpoznanych pieniedzy
            calib = {"points": 0, "pts": [], "img": img.copy()}
            win_name = f"Zaznacz odcinek 1cm: {img_name}"

            cv.namedWindow(win_name, cv.WINDOW_NORMAL)
            cv.setMouseCallback(win_name, mouse_callback, calib)

            while calib["points"] < 2:
                cv.imshow(win_name, calib["img"])
                if cv.waitKey(1) & 0xFF == 27:  # ESC aby pominąć
                    break

            cv.destroyAllWindows()
            if calib["points"] < 2:
                continue

            # przeliczenie pikseli na cm
            p1 = np.array(calib["pts"][0])
            p2 = np.array(calib["pts"][1])
            px_per_cm = np.linalg.norm(p1 - p2)

            # dodanie zaznaczonego odcinka do zdjecia na stale, w celu lepszej analizy wynikow
            img_with_scale = img.copy()

            mid_x = int((calib["pts"][0][0] + calib["pts"][1][0]) / 2)
            mid_y = int((calib["pts"][0][1] + calib["pts"][1][1]) / 2)
            cv.line(img_with_scale, tuple(calib["pts"][0]), tuple(calib["pts"][1]), (0, 255, 0), 5)
            cv.circle(img_with_scale, tuple(calib["pts"][0]), 10, (0, 0, 255), -1)
            cv.circle(img_with_scale, tuple(calib["pts"][1]), 10, (0, 0, 255), -1)
            cv.putText(img_with_scale, "1cm",
                       (mid_x - 40, mid_y - 15),
                       cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3
                       )

            # max i min promień
            # zakres srednic monet z lekką tolerancją
            min_d_mm = min(coins.values()) - 1.5
            max_d_mm = max(coins.values()) + 1.5

            # przeliczenie na piksele
            minR = int((min_d_mm / 10 * px_per_cm) / 2)
            maxR = int((max_d_mm / 10 * px_per_cm) / 2)

            # algorytm Hough
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

            preview = img_with_scale.copy()
            total_value = 0.0
            detected = 0

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for cx, cy, r in circles[0]:
                    diameter_mm = (2 * r) / px_per_cm * 10
                    diffs = {k: abs(v - diameter_mm) for k, v in coins.items()}
                    best = min(diffs, key=diffs.get)

                    tolerance = 0.8 + 0.04 * diameter_mm

                    # Analiza koloru/nasycenia dla rozróżnienia 1zł i 2zł
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

                        # Aktualizacja statystyk zbiorczych
                        summary[nazwa_wariantu]["count"] += 1
                        summary[nazwa_wariantu]["total_cash"] += values[best]

                        # rysowanie rozpoznanej monety
                        cv.circle(preview, (cx, cy), r, color, 4)
                        cv.putText(
                            preview, label,
                            (cx - 40, cy - 10),
                            cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 3
                        )

            # umieszczenie zliczonej sumy na zdjeciu
            cv.putText(
                preview,
                f"SUMA: {total_value:.2f} PLN",
                (30, 150),
                cv.FONT_HERSHEY_SIMPLEX, 5.0,
                (0, 0, 255), 10
            )

            # zapisanie obrazu wynikowego
            out_name = f"wynik_{nazwa_wariantu}_{idx}.jpg"
            out_path = os.path.join(variant_results_dir, out_name)
            cv.imwrite(out_path, preview)

            # zapis do pliku tekstowego
            wyniki_txt.write(
                f"wariant={nazwa_wariantu}, plik={img_name}, "
                f"suma={total_value:.2f}, monety={detected}\n"
            )
            wyniki_txt.write("\n")

            print(f"  {img_name} -> {total_value:.2f} PLN")

            # Wyswietlenie wynikow
            show_name = f"Wynik: {img_name} (Nacisnij klawisz)"
            cv.namedWindow(show_name, cv.WINDOW_NORMAL)
            cv.imshow(show_name, preview)
            key = cv.waitKey(0)
            cv.destroyWindow(show_name)

            if key == 27:  # esc przerywa caly program
                print("Przerwano przez użytkownika.")
                exit()

    # Podsumowanie wynikow
    print("\nPodsumowanie:")
    wyniki_txt.write("\nPodsumowanie::\n")

    for w, stats in summary.items():
        podsumowanie_str = (f"Wariant: {w:10} | "
                            f"Łącznie monet: {stats['count']:3} | "
                            f"Łączna wartość: {stats['total_cash']:6.2f} PLN")
        print(podsumowanie_str)
        wyniki_txt.write(podsumowanie_str + "\n")

print("\nKoniec programu")