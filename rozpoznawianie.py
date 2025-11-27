import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

scale = 0.4

# rozmiar bilonu w Polsce
grosz = 15.5
dwa_grosze = 17.5
piec_groszy = 19.5
dziesiec_groszy = 16.5
dwadziescia_groszy = 18.5
piecdzisiat_groszy = 20.5
zlotowka = 23
dwa_zlote = 21.5
piec_zlotych = 24



def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param["lista"].append((x,y))
        param["points"] += 1
        cv.circle(param["img"], (x, y), 5, (0, 0, 255), -1)


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


while True:
    cv.imshow("Zaznacz odcinek 5cm na obrazku",img)
    if param["points"] == 2:
        cv.destroyAllWindows()
        break


    cv.setMouseCallback("Zaznacz odcinek 5cm na obrazku", mouse_callback, param)



    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        cv.destroyAllWindows()
        break

print(param["lista"])

circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,200,
param1=150,param2=130,minRadius=50,maxRadius=500)

circles = np.uint16(np.around(circles))
print(f"Ilość kół: {circles}")
for i in circles[0,:]:
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)



cv.imshow('detected circles',cimg)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('wynik_detekcji.png',cimg)


# if __name__ == '__main__':
