import cv2 as cv
import numpy as np
import os

scale = 0.4

img_path = ".//photos/IMG_20221011_172837.jpg"
print(img_path)


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

img = cv.imread(img_path, 0)
img = cv.resize(img,None, fx=scale, fy=scale)
img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
img = cv.medianBlur(img,3)
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
 
cv.imshow('detected circles',img)
cv.waitKey(0)
cv.destroyAllWindows()
 
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