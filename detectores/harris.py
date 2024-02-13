import numpy as np
import cv2 as cv
filename = 'imgs/ex1.png'
img = cv.imread(filename)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,5,3,0.09)
print("Quantidade de pontos: ", len(dst))
dst = cv.dilate(dst,None)
img[dst>0.001*dst.max()]=[0,0,255]
cv.imshow('dst',img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()