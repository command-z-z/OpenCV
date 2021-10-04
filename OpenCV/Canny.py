import cv2
import numpy as np

img_input = cv2.imread('screw1.png')

gray = cv2.cvtColor(img_input,cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray,100,200)


cv2.imshow('result1', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()