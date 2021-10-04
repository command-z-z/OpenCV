import numpy as np
import cv2
from matplotlib import pyplot as plt

a= cv2.imread('shim.jpg',0)

equ = cv2.equalizeHist(a)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
CLAHE = CLAHE.apply(a)
res = np.hstack((a,equ,CLAHE))
#stacking images side-by-side
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()

