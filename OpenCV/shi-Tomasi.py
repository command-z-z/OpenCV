import numpy as np
import cv2
from matplotlib import pyplot as plt
# img = cv2.imread('screw1.png')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# corners = cv2.goodFeaturesToTrack(gray,4,0.5,10)
# # 返回的结果是 [[ 311., 250.]] 两层括号的数组。
# corners = np.int0(corners)
# print(corners)
# for i in corners:
#     x,y = i.ravel()
#     cv2.circle(img,(x,y),3,255,-1)
#
# cv2.imshow('rea',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np
img = cv2.imread('screw2.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT()
kp = sift.detect(gray,None)
cv2.drawKeypoints(gray,kp)

cv2.imshow('res',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()