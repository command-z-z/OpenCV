import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('shelter2.jpg')
img = cv2.resize(img,(0,0),fx=0.1,fy=0.1)
h,w =img.shape[:2]
cv2.imshow('re',img)
cv2.waitKey()
cv2.destroyAllWindows()
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (80,40,230,120)
# 函数的返回值是更新的 mask, bgdModel, fgdModel
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()