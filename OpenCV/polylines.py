import cv2
import numpy as np

img = np.zeros((200, 200, 3), dtype=np.uint8)
pts = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], np.int32)  # 数据类型必须为 int32
pts = pts.reshape((-1, 1, 2))
# 绘制未填充的多边形
cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 0), thickness=1)
# 绘制填充的多边形
# cv2.fillPoly(img, [pts], color=(255, 255, 0))
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()