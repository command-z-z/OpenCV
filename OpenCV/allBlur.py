import cv2
import numpy as np

img1 = cv2.imread('screw1.png')

# gray
gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

# all kinds of blur way
img_box = cv2.boxFilter(gray,-1,(5,5))
img_blur = cv2.blur(gray,(5,5))
img_gauss = cv2.GaussianBlur(gray,(5,5),0)
img_median = cv2.medianBlur(gray,5)
img_bilateral = cv2.bilateralFilter(gray,0,75,75)

# threshold
ret, img_threshold = cv2.threshold(img_bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(ret)

# open operation
kernel = np.ones((2, 2), np.uint8)
img_erode = cv2.erode(img_threshold, kernel, iterations=1)
img = cv2.dilate(img_erode, kernel, iterations=1)

# invert
img = ~img

# findcontours
contours, nada = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# find max contour
maxArea = 0
for c in contours:
    if cv2.contourArea(c)>maxArea:
        maxArea=cv2.contourArea(c)
        max_contour = c

x, y, w, h = cv2.boundingRect(max_contour)

#make contours frame
frame = img1.copy()

# plot contours
cv2.drawContours(frame, [max_contour], 0, (0, 0, 255), 2)
cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

print("top_left:", (x, y))
print("top_right:", (x + w, y))
print("bottom_left:", (x, y + h))
print("bottom_right:", (x + w, y + h))
# cv2.circle(frame, (cx, cy), 2, (0, 0, 255), 2)
# cv2.circle(frame, (rx, ry), 2, (0, 255, 0), 2)

cv2.imshow('result1',img)
cv2.imshow('result',frame)

cv2.imshow('img_box',img_box)
cv2.imshow('img_blur',img_blur)
cv2.imshow('img_gauss',img_gauss)
cv2.imshow('img_median',img_median)
cv2.imshow('img_bilateral',img_bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()
