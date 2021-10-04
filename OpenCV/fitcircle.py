import cv2
import numpy as np

img_input = cv2.imread('shim2.jpg')

# gray
gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

# gauss blur
img_blur = cv2.GaussianBlur(gray, (3, 3), 0)

# threshold
ret, img_threshold = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# open operation
kernel = np.ones((2, 2), np.uint8)
img_erode = cv2.erode(img_threshold, kernel, iterations=1)
img = cv2.dilate(img_erode, kernel, iterations=1)

# invert
img = ~img

# find contours
outer_contours, nada = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# find max outer_contour
maxOuterArea = 0
for c in outer_contours:
    if cv2.contourArea(c) > maxOuterArea:
        maxOuterArea = cv2.contourArea(c)
        max_outer_contour = c

#  get focus
M = cv2.moments(max_outer_contour)
# print( M )
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])

# get centre of mass
centre_point = [cx, cy]


x1, y1, w1, h1 = cv2.boundingRect(max_outer_contour)

# make contours frame
frame = img_input.copy()

print(max_outer_contour)
print(len(max_outer_contour))
# plot outer contours
cv2.drawContours(frame, [max_outer_contour], 0, (0, 0, 255), 2)
cv2.circle(frame,(cx,cy),2,(0,255,0),2)
# cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

cv2.imshow('result1', img)
cv2.imshow('result', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
