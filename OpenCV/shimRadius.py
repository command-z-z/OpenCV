import cv2
import numpy as np

def get_outer_radius(img_input,blur):
    # gray
    gray = cv2.cvtColor(img_input,cv2.COLOR_BGR2GRAY)

    # gauss blur
    img_blur = cv2.GaussianBlur(gray,(blur,blur),0)

    # threshold
    ret,img_threshold =cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # open operation
    kernel = np.ones((2, 2), np.uint8)
    img_erode = cv2.erode(img_threshold, kernel, iterations=1)
    img = cv2.dilate(img_erode, kernel, iterations=1)

    # invert
    img=~img

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
    frame = img_input.copy()

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
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_inner_radius(img_input,blur):

    # gray
    gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

    # gauss blur
    img_blur = cv2.GaussianBlur(gray, (blur, blur), 0)

    # threshold
    ret,img_threshold = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # open operation
    kernel = np.ones((2, 2), np.uint8)
    img_erode = cv2.erode(img_threshold, kernel, iterations=1)
    img = cv2.dilate(img_erode, kernel, iterations=1)

    # invert
    img = ~img

    # findcontours
    all_contours, nada = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    outer_contours, nada = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(len(all_contours))
    print(len(outer_contours))

    # get inner contours
    inner_contours = all_contours
    for i in range(len(outer_contours)):
        outer_size = len(outer_contours[i])
        for j in range(len(all_contours)):
            all_size =len(all_contours[j])
            if (outer_size == all_size):
                inner_contours.remove(all_contours[j])
                break

    # find max inner_contour
    maxArea = 0
    for c in inner_contours:
        if cv2.contourArea(c) > maxArea:
            maxArea = cv2.contourArea(c)
            max_contour = c

    x, y, w, h = cv2.boundingRect(max_contour)

    # make contours frame
    frame = img_input.copy()

    # plot contours
    cv2.drawContours(frame, [max_contour], 0, (0, 0, 255), 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print("top_left:", (x, y))
    print("top_right:", (x + w, y))
    print("bottom_left:", (x, y + h))
    print("bottom_right:", (x + w, y + h))
    # cv2.circle(frame, (cx, cy), 2, (0, 0, 255), 2)
    # cv2.circle(frame, (rx, ry), 2, (0, 255, 0), 2)

    cv2.imshow('result1', img)
    cv2.imshow('result', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('shim3.png')

get_outer_radius(img,5)
# get_inner_radius(img,5)



