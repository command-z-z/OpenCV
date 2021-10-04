import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_contours(img_input,blur,threshold):

    # gray
    gray = cv2.cvtColor(img_input,cv2.COLOR_BGR2GRAY)

    # guass
    img_blur = cv2.GaussianBlur(gray,(blur,blur),0)

    # threshold
    img = cv2.threshold(img_blur,threshold,255,cv2.THRESH_OTSU)[1]

    # open operation
    kernel = np.ones((2,2),np.uint8)
    img_erode = cv2.erode(img,kernel,iterations=1)
    img = cv2.dilate(img_erode,kernel,iterations=1)

    # invert
    img = ~img

    # findcontours
    # parameter cv2.CHAIN_APPROX_SIMPLE
    contours,nada = cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


    # make coutour frame
    frame = img_input.copy()
    print(len(contours))

    for c in contours:

        if cv2.contourArea(c) < 500:
            continue


        M = cv2.moments(c)
        # print( M )
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        x, y, w, h = cv2.boundingRect(c)
        rx = x + int(w / 2)
        ry = y + int(h / 2)

        # plot contours
        cv2.drawContours(frame, [c], 0, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame,(cx,cy),5,(0, 255, 0), 2)

        print("top_left:",(x,y))
        print("top_right:",(x+w,y))
        print("bottom_left:", (x, y+h))
        print("bottom_right:", (x + w, y+h))
        # cv2.circle(frame, (cx, cy), 2, (0, 0, 255), 2)
        # cv2.circle(frame, (rx, ry), 2, (0, 255, 0), 2)

    cv2.imshow('result1',img)
    cv2.imshow('result',frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('screw3.png')
find_contours(img,5,127)


