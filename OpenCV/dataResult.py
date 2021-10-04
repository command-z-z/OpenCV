import math

import cv2
import numpy as np
# import remove_shadow
import time

def calc_diff(pixel,bg_color):
    '''
    计算pixel与背景的离差平方和，作为当前像素点与背景相似程度的度量
    '''
    b = int(bg_color[0])
    g = int(bg_color[1])
    r = int(bg_color[2])
    return (pixel[0] - b) ** 2 + (pixel[1] - g) ** 2 + (pixel[2] - r) ** 2

def remove_bg(img,threshold=4000):
    # get bg color
    bg_color = img[0][0]

    # 将图像转成带透明通道的BGRA格式
    img_res = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    h, w = img_res.shape[0:2]
    for i in range(h):
        for j in range(w):
            if calc_diff(img_res[i][j],bg_color) < threshold:
                # img_res[i][j]为背景，将其颜色设为白色，且完全透明
                img_res[i][j][0] = 255
                img_res[i][j][1] = 255
                img_res[i][j][2] = 255
                img_res[i][j][3] = 0
    return img_res
    # cv2.imshow('res',img_res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# get two points distance
def distance(A,B):
    return (A[0]-B[0]) **2 + (A[1]-B[1]) **2

def get_screw_result(img_input,blur):
    rows, cols = img_input.shape[:2]
    # gray
    gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

    # gauss blur
    img_blur = cv2.GaussianBlur(gray, (blur, blur), 0)

    # threshold,use Otsu algorithm
    ret, img_threshold = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
        if cv2.contourArea(c) > maxArea:
            maxArea = cv2.contourArea(c)
            max_contour = c

    #  get focus
    M = cv2.moments(max_contour)
    # print( M )
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # get centre of mass
    centre_point = [cx, cy]

    # get closest point
    closest_distance = 100000
    for point in max_contour:
        if distance(point[0], centre_point) < closest_distance:
            closest_distance = distance(point[0], centre_point)
            closest_point = [point[0][0], point[0][1]]

    # get coordinate
    #  get min Rect
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # make contours frame
    frame = img_input.copy()

    # plot contours
    cv2.drawContours(frame, [max_contour], 0, (0, 0, 0), 2)
    cv2.line(frame, box[0], box[1], (0, 255, 0), 2)
    cv2.line(frame, box[1], box[2], (0, 255, 0), 2)
    cv2.line(frame, box[2], box[3], (0, 255, 0), 2)
    cv2.line(frame, box[3], box[0], (0, 255, 0), 2)
    cv2.circle(frame,(cx,cy),2,(0,0,255),2)
    # cv2.line(frame,centre_point,closest_point,(0, 255, 0), 2)
    [vx, vy, x, y] = cv2.fitLine(max_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    k=vy/vx
    print(k)
    # print(math.degrees(k))
    # print(vx,vy,x,y)
    cv2.circle(frame,(int(x),int(y)),2,(0,255,0),2)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(frame, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)

    ellipse = cv2.fitEllipse(max_contour)
    cv2.ellipse(frame, ellipse, (120, 255, 120), 2)

    hull = cv2.convexHull(max_contour)
    cv2.polylines(frame,[hull],True,(0,0,255),2)

    # judge length and width
    length1 = distance(box[0],box[1])
    length2 = distance(box[0],box[3])
    if length1 >= length2:
        radius_list = [box[0],box[3]]
        length_list = [box[0],box[1]]
    else:
        radius_list = [box[0], box[1]]
        length_list = [box[0], box[3]]

    cv2.imshow('result1', img)
    cv2.imshow('result', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # return only two coordinate like [(),()]*2
    return radius_list,length_list


def get_shim_result(img_input,blur):
    # gray
    gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

    # gauss blur
    img_blur = cv2.GaussianBlur(gray, (blur, blur), 0)

    # threshold
    ret, img_threshold = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # open operation
    kernel = np.ones((2, 2), np.uint8)
    img_erode = cv2.erode(img_threshold, kernel, iterations=1)
    img = cv2.dilate(img_erode, kernel, iterations=1)

    # invert
    img = ~img

    # find contours
    all_contours, nada = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    outer_contours, nada = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # don't find inner contour how to optimize
    if len(all_contours) == len(outer_contours):
        print('first dont find inner contour')

        CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        CLAHE = CLAHE.apply(img_blur)

        # change ret how to optimize
        # ret = ret + (255-ret)*0.6
        # threshold
        ret, img_threshold = cv2.threshold(CLAHE, ret, 255, cv2.THRESH_BINARY)

        # open operation
        kernel = np.ones((2, 2), np.uint8)
        img_erode = cv2.erode(img_threshold, kernel, iterations=1)
        img = cv2.dilate(img_erode, kernel, iterations=1)

        # invert
        img = ~img

        # find contours
        all_contours, nada = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        outer_contours, nada = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # get inner contours
    inner_contours = all_contours
    for i in range(len(outer_contours)):
        outer_size = len(outer_contours[i])
        for j in range(len(all_contours)):
            all_size = len(all_contours[j])
            if (outer_size == all_size):
                inner_contours.remove(all_contours[j])
                break

    # find max outer_contour
    maxOuterArea = 0
    for c in outer_contours:
        if cv2.contourArea(c) > maxOuterArea:
            maxOuterArea = cv2.contourArea(c)
            max_outer_contour = c

    # find max inner_contour
    maxInnerArea = 0
    for c in inner_contours:
        if cv2.contourArea(c) > maxInnerArea:
            maxInnerArea = cv2.contourArea(c)
            max_inner_contour = c

    x1, y1, w1, h1 = cv2.boundingRect(max_outer_contour)
    x2, y2, w2, h2 = cv2.boundingRect(max_inner_contour)

    # make contours frame
    frame = img_input.copy()

    # plot outer contours
    cv2.drawContours(frame, [max_outer_contour], 0, (0, 0, 255), 2)
    cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

    # plot inner contours
    cv2.drawContours(frame, [max_inner_contour], 0, (0, 0, 255), 2)
    cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)

    #get coordinate
    outer_list = [(x1, y1),(x1 + w1, y1),(x1, y1 + h1),(x1 + w1, y1 + h1)]
    inner_list = [(x2, y2),(x2 + w2, y2),(x2, y2 + h2),(x2 + w2, y2 + h2)]

    cv2.imshow('result1', img)
    cv2.imshow('result', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # return like this [top_left, top_right, bottom_left, bottom_right]*2
    return outer_list,inner_list

img = cv2.imread('screw3.png')
# img = remove_bg(img)
# get_shim_result(img,3)
get_screw_result(img,3)

