import cv2
import numpy as np
import math


def max_filtering(N, I_temp):
    wall = np.full((I_temp.shape[0] + (N // 2) * 2, I_temp.shape[1] + (N // 2) * 2), -1)
    wall[(N // 2):wall.shape[0] - (N // 2), (N // 2):wall.shape[1] - (N // 2)] = I_temp.copy()
    temp = np.full((I_temp.shape[0] + (N // 2) * 2, I_temp.shape[1] + (N // 2) * 2), -1)
    for y in range(0, wall.shape[0]):
        for x in range(0, wall.shape[1]):
            if wall[y, x] != -1:
                window = wall[y - (N // 2):y + (N // 2) + 1, x - (N // 2):x + (N // 2) + 1]
                num = np.amax(window)
                temp[y, x] = num
    A = temp[(N // 2):wall.shape[0] - (N // 2), (N // 2):wall.shape[1] - (N // 2)].copy()
    return A


def min_filtering(N, A):
    wall_min = np.full((A.shape[0] + (N // 2) * 2, A.shape[1] + (N // 2) * 2), 300)
    wall_min[(N // 2):wall_min.shape[0] - (N // 2), (N // 2):wall_min.shape[1] - (N // 2)] = A.copy()
    temp_min = np.full((A.shape[0] + (N // 2) * 2, A.shape[1] + (N // 2) * 2), 300)
    for y in range(0, wall_min.shape[0]):
        for x in range(0, wall_min.shape[1]):
            if wall_min[y, x] != 300:
                window_min = wall_min[y - (N // 2):y + (N // 2) + 1, x - (N // 2):x + (N // 2) + 1]
                num_min = np.amin(window_min)
                temp_min[y, x] = num_min
    B = temp_min[(N // 2):wall_min.shape[0] - (N // 2), (N // 2):wall_min.shape[1] - (N // 2)].copy()
    return B


def background_subtraction(I, B):
    O = I - B
    norm_img = cv2.normalize(O, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    return norm_img


def min_max_filtering(M, N, I):
    if M == 0:
        # max_filtering
        A = max_filtering(N, I)
        # min_filtering
        B = min_filtering(N, A)
        # subtraction
        normalised_img = background_subtraction(I, B)
        normalised_img = np.array(normalised_img, dtype=np.uint8)
    elif M == 1:
        # min_filtering
        A = min_filtering(N, I)
        # max_filtering
        B = max_filtering(N, A)
        # subtraction
        normalised_img = background_subtraction(I, B)
        normalised_img = np.array(normalised_img, dtype=np.uint8)
    return normalised_img


def calc_diff(pixel, bg_color):
    '''
    计算pixel与背景的离差平方和，作为当前像素点与背景相似程度的度量
    '''
    b = int(bg_color[0])
    g = int(bg_color[1])
    r = int(bg_color[2])
    return (pixel[0] - b) ** 2 + (pixel[1] - g) ** 2 + (pixel[2] - r) ** 2


def remove_bg(img, threshold=4000):
    # get bg color
    bg_color = img[0][0]

    # 将图像转成带透明通道的BGRA格式
    img_res = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    h, w = img_res.shape[0:2]
    for i in range(h):
        for j in range(w):
            if calc_diff(img_res[i][j], bg_color) < threshold:
                # img_res[i][j]为背景，将其颜色设为白色，且完全透明
                img_res[i][j][0] = 255
                img_res[i][j][1] = 255
                img_res[i][j][2] = 255
                img_res[i][j][3] = 0

    # cv2.imshow('res',img_res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img_res


def distance(A, B):
    return (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2


def find_Screw_contours(img_input, blur=3):
    # remove bg
    # img_input = remove_bg(img_input)

    # gray
    gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

    # remove shadow
    # gray = min_max_filtering(M=0, N=20, I=gray)
    # gauss blur
    img_blur = cv2.GaussianBlur(gray, (blur, blur), 0)

    # threshold,use Otsu algorithm
    ret, img_threshold = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(ret)

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

    # get focus
    M = cv2.moments(max_contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # get center point
    center_point = [cx, cy]

    # get closest point
    close_distance = 100000
    for point in max_contour:
        if distance(point[0], center_point) < close_distance:
            close_distance = distance(point[0], center_point)
            close_point = [point[0][0], point[0][1]]

    #  get min Rect
    rect = cv2.minAreaRect(max_contour)

    print('w,h:',rect[1])
    print('angle:',rect[2])

    # get coordinate
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # make contours frame
    frame = img_input.copy()

    # plot contours
    cv2.drawContours(frame, [max_contour], 0, (0, 0, 255), 1)
    cv2.line(frame, tuple(box[0]), tuple(box[1]), (0, 255, 0), 1)
    cv2.line(frame, tuple(box[1]), tuple(box[2]), (0, 255, 0), 1)
    cv2.line(frame, tuple(box[2]), tuple(box[3]), (0, 255, 0), 1)
    cv2.line(frame, tuple(box[3]), tuple(box[0]), (0, 255, 0), 1)
    cv2.circle(frame, tuple(center_point), 2, (0, 255, 0), 1)
    # cv2.line(frame,center_point,close_point,(0,255,0),1)
    # judge length and width
    length1 = distance(box[0], box[1])
    length2 = distance(box[0], box[3])
    print('len:',math.sqrt(length1))
    if length1 >= length2:
        radius_list = [[box[0][0], box[0][1]], [box[3][0], box[3][1]]]
        length_list = [[box[0][0], box[0][1]], [box[1][0], box[1][1]]]
    else:
        radius_list = [[box[0][0], box[0][1]], [box[1][0], box[1][1]]]
        length_list = [[box[0][0], box[0][1]], [box[3][0], box[3][1]]]

    # 改用找斜率的算法
    k = (radius_list[0][0] - radius_list[1][0]) / (radius_list[0][1] - radius_list[1][1])
    min_left = 1e7
    min_right = 1e7
    left_point = [0, 0]
    right_point = [0, 0]
    for contour in max_contour:
        temp = (cx - contour[0][0]) / (cy - contour[0][1])
        if abs(temp - k) < min_right:
            min_right = abs(temp - k)
            # print(temp)
            right_point[0] = contour[0][0]
            right_point[1] = contour[0][1]
    for contour in max_contour:
        temp = (cx - contour[0][0]) / (cy - contour[0][1])
        if abs(temp - k) < min_left and (
                (right_point[0] - cx) / (contour[0][0] - cx) < 0 or (right_point[1] - cy) / (contour[0][1] - cy) < 0):
            min_left = abs(temp - k)
            # print(temp)
            left_point[0] = contour[0][0]
            left_point[1] = contour[0][1]

    #  print(left_point,right_point)
    # way1 to measure radius
    # point1,point2 = radius_list
    # center_point = [(point1[0]+point2[0])/2,(point1[1]+point2[1])/2]
    # radius_list = [point1,center_point]

    # way2 to measure radius
    # radius_list = [center_point,close_point]

    # way3 to measure radius
    radius_list = [left_point, right_point]
    cv2.line(frame, tuple(left_point), tuple(right_point), (0, 255, 0), 1)

    cv2.imshow('result1', img)
    cv2.imshow('result', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return only two coordinate like [(),()]*2
    return radius_list, length_list


def find_Shim_contours(img_input, blur=3):
    # remove bg
    img_input = remove_bg(img_input)

    # gray
    gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

    # gauss blur
    img_blur = cv2.GaussianBlur(gray, (blur, blur), 0)

    # threshold
    ret, img_threshold = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(ret)

    # open operation
    kernel = np.ones((2, 2), np.uint8)
    img_erode = cv2.erode(img_threshold, kernel, iterations=1)
    img = cv2.dilate(img_erode, kernel, iterations=1)

    # invert
    img = ~img

    # findcontours
    all_contours, nada = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    outer_contours, nada = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # don't find inner contour how to optimize
    if len(all_contours) == len(outer_contours):
        print('first dont find inner contour')

        # CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # CLAHE = CLAHE.apply(img_blur)

        # change ret how to optimize
        ret = ret + (255 - ret) * 0.8
        # threshold
        ret, img_threshold = cv2.threshold(img_blur, ret, 255, cv2.THRESH_BINARY)

        # open operation
        kernel = np.ones((2, 2), np.uint8)
        img_erode = cv2.erode(img_threshold, kernel, iterations=1)
        img = cv2.dilate(img_erode, kernel, iterations=1)

        # invert
        img = ~img

        # findcontours
        all_contours, nada = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        outer_contours, nada = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # get inner contours
    # inner_contours = all_contours
    # for i in range(len(outer_contours)):
    #     outer_size = len(outer_contours[i])
    #     for j in range(len(all_contours)):
    #         all_size = len(all_contours[j])
    #         if (outer_size == all_size):
    #             print("inner_contours:",inner_contours)
    #             print("contours:",all_contours[j])
    #             inner_contours.remove(all_contours[j])
    #             break
    inner_contours = [x for x in all_contours if
                      len(outer_contours) == len([y for y in outer_contours if not np.array((y == x)).all()])]
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

    frame = img_input.copy()
    # 获取外轮廓外接矩形
    try:
        x1, y1, w1, h1 = cv2.boundingRect(max_outer_contour)
        # plot outer contours
        cv2.drawContours(frame, [max_outer_contour], 0, (0, 0, 255), 1)
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
        # get coordinate
        outer_list = [[x1, y1], [x1 + w1, y1], [x1, y1 + h1], [x1 + w1, y1 + h1]]
        # print(outer_list)
    except:
        # 如果找不到外轮廓
        print("No outer contour!Use input img!")
        # 采用原图各点缩进10个像素的方法
        x = img_input.shape[0]
        y = img_input.shape[1]
        outer_list = [[10, 10], [x - 10, 10], [10, y - 10], [x - 10, y - 10]]

    # 尝试获取内轮廓外接矩形
    try:
        x2, y2, w2, h2 = cv2.boundingRect(max_inner_contour)
        # 如果内外轮廓相近，则仍然选择用0.5的方法
        if (w2 / w1 > 0.8 or h2 / h1 > 0.8):
            print("Error inner contour!Multiply by parameter 0.5!")
            raise
        # plot inner contours
        cv2.drawContours(frame, [max_inner_contour], 0, (0, 0, 255), 1)
        cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)
        # get coordinate
        inner_list = [[x2, y2], [x2 + w2, y2], [x2, y2 + h2], [x2 + w2, y2 + h2]]
    except:
        # 如果内轮廓图为空，则提示没有内轮廓
        print("No inner contour!Multiply by parameter 0.5!")
        # 采取外轮廓点乘以参数的办法
        parameter = 0.5
        inner_list = [[int(x1 + w1 * (1 - parameter) / 2), int(y1 + h1 * (1 - parameter) / 2)],
                      [int(x1 - w1 * (-1 - parameter) / 2), int(y1 + h1 * (1 - parameter) / 2)],
                      [int(x1 + w1 * (1 - parameter) / 2), int(y1 - h1 * (-1 - parameter) / 2)],
                      [int(x1 - w1 * (-1 - parameter) / 2), int(y1 - h1 * (-1 - parameter) / 2)]]
        print((x1 + w1 * (1 - parameter) / 2, y1 + h1 * (1 - parameter) / 2),
              (x1 - w1 * (-1 - parameter) / 2, y1 - h1 * (-1 - parameter) / 2))
        cv2.rectangle(frame, (int(x1 + w1 * (1 - parameter) / 2), int(y1 + h1 * (1 - parameter) / 1)),
                      (int(x1 - w1 * (-1 - parameter) / 2), int(y1 - h1 * (-1 - parameter) / 2)), (0, 255, 0), 1)

    # show img
    cv2.imshow('result1', img)
    cv2.imshow('result', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # print(outer_list)
    # print(outer_list[0])
    # print(inner_list)

    # return like this [top_left, top_right, bottom_left, bottom_right]*2
    return outer_list, inner_list



img = cv2.imread('screw1.png')
find_Screw_contours(img)
cv2.waitKey(0)
cv2.destroyAllWindows()