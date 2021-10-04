import cv2
import numpy as np
import time

def max_filtering(N, I_temp):
    wall = np.full((I_temp.shape[0]+(N//2)*2, I_temp.shape[1]+(N//2)*2), -1)
    wall[(N//2):wall.shape[0]-(N//2), (N//2):wall.shape[1]-(N//2)] = I_temp.copy()
    temp = np.full((I_temp.shape[0]+(N//2)*2, I_temp.shape[1]+(N//2)*2), -1)
    for y in range(0,wall.shape[0]):
        for x in range(0,wall.shape[1]):
            if wall[y,x]!=-1:
                window = wall[y-(N//2):y+(N//2)+1,x-(N//2):x+(N//2)+1]
                num = np.amax(window)
                temp[y,x] = num
    A = temp[(N//2):wall.shape[0]-(N//2), (N//2):wall.shape[1]-(N//2)].copy()
    return A

def min_filtering(N, A):
    wall_min = np.full((A.shape[0]+(N//2)*2, A.shape[1]+(N//2)*2), 300)
    wall_min[(N//2):wall_min.shape[0]-(N//2), (N//2):wall_min.shape[1]-(N//2)] = A.copy()
    temp_min = np.full((A.shape[0]+(N//2)*2, A.shape[1]+(N//2)*2), 300)
    for y in range(0,wall_min.shape[0]):
        for x in range(0,wall_min.shape[1]):
            if wall_min[y,x]!=300:
                window_min = wall_min[y-(N//2):y+(N//2)+1,x-(N//2):x+(N//2)+1]
                num_min = np.amin(window_min)
                temp_min[y,x] = num_min
    B = temp_min[(N//2):wall_min.shape[0]-(N//2), (N//2):wall_min.shape[1]-(N//2)].copy()
    return B

def background_subtraction(I, B):
    O = I - B
    norm_img = cv2.normalize(O, None, 0,255, norm_type=cv2.NORM_MINMAX)
    return norm_img

def min_max_filtering(M, N, I):
    if M == 0:
        #max_filtering
        A = max_filtering(N, I)
        #min_filtering
        B = min_filtering(N, A)
        #subtraction
        normalised_img = background_subtraction(I, B)
        normalised_img =np.array(normalised_img,dtype=np.uint8)
    elif M == 1:
        #min_filtering
        A = min_filtering(N, I)
        #max_filtering
        B = max_filtering(N, A)
        #subtraction
        normalised_img = background_subtraction(I, B)
        normalised_img = np.array(normalised_img, dtype=np.uint8)
    return normalised_img

# get two points distance
def distance(A,B):
    return (A[0]-B[0]) **2 + (A[1]-B[1]) **2

def get_screw_length(img_input,blur,threshold=0):
    # gray
    gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

    # gauss blur
    img_blur = cv2.GaussianBlur(gray, (blur, blur), 0)

    # threshold
    ret,img_threshold = cv2.threshold(img_blur, threshold, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
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

    cv2.imshow('result1', img)
    cv2.imshow('result', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_screw_radius(img_input,blur,threshold=0):
    # gray
    gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

    # gauss blur
    img_blur = cv2.GaussianBlur(gray, (blur, blur), 0)

    # threshold,use Otsu algorithm
    ret, img_threshold = cv2.threshold(img_blur, threshold, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

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
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # get centre of mass
    centre_point = [cx,cy]

    # get closest point
    closest_distance = 100000
    for point in max_contour:
        if distance(point[0],centre_point)<closest_distance:
            closest_distance = distance(point[0],centre_point)
            closest_point = [point[0][0],point[0][1]]

    #  get min Rect
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # make contours frame
    frame = img_input.copy()

    # plot contours
    cv2.drawContours(frame, [max_contour], 0, (0, 0, 255), 2)
    cv2.line(frame, box[0], box[1], (0, 255, 0), 2)
    cv2.line(frame, box[1], box[2], (0, 255, 0), 2)
    cv2.line(frame, box[2], box[3], (0, 255, 0), 2)
    cv2.line(frame, box[3], box[0], (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 2, (0, 255, 0), 2)
    cv2.line(frame,centre_point,closest_point,(0, 255, 0), 2)


    cv2.imshow('result1', img)
    cv2.imshow('result',frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# start = time.time()
img2 = cv2.imread('screw2.png')
# get_screw_length(img2,5)
get_screw_radius(img2,5)
# end = time.time()
# print('time :',end-start)
