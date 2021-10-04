import cv2

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
            print(calc_diff(img_res[i][j],bg_color))
            if calc_diff(img_res[i][j],bg_color) < threshold:
                # img_res[i][j]为背景，将其颜色设为白色，且完全透明
                img_res[i][j][0] = 255
                img_res[i][j][1] = 255
                img_res[i][j][2] = 255
                img_res[i][j][3] = 0

    cv2.imshow('res',img_res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def change_parameter():


img = cv2.imread('red_bg.jpg')
remove_bg(img)
