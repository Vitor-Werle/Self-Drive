import cv2
import numpy as np
import matplotlib.pyplot as plt


def average_slop_intercept(image,lines):
    pass



def canny(image):
    # Aplica o filtro cinza
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def display_lines(image, lines):
    lines_img = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lines_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_img


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 255)]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img

image = cv2.imread('../img/test_image.jpg')

# cria um cópia do array da imagem
lane_img = np.copy(image)
canny_img = canny(lane_img)
cropped_img = region_of_interest(canny_img)
lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average_slop_intercept(lane_img, lines)
lines_img = display_lines(lane_img, lines)
combo_img = cv2.addWeighted(lane_img, 0.8, lines_img, 1, 1)
cv2.imshow('result', combo_img)
cv2.waitKey(0)