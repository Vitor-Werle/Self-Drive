import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_algorithm(image):
    # Convert the imagem to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply the Gaussian Blur 
    blur = cv2.GaussianBlur(gray, [5, 5], 0)
    # Detect the edges
    canny = cv2.Canny(blur, 50, 150) # parameters: (img, low, high)
    return canny

def region_of_interest(image):
    # returns the image dimensions 
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    # create a black imagem from my image
    mask = np.zeros_like(image)
    # Prenche a região do polígno (0 - dentr0, 255 - fora)
    cv2.fillPoly(mask, polygons, 255)
    #O cv2.bitwise_and() faz uma operação AND bit a bit entre os pixels das duas imagens
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

#It returns a NumPy ndarray representing the pixel values of the image
image = cv2.imread('../img/test_image.jpg')

# Create a copy of image
lane_image = np.copy(image)

#call the function with canny algorithm
canny = canny_algorithm(lane_image)

cropped_img = region_of_interest(canny)
cv2.imshow('result', cropped_img)
cv2.waitKey(0)
"""
# Visualize the X and Y to get the coordinates
plt.imshow(canny)
plt.show()
"""