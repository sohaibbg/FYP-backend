import cv2
import numpy as np


def sharpen_image(img):
    # Define the sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    # Apply the kernel to the image
    sharpened = cv2.filter2D(img, -1, kernel)

    return sharpened
