from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from scipy.ndimage import convolve
import lib.myMetrics as metrics
# import myAverage as average


def green(x, y, width, height, r, g, b, img, ave_green, ave_brightness, factor):
    # if (metrics.green(r, g, b)*factor) > ave_green:
    if (g*factor) > ave_green and metrics.brightness(r, g, b) < ave_brightness:
        return 255, 255, 255
    return r, g, b


def for_brightness(x, y, width, height, r, g, b, img, ave_green, ave_brightness, factor):
    if metrics.brightness(r, g, b) * factor > ave_brightness:
        return 255, 255, 255
    return r, g, b


def for_dark_green(x, y, width, height, r, g, b, img, ave_green, ave_brightness, factor):
    if metrics.dark_green(r, g, b) * factor > ave_brightness:
        return 255, 255, 255
    return r, g, b


def saturation_up(img, factor):
    filter = ImageEnhance.Color(img)
    return filter.enhance(factor)


def contrast(img, factor):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def white_point_up(img, factor):
    img_arr = np.array(img)
    img = np.clip(img_arr + factor, 0, 255)
    return Image.fromarray(img)


def sharpen(img, factor):
    while factor != 0:
        img = img.filter(ImageFilter.SHARPEN)
        factor -= 1
    return img
    # # Convert the image to grayscale
    # grayscale_image = img.convert("L")

    # # Convert the grayscale image to a numpy array
    # grayscale_array = np.array(grayscale_image)

    # # Define the sharpening kernel
    # kernel = np.array([[-2, -2, -2],
    #                    [-2, 32, -2],
    #                    [-2, -2, -2]])

    # # Apply the kernel to the grayscale array
    # sharpened_array = convolve(grayscale_array, kernel)

    # # Convert the sharpened array back to PIL image
    # return Image.fromarray(sharpened_array)
