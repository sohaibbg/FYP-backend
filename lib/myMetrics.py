import numpy as np
import scipy.stats as stats
import cv2


def calculate_bell_curve_position(value):
    # Define the mean and standard deviation of the bell curve
    mu = 0.5  # Mean
    sigma = 0.125  # Standard deviation

    # Calculate the PDF (probability density function) at the given value
    position = stats.norm.pdf(value, mu, sigma)

    return position


def brightness(r, g, b):
    return max(r, g, b)/255


def centrality(x_percent, y_percent):
    return calculate_bell_curve_position(x_percent) + calculate_bell_curve_position(y_percent)

def dark_green(r,g,b):
    if g >= 100 and r+g+b<=200:
        return True
    return False

def green(r, g, b):
    # Calculate the brightness value (V) using the maximum RGB component
    v = max(r, g, b) / 255.0

    # Calculate the greenness value (G) relative to the sum of RGB components
    total = r + g + b
    g_relative = g / total if total > 0 else 0

    # Calculate the light green index by combining brightness and greenness
    index = v * g_relative

    return index


def blur(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the Laplacian variance
    return cv2.Laplacian(gray, cv2.CV_64F).var()
