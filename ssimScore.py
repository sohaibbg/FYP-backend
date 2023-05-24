from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np


def greyscale_and_remove_whitespace(image):
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find the coordinates of the first black pixel
    left = np.argmax(np.min(grayscale_image, axis=0) < 128)
    right = grayscale_image.shape[1] - \
        np.argmax(np.flip(np.min(grayscale_image, axis=0), axis=0) < 128)
    top = np.argmax(np.min(grayscale_image, axis=1) < 128)
    bottom = grayscale_image.shape[0] - np.argmax(
        np.flip(np.min(grayscale_image, axis=1), axis=0) < 128)
    # Crop the image based on the coordinates
    cropped_image = image[top:bottom, left:right]
    return cropped_image


def ssim_index(img1, img2):
    img1 = cv2.resize(img1, (100, 100))
    img2 = cv2.resize(img2, (100, 100))

    # output displayed as a double number
    index = ssim(img1, img2, channel_axis=-1)
    return index


def main():
    img_path1 = 'assets/test/real1.png'
    img1 = cv2.imread(img_path1)
    img1 = greyscale_and_remove_whitespace(img1)

    img_path2 = 'assets/test/real2.png'
    img2 = cv2.imread(img_path2)
    img2 = greyscale_and_remove_whitespace(img2)

    print(ssim_index(img1, img2))


if __name__ == '__main__':
    main()
