import cv2
import numpy as np
import os


def display_img(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def crop(target_img):
    roi = [(900, 600), (1200, 735), 'box', 'signature']
    target_img = target_img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
    return target_img


def orient(target_img):
    # params
    per = 25
    roi = [(900, 600), (1200, 735), 'box', 'signature']

    # # open target img
    # img = cv2.imread(img_path)
    # open calibration img
    calibration_img = cv2.imread('assets/calibrated.jpeg')

    # calibration object
    orb = cv2.ORB_create(5000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2)
    # initialize calibrator
    kp1, des1 = orb.detectAndCompute(calibration_img, None)
    # initialize calibratee
    kp2, des2 = orb.detectAndCompute(target_img, None)
    matches = bf.match(des2, des1)

    """Prints the matches in the console"""
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:int(len(matches)*(per/100))]
    # imgMatch = cv2.drawMatches(
    #     target_img, kp2, calibration_img, kp1, good[:100], None, flags=2)
    srcPoints = np.float32(
        [kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32(
        [kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # correct orientation
    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    target_img = cv2.warpPerspective(
        target_img, M, (calibration_img.shape[1], calibration_img.shape[0]))
    return target_img


def main():
    path = '/home/sohaib/Downloads/Photos-001/'
    for name in os.listdir(path):
        x = cv2.imread(path+name)
        img = crop(orient(x))
        cv2.imwrite("assets/correct/"+name, img)
        # display_img(img)


if __name__ == '__main__':
    main()
