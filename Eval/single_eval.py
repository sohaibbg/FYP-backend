from typing import overload
import pandas as pd
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from Eval.metrics import dice_coef, dice_loss, iou
from Eval.data import create_dir
from Eval.load_data import load_data

import os

from Utils.file_path import get_directory_path
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

"""Global Variables"""
H = 1024
W = 1024


def save_mask(mask, path):
    """Getting Directory to save the mask image"""
    dir_path = get_directory_path(path)

    """Create Directory for storing files"""
    create_dir(dir_path)

    # Ensuring mask is in the range 0-255 (as uint8)
    result = (mask * 255).astype('uint8')

    # Invert the image
    mask_inv = 255 - result
    # Saving the image
    cv2.imwrite(path, mask_inv)
    # cv2.imshow('img', mask_inv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def save_before_after(image, mask, y_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 128

    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    y_pred = y_pred * 255

    cat_image = np.concatenate([image, line, mask, line, y_pred], axis=1)

    # save_image_path = f"assets/results/before_after/{name}.png"
    cv2.imwrite(save_image_path, cat_image)


def convert_4_channel_to_3(img):
    # Check if the image has an alpha channel
    if img.shape[2] == 4:
        # Split the image channels
        b, g, r, a = cv2.split(img)

        # Create a 3-channel image (RGB) by merging the first three channels
        img_3_channel = cv2.merge((b, g, r))

        return img_3_channel

    return img


def maskify(img, model):
    """Testing new Images"""
    if (img.shape[1] != 1024 or img.shape[2] != 1024):
        img = cv2.resize(img, (W, H))
    img = convert_4_channel_to_3(img)
    # cv2.imshow('h',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    x = img/255.0
    x = np.expand_dims(x, axis=0)
    """Prediction"""
    y_pred = model.predict(x)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)

    return y_pred


def init_model():
    """Seeding"""
    np.random.seed(42)
    tf.random.set_seed(42)

    """Create Directory for storing files"""
    # create_dir("assets/results/before_after")
    create_dir("Eval/assets/results/masks")

    """Loading the Model"""
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("Eval/files/save_model.h5")

    return model


def img_to_mask(img):
    """takes cv2 image or filepath, returns cv2 image"""
    model = init_model()
    if isinstance(img, str):
        img = cv2.imread(img)
    return maskify(img, model)


# reads from local path and saves img
def main():
    img_path = 'Eval/assets/test/img.jpg'
    mask = img_to_mask(img_path)
    filename = img_path.split("/")[-1]
    signature_path = f"Eval/assets/results/masks/{filename}"
    save_mask(mask, signature_path)


if __name__ == "__main__":
    main()
