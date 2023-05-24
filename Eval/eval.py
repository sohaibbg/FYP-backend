import pandas as pd
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_coef, dice_loss, iou
from data import create_dir
from load_data import load_data

import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

"""Global Variables"""
H = 1024
W = 1024


def save_prediction(y_pred, path):
    # Ensuring y_pred is in the range 0-255 (as uint8)
    result = (y_pred * 255).astype('uint8')

    # Invert the image
    y_pred_inv = 255 - result
    # Saving the image
    cv2.imwrite(path, y_pred_inv)


def save_results(image, mask, y_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 128

    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    y_pred = y_pred * 255

    cat_image = np.concatenate([image, line, mask, line, y_pred], axis=1)

    cv2.imwrite(save_image_path, cat_image)


if __name__ == "__main__":
    """Seeding"""
    np.random.seed(42)
    tf.random.set_seed(42)

    """Create Directory for storing files"""
    create_dir("results")
    create_dir("assets/results/masks")

    """Loading the Model"""
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/save_model.h5")
        # model.summary()

    """Load the dataset"""
    dataset_path = "new_data"
    valid_path = os.path.join(dataset_path, "cnic_scan")
    test_x, test_y = load_data(valid_path)
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """Evaluation and Prediction"""
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """Extract the Name"""
        name = x.split("/")[-1].split(".")[0]
        # print(name)

        """Reading the Image and Mask"""
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        """Testing new Images"""
        if (image.shape[1] != 1024 or image.shape[2] != 1024):
            # print(f"Shape of the Image : {image.shape}")
            image = cv2.resize(image, (W, H))
            mask = cv2.resize(mask, (H, W))
        x = image/255.0
        x = np.expand_dims(x, axis=0)

        """Prediction"""
        y_pred = model.predict(x)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)

        """Saving the prediction"""
        save_image_path = f"results/{name}.png"
        save_signature_path = f"assets/results/masks/{name}.png"
        save_results(image, mask, y_pred, save_image_path)
        save_prediction(y_pred, save_signature_path)

        # break
