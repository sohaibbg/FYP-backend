import cv2
import numpy as np


def compare_rotations(image1, image2, similarity_function):

    # Get the center of the image2
    center = tuple(np.array(image2.shape[1::-1]) / 2)

    max_similarity = float('-inf')

    for angle in [-45, 45]:
        # Rotate the image
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image2, rotation_matrix, image2.shape[1::-1], flags=cv2.INTER_LINEAR)

        # Compare to the original image
        similarity = similarity_function(image1, rotated)

        # Update max_similarity
        max_similarity = max(max_similarity, similarity)

    return max_similarity
