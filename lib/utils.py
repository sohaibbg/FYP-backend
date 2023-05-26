import os
from PIL import Image

def combine_test_results():
    test_folder="assets/test"
    results_folder="assets/results"
    comparison_folder="assets/comparison"
    # Iterate over the images in the test folder
    for test_file_name in os.listdir(test_folder):
        test_image_path = os.path.join(test_folder, test_file_name)

        # Check if the corresponding image exists in the results folder
        result_image_path = os.path.join(results_folder, test_file_name)
        if not os.path.isfile(result_image_path):
            continue
        # Open the test and result images
        test_image = Image.open(test_image_path)
        result_image = Image.open(result_image_path)

        # Resize the images to have the same height
        height = min(test_image.size[1], result_image.size[1])
        test_image = test_image.resize(
            (int(test_image.size[0] * height / test_image.size[1]), height))
        result_image = result_image.resize(
            (int(result_image.size[0] * height / result_image.size[1]), height))

        # Create a new image with the combined width
        combined_width = test_image.size[0] + result_image.size[0]
        combined_image = Image.new("RGB", (combined_width, height))

        # Paste the test and result images side by side
        combined_image.paste(test_image, (0, 0))
        combined_image.paste(result_image, (test_image.size[0], 0))

        # Save the combined image to the comparison folder
        comparison_image_path = os.path.join(comparison_folder, test_file_name)
        combined_image.save(comparison_image_path)
