# import os
from PIL import Image
# from lib.utils import combine_test_results
import lib.myMetrics as metrics
import lib.myAverage as average
import lib.filters as filter

# folder = "assets/test"
# essa = 'assets/test/essa.png'
# soh = 'assets/test/sohaib.jpg'


def process_every_pixel(img, func, ave_green, ave_brightness, factor):
    width, height = img.size
    modified_image = Image.new("RGB", (width, height))
    # Iterate over each pixel in the original image
    for y in range(height):
        for x in range(width):
            # Get the RGB values of the current pixel
            r, g, b = img.getpixel((x, y))
            # Calculate the new value for the pixel
            new_pixel_value = func(
                x, y, width, height, r, g, b, img, ave_green, ave_brightness, factor
            )
            # place in image
            modified_image.putpixel((x, y), new_pixel_value)
    return modified_image


def model1(img):
    img = filter.contrast(img, 2)
    # while (metrics.blur(img) < 40):
    #     img = filter.sharpen(img, 1)
    ave_green = average.green(img)
    ave_brightness = average.brightness(img)
    img = process_every_pixel(img, filter.green, ave_green, ave_brightness, 1.25)
    # remove bright
    ave_brightness = average.brightness(img)
    img = process_every_pixel(
        img, filter.for_brightness, ave_green, ave_brightness, 1.01
    )
    ave_green = average.green(img)
    ave_brightness = average.brightness(img)
    img = process_every_pixel(img, filter.green, ave_green, ave_brightness, 1.4)
    # img = filter.sharpen(img, 1)
    ave_brightness = average.brightness(img)
    img = process_every_pixel(
        img, filter.for_brightness, ave_green, ave_brightness, 1.01
    )
    return img


def process_img(img_path):
    # Open the image
    img = Image.open(img_path).convert("RGB")
    img= model1(img)
    # sharpen with auto stop on blur correction
        # while (metrics.blur(img) < 100):
        #     img = filter.sharpen(img, 1)
        # # remove bright
        # ave_brightness = average.brightness(img)
        # img = process_every_pixel(
        #     img, filter.for_brightness, 0, ave_brightness, 1.005
        # )
        # # sharpen with auto stop on blur correction
        # # while (metrics.blur(img) < 150):
        # img = filter.sharpen(img, 1)
        # # remove bright
        # ave_brightness = average.brightness(img)
        # img = process_every_pixel(
        #     img, filter.for_brightness, 0, ave_brightness, 1.005
        # )
        # # remove green
        # ave_green = average.green(img)
        # img = process_every_pixel(img, filter.green, ave_green, 0, 1)
        # img = filter.sharpen(img, 1)
    # ave_brightness = average.brightness(img)
    # img = process_every_pixel(
    #     img, filter.for_brightness, 0, ave_brightness, 1.005
    # )
    # img = process_every_pixel(
    #     img, filter.for_brightness, 0, ave_brightness, 1.005
    # )
    # ave_green = average.green(img)
    # img = process_every_pixel(img, filter.green, ave_green, 0, 1.15)
    
    return img


# def process():
#     return model1(img)
#     # for file_name in os.listdir(folder):
    #     file_path = os.path.join(folder, file_name)
    #     processed = process_img(file_path)
    #     processed.save(folder+"/../results/"+file_name)
    # combine_test_results()


if __name__ == "__main__":
    main()
