import lib.myMetrics as myMetrics


def green(img):
    # Get the dimensions of the image
    width, height = img.size

    # Get the average green value in the image
    pixel_count = height*width
    total_light_green_index = 0
    for y in range(height):
        for x in range(width):
            # Get the RGB values of the current pixel
            r, g, b = img.getpixel((x, y))
            # total_light_green_index += myMetrics.green(r, g, b)
            total_light_green_index += g

    average_light_green_index = total_light_green_index / pixel_count
    return average_light_green_index


def brightness(img):
    # Get the dimensions of the image
    width, height = img.size

    # Get the average green value in the image
    total_b = 0.0
    pixel_count = 0

    for y in range(height):
        for x in range(width):
            # Get the RGB values of the current pixel
            r, g, b = img.getpixel((x, y))
            if (r, g, b) != (255, 255, 255):
                total_b += myMetrics.brightness(r, g, b)
                pixel_count += 1

    return total_b/pixel_count
