from io import BytesIO
import cairosvg
from PIL import Image
import os


def main():
    folder = 'assets/'
    for name in os.listdir(folder):
        if name[-4:]!=".svg":
            continue
        # Provide the path to the SVG file
        svg_file_path = folder+name

        # Convert SVG to PNG using cairosvg
        png_data = BytesIO(cairosvg.svg2png(url=svg_file_path))

        # Create a PIL image from the PNG data
        image = Image.open(png_data)

        # Convert PIL image to WebP format and save it
        webp_file_path = folder+name+'.webp'
        image.save(webp_file_path, "WebP")


if __name__ == '__main__':
    main()
