import xml.etree.ElementTree as ET


def delete_strokes(svg_file_path, min_stroke_width):
    # Parse the SVG file
    tree = ET.parse(svg_file_path)
    root = tree.getroot()

    # Define the SVG namespace
    namespace = {"svg": "http://www.w3.org/2000/svg"}

    # Find all path elements with stroke width less than min_stroke_width
    path_elements = root.findall(".//svg:path", namespace)
    for path_element in path_elements:
        stroke_width = path_element.get("stroke-width")
        # print(stroke_width)
        if stroke_width is not None and float(stroke_width) < min_stroke_width:
            # Remove the path element
            root.remove(path_element)

    # Save the modified SVG file
    tree.write(svg_file_path+".non-thin.svg")


# Provide the path to the SVG file
svg_file_path = "assets/single(2).svg"

# Delete strokes with stroke width less than 3 and save the modified SVG file
delete_strokes(svg_file_path, 3.8)
