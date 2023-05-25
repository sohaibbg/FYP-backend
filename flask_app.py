from flask import Flask, request, jsonify, send_file
from ssimScore import ssim_index
from normalize_nic import orient, crop
from orient_nic import orient_nic_img
from io import BytesIO
from base64 import encodebytes
import numpy as np
import cv2
import os
from Eval.single_eval import img_to_mask, save_mask
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/extractMask', methods=['POST'])
def extract():
    # receive img into string data
    img = request.files['img']
    img_extension = os.path.splitext(img.filename)[1]
    img_str = img.read()
    # convert string data to numpy array
    file_bytes = np.frombuffer(img_str, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    """Reorient the image"""
    img = orient_nic_img(img)

    """Cropout the Signature"""
    img = crop(img)

    # process img
    mask = img_to_mask(img)

    mask_path = os.path.join('assets', 'masks', 'mask.png')
    # mask_path = 'assets\masks\mask.png'
    save_mask(mask, mask_path)

    image_path = mask_path
    # Read the image
    img = cv2.imread(image_path)
    # Encode the image
    img_encoded = cv2.imencode('.png', img)[1].tobytes()
    # Convert to a file-like object
    img_io = BytesIO(img_encoded)
    # Return as a response
    return send_file(img_io, mimetype='image/png')


if __name__ == '__main__':
    """Essa App"""
    os.chdir('~/FYP-backend')
    app.run(debug=True)
