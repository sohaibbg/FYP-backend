from flask import Flask, request, jsonify, send_file
from ssimScore import ssim_index
from normalize_nic import orient, crop
from orient_nic import orient_nic_img
from io import BytesIO
from base64 import encodebytes
import numpy as np
import cv2
import os
from Eval.single_eval import img_to_mask
app = Flask(__name__)


@app.route('/ssim-index', methods=['POST'])
def post_ssim_index():
    # receive img into string data
    img1 = request.files['img1']
    img_extension = os.path.splitext(img1.filename)[1]
    img_str = img1.read()
    # convert string data to numpy array
    file_bytes = np.frombuffer(img_str, np.uint8)
    # convert numpy array to image
    img1 = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    # receive img into string data
    img2 = request.files['img2']
    img_extension = os.path.splitext(img2.filename)[1]
    img_str = img2.read()
    # convert string data to numpy array
    file_bytes = np.frombuffer(img_str, np.uint8)
    # convert numpy array to image
    img2 = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    # process img
    # return 'x'
    index = ssim_index(img1, img2)
    return {'response': str(index)}


@app.route('/orient-and-crop', methods=['POST'])
def orient_crop():
    # receive img into string data
    img = request.files['img']
    img_extension = os.path.splitext(img.filename)[1]
    img_str = img.read()
    # convert string data to numpy array
    file_bytes = np.frombuffer(img_str, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    # process img
    oriented_img = orient(img)
    cropped_img = crop(oriented_img)

    # encode
    buffer1 = cv2.imencode(img_extension, oriented_img)[1]
    buffer1 = encodebytes(BytesIO(buffer1).getvalue()).decode('ascii')
    buffer2 = cv2.imencode(img_extension, cropped_img)[1]
    buffer2 = encodebytes(BytesIO(buffer2).getvalue()).decode('ascii')

    return jsonify({
        "oriented": buffer1,
        'cropped': buffer2
    })

# @app.route('/orient', methods=['POST'])
# def orient():
#     # receive img into string data
#     img = request.files['img']
#     img_extension = os.path.splitext(img.filename)[1]
#     img_str = img.read()
#     # convert string data to numpy array
#     file_bytes = np.frombuffer(img_str, np.uint8)
#     # convert numpy array to image
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

#     # process img
#     img = orient_nic_img(img)
#     # encode
#     buffer = cv2.imencode(img_extension, img)[1]
#     io_buf = BytesIO(buffer)

#     return send_file(io_buf, mimetype='image')


@app.route('/maskify', methods=['POST'])
def upload():
    # receive img into string data
    img = request.files['img']
    img_extension = os.path.splitext(img.filename)[1]
    img_str = img.read()
    # convert string data to numpy array
    file_bytes = np.frombuffer(img_str, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    # process img
    mask = img_to_mask(img)

    # encode
    buffer = cv2.imencode(img_extension, mask)[1]
    io_buf = BytesIO(buffer)
    io_buf.seek(0)

    return send_file(io_buf, 'image', download_name=f'mask.{img_extension}')


if __name__ == '__main__':
    app.run(debug=True)
