from flask import Flask, request
from PIL import Image
import tensorflow as tf
import json
import numpy as np
import base64
import io
import os
from os.path import join


# logo detector model path
LD_MODEL_PATH = '/home/irizqy/ml_ws/bangkit-ws/src/logo-detector/efficientnet512_ft/saved_model'
LD_LABEL_PATH = '/home/irizqy/ml_ws/bangkit-ws/data/label_map.pbtxt'

FR_MODEL_PATH = None

ld_detector = tf.saved_model.load(LD_MODEL_PATH)

app = Flask(__name__)

@app.route('/')
def root_response():
    return json.dumps({'greeting': 'Hello World!'})

@app.route('/ld_predict', methods=['POST'])
def detect_logo():
    req = request.json

    img_bytes = base64.b64decode(req.get('data').encode('utf-8'))
    im = Image.open(io.BytesIO(img_bytes))
    im_arr = np.asarray(im)

    input_tensor = tf.convert_to_tensor(im_arr)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = ld_detector(input_tensor)

    width, height, _ = im_arr.shape
    print(detections['detection_boxes'][0][0])
    ymin, xmin, ymax, xmax = detections['detection_boxes'][0][0]
    (left, right, top, bottom) = (xmin*width, xmax*width, ymin*height, ymax*height)
    cropped_im = im_arr[int(top):int(bottom), int(left):int(right)]
    Image.fromarray(cropped_im).save(os.path.join(os.getcwd(), 'cropped-logo.jpg'))

    response_json = {
        "data": 'success' 
    }

    return json.dumps(response_json)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3000)