from flask import Flask, request
from PIL import Image
from tensorflow.saved_model import load
import json
import numpy as np
import base64
import io

# logo detector model path
LD_MODEL_PATH = '/home/irizqy/ml_ws/bangkit-ws/src/logo-detector/efficientnet512_ft/saved_model'
LD_LABEL_PATH = '/home/irizqy/ml_ws/bangkit-ws/data/label_map.pbtxt'

FR_MODEL_PATH = None

ld_detector = load(LD_MODEL_PATH)

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
    # im_bytes = base64.decodebytes(bytes(req.get('data'), 'utf-8'))
    # im_arr = np.frombuffer(im_bytes)

    response_json = {
        "data": 'success' 
    }

    print(im_arr)

    return json.dumps(response_json)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3000)