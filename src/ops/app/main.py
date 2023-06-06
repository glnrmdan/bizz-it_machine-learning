from fastapi import FastAPI
from PIL import Image
from app.model import ImageParser
from app.utils import cropped_detected_im
import tensorflow as tf
import numpy as np
import base64
import io
import json


# logo detector model path
LD_MODEL_PATH = './app/model/model_v1/saved_model/'
LD_LABEL_PATH = '/home/irizqy/ml_ws/bangkit-ws/data/label_map.pbtxt'

FR_MODEL_PATH = None

ld_detector = tf.saved_model.load(LD_MODEL_PATH)


app = FastAPI()

@app.get('/')
def index():
    return {'greetings': 'Hello World!'}


@app.post('/ld_predict', status_code=200)
async def detect_logo(im: ImageParser):

    response_json = {
            'status': 400,
            'message': 'Incorrect data format'
        }
    
    try:
        img_bytes = base64.b64decode(im.image.encode('utf-8'))
        im = Image.open(io.BytesIO(img_bytes))
        im_arr = np.asarray(im)

        # take only RGB values from RGBA images
        if im_arr.shape[2] == 4:
            im_arr = im_arr[:,:,:3]

        input_tensor = tf.convert_to_tensor(im_arr)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = ld_detector(input_tensor)

        cropped_ims = cropped_detected_im(detections, im_arr)
    
        response_json = {
            'status': 1,
            'length': len(cropped_ims)
        }
    except Exception:
        pass

    return json.dumps(response_json)
