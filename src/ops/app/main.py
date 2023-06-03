from fastapi import FastAPI
from PIL import Image
from app.model import ImageParser
import tensorflow as tf
import numpy as np
import base64
import io
import os


# logo detector model path
LD_MODEL_PATH = './app/model/model_v1/saved_model/'
LD_LABEL_PATH = '/home/irizqy/ml_ws/bangkit-ws/data/label_map.pbtxt'

FR_MODEL_PATH = None

ld_detector = tf.saved_model.load(LD_MODEL_PATH)


app = FastAPI()

@app.get('/')
def index():
    return {'greetings': 'Hello World!'}


@app.post('/ld_predict')
async def detect_logo(im: ImageParser):
    img_bytes = base64.b64decode(im.image.encode('utf-8'))
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
        "status": 1 
    }
    return response_json
