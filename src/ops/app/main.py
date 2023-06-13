from fastapi import FastAPI
from fastapi.responses import JSONResponse
from model import ImageParser
import functions as utils
import tensorflow as tf
import numpy as np
import base64
import tensorflow.keras.backend as K
import uvicorn
import tensorflow as tf
import cv2 as cv


# logo detector model path
LD_MODEL_PATH = './app/model/model_v1/saved_model/'
IS_MODEL_PATH = './app/model/im_similar_v1/'
FR_MODEL_PATH = None


ld_detector = tf.saved_model.load(LD_MODEL_PATH)
im_similarity = tf.keras.models.load_model(IS_MODEL_PATH)

app = FastAPI()


@app.get('/')
def index():
    return {'greetings': 'Hello World!'}


@app.post('/ld_predict')
async def detect_logo(im: ImageParser):
    try:
        img_bytes = base64.b64decode(im.image.encode('utf-8'))
        jpg_as_np = np.frombuffer(img_bytes, dtype=np.uint8)
        im_arr = cv.imdecode(jpg_as_np, flags=1)
        im_arr = cv.cvtColor(im_arr, cv.COLOR_BGR2RGB)
        im_arr = cv.resize(im_arr, (512, 512))

        input_tensor = tf.convert_to_tensor(im_arr)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = ld_detector(input_tensor)

        cropped_ims = utils.cropped_detected_im(detections, im_arr)
        utils.check_franchise_availability(cropped_ims, im_similarity)

        if not cropped_ims:
            return JSONResponse(status_code=200, content={'message': 'No Logo Found in the Image'})

        response_json = {
            'message': 'Ok',
            'data': {
                'franchise_id: ': len(cropped_ims),
                # 'confidence_score': confidence,
                # ''
            }
        }

        return JSONResponse(status_code=200, content=response_json)
    
    except Exception:
        JSONResponse(status_code=400, content={'message': 'Bad Request'})

if __name__ == '__main__':
    uvicorn.run("main:app", port=3000, log_level="info", host='0.0.0.0')