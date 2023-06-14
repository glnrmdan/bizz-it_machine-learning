import numpy as np
import cv2 as cv
import os
import requests
import json


API_URL = 'https://bizzit-387412.as.r.appspot.com/franchises'


def cropped_detected_im(detections, im, save_path=os.getcwd()):
    width, height, _ = im.shape
    dets_index = np.where(detections['detection_scores'][0] >= .5)[0]

    if dets_index.shape[0] == 0:
        return 0

    cropped_ims = []

    for i in dets_index:
        ymin, xmin, ymax, xmax = detections['detection_boxes'][0][i]
        (left, right, top, bottom) = (xmin*width, xmax*width, ymin*height, ymax*height)
        cropped_im = im[int(top):int(bottom), int(left):int(right)]
        cropped_ims.append(cropped_im)

    return cropped_ims[0]


def get_franchise_data(uri=API_URL):
    franchises = json.loads(requests.get(uri).content.decode())['data']
    _data = []

    for franchise in franchises:
        _data.append({'id': franchise['id'], 'logo': franchise['logo']})

    return _data


def adjust_im(im_url, size, normalize=True, use_gray=False):
    logo = requests.get(im_url['logo']).content
    jpg_as_np = np.frombuffer(logo, dtype=np.uint8)
    im = cv.imdecode(jpg_as_np, flags=1)
    im = cv.resize(im, size)
    
    if use_gray:
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    else:
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

    if normalize:
        im = im / 255

    return np.expand_dims(im, 0)


def check_franchise_availability(source_im, model):
    # get all franchise data
    preds = []
    franchises = get_franchise_data()
    source_im = cv.resize(source_im, (150, 150))
    source_im = source_im / 255
    source_im = np.expand_dims(source_im, 0)
    for franchise in franchises:
        im_target = adjust_im(franchise, (150, 150))
        pred = model.predict((source_im, im_target))[0][0]
        preds.append(pred)

    conf_pred = franchises[np.argmax(preds)].get('id')
    conf_score = preds[np.argmax(preds)]
    preds.pop(np.argmax(preds))
    other_preds = list(map(float, preds))

    return conf_pred, float(conf_score), other_preds