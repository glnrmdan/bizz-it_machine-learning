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

    return cropped_ims


def get_franchise_data(uri=API_URL):
    franchises = json.loads(requests.get(uri).content.decode())['data']
    _data = {}

    for franchise in franchises:
        _data['id'] = franchise['id']
        _data['logo'] = franchise['logo']

    return _data



def check_franchise_availability(cropped_ims, model):
    # get all franchise data
    franchises = get_franchise_data()
