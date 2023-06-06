import numpy as np
from PIL import Image
import os

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
        Image.fromarray(cropped_im).save(f'{save_path}/cropped-logo{i}.jpg')

    return cropped_ims


if __name__ == '__main__':
    pass