import cv2 as cv
import numpy as np


def adjust_im(im_path: str, size: tuple, normalize:bool = True, use_gray:bool = False):
    im = cv.imread(im_path)
    im = cv.resize(im, size)
    
    if use_gray:
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    else:
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

    if normalize:
        im = im / 255

    return np.expand_dims(im, 0)