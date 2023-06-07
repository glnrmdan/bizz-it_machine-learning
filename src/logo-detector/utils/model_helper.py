import cv2 as cv
import numpy as np


def adjust_im(im_path: str, size: tuple, normalize:bool = True, use_gray:bool = False):
    im1 = cv.imread(im_path)
    im1 = cv.resize(im1, size)
    
    if use_gray:
        im1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    else:
        im1 = cv.cvtColor(im1, cv.COLOR_BGR2RGB)

    if normalize:
        im1 = im1 / 255

    return np.expand_dims(im1, 0)