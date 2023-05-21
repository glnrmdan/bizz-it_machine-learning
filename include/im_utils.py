import matplotlib.pyplot as plt
import cv2 as cv
import os
import numpy as np
from PIL import Image

ROOT_PATH = os.getcwd()
TRAIN_DATA_PATH = os.path.join(ROOT_PATH, 'data/Logos/train')
# TRAIN_DATA_TARGET_PATH = os.path.join(ROOT_PATH, 'data/bizit-dev_train-data')

# train_image_demo = os.listdir(TRAIN_DATA_PATH)[:20]

def into_jpg_format(im, filename, output_path=ROOT_PATH):
    pil_image = Image.fromarray(im)
    rgb_image = pil_image.convert('RGB')
    rgb_image.save(os.path.join(output_path, f'{filename}'))


def resize_im(im_path):
        im = Image.open(im_path)
        im = im.resize((320, 320))
        return np.asarray(im)

def plot_image_to_figure(images, rows, columns):
    if len(images) > (rows*columns):
        raise ValueError('Number of images is beyond rows and columns')

    fig = plt.figure(figsize=(rows, columns))
    rows, columns = (rows, columns)

    for i, image in enumerate(images):
        _, file_format = os.path.splitext(image)
        if file_format == '.xml':
            continue
        cv_image = cv.imread(os.path.join(TRAIN_DATA_PATH, image))
        rgb_image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.title(rgb_image.shape)

    plt.show()

def im_rotate(im_path, theta):
    im = Image.open(im_path)
    return np.asarray(im.rotate(theta))

if __name__ == '__main__':
    im = im_rotate('/home/irizqy/ml_ws/bangkit-ws/data/bizit-dev_test-data/0_20230510_162226.jpg', -90)
    Image.fromarray(im).show()