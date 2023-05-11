import cv2 as cv
from pathlib import Path
import os

count = 573

path = os.path.join(Path(__file__).parent.resolve(), 'data')
output_path = os.path.join(Path(__file__).parent.resolve(), 'output')
for file in os.listdir(path):
    try:
        img = cv.imread(os.path.join(path, file))
        resize_img = cv.resize(img, (416, 416))

        fileo = f'{count}.jpg'

        output = os.path.join(output_path, fileo)

        # cv.imwrite(output, img)
        count += 1
        cv.imwrite(output, resize_img)
    except Exception:
        pass
