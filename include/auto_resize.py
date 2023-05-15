import cv2 as cv
import os
import matplotlib.pyplot as plt

ROOT_PATH = os.getcwd()
TRAIN_DATA_PATH = os.path.join(ROOT_PATH, 'data/Logos')

train_image_demo = os.listdir(TRAIN_DATA_PATH)[:20]

fig = plt.figure(figsize=(4, 5))
rows, columns = (4, 5)

for i, image in enumerate(train_image_demo):
    filename, file_format = os.path.splitext(image)
    if file_format == '.xml':
        continue
    cv_image = cv.imread(os.path.join(TRAIN_DATA_PATH, image))
    rgb_image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.title('Image')

plt.show()

# count = 573

# path = os.path.join(Path(__file__).parent.resolve(), 'data')
# output_path = os.path.join(Path(__file__).parent.resolve(), 'output')
# for file in os.listdir(path):
#     try:
#         img = cv.imread(os.path.join(path, file))
#         resize_img = cv.resize(img, (416, 416))

#         fileo = f'{count}.jpg'

#         output = os.path.join(output_path, fileo)

#         # cv.imwrite(output, img)
#         count += 1
#         cv.imwrite(output, resize_img)
#     except Exception:
#         pass
