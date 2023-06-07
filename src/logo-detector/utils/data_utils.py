import cv2 as cv
import numpy as np
import os

DATASET_PATH = '/home/irizqy/ml_ws/bangkit-ws/data/bizz.it-sim_dataset'

im_path_arr = os.listdir(DATASET_PATH)
im_classes_arr = []

for file in im_path_arr:
    im_classes_arr.append(file.split('_')[0])

im_classes_arr = np.asarray(im_classes_arr)

classes = np.unique(im_classes_arr)

dict_keys = {val:key for key, val in enumerate(classes.flatten())}

grouped_im_path = [np.where(im_classes_arr == cls)[0] for cls in classes]

def make_train_test_pairs(ims_arr, percentage):
    train_pairs = []
    train_labels = []
    test_pairs = []
    test_labels = []

    for file in ims_arr:
        file_cls = file.split('_')[0]
        cls = dict_keys[file_cls]

        current_im = cv.imread(os.path.join(DATASET_PATH, file))
        current_im = cv.cvtColor(current_im, cv.COLOR_BGR2RGB)
        # current_im = cv.cvtColor(current_im, cv.COLOR_BGR2GRAY)
        current_im = cv.resize(current_im, (150, 150))
        current_im = current_im / 255

        # pos_idx = np.random.choice(np.where(im_classes_arr != file_cls)[0], 1)[0]
        pos_idx = np.random.choice(grouped_im_path[cls], 1)[0]
        pos_pair_im = cv.imread(os.path.join(DATASET_PATH, im_path_arr[pos_idx]))
        pos_pair_im = cv.cvtColor(pos_pair_im, cv.COLOR_BGR2RGB)
        # pos_pair_im = cv.cvtColor(pos_pair_im, cv.COLOR_BGR2GRAY)
        pos_pair_im = cv.resize(pos_pair_im, (150, 150))
        pos_pair_im = pos_pair_im / 255

        train_pairs.append((current_im, pos_pair_im))
        train_labels.append(1)

        neg_idx = np.random.choice(np.where(im_classes_arr != file_cls)[0], 1)[0]
        neg_pair_im = cv.imread(os.path.join(DATASET_PATH, im_path_arr[neg_idx]))
        neg_pair_im = cv.cvtColor(neg_pair_im, cv.COLOR_BGR2RGB)
        # neg_pair_im = cv.cvtColor(neg_pair_im, cv.COLOR_BGR2GRAY)
        neg_pair_im = cv.resize(neg_pair_im, (150, 150))
        neg_pair_im = neg_pair_im / 255

        train_pairs.append((current_im, neg_pair_im))
        train_labels.append(0)

    arr_length = len(train_pairs)
    num_of_data = int(percentage * arr_length)

    for i in range(num_of_data):
        rand_i = np.random.randint(arr_length - 1)
        test_pairs.append(train_pairs[rand_i])
        train_pairs.pop(rand_i)
        test_labels.append(train_labels[rand_i])
        train_labels.pop(rand_i)

        arr_length -= 1

    return np.asarray(train_pairs), np.asarray(train_labels), np.asarray(test_pairs), np.asarray(test_labels)