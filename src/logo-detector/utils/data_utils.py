import cv2 as cv
import numpy as np
import os


def grouped_file_class(files_arr: str):
    im_classes_arr = [file.split('_')[0] for file in files_arr]

    return np.asarray(im_classes_arr)


def make_train_test_pairs(data_path: str, percentage: float, use_gray:bool = False):
    train_pairs = []
    train_labels = []
    test_pairs = []
    test_labels = []

    im_path_arr = os.listdir(data_path)
    im_classes_arr = grouped_file_class(im_path_arr)

    for file in im_path_arr:
        file_cls = file.split('_')[0]

        current_im = cv.imread(os.path.join(data_path, file))

        if use_gray:
            current_im = cv.cvtColor(current_im, cv.COLOR_BGR2GRAY)
        else:
            current_im = cv.cvtColor(current_im, cv.COLOR_BGR2RGB)

        current_im = cv.resize(current_im, (150, 150))
        current_im = current_im / 255

        pos_idx = np.random.choice(np.where(im_classes_arr == file_cls)[0], 1)[0]
        pos_pair_im = cv.imread(os.path.join(data_path, im_path_arr[pos_idx]))

        if use_gray:
            pos_pair_im = cv.cvtColor(pos_pair_im, cv.COLOR_BGR2GRAY)
        else:
            pos_pair_im = cv.cvtColor(pos_pair_im, cv.COLOR_BGR2RGB)

        pos_pair_im = cv.resize(pos_pair_im, (150, 150))
        pos_pair_im = pos_pair_im / 255

        train_pairs.append((current_im, pos_pair_im))
        train_labels.append(1)

        neg_idx = np.random.choice(np.where(im_classes_arr != file_cls)[0], 1)[0]
        neg_pair_im = cv.imread(os.path.join(data_path, im_path_arr[neg_idx]))

        if use_gray:
            neg_pair_im = cv.cvtColor(neg_pair_im, cv.COLOR_BGR2GRAY)
        else:
            neg_pair_im = cv.cvtColor(neg_pair_im, cv.COLOR_BGR2RGB)

        neg_pair_im = cv.resize(neg_pair_im, (150, 150))
        neg_pair_im = neg_pair_im / 255

        train_pairs.append((current_im, neg_pair_im))
        train_labels.append(0)

    arr_length = len(train_pairs)
    num_of_data = int(percentage * arr_length)

    for _ in range(num_of_data):
        rand_i = np.random.randint(arr_length - 1)
        test_pairs.append(train_pairs[rand_i])
        train_pairs.pop(rand_i)
        test_labels.append(train_labels[rand_i])
        train_labels.pop(rand_i)

        arr_length -= 1

    return (np.asarray(train_pairs), np.asarray(train_labels)), (np.asarray(test_pairs), np.asarray(test_labels))


if __name__ == '__main__':
    DATASET_PATH = '/home/irizqy/ml_ws/bangkit-ws/data/bizz.it-sim_dataset'

    (train_pairs, train_labels), (test_pairs, test_labels) = make_train_test_pairs(DATASET_PATH, .2)
