import os
import numpy as np
import argparse
from im_utils import into_jpg_format, resize_im

# Initiate argument parser
parser = argparse.ArgumentParser(
    description='Utility script to get random shuffle images'
)
parser.add_argument('-x',
                    '--image_input_dir',
                    help='Path to the image folder to be process',
                    type=str
                    )
parser.add_argument('-o',
                    '--output_path',
                    help='Path to the folder where the input image files are stored',
                    type=str
                    )
parser.add_argument('-n',
                    '--num_train_data',
                    help='Number of train data to be taken for training',
                    type=int
                    )
parser.add_argument('-ns',
                    '--num_subdir',
                    help='Number of sub-directory that the train data will be taken (if sub-directory exist)',
                    type=int, default=1
                    )
args = parser.parse_args()

ROOT_PATH = os.getcwd()


def get_subset_data(im_path, n_subset, n_subdir):
    # iterate directory
    if n_subdir > 1:
        for dir in os.listdir(im_path):
            # check if it's containing sub-dir
            if os.path.isdir(os.path.join(im_path, dir)):
                num_im_taken = int(n_subset / (len(os.listdir(im_path))*n_subdir))
                subdir_contents = np.random.choice(os.listdir(os.path.join(im_path, dir)), n_subdir)
                # iterate sub-directory
                for sub_dir in subdir_contents:
                    # get random samples of image in sub-directory
                    im_contents = np.random.choice(os.listdir(os.path.join(im_path, dir, sub_dir)), num_im_taken)
                    for im in im_contents:
                        _im = resize_im(os.path.join(im_path, dir, sub_dir, im))
                        into_jpg_format(_im, f'{sub_dir}-{im}', args.output_path)

    else:
        im_contents = np.random.choice(os.listdir(im_path), n_subset)
        print(len(im_contents))
        for i, im in enumerate(im_contents):
            _im = resize_im(os.path.join(im_path, im))
            into_jpg_format(_im, f'{i}_{im}', args.output_path)


def main():
    get_subset_data(args.image_input_dir, args.num_train_data, args.num_subdir)


if __name__ == '__main__':
    print('Running...') 
    main()