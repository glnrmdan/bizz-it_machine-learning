import os
from PIL import Image
import numpy as np
import argparse

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
parser.add_argument('-n_subdir',
                    '--num_subdir',
                    help='Number of sub-directory that the train data will be taken (if sub-directory exist)',
                    type=int, default=2
                    )
args = parser.parse_args()

ROOT_PATH = os.getcwd()


def into_jpg_format(im, filename, output_path=ROOT_PATH):
    pil_image = Image.fromarray(im)
    rgb_image = pil_image.convert('RGB')
    rgb_image.save(os.path.join(output_path, f'{filename}'))


def resize_im(im_path):
    im = Image.open(im_path)
    im = im.resize((320, 320))
    return np.asarray(im)

def get_subset_data(im_path, n_subset, n_subdir=2):
    # iterate directory
    for dir in os.listdir(im_path):
        # check if it's containing sub-dir
        if os.path.isdir(os.path.join(im_path, dir)):
            num_im_taken = int(n_subset / (len(os.listdir(im_path))*n_subdir))
            subdir_contents = np.random.choice(os.listdir(os.path.join(im_path, dir)), n_subdir)
            # iterate sub-directory
            for sub_dir in subdir_contents:
                im_contents = np.random.choice(os.listdir(os.path.join(im_path, dir, sub_dir)), num_im_taken)
                for im in im_contents:
                    _im = resize_im(os.path.join(im_path, dir, sub_dir, im))
                    into_jpg_format(_im, im, args.output_path)

        # images_path = os.path.join(args.image_input_dir, dir)
        # dir_len = len(os.listdir(images_path))
        # n_rand = np.random.randint(dir_len)
        # print(n_rand)

def main():
    get_subset_data(args.image_input_dir, args.num_train_data, args.num_subdir)


if __name__ == '__main__':
    print(args) 
    main()
    # for dir in os.listdir(TRAIN_DATA_PATH):
    #     for subdir in os.listdir(os.path.join(TRAIN_DATA_PATH, dir)):
    #         for file in os.listdir(os.path.join(TRAIN_DATA_PATH, dir, subdir)):
    #             im = resize_im(os.path.join(TRAIN_DATA_PATH, dir, subdir, file))
    #             into_jpg_format(im, f'{subdir}-{file}', TRAIN_DATA_TARGET_PATH)
