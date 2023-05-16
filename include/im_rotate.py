import argparse
import os
from PIL import Image
from im_utils import im_rotate, into_jpg_format

parser = argparse.ArgumentParser(
    description='Rotate image scripts (can exclude some files)'
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
parser.add_argument('-t',
                    '--theta',
                    help='Rotation angle applied on images',
                    type=int
                    )
parser.add_argument('-e',
                    '--excluded_file',
                    help='Files in the directory to be excluded',
                    type=str, default=None
                    )


args = parser.parse_args()

def rotate_images(im_dir, ex_im, output_path, theta):
    for im in os.listdir(im_dir):
        if im in ex_im:
            continue
        _im = im_rotate(os.path.join(im_dir, im), -theta)
        into_jpg_format(_im, im, output_path)


if __name__ == '__main__':
    excluded_files = args.excluded_file.split(',')
    rotate_images(args.image_input_dir, excluded_files, args.output_path, args.theta)