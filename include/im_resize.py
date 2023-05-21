from im_utils import resize_im, into_jpg_format
import argparse
import os

parser = argparse.ArgumentParser(
    description='Resize all image in the input folder')
parser.add_argument('-x',
                    '--input_dir',
                    help='Path to the image folder to be process',
                    type=str
                    )
parser.add_argument('-o',
                    '--output_dir',
                    help='Path to the image output folder',
                    type=str, default=None
                    )

args = parser.parse_args()

if __name__ == '__main__':
    if args.output_dir == None:
        args.output_dir = args.input_dir
    for file in os.listdir(args.input_dir):
        im = resize_im(os.path.join(args.input_dir, file))
        into_jpg_format(im, file, args.output_dir)