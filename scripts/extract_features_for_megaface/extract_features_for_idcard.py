#! /usr/bin/env python
import sys
from extract_features import extract_features
import argparse

def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-json', type=str, help='config file')
    parser.add_argument('--image-dir', type=str,
                        help='image root dir if image list contains relative paths')
    parser.add_argument('--save-dir', type=str, default='./rlt-features',
                        help='where to save the features')
    parser.add_argument('--image-list-file', type=str, help='image list file')
    parser.add_argument('--gpu', type=int, help='', default=0)
    return parser.parse_args(argv)

def main(args):
    print('===> args:\n', args)
    config_json = args.config_json
    image_dir = args.image_dir
    save_dir = args.save_dir
    image_list_file = args.image_list_file
    gpu_id = args.gpu
    extract_features(config_json, save_dir, image_list_file, image_dir, gpu_id=gpu_id)

if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
