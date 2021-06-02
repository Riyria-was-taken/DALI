#!/usr/bin/env python

import argparse
import os

cwd = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('--tag', default="dali-effdet", help='Image tag')
parser.add_argument('--name', default="dali-effdet-gentf", help="Container name")
parser.add_argument('--coco', default=f"{cwd}/mnt/coco_dir", help="Mount path for dataset")
parser.add_argument('--tfrecord_dir', default=f"{cwd}/mnt/tfrecord_dir", help="Output directory for tfrecord files")
args = parser.parse_args()


os.system(f"docker run --gpus all --rm -v {args.tfrecord_dir}:/tfrec -v {args.coco}:/coco --name {args.name} {args.tag} bash -c \"python3 dataset/create_tfrecord_indexes.py\"")
