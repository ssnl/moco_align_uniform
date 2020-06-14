# Copyright (c) 2020 Tongzhou Wang
import os
import re
import errno
import argparse


def create_subset(class_list, full_imagenet_path, subset_imagenet_path, *,
                  splits=('train', 'val'), force=False):
    def symlink(src, dst, **kwargs):
        try:
            os.symlink(src, dst, **kwargs)
        except OSError as e:
            if force and e.errno == errno.EEXIST:
                if not os.path.islink(dst):
                    raise RuntimeError(
                        f'--force cannot overwrite existing {dst} because it is not a symlink') from e
                os.remove(dst)
                os.symlink(src, dst, **kwargs)
            else:
                raise e

    full_imagenet_path = os.path.abspath(full_imagenet_path)
    subset_imagenet_path = os.path.abspath(subset_imagenet_path)
    os.makedirs(subset_imagenet_path, exist_ok=True)
    for split in splits:
        os.makedirs(os.path.join(subset_imagenet_path, split), exist_ok=True)
    for c in class_list:
        if re.match(r"n[0-9]{8}", c) is None:
            raise ValueError(
                f"Expected class names to be of the format nXXXXXXXX, where "
                f"each X represents a numerical number, e.g., n04589890, but "
                f"got {c}")
        for split in splits:
            symlink(
                os.path.join(full_imagenet_path, split, c),
                os.path.join(subset_imagenet_path, split, c),
                target_is_directory=True,
            )
    print(f'Finished creating ImageNet subset at {subset_imagenet_path}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Subset Creation')
    parser.add_argument('full_imagenet_path', metavar='IMAGENET_DIR',
                        help='path to the existing full ImageNet dataset')
    parser.add_argument('subset_imagenet_path', metavar='SUBSET_DIR',
                        help='path to create the ImageNet subset dataset')
    parser.add_argument('-s', '--subset', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'imagenet100_classes.txt'),
                        help='file contains a list of subset classes')
    parser.add_argument('-f', '--force', action='store_true',
                        help='whether to overwrite if symlinks already exist')
    args = parser.parse_args()

    print(f'Using class names specified in {args.subset}.')
    with open(args.subset, 'r') as f:
        class_list = [l.strip() for l in f.readlines()]

    create_subset(class_list, args.full_imagenet_path, args.subset_imagenet_path, force=args.force)
