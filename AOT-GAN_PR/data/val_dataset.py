import os
import math
import numpy as np
from glob import glob

from random import shuffle
from PIL import Image, ImageFilter

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    images = sorted(images)

    return images[:min(max_dataset_size, len(images))]


class InpaintingValData(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.w = self.h = args.image_size
        self.mask_type = args.mask_type

        # image and mask

        self.image_path = make_dataset(dir=args.dir_input_val)  # 输入图片
        self.GT_path = make_dataset(dir=args.dir_GT_val)  # 真实图片
        self.mask_path = make_dataset(dir=args.dir_mask_val)  # 蒙版
        self.number_input = len(self.image_path)
        self.number_mask = len(self.mask_path)
        # augmentation
        self.input_trans = transforms.Compose([
            transforms.Resize((args.image_size,args.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor()])
        self.GT_trans = transforms.Compose([
            transforms.Resize((args.image_size,args.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor()])
        self.mask_trans = transforms.Compose([
            transforms.Resize((args.image_size,args.image_size), interpolation=transforms.InterpolationMode.BILINEAR)])

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        # load image
        input = Image.open(self.image_path[index % self.number_input]).convert('RGB')
        GT = Image.open(self.GT_path[index % self.number_input]).convert('RGB')
        mask = Image.open(self.mask_path[index % self.number_mask]).convert('L')

        # augment
        input = self.input_trans(input) * 2. - 1.
        GT = self.GT_trans(GT) * 2. - 1.
        mask = F.to_tensor(self.mask_trans(mask))

        return input, mask, GT


if __name__ == '__main__':
    from attrdict import AttrDict

    args = {
        'dir_image': '../../../dataset',
        'data_train': 'places2',
        'dir_mask': '../../../dataset',
        'mask_type': 'pconv',
        'image_size': 512
    }
    args = AttrDict(args)

    data = InpaintingValData(args)
    print(len(data), len(data.mask_path))
    img, mask, filename = data[0]
    print(img.size(), mask.size(), filename)