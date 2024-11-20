import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from datasets.transform import mask_transforms, image_transforms,GT_transforms
from datasets.folder import make_dataset



class ImageDataset(Dataset):

    def __init__(self, image_root, mask_root,GT_root,load_size, mode='test'):
        super(ImageDataset, self).__init__()

        self.image_files = make_dataset(dir=image_root)
        if GT_root == '':
            self.GT_files = None
        else:
            self.GT_files = make_dataset(dir=GT_root)
        self.mask_files = make_dataset(dir=mask_root)

        self.number_image = len(self.image_files)
        if GT_root == '':
            self.number_GT = None
        else:
            self.number_GT = len(self.GT_files)
        self.number_mask = len(self.mask_files)

        self.mode = mode

        self.load_size = load_size

        self.image_files_transforms = image_transforms(load_size)
        if self.number_GT != None and self.GT_files != None:
            self.GT_files_transforms = GT_transforms(load_size)
        self.mask_files_transforms = mask_transforms(load_size)

    def __getitem__(self, index):

        image = Image.open(self.image_files[index % self.number_image])
        image = self.image_files_transforms(image.convert('RGB'))
        if self.number_GT != None and self.GT_files!=None:
            GT = Image.open(self.GT_files[index % self.number_GT])
            GT = self.image_files_transforms(GT.convert('RGB'))

        if self.mode == 'train':
            mask = Image.open(self.mask_files[index % self.number_mask])
        else:
            mask = Image.open(self.mask_files[index % self.number_mask])

        mask = self.mask_files_transforms(mask)
        threshold = 0.00004
        ones = mask < threshold
        zeros = mask > threshold

        mask.masked_fill_(ones, 1.0)
        mask.masked_fill_(zeros, 0.0)

        mask = 1 - mask

        file_path = self.mask_files[index % self.number_mask]

        if self.number_GT != None and self.GT_files != None:
            return image, mask, GT
        else:
            return image, mask

    def __len__(self):

        return self.number_image


def create_image_dataset(opts):
    image_dataset = ImageDataset(
        image_root=opts.image_root,
        mask_root=opts.mask_root,
        GT_root=opts.GT_root,
        load_size=opts.load_size,
        mode=opts.mode
    )

    return image_dataset
def create_val_image_dataset(opts):
    image_dataset = ImageDataset(
        image_root=opts.val_image_root,
        mask_root=opts.val_mask_root,
        GT_root=opts.val_GT_root,
        load_size=opts.load_size,
        mode=opts.mode
    )

    return image_dataset