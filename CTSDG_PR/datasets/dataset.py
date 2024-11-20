import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.canny import image_to_edge
from datasets.transform import mask_transforms, image_transforms
from datasets.folder import make_dataset


class ImageDataset(Dataset):

    def __init__(self, input_root, GT_root, mask_root, load_size, sigma=2., mode='test'):
        super(ImageDataset, self).__init__()

        self.input_files = make_dataset(dir=input_root)
        self.GT_files = make_dataset(dir=GT_root)
        self.mask_files = make_dataset(dir=mask_root)

        self.number_input = len(self.input_files)
        self.number_mask = len(self.mask_files)

        self.sigma = sigma
        self.mode = mode

        self.load_size = load_size

        self.input_files_transforms = image_transforms(load_size)
        self.GT_files_transforms = image_transforms(load_size)
        self.mask_files_transforms = mask_transforms(load_size)
        file = os.listdir(input_root)
        self.number_image = len(file)
    def __getitem__(self, index):

        input = Image.open(self.input_files[index % self.number_input])
        input = self.input_files_transforms(input.convert('RGB'))
        GT = Image.open(self.GT_files[index % self.number_input])
        GT = self.GT_files_transforms(GT.convert('RGB'))

        if self.mode == 'train':
            mask = Image.open(self.mask_files[index % self.number_mask])
        else:
            mask = Image.open(self.mask_files[index % self.number_mask])

        mask = self.mask_files_transforms(mask)

        threshold = 0.003
        ones = mask >= threshold
        zeros = mask < threshold

        mask.masked_fill_(ones, 1.0)
        mask.masked_fill_(zeros, 0.0)

        mask = 1 - mask

        input_edge, input_gray_image = image_to_edge(input, sigma=self.sigma)
        GT_edge, GT_gray_image = image_to_edge(GT, sigma=self.sigma)

        return input, input_edge,input_gray_image,GT,GT_edge, GT_gray_image, mask

    def __len__(self):

        return self.number_input


def create_image_dataset(opts):
    image_dataset = ImageDataset(
        opts.input_root,
        opts.GT_root,
        opts.mask_root,
        opts.load_size,
        opts.sigma,
        opts.mode
    )
    return image_dataset

def create_val_image_dataset(opts):
    val_image_dataset = ImageDataset(
        opts.val_input_root,
        opts.val_GT_root,
        opts.val_mask_root,
        opts.load_size,
        opts.sigma,
        opts.mode
    )
    return val_image_dataset

