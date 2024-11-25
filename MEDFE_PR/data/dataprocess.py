import random
import torch
import torch.utils.data
from PIL import Image
from glob import glob
import numpy as np
import torchvision.transforms as transforms

class DataProcess(torch.utils.data.Dataset):
    def __init__(self, de_root, st_root, input_root,mask_root, opt, train=True):
        super(DataProcess, self).__init__()
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256),interpolation=Image.BILINEAR),
            transforms.ToTensor(),
        ])
        # mask should not normalize, is just have 0 or 1
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256),interpolation=Image.BILINEAR),
            transforms.ToTensor()
        ])
        self.Train = False
        self.opt = opt

        if train:
            self.de_paths = sorted(glob('{:s}/*'.format(de_root), recursive=True))
            self.st_paths = sorted(glob('{:s}/*'.format(st_root), recursive=True))
            self.input_paths = sorted(glob('{:s}/*'.format(input_root), recursive=True))
            self.mask_paths = sorted(glob('{:s}/*'.format(mask_root), recursive=True))
            self.Train=True
        self.N_mask = len(self.mask_paths)
        print(self.N_mask)
    def __getitem__(self, index):

        de_img = Image.open(self.de_paths[index])
        st_img = Image.open(self.st_paths[index])
        input_img = Image.open(self.input_paths[index])
        mask_img = Image.open(self.mask_paths[index])

        de_img = self.img_transform(de_img.convert('RGB'))
        st_img = self.img_transform(st_img .convert('RGB'))
        input_img = self.img_transform(input_img .convert('RGB'))
        mask_img = self.mask_transform(mask_img.convert('RGB'))


        return de_img, st_img,input_img, mask_img

    def __len__(self):
        return len(self.de_paths)
