from PIL import Image

from torchvision import transforms


def image_transforms(load_size):

    return transforms.Compose([
        # transforms.CenterCrop(size=(178, 178)),  # for CelebA
        transforms.Resize(size=load_size, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#做归一化更好的训练，也就是数据的度量统一
    ])

def GT_transforms(load_size):

    return transforms.Compose([
        # transforms.CenterCrop(size=(178, 178)),  # for CelebA
        transforms.Resize(size=load_size, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#做归一化更好的训练，也就是数据的度量统一
    ])


def mask_transforms(load_size):

    return transforms.Compose([
        transforms.Resize(size=load_size, interpolation=Image.BILINEAR),
        transforms.ToTensor()
    ])