from PIL import Image

from torchvision import transforms


def image_transforms(load_size):

    return transforms.Compose([
        # transforms.CenterCrop(size=(178, 178)),  # for CelebA
        transforms.Resize(size=(256,256), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def GT_transforms(load_size):

    return transforms.Compose([
        # transforms.CenterCrop(size=(178, 178)),  # for CelebA
        transforms.Resize(size=(256,256), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def mask_transforms(load_size):

    return transforms.Compose([
        transforms.Resize(size=(256,256), interpolation=Image.BILINEAR),
        transforms.ToTensor()
    ])