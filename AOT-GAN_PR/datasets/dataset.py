from PIL import Image
from torch.utils.data import Dataset

from datasets.transform import mask_transforms, GT_transforms
from datasets.folder import make_dataset



class ImageDataset(Dataset):

    def __init__(self,mask_root,GT_root,input_root,load_size, mode='test'):
        super(ImageDataset, self).__init__()

        self.GT_files = make_dataset(dir=GT_root)
        self.input_files = make_dataset(dir=input_root)
        self.mask_files = make_dataset(dir=mask_root)
        self.number_input = len(self.input_files)
        self.number_GT = len(self.GT_files)
        self.number_mask = len(self.mask_files)
        self.mode = mode
        self.load_size = load_size
        self.GT_files_transforms = GT_transforms(load_size)
        
        self.mask_files_transforms = mask_transforms(load_size)

    def __getitem__(self, index):

        input = Image.open(self.input_files[index % self.number_input])
        input = self.GT_files_transforms(input.convert('RGB'))
        GT = Image.open(self.GT_files[index % self.number_GT])
        GT = self.GT_files_transforms(GT.convert('RGB'))

        mask = Image.open(self.mask_files[index % self.number_mask])

        mask = self.mask_files_transforms(mask)
        threshold = 0.00004
        ones = mask < threshold
        zeros = mask > threshold

        mask.masked_fill_(ones, 1.0)
        mask.masked_fill_(zeros, 0.0)

        mask = 1 - mask

        file_path = self.mask_files[index % self.number_mask]
        return input,mask, GT


    def __len__(self):

        return self.number_input


def create_image_dataset(opts):
    image_dataset = ImageDataset(
        mask_root=opts.dir_mask,
        GT_root=opts.dir_image,
        load_size=[512,512],
    )

    return image_dataset
def create_val_image_dataset(opts):
    image_dataset = ImageDataset(
        mask_root=opts.dir_mask_val,
        GT_root=opts.dir_GT_val,
        input_root=opts.dir_input_val,
        load_size=[512,512],
    )

    return image_dataset