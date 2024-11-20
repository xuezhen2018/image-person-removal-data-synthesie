from .dataset import InpaintingData
from .val_dataset import InpaintingValData

from torch.utils.data import DataLoader


def sample_data(loader): 
    while True:
        for batch in loader:
            yield batch


def create_loader(args): 
    dataset = InpaintingData(args)
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    return sample_data(data_loader)


def create_val_loader(args):
    dataset = InpaintingValData(args)
    data_loader = DataLoader(
        dataset, batch_size=1,
        shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return sample_data(data_loader)