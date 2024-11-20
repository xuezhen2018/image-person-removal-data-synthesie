import warnings
import time,os,torch
warnings.filterwarnings("ignore")
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils import data
from utils.misc import sample_data,adjust_learning_rate
from train_options import Options
from datasets.datasetNM import create_image_dataset,create_val_image_dataset
from utils.ddp import data_sampler
import models as models

def main(args):
    args.seed = 1
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    lr = args.lr
    # dataset
    #traindata
    image_dataset = create_image_dataset(args)
    len_data = image_dataset.__len__()
    image_data_loader = data.DataLoader(
        image_dataset,
        batch_size=args.train_batch,
        sampler=data_sampler(
            image_dataset, shuffle=True
        ),
        drop_last=True, num_workers=args.num_workers, pin_memory=True
    )
    image_data_loader = sample_data(image_data_loader)

    #valdata
    val_dataset = create_val_image_dataset(args)
    val_len_data = val_dataset.__len__()
    val_image_data_loader = data.DataLoader(
        val_dataset,
        batch_size=args.test_batch,
        sampler=data_sampler(
            val_dataset, shuffle=False
        ),
        drop_last=True, num_workers=args.num_workers, pin_memory=True
    )
    val_image_data_loader = sample_data(val_image_data_loader)
    print("len_data",len_data)
    print("val_data",val_len_data)
    model = models.__dict__[args.models](train_datasets=image_data_loader,val_datasets=val_image_data_loader, args=args)
    print('============================ Initization Finish && Training Start =============================================')
    for epoch in range(model.args.epochs):
        # lr = adjust_learning_rate(image_dataset, model, epoch, lr, args)
        print('\nEpoch: %d' % (epoch + 1))
        model.record('lr',lr, epoch)
        model.train(epoch,len_data)
        if args.freq < 0:
            model.validate(epoch,val_len_data)
            model.flush()
            model.save_checkpoint()
if __name__ == '__main__':

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        print('Cuda is available')
        print(torch.cuda.get_device_capability(device=None), torch.cuda.get_device_name(device=None))
        cudnn.enable = True
        cudnn.benchmark = True

    print('==================================== Person Removal =============================================')
    print('==> {:50}: {:<}'.format("Start Time",time.ctime(time.time())))
    args=Options().parse
    filename = './' + '{:s}'.format(args.save_dir).strip()
    os.makedirs(filename, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    print('==================================== Start Init Model  ===============================================')
    main(args)
    print('==================================== FINISH WITHOUT ERROR ============================================')