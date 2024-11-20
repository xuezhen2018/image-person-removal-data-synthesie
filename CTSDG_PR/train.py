import warnings
warnings.filterwarnings("ignore")

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data

from models.discriminator.discriminator import Discriminator
from models.generator.generator import Generator
from models.generator.vgg16 import VGG16FeatureExtractor
from options.train_options import TrainOptions
from datasets.dataset import create_image_dataset,create_val_image_dataset
from utils.distributed import synchronize
from utils.ddp import data_sampler
from trainer import train


opts = TrainOptions().parse

os.makedirs('{:s}'.format(opts.save_dir), exist_ok=True)

is_cuda = torch.cuda.is_available()
if is_cuda:
    
    print('Cuda is available')
    cudnn.enable = True
    cudnn.benchmark = True

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    print('GPU number: ', n_gpu)
    opts.distributed = n_gpu > 1
    # if opts.distributed:
    #     torch.cuda.set_device(opts.local_rank)
    #     torch.distributed.init_process_group(backend="nccl", init_method="env://")
    #     synchronize()

# model & load model
generator = Generator(image_in_channels=3, edge_in_channels=2, out_channels=3)
discriminator = Discriminator(image_in_channels=3, edge_in_channels=2)
extractor = VGG16FeatureExtractor()

# cuda
if is_cuda:
    generator, discriminator, extractor = generator.cuda(), discriminator.cuda(), extractor.cuda()

# optimizer
if opts.finetune == True:
    print('Fine tune...')
    lr = opts.lr_finetune
    generator.freeze_ec_bn = True
else:
    lr = opts.gen_lr

generator_optim = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=lr)
discriminator_optim = optim.Adam(discriminator.parameters(), lr=lr * opts.D2G_lr)

# load checkpoints
if opts.pre_trained != '':
    ckpt_dict = torch.load(opts.pre_trained, map_location=lambda storage, loc: storage)
    opts.start_iter = ckpt_dict['n_iter']
    generator.load_state_dict(ckpt_dict['generator'])
    discriminator.load_state_dict(ckpt_dict['discriminator'])

    print('Starting from iter', opts.start_iter)
else:
    print('Starting from iter', opts.start_iter)


# dataset
image_dataset = create_image_dataset(opts)
val_image_dataset = create_val_image_dataset(opts)
train_len = image_dataset.__len__()
val_len = val_image_dataset.__len__()
print(image_dataset.__len__())
print(val_image_dataset.__len__())

image_data_loader = data.DataLoader(
    image_dataset,
    batch_size=opts.batch_size,
    sampler=data_sampler(
        image_dataset, shuffle=True, distributed=opts.distributed
    ),
    drop_last=True
)
val_data_loader = data.DataLoader(
    val_image_dataset,
    batch_size=1,
    sampler=data_sampler(
        val_image_dataset, shuffle=False, distributed=opts.distributed
    ),
    drop_last=True
)

# training
train(opts, image_data_loader,train_len,val_data_loader,val_len, generator, discriminator, extractor, generator_optim, discriminator_optim, is_cuda)



