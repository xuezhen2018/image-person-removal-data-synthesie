from __future__ import print_function, absolute_import

import argparse
import torch
import os
from math import log10


torch.backends.cudnn.benchmark = True

import datasets as datasets
import models as models
from test_option import Options
import torch.nn.functional as F
import pytorch_ssim
from evaluation import  FScore, AverageMeter, compute_RMSE, normPRED
from datasets.dataset import create_image_dataset,create_val_image_dataset
from utils.ddp import data_sampler
from torch.utils import data
from utils.misc import sample_data
from skimage.measure import compare_ssim as ssim
import time


def is_dic(x):
    return type(x) == type([])




def main(args):
    args.dataset = args.dataset.lower()

    # valdata
    val_dataset = create_val_image_dataset(args)
    val_len_data = val_dataset.__len__()
    val_image_data_loader = data.DataLoader(
        val_dataset,
        batch_size=args.test_batch,
        sampler=data_sampler(
            val_dataset, shuffle=True
        ),
        drop_last=True, num_workers=args.num_workers, pin_memory=True
    )
    val_image_data_loader = sample_data(val_image_data_loader)


    Machine = models.__dict__[args.models](val_datasets=val_image_data_loader, args=args)
    Machine.test(val_len_data)

if __name__ == '__main__':
    args=Options().parse
    main(args)
    
