import time
import pdb
from options.test_options import TestOptions
from data.dataprocess import DataProcess
from models.models import create_model
import torchvision
from torch.utils import data
#from torch.utils.tensorboard import SummaryWriter
import os
import torch
from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm
import torchvision.transforms as transforms
import cv2
from utils.ddp import data_sampler
from torch.utils import data
from utils.misc import sample_data
from math import log10
from evaluation import AverageMeter, FScore,compute_RMSE
import pytorch_ssim
import lpips
import torch.nn.functional as F

def tensor2np(x, isMask=False):
    if isMask:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = ((x.cpu().detach())) * 255
    else:
        x = x.cpu().detach()
        mean = 0
        std = 1
        x = (x * std + mean) * 255
    return x.numpy().transpose(0, 2, 3, 1).astype(np.uint8)


if __name__ == "__main__":

    img_transform = transforms.Compose([
        transforms.Resize((256, 256),interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256),interpolation=Image.BILINEAR),
        transforms.ToTensor()
    ])
    psnr_meter = AverageMeter()
    fpsnr_meter = AverageMeter()
    LPIPS_meter = []
    ssim_meter = AverageMeter()
    rmse_meter = AverageMeter()
    rmsew_meter = AverageMeter()
    loss_fn = lpips.LPIPS(net='vgg', spatial=True)
    loss_fn.cuda()
    opt = TestOptions().parse()
    de_paths = glob('{:s}/*'.format(opt.eval_GT_root))
    fake_path = glob('{:s}/*'.format(opt.eval_fake_root))
    image_len = len(de_paths )
    for i in tqdm(range(image_len)):
        # only use one mask for all image
        path_d = de_paths[i]
        path_f = fake_path[i]

        detail = Image.open(path_d).convert("RGB")
        fake = Image.open(path_f).convert("RGB")

        detail = img_transform(detail)
        fake = img_transform(fake)

        detail = torch.unsqueeze(detail, 0)
        fake = torch.unsqueeze(fake,0)

        with torch.no_grad():
            detail, fake= detail.cuda(), fake.cuda()
            eps = 1e-6

            LPIPS = loss_fn.forward(detail, fake)
            LPIPS_meter.append(LPIPS.mean().item())


            psnr = 10 * log10(1 / F.mse_loss(fake, detail).item())

            ssim = pytorch_ssim.ssim(fake, detail)
            psnr_meter.update(psnr, detail.size(0))
            ssim_meter.update(ssim, detail.size(0))
    print("PSNR:%.4f,SSIM:%.4f,LPIPS:%.4f" % (
    psnr_meter.avg, ssim_meter.avg, sum(LPIPS_meter) / 150))