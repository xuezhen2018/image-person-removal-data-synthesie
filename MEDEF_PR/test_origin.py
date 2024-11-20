import time
import pdb
from options.test_options import TestOptions
from data.dataprocess import DataProcess
from models.models import create_model
import torchvision
from torch.utils import data
# from torch.utils.tensorboard import SummaryWriter
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
from evaluation import AverageMeter, FScore, compute_RMSE
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
        transforms.Resize((256, 256), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.BILINEAR),
        transforms.ToTensor()
    ])
    psnr_meter = AverageMeter()
    fpsnr_meter = AverageMeter()
    LPIPS_meter = []
    ssim_meter = AverageMeter()
    rmse_meter = AverageMeter()
    rmsew_meter = AverageMeter()
    loss_fn = lpips.LPIPS(net='vgg', spatial=True)
    loss_fn.cpu()
    opt = TestOptions().parse()
    model = create_model(opt)
    model.netEN.module.load_state_dict(torch.load("EN.pkl"))
    model.netDE.module.load_state_dict(torch.load("DE.pkl"))
    model.netMEDFE.module.load_state_dict(torch.load("MEDEF.pkl"))
    dir = r'models/result/MEDFE_3dre7500'
    if not os.path.exists(dir):
        os.mkdir(dir)
    results_dir_fake = r'models/result/MEDFE_3dre7500/fake'
    results_dir_real = r'models/result/MEDFE_3dre7500/real'
    if not os.path.exists(results_dir_fake):
        os.mkdir(results_dir_fake)
    if not os.path.exists(results_dir_real):
        os.mkdir(results_dir_real)

    mask_paths = glob('{:s}/*'.format(opt.val_mask_root))
    de_paths = glob('{:s}/*'.format(opt.val_GT_root))
    st_path = glob('{:s}/*'.format(opt.val_st_root))
    image_len = len(de_paths)
    with torch.no_grad():
        for i in tqdm(range(image_len)):
            # only use one mask for all image
            path_m = mask_paths[i]
            path_d = de_paths[i]
            path_s = de_paths[i]

            mask = Image.open(path_m).convert("RGB")
            detail = Image.open(path_d).convert("RGB")
            structure = Image.open(path_s).convert("RGB")
            # mask trans
            mask = mask_transform(mask)
            threshold = 0.00004
            ones = mask < threshold
            zeros = mask > threshold

            mask.masked_fill_(ones, 0.0)
            mask.masked_fill_(zeros, 1.0)

            detail = img_transform(detail)

            structure = img_transform(structure)
            mask = torch.unsqueeze(mask, 0)
            detail = torch.unsqueeze(detail, 0)
            structure = torch.unsqueeze(structure, 0)

            filepath1 = results_dir_fake + '/' + str(i) + ".png"
            filepath2 = results_dir_real + '/' + str(i) + ".png"
            # structure=structure.cuda()
            # mask= mask.cuda()
            output2 = cv2.cvtColor(tensor2np(detail)[0], cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath2, output2)
            model.set_input(detail, structure, mask)

            model.forward()
            fake_out = model.fake_out
            imagefine = fake_out.cpu() * mask + detail * (1 - mask)
            eps = 1e-6
            # LPIPS
            LPIPS = loss_fn.forward(detail, imagefine)
            LPIPS_meter.append(LPIPS.mean().item())


            psnr = 10 * log10(1 / F.mse_loss(imagefine, detail).item())

            fmse = F.mse_loss(imagefine * 1 - mask, detail * 1 - mask, reduction='none').sum(dim=[1, 2, 3]) / (
                    1 - mask.sum(dim=[1, 2, 3]) * 3 + eps)
            fpsnr = 10 * torch.log10(1 / fmse).mean().item()
            ssim = pytorch_ssim.ssim(imagefine, detail)
            psnr_meter.update(psnr, detail.size(0))
            fpsnr_meter.update(fpsnr, detail.size(0))
            ssim_meter.update(ssim, detail.size(0))
            rmse_meter.update(compute_RMSE(imagefine, detail, 1 - mask), detail.size(0))
            rmsew_meter.update(compute_RMSE(imagefine, detail, 1 - mask, is_w=True), detail.size(0))
            output1 = cv2.cvtColor(tensor2np(imagefine)[0], cv2.COLOR_RGB2BGR)

            cv2.imwrite(filepath1, output1)

    print("PSNR:%.4f,SSIM:%.4f,LPIPS:%.4f,RMSE:%.4f,RMSER:%.4f" % (
        psnr_meter.avg, ssim_meter.avg, sum(LPIPS_meter) / 150, rmse_meter.avg, rmsew_meter.avg))