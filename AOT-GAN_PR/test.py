import os
import argparse
import importlib
import numpy as np
from PIL import Image
from glob import glob

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
# from datasets.dataset import create_image_dataset,create_val_image_dataset
from utils.ddp import data_sampler
from torch.utils import data
from utils.misc import sample_data
from math import log10
from evaluation import AverageMeter, FScore,compute_RMSE
from utils.util import *
import torch.nn.functional as F
import pytorch_ssim
import lpips
from data import create_loader,create_val_loader


from utils.option import args 


def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cuda().numpy().astype(np.uint8)
    return Image.fromarray(image)


def main_worker(args, use_gpu=True):
    val_dataloader = create_val_loader(args)
    print("==> testing VM model ")
    psnr_meter = AverageMeter()
    fpsnr_meter = AverageMeter()
    LPIPS_meter = []
    ssim_meter = AverageMeter()
    rmse_meter = AverageMeter()
    rmsew_meter = AverageMeter()

    # val_dataset = create_val_image_dataset(args)
    # val_len_data = val_dataset.__len__()
    # val_image_data_loader = data.DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size,
    #     sampler=data_sampler(
    #         val_dataset, shuffle=False
    #     ),
    #     drop_last=True, num_workers=args.num_workers, pin_memory=True
    # )  # 设置DataLoader,pin_memory 可以解决显存不足的问题。
    
    # val_image_data_loader = sample_data(val_image_data_loader)  # 根据batch值给出数据
    device = torch.device('cuda') \
        # if use_gpu else torch.device('cpu')
    
    # Model and version
    net = importlib.import_module('model.'+args.model)
    model = net.InpaintGenerator(args).cuda()
    model.load_state_dict(torch.load(args.pre_train, map_location='cuda'))
    model.eval()
    #
    loss_fn = lpips.LPIPS(net='vgg', spatial=True)
    loss_fn.cuda()
    # # prepare dataset
    # image_paths = []
    # for ext in ['.jpg', '.png']:
    #     image_paths.extend(glob(os.path.join(args.dir_image, '*'+ext)))
    # image_paths.sort()
    # mask_paths = sorted(glob(os.path.join(args.dir_mask, '*.png')))
    # os.makedirs(args.outputs, exist_ok=True)
    #
    # iteration through datasets
    # for ipath, mpath in zip(image_paths, mask_paths):
        # image = ToTensor()(Image.open(ipath).convert('RGB'))
        # image = (image * 2.0 - 1.0).unsqueeze(0)
        # mask = ToTensor()(Image.open(mpath).convert('L'))
        # mask = mask.unsqueeze(0)
        # image, mask = image.cuda(), mask.cuda()
    prediction_dir = os.path.join("../experiments/AOT_BL", 'image')
    if not os.path.exists(prediction_dir): os.makedirs(prediction_dir)
    # print(prediction_dir)s
    with torch.no_grad():
        for i in range(150):
            val_images, val_masks, val_GT = next(val_dataloader)
            # val_mask_m = 1 - val_mask
            # val_inputs = val_inputs * val_mask_m
            val_images, val_masks,val_GT = val_images.float().cuda(), val_masks.float().cuda(),val_GT.float().cuda()
            # image_masked = val_GT * (1 - val_masks.float()) + val_masks
            image_masked = val_GT
            pred_img = model(image_masked, val_masks)
            
            imfinal = (1 - val_masks) * val_GT + val_masks * pred_img
            # print(imfinal.shape)
            imfinal = (imfinal+1.0)/2.0
            val_GT = (val_GT+1.0)/2.0
            val_images = (val_images+1.0)/2.0
            eps = 1e-6
            # LPIPS
            LPIPS = loss_fn.forward(val_GT, imfinal)
            LPIPS_meter.append(LPIPS.mean().item())


            psnr = 10 * log10(1 / F.mse_loss(imfinal, val_GT).item())


            fmse = F.mse_loss(imfinal * val_masks, val_GT * val_masks, reduction='none').sum(dim=[1, 2, 3]) / (
                                val_masks.sum(dim=[1, 2, 3]) * 3 + eps)
            fpsnr = 10 * torch.log10(1 / fmse).mean().item()
            ssim = pytorch_ssim.ssim(imfinal, val_GT)

            psnr_meter.update(psnr, val_GT.size(0))
            fpsnr_meter.update(fpsnr, val_GT.size(0))
            ssim_meter.update(ssim, val_GT.size(0))
            # file_name1 = "../../third/500_R_R"+ str(i)+'.png'
            # print(file_name1)
            # print(imfinal.type)
            # print(val_images.type)
            # print(val_masks.type)
            # print(val_GT.type)
            save_output(
                    inputs={'I': val_images, 'mask': val_masks,'GT':val_GT},
                    preds={'bg': imfinal},
                    save_dir=prediction_dir,
                    num = i,
                    verbose=False)
        print("PSNR:%.4f,SSIM:%.4f,LPIPS:%.4f" % (psnr_meter.avg, ssim_meter.avg, sum(LPIPS_meter) / 150))

if __name__ == '__main__':
    main_worker(args)
