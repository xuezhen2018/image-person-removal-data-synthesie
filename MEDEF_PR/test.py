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
    opt = TestOptions().parse()
    dataset = DataProcess(opt.val_GT_root,opt.val_st_root,opt.val_input_root,opt.val_mask_root,opt.isTrain)
    iterator_val = (data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers))
    psnr_meter = AverageMeter()
    fpsnr_meter = AverageMeter()
    LPIPS_meter = []
    ssim_meter = AverageMeter()
    rmse_meter = AverageMeter()
    rmsew_meter = AverageMeter()
    loss_fn = lpips.LPIPS(net='vgg', spatial=True)
    loss_fn.cuda()
    
    model = create_model(opt)
    # model.netEN.module.load_state_dict(torch.load("checkpoints_save/500_R_R/best_net_EN.pth"))
    # model.netDE.module.load_state_dict(torch.load("checkpoints_save/500_R_R/best_net_DE.pth"))
    # model.netMEDFE.module.load_state_dict(torch.load("checkpoints_save/500_R_R/best_net_MEDEF.pth"))
    results_dir = r'models/result/NLR_3dre_R'
    if not os.path.exists( results_dir):
        os.mkdir(results_dir)
    
    mask_paths = glob('{:s}/*'.format(opt.val_mask_root))
    de_paths = glob('{:s}/*'.format(opt.val_input_root))
    gt_paths = glob('{:s}/*'.format(opt.val_GT_root))
    st_path = glob('{:s}/*'.format(opt.val_st_root))
    image_len = len(de_paths )
    i = 0
    with torch.no_grad():
        for val_detail, val_structure,val_input, val_mask in iterator_val:
            model.set_input(val_detail, val_structure,val_input, val_mask)
            model.forward()
            fake_out = model.fake_out
            val_mask,val_detail,val_input = val_mask.cuda(),val_detail.cuda(),val_input.cuda()
            imagefine = fake_out * val_mask + val_detail * (1 - val_mask)
            eps = 1e-6
            # LPIPS
            LPIPS = loss_fn.forward(imagefine, val_detail)
            LPIPS_meter.append(LPIPS.mean().item())


            psnr = 10 * log10(1 / F.mse_loss(imagefine, val_detail).item())

            fmse = F.mse_loss(imagefine * 1 - val_mask, val_detail * 1 - val_mask, reduction='none').sum(dim=[1, 2, 3]) / (
                    1 - val_mask.sum(dim=[1, 2, 3]) * 3 + eps)
            fpsnr = 10 * torch.log10(1 / fmse).mean().item()
            ssim = pytorch_ssim.ssim(imagefine, val_detail)

            psnr_meter.update(psnr, val_detail.size(0))
            fpsnr_meter.update(fpsnr, val_detail.size(0))
            ssim_meter.update(ssim, val_detail.size(0))
            rmse_meter.update(compute_RMSE(imagefine, val_detail, 1 - val_mask), val_detail.size(0))
            rmsew_meter.update(compute_RMSE(imagefine, val_detail, 1 - val_mask, is_w=True), val_detail.size(0))
            output1 = cv2.cvtColor(tensor2np(imagefine)[0], cv2.COLOR_RGB2BGR)
        
            filepath1 = results_dir+'/'+str(i)+"_p"+".png"


            print(filepath1)
            cv2.imwrite(filepath1, output1)
            i=i+1

    print("PSNR:%.4f,SSIM:%.4f,LPIPS:%.4f,RMSE:%.4f,RMSER:%.4f" % (
    psnr_meter.avg, ssim_meter.avg, sum(LPIPS_meter) / 150, rmse_meter.avg, rmsew_meter.avg))