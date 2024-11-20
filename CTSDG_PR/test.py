import warnings
warnings.filterwarnings("ignore")


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils import data
from torchvision.utils import save_image
import cv2
from models.generator.generator import Generator
from datasets.dataset import create_image_dataset
from options.test_options import TestOptions
from utils.misc import sample_data, postprocess
import os
import pytorch_ssim
import torch
import numpy as np
from tqdm import tqdm
from math import log10
from tensorboardX import SummaryWriter
from evaluation import AverageMeter, FScore,compute_RMSE
import torch.nn.functional as F
import torchvision
from utils.distributed import get_rank, reduce_loss_dict
from utils.misc import requires_grad, sample_data,postprocess
from criteria.loss import generator_loss_func, discriminator_loss_func
from models.generator.vgg16 import VGG16FeatureExtractor
import lpips

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
is_cuda = torch.cuda.is_available()
if is_cuda:
    print('Cuda is available')
    cudnn.enable = True
    cudnn.benchmark = True

opts = TestOptions().parse
os.makedirs('{:s}'.format(opts.result_root), exist_ok=True)
extractor = VGG16FeatureExtractor()
# model & load model
generator = Generator(image_in_channels=3, edge_in_channels=2, out_channels=3)
if opts.pre_trained != '':
    generator.load_state_dict(torch.load(opts.pre_trained)['generator'])
else:
    print('Please provide pre-trained model!')

if is_cuda:
    generator,extractor = generator.cuda(),extractor.cuda()

# dataset
image_dataset = create_image_dataset(opts)
val_len = image_dataset.__len__()
image_data_loader = data.DataLoader(
    image_dataset,
    batch_size=opts.batch_size,
    shuffle=False,
    num_workers=opts.num_workers,
    drop_last=False
)
test_image_data_loader = sample_data(image_data_loader)
pbar = range(val_len)
pbar = tqdm(pbar, initial=1, dynamic_ncols=True, smoothing=0.01)
writer = SummaryWriter(opts.log_dir)
loss_fn = lpips.LPIPS(net='vgg', spatial=True)
loss_fn.cuda()

print('start test...')
with torch.no_grad():
    psnr_meter = AverageMeter()
    fpsnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    LPIPS_meter = []
    rmse_meter = AverageMeter()
    rmsew_meter = AverageMeter()
    val_lenth = 0
    generator.eval()
    for j in pbar:
        val_input, val_input_edge, val_input_gray_image, val_ground_truth, val_GT_edge, val_GT_gray_image, val_mask = next(
            test_image_data_loader)
        if is_cuda:
            val_input, val_input_edge, val_input_gray_image, val_ground_truth, val_GT_edge, val_GT_gray_image, val_mask = val_input.cuda(), val_input_edge.cuda(), val_input_gray_image.cuda(), val_ground_truth.cuda(), val_GT_edge.cuda(), val_GT_gray_image.cuda(), val_mask.cuda()

        output, projected_image, projected_edge = generator(val_input, torch.cat(
            (val_input_edge, val_input_gray_image), dim=1), val_mask)
        comp = val_ground_truth * val_mask + output * (1 - val_mask)
        imfinal = output * (1 - val_mask) + val_input * val_mask
        eps = 1e-6

        LPIPS = loss_fn.forward(val_ground_truth, imfinal)
        LPIPS_meter.append(LPIPS.mean().item())

        psnr = 10 * log10(1 / F.mse_loss(imfinal, val_ground_truth).item())

        fmse = F.mse_loss(imfinal * (1 - val_mask), val_ground_truth * (1 - val_mask),
                          reduction='none').sum(dim=[1, 2, 3]) / (
                       (1 - val_mask).sum(dim=[1, 2, 3]) * 3 + eps)
        fpsnr = 10 * torch.log10(1 / fmse).mean().item()
        ssim = pytorch_ssim.ssim(imfinal, val_ground_truth)
        psnr_meter.update(psnr, val_input.size(0))
        fpsnr_meter.update(fpsnr, val_input.size(0))
        ssim_meter.update(ssim, val_input.size(0))
        rmse_meter.update(compute_RMSE(imfinal, val_ground_truth, (1 - val_mask)), val_input.size(0))
        rmsew_meter.update(compute_RMSE(imfinal, val_ground_truth, (1 - val_mask), is_w=True),
                           val_input.size(0))
        pbar.set_description((
            f'fpsnr: {fpsnr_meter.avg:.4f} '
            f'ssim: {ssim_meter.avg:.4f} '
            f'rmse: {rmse_meter.avg:.4f} '
            f'rmsew: {rmsew_meter.avg:.4f} '
        ))
        
        image, mask_gt,input_gt = val_input , val_mask,val_ground_truth

        image = cv2.cvtColor(tensor2np(image)[0], cv2.COLOR_RGB2BGR)

        mask_gt = tensor2np(mask_gt, isMask=True)[0]
        input_gt = cv2.cvtColor(tensor2np(input_gt)[0], cv2.COLOR_RGB2BGR)

        bg_pred = imfinal
        bg_pred = cv2.cvtColor(tensor2np(bg_pred)[0], cv2.COLOR_RGB2BGR)

        outs = [image, bg_pred,input_gt, mask_gt]
        outimg = np.concatenate(outs, axis=1)

        filrname =str(j)+".jpg"
        out_fn = os.path.join(opts.result_root,filrname)
        cv2.imwrite(out_fn, outimg)

print("PSNR:",psnr_meter.avg,"SSIM：",ssim_meter.avg,"LPIPS:",sum(LPIPS_meter) / val_len)

show_size = 5 if val_input.shape[0] > 5 else val_input.shape[0]
image_display = torch.cat([
    val_input[0:show_size].detach().cpu(),  # input image
    val_ground_truth[0:show_size].detach().cpu(),  # ground truth
    comp[0:show_size].detach().cpu(),  # refine out
    val_mask[0:show_size].detach().cpu().repeat(1, 3, 1, 1),
], dim=0)
image_dis = torchvision.utils.make_grid(image_display, nrow=show_size)
writer.add_image('val_Image', image_dis,1)