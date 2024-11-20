import os
import importlib
from tqdm import tqdm
from glob import glob

import torch
import torch.optim as optim
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from data import create_loader,create_val_loader
from loss import loss as loss_module
from .common import timer, reduce_loss_dict
from math import log10
from evaluation import AverageMeter, FScore,compute_RMSE
from utils.util import *
import torch.nn.functional as F
import pytorch_ssim
import lpips

class Trainer():
    def __init__(self, args):
        self.args = args 
        self.iteration = 0

        # setup data set and data loader
        self.dataloader = create_loader(args)
        self.val_dataloader = create_val_loader(args)
        self.loss_fn = lpips.LPIPS(net='vgg', spatial=True)
        self.loss_fn.cuda()
        # set up losses and metrics
        self.rec_loss_func = {
            key: getattr(loss_module, key)() for key, val in args.rec_loss.items()}
        self.adv_loss = getattr(loss_module, args.gan_type)()

        # Image generator input: [rgb(3) + mask(1)], discriminator input: [rgb(3)]
        net = importlib.import_module('model.'+args.model)

        self.netG = net.InpaintGenerator(args).cuda()
        self.optimG = torch.optim.Adam(
            self.netG.parameters(), lr=args.lrg, betas=(args.beta1, args.beta2))

        self.netD = net.Discriminator().cuda()
        self.optimD = torch.optim.Adam(
            self.netD.parameters(), lr=args.lrd, betas=(args.beta1, args.beta2))
        
        # self.load()
        if args.distributed:
            self.netG = DDP(self.netG, device_ids= [args.local_rank], output_device=[args.local_rank])
            self.netD = DDP(self.netD, device_ids= [args.local_rank], output_device=[args.local_rank])
        
        if args.tensorboard: 
            self.writer = SummaryWriter(os.path.join(args.save_dir, 'log'))
            

    def load(self):
        try: 
            gpath = sorted(list(glob(os.path.join(self.args.save_dir, 'G*.pt'))))[-1]
            self.netG.load_state_dict(torch.load(gpath, map_location='cuda'))
            self.iteration = int(os.path.basename(gpath)[1:-3])
            if self.args.global_rank == 0: 
                print(f'[**] Loading generator network from {gpath}')
        except: 
            pass 
        
        try: 
            dpath = sorted(list(glob(os.path.join(self.args.save_dir, 'D*.pt'))))[-1]
            self.netD.load_state_dict(torch.load(dpath, map_location='cuda'))
            if self.args.global_rank == 0: 
                print(f'[**] Loading discriminator network from {dpath}')
        except: 
            pass
        
        try: 
            opath = sorted(list(glob(os.path.join(self.args.save_dir, 'O*.pt'))))[-1]
            data = torch.load(opath, map_location='cuda')
            self.optimG.load_state_dict(data['optimG'])
            self.optimD.load_state_dict(data['optimD'])
            if self.args.global_rank == 0: 
                print(f'[**] Loading optimizer from {opath}')
        except: 
            pass


    def save(self, ):
        if self.args.global_rank == 0:
            print(f'\nsaving {self.iteration} model to {self.args.save_dir} ...')
            torch.save(self.netG.state_dict(), 
                os.path.join(self.args.save_dir, f'G{str(self.iteration).zfill(7)}.pt'))
            torch.save(self.netD.state_dict(), 
                os.path.join(self.args.save_dir, f'D{str(self.iteration).zfill(7)}.pt'))
            torch.save(
                {'optimG': self.optimG.state_dict(), 'optimD': self.optimD.state_dict()}, 
                os.path.join(self.args.save_dir, f'O{str(self.iteration).zfill(7)}.pt'))
            

    def train(self):
        pbar = range(self.iteration, self.args.iterations)
        best_loss = 0
        if self.args.global_rank == 0: 
            pbar = tqdm(range(self.args.iterations), initial=self.iteration, dynamic_ncols=True, smoothing=0.01)
            timer_data, timer_model = timer(), timer()
        for idx in pbar:
            self.iteration += 1
            images, masks, GT = next(self.dataloader)
            images, masks,GT = images.cuda(), masks.cuda(),GT.cuda()
            # images_masked = (images * (1 - masks).float()) + masks

            if self.args.global_rank == 0: 
                timer_data.hold()
                timer_model.tic()

            # in: [rgb(3) + edge(1)]
            pred_img = self.netG(images, masks)
            # print(images.shape)
            # print(masks.shape)
            # print(GT.shape)
            # print(pred_img.shape)
            comp_img = (1 - masks) * GT + masks * pred_img

            # reconstruction losses 
            losses = {}
            for name, weight in self.args.rec_loss.items(): 
                losses[name] = weight * self.rec_loss_func[name](pred_img, GT)
            
            # adversarial loss 
            dis_loss, gen_loss = self.adv_loss(self.netD, comp_img, GT, masks)
            losses[f"advg"] = gen_loss * self.args.adv_weight
            
            # backforward 
            self.optimG.zero_grad()
            self.optimD.zero_grad()
            sum(losses.values()).backward()
            losses[f"advd"] = dis_loss 
            dis_loss.backward()
            self.optimG.step()
            self.optimD.step()

            if self.args.global_rank == 0:
                timer_model.hold()
                timer_data.tic()

            # logs
            scalar_reduced = reduce_loss_dict(losses, self.args.world_size)
            if self.args.global_rank == 0 and (self.iteration % self.args.print_every == 0): 
                pbar.update(self.args.print_every)
                description = f'mt:{timer_model.release():.1f}s, dt:{timer_data.release():.1f}s, '
                for key, val in losses.items(): 
                    description += f'{key}:{val.item():.3f}, '
                    if self.args.tensorboard: 
                        self.writer.add_scalar(key, val.item(), self.iteration)
                pbar.set_description((description))
                if self.args.tensorboard: 
                    self.writer.add_image('mask', make_grid(masks), self.iteration)
                    self.writer.add_image('orig', make_grid((GT+1.0)/2.0), self.iteration)
                    self.writer.add_image('pred', make_grid((pred_img+1.0)/2.0), self.iteration)
                    self.writer.add_image('comp', make_grid((comp_img+1.0)/2.0), self.iteration)
                    
            
            if self.args.global_rank == 0 and (self.iteration % self.args.save_every) == 0:
                print("==> val VM model ")
                psnr_meter = AverageMeter()
                fpsnr_meter = AverageMeter()
                LPIPS_meter = []
                ssim_meter = AverageMeter()
                rmse_meter = AverageMeter()
                rmsew_meter = AverageMeter()

                with torch.no_grad():
                    self.netG.eval()
                    for i in range(150):
                        val_images, val_masks, val_GT = next(self.val_dataloader)
                        # val_mask_m = 1 - val_mask
                        # val_inputs = val_inputs * val_mask_m
                        val_images, val_masks,val_GT = val_images.cuda(), val_masks.cuda(),val_GT.cuda()
                        pred_img = self.netG(val_images, val_masks)

                        imfinal = (1 - val_masks) * val_GT + val_masks * pred_img
                        # print(imfinal.shape)

                        eps = 1e-6
                        # LPIPS
                        LPIPS = self.loss_fn.forward(val_GT, imfinal)
                        LPIPS_meter.append(LPIPS.mean().item())

                        # psnr是“Peak Signal to Noise Ratio”的缩写，即峰值信噪比，是一种评价图像的客观标准
                        psnr = 10 * log10(1 / F.mse_loss(imfinal, val_GT).item())

                        # 均方误差
                        fmse = F.mse_loss(imfinal * val_masks, val_GT * val_masks, reduction='none').sum(dim=[1, 2, 3]) / (
                                val_masks.sum(dim=[1, 2, 3]) * 3 + eps)
                        fpsnr = 10 * torch.log10(1 / fmse).mean().item()
                        ssim = pytorch_ssim.ssim(imfinal, val_GT)

                        psnr_meter.update(psnr, val_GT.size(0))
                        fpsnr_meter.update(fpsnr, val_GT.size(0))
                        ssim_meter.update(ssim, val_GT.size(0))
                    print("PSNR:%.4f,SSIM:%.4f,LPIPS:%.4f" % (
                        psnr_meter.avg, ssim_meter.avg, sum(LPIPS_meter) / 150))
                    metric = psnr_meter.avg
                    is_best = True if best_loss < metric else False
                    if is_best:
                        best_loss = metric
                    if is_best:
                        self.save()
                self.netG.eval()


