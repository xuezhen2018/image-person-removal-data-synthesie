import time
from options.train_options import TrainOptions
from data.dataprocess import DataProcess
from models.models import create_model
import torchvision
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import os
import torch
from math import log10
from evaluation import AverageMeter, FScore,compute_RMSE
import pytorch_ssim
import lpips
import torch.nn.functional as F
import csv
if __name__ == "__main__":

    opt = TrainOptions().parse()
    # define the dataset

    dataset = DataProcess(opt.GT_root,opt.st_root,opt.input_root,opt.mask_root,opt.isTrain)
    iterator_train = (data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.num_workers))
    dataset = DataProcess(opt.val_GT_root,opt.val_st_root,opt.val_input_root,opt.val_mask_root,opt.isTrain)
    iterator_val = (data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers))
    # Create model
    model = create_model(opt)
    total_steps=0
    best_loss=0
    epoch_iter = 0
    psnr_meter = AverageMeter()
    fpsnr_meter = AverageMeter()
    LPIPS_meter = []
    ssim_meter = AverageMeter()
    rmse_meter = AverageMeter()
    rmsew_meter = AverageMeter()
    total_loss_epoch = AverageMeter()
    total_loss_iters = AverageMeter()
    csv_filename_iters = os.path.join(opt.checkpoints_dir, opt.name, 'iters_loss_log.csv')
    csv_filename_epoch = os.path.join(opt.checkpoints_dir, opt.name, 'epoch_loss_log.csv')
    loss_fn = lpips.LPIPS(net='vgg', spatial=True)
    loss_fn.cuda()
    # Create the logs
    dir = os.path.join(opt.log_dir, opt.name,'log').replace('\\', '/')
    if not os.path.exists(dir):
        os.mkdir(dir)
    writer = SummaryWriter(log_dir=dir, comment=opt.name)
    # Start Training
    for epoch in range (opt.epoch_count, 100):
        epoch_start_time = time.time()
        epoch_iter += opt.batchSize
        for detail, structure,input, mask in iterator_train:

            iter_start_time = time.time()
            total_steps += opt.batchSize

            model.set_input(detail, structure,input, mask)
            model.optimize_parameters()

            errors = model.get_current_errors()
            dis_loss = errors['G_L1']
            total_loss_iters.update(dis_loss.item(), detail.size(0))
            total_loss_epoch.update(dis_loss.item(), detail.size(0))

            with open(csv_filename_iters, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([total_steps, total_loss_iters.avg])




            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time)

                # writer.add_scalar('G_GAN', errors['G_GAN'], total_steps + 1)
                # writer.add_scalar('G_L1', errors['G_L1'], total_steps + 1)
                # writer.add_scalar('G_stde', errors['G_stde'], total_steps + 1)
                # writer.add_scalar('D_loss', errors['D'], total_steps + 1)
                # writer.add_scalar('F_loss', errors['F'], total_steps + 1)
                print('iteration time: %d' % t)
        with torch.no_grad():
            for val_detail, val_structure,val_input, val_mask in iterator_val:
                model.set_input(val_detail, val_structure,val_input, val_mask)
                model.forward()
                fake_out = model.fake_out
                val_mask,val_detail,val_input = val_mask.cuda(),val_detail.cuda(),val_input.cuda()
                imagefine = fake_out * val_mask + val_detail * (1 - val_mask)
                eps = 1e-6
                # LPIPS
                LPIPS = loss_fn.forward(val_detail, imagefine)
                LPIPS_meter.append(LPIPS.mean().item())


                psnr = 10 * log10(1 / F.mse_loss(imagefine, val_detail).item())
                # 均方误差
                fmse = F.mse_loss(imagefine * 1 - val_mask, val_detail * 1 - val_mask, reduction='none').sum(dim=[1, 2, 3]) / (
                        1 - val_mask.sum(dim=[1, 2, 3]) * 3 + eps)
                fpsnr = 10 * torch.log10(1 / fmse).mean().item()
                ssim = pytorch_ssim.ssim(imagefine, val_detail)

                psnr_meter.update(psnr, val_detail.size(0))
                fpsnr_meter.update(fpsnr, val_detail.size(0))
                ssim_meter.update(ssim, val_detail.size(0))
                rmse_meter.update(compute_RMSE(imagefine, val_detail, 1 - val_mask), val_detail.size(0))
                rmsew_meter.update(compute_RMSE(imagefine, val_detail, 1 - val_mask, is_w=True), val_detail.size(0))
            metric = psnr_meter.avg
            is_best = True if best_loss < metric else False
            if is_best:
                best_loss = metric
            if is_best:
                print('saving the model at the end of epoch %d, iters %d' %
                    (epoch, total_steps))
                model.save_networks(epoch)
                print("PSNR:", psnr_meter.avg, "SSIM：", ssim_meter.avg, "RMSEw:", rmsew_meter.avg)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, 400, time.time() - epoch_start_time))

        with open(csv_filename_epoch, mode='a', newline='') as file:
            writer_epoch = csv.writer(file)
            writer_epoch.writerow([epoch_iter, total_loss_epoch.avg])
    writer.close()
