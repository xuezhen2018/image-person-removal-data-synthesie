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

import csv
def denorm(x):
        return x


def norm(x):
        return x

def train(opts, image_data_loader,train_len,val_image_data_loader,val_len,  generator, discriminator, extractor, generator_optim, discriminator_optim, is_cuda):



    image_data_loader = sample_data(image_data_loader)
    val_image_data_loader = sample_data(val_image_data_loader)

    print(train_len)
    print(val_len)
    best_loss = 0
    epoch = 0
    pbar = range(opts.train_iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=opts.start_iter, dynamic_ncols=True, smoothing=0.01)
    if opts.distributed:
        generator_module, discriminator_module = generator.module, discriminator.module
    else:
        generator_module, discriminator_module = generator, discriminator
    generator.train()
    discriminator.train()
    writer = SummaryWriter(opts.log_dir)
    print("=============start train===========")
    total_loss_epoch = AverageMeter()
    total_loss_iters = AverageMeter()
    csv_filename_iters = os.path.join(opts.save_dir, 'iters_loss_log.csv')
    csv_filename_epoch = os.path.join(opts.save_dir, 'epoch_loss_log.csv')
    csv_best_epoch = os.path.join(opts.save_dir, 'epoch_best_log.csv')
    for index in pbar:
        i = index + opts.start_iter+1
        if i > opts.train_iter:
            print('Done...')
            break

        input, input_edge, input_gray_image, ground_truth, GT_edge, GT_gray_image, mask = next(image_data_loader)
        print(input_edge.shape)
        print(input_gray_image.shape)
        print(GT_edge.shape)
        print(GT_gray_image.shape)
        if is_cuda:
            input, input_edge, input_gray_image, ground_truth, GT_edge, GT_gray_image, mask = input.cuda(), input_edge.cuda(), input_gray_image.cuda(), ground_truth.cuda(), GT_edge.cuda(), GT_gray_image.cuda(), mask.cuda()

        print(input_edge.shape)
        print(input_gray_image.shape)

        # ---------
        # Generator
        # ---------
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        output, projected_image, projected_edge = generator(input, torch.cat((input_edge, input_gray_image), dim=1), mask)
        comp = ground_truth * mask + output * (1 - mask)
  
        output_pred, output_edge = discriminator(output, GT_gray_image, GT_edge, is_real=False)
        
        vgg_comp, vgg_output, vgg_ground_truth = extractor(comp), extractor(output), extractor(ground_truth)

        generator_loss_dict = generator_loss_func(
            mask, output, ground_truth, GT_edge, output_pred,
            vgg_comp, vgg_output, vgg_ground_truth, 
            projected_image, projected_edge,
            output_edge,mode="train"
        )
        generator_loss = generator_loss_dict['loss_hole'] * opts.HOLE_LOSS + \
                         generator_loss_dict['loss_valid'] * opts.VALID_LOSS + \
                         generator_loss_dict['loss_perceptual'] * opts.PERCEPTUAL_LOSS + \
                         generator_loss_dict['loss_style'] * opts.STYLE_LOSS + \
                         generator_loss_dict['loss_adversarial'] * opts.ADVERSARIAL_LOSS + \
                         generator_loss_dict['loss_intermediate'] * opts.INTERMEDIATE_LOSS
        generator_loss_dict['loss_joint'] = generator_loss
        
        generator_optim.zero_grad()

        generator_loss.backward()
        generator_optim.step()

        total_loss_iters.update(generator_loss.item(), input.size(0))
        total_loss_epoch.update(generator_loss.item(), input.size(0))
        ################################loss################################
        with open(csv_filename_iters, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, total_loss_iters.avg])


        # -------------
        # Discriminator
        # -------------
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        real_pred, real_pred_edge = discriminator(ground_truth, GT_gray_image, GT_edge, is_real=True)
        fake_pred, fake_pred_edge = discriminator(output.detach(), GT_gray_image, GT_edge, is_real=False)

        discriminator_loss_dict = discriminator_loss_func(real_pred, fake_pred, real_pred_edge, fake_pred_edge, GT_edge)
        discriminator_loss = discriminator_loss_dict['loss_adversarial']
        discriminator_loss_dict['loss_joint'] = discriminator_loss

        discriminator_optim.zero_grad()
        discriminator_loss.backward()
        discriminator_optim.step()

        # ---
        # log
        # ---
        generator_loss_dict_reduced, discriminator_loss_dict_reduced = reduce_loss_dict(generator_loss_dict), reduce_loss_dict(discriminator_loss_dict)

        pbar_g_loss_hole = generator_loss_dict_reduced['loss_hole'].mean().item()
        pbar_g_loss_valid = generator_loss_dict_reduced['loss_valid'].mean().item()
        pbar_g_loss_perceptual = generator_loss_dict_reduced['loss_perceptual'].mean().item()
        pbar_g_loss_style = generator_loss_dict_reduced['loss_style'].mean().item()
        pbar_g_loss_adversarial = generator_loss_dict_reduced['loss_adversarial'].mean().item()
        pbar_g_loss_intermediate = generator_loss_dict_reduced['loss_intermediate'].mean().item()
        pbar_g_loss_joint = generator_loss_dict_reduced['loss_joint'].mean().item()

        pbar_d_loss_adversarial = discriminator_loss_dict_reduced['loss_adversarial'].mean().item()
        pbar_d_loss_joint = discriminator_loss_dict_reduced['loss_joint'].mean().item()





        if get_rank() == 0:

            pbar.set_description((
                f'g_loss_joint: {pbar_g_loss_joint:.4f} '
                f'd_loss_joint: {pbar_d_loss_joint:.4f}'
            ))



        val_g_loss_holeb =0.0
        val_g_loss_valid =0.0
        val_g_loss_perceptual =0.0
        val_g_loss_style =0.0
        val_g_loss_intermediate =0.0
        val_g_loss_joint =0.0
        psnr_meter = AverageMeter()
        fpsnr_meter = AverageMeter()
        ssim_meter = AverageMeter()
        rmse_meter = AverageMeter()
        rmsew_meter = AverageMeter()
        val_lenth = 0
        if i % 350 == 0:


            epoch += 1
            with open(csv_filename_epoch, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, total_loss_epoch.avg])


            with torch.no_grad():
                generator.eval()

                for j in tqdm(range(val_len)):

                    val_input, val_input_edge, val_input_gray_image, val_ground_truth, val_GT_edge, val_GT_gray_image, val_mask = next(val_image_data_loader)

                    if is_cuda:
                        val_input, val_input_edge, val_input_gray_image, val_ground_truth,val_GT_edge, val_GT_gray_image, val_mask = val_input.cuda(), val_input_edge.cuda(), val_input_gray_image.cuda(), val_ground_truth.cuda(), val_GT_edge.cuda(), val_GT_gray_image.cuda(), val_mask.cuda()

                    output, projected_image, projected_edge = generator(val_input,torch.cat((val_input_edge, val_input_gray_image), dim=1), val_mask)
                    comp = val_ground_truth * val_mask + output * (1 - val_mask)
                    vgg_comp, vgg_output, vgg_ground_truth = extractor(comp), extractor(output), extractor(val_ground_truth)

                    generator_loss_dict = generator_loss_func(
                        val_mask, output, val_ground_truth, val_GT_edge, None,
                        vgg_comp, vgg_output, vgg_ground_truth,
                        projected_image, projected_edge,
                        None, mode="valid"
                    )
                    generator_loss = generator_loss_dict['loss_hole'] * opts.HOLE_LOSS + \
                                     generator_loss_dict['loss_valid'] * opts.VALID_LOSS + \
                                     generator_loss_dict['loss_perceptual'] * opts.PERCEPTUAL_LOSS + \
                                     generator_loss_dict['loss_style'] * opts.STYLE_LOSS + \
                                     generator_loss_dict['loss_intermediate'] * opts.INTERMEDIATE_LOSS
                    generator_loss_dict['loss_joint'] = generator_loss
                    generator_loss_dict_reduced= reduce_loss_dict(generator_loss_dict)

                    pbar_g_loss_hole = generator_loss_dict_reduced['loss_hole'].mean().item()
                    pbar_g_loss_valid = generator_loss_dict_reduced['loss_valid'].mean().item()
                    pbar_g_loss_perceptual = generator_loss_dict_reduced['loss_perceptual'].mean().item()
                    pbar_g_loss_style = generator_loss_dict_reduced['loss_style'].mean().item()
                    pbar_g_loss_intermediate = generator_loss_dict_reduced['loss_intermediate'].mean().item()
                    pbar_g_loss_joint = generator_loss_dict_reduced['loss_joint'].mean().item()


                    val_g_loss_holeb += pbar_g_loss_hole*val_input.size(0)
                    val_g_loss_valid += pbar_g_loss_valid*val_input.size(0)
                    val_g_loss_perceptual += pbar_g_loss_perceptual*val_input.size(0)
                    val_g_loss_style += pbar_g_loss_style*val_input.size(0)
                    val_g_loss_intermediate += pbar_g_loss_intermediate*val_input.size(0)
                    val_g_loss_joint += pbar_g_loss_joint*val_input.size(0)
                    val_lenth += val_input.size(0)
                    imfinal = output * (1 - val_mask) + val_input * val_mask

                    eps = 1e-6

                    psnr = 10 * log10(1 / F.mse_loss(imfinal, val_ground_truth).item())

                    fmse = F.mse_loss(imfinal * (1-val_mask), val_ground_truth * (1-val_mask), reduction='none').sum(dim=[1, 2, 3]) / (
                        (1-val_mask).sum(dim=[1, 2, 3]) * 3 + eps)
                    fpsnr = 10 * torch.log10(1 / fmse).mean().item()
                    ssim = pytorch_ssim.ssim(imfinal, val_ground_truth)

                    psnr_meter.update(psnr, val_input.size(0))
                    fpsnr_meter.update(fpsnr, val_input.size(0))
                    ssim_meter.update(ssim, val_input.size(0))
                    rmse_meter.update(compute_RMSE(imfinal, val_ground_truth, (1-val_mask)), val_input.size(0))
                    rmsew_meter.update(compute_RMSE(imfinal, val_ground_truth, (1-val_mask), is_w=True), val_input.size(0))

                    pbar.set_description((
                        f'g_loss_joint: {pbar_g_loss_joint:.4f} '
                        f'psnr: {psnr_meter.avg:.4f} '
                        f'fpsnr: {fpsnr_meter.avg:.4f} '
                        f'ssim: {ssim_meter.avg:.4f} '
                        f'rmse: {rmse_meter.avg:.4f} '
                        f'rmsew: {rmsew_meter.avg:.4f} '
                    ))




            writer.add_scalar('val/g_loss_hole', val_g_loss_holeb/val_lenth, i)
            writer.add_scalar('val/g_loss_valid', val_g_loss_valid/val_lenth, i)
            writer.add_scalar('val/g_loss_perceptual', val_g_loss_perceptual/val_lenth, i)
            writer.add_scalar('val/g_loss_style', val_g_loss_style/val_lenth, i)
            writer.add_scalar('val/g_loss_intermediate', val_g_loss_intermediate/val_lenth, i)
            writer.add_scalar('val/g_loss_joint', val_g_loss_joint/val_lenth, i)
            writer.add_scalar('val/PSNR', psnr_meter.avg, i)
            writer.add_scalar('val/SSIM', ssim_meter.avg, i)
            writer.add_scalar('val/RMSEw', rmsew_meter.avg, i)

            show_size = 5 if val_input.shape[0] > 5 else val_input.shape[0]

            image_display = torch.cat([
                val_input[0:show_size].detach().cpu(),  # input image
                val_ground_truth[0:show_size].detach().cpu(),  # ground truth
                comp[0:show_size].detach().cpu(),  # refine out
                val_mask[0:show_size].detach().cpu().repeat(1, 3, 1, 1),

            ], dim=0)
            if i % 350 == 0:

                image_dis = torchvision.utils.make_grid(image_display, nrow=show_size)
                writer.add_image('val_Image', image_dis, i)
            metric = psnr_meter.avg
            is_best = True if best_loss < metric else False
            if is_best:
                best_loss = metric

                ##################################loss#############################
                with open(csv_best_epoch, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch, "last_best"])
            if is_best:
                torch.save(
                    {
                        'n_iter': i,
                        'generator': generator_module.state_dict(),
                        'discriminator': discriminator_module.state_dict()
                    },
                    f"{opts.save_dir}/{str(i).zfill(6)}.pt",
                )
            generator.train()



