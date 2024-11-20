import torch
import torch.nn as nn
from progress.bar import Bar
import sys, time, os, shutil
import networks as nets
from math import log10
from evaluation import AverageMeter, FScore, compute_RMSE
import torch.nn.functional as F
from utils.losses import VGGLoss, l1_relative, is_dic
from utils.osutils import mkdir_p, isdir
import torchvision
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pytorch_ssim
import cv2
import numpy as np
# from skimage.metrics import peak_signal_noise_ratio
# from skimage.metrics import structural_similarity

import lpips


class Losses(nn.Module):
    def __init__(self, argx, device, norm_func, denorm_func):
        super(Losses, self).__init__()
        self.args = argx
        self.masked_l1_loss = l1_relative
        self.l1_loss = nn.L1Loss()

        if self.args.lambda_content > 0:
            self.vgg_loss = VGGLoss(self.args.sltype, style=self.args.lambda_style > 0).to(device)
        self.gamma = 0.5
        self.norm = norm_func
        self.denorm = denorm_func

    def forward(self, synthesis, pred_ims, target, mask):
        pixel_loss, refine_loss, vgg_loss = [0] * 3
        pred_ims = pred_ims if is_dic(pred_ims) else [pred_ims]

        # reconstruction loss
        pixel_loss += self.masked_l1_loss(pred_ims[-1], target, mask)  # coarse stage
        if len(pred_ims) > 1:
            refine_loss = self.masked_l1_loss(pred_ims[0], target, mask)  # refinement stage

        recov_imgs = [self.denorm(pred_im * mask + (1 - mask) * self.norm(target)) for pred_im in pred_ims]
        pixel_loss += sum([self.l1_loss(im, target) for im in recov_imgs]) * 1.5

        if self.args.lambda_content > 0:
            vgg_loss = [self.vgg_loss(im, target, mask) for im in recov_imgs]
            vgg_loss = sum([vgg['content'] for vgg in vgg_loss]) * self.args.lambda_content + \
                       sum([vgg['style'] for vgg in vgg_loss]) * self.args.lambda_style
        return pixel_loss, refine_loss, vgg_loss


class FFPRN(object):
    def __init__(self, train_datasets=None, val_datasets=None, models=None, args=None, **kwargs):
        super(FFPRN, self).__init__()
        self.args = args

        # 调用了BasicModel中的代码
        print("==> creating model ")
        self.model = nets.__dict__[self.args.nets](args=args)
        print("==> creating model [Finish]")
        self.train_loader = train_datasets
        self.val_loader = val_datasets
        self.device = torch.device('cuda')
        self.loss = Losses(self.args, self.device, self.norm, self.denorm)
        self.loss_fn = lpips.LPIPS(net='vgg', spatial=True)
        self.loss_fn.cuda()
        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        self.model.set_optimizers()
        if self.args.resume != '':
            self.resume(self.args.resume)
        if not isdir(self.args.checkpoint):
            mkdir_p(self.args.checkpoint)

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=args.lr,
                                          betas=(args.beta1, args.beta2),
                                          weight_decay=args.weight_decay)

        if not self.args.evaluate:
            self.writer = SummaryWriter(self.args.checkpoint + '/' + 'ckpt')

        self.best_acc = 0
        self.is_best = False
        self.current_epoch = 0
        self.metric = -100000
        self.model.to(self.device)
        self.loss.to(self.device)

    def train(self, epoch, len_data):
        pbar = range(len_data)
        pbar = tqdm(pbar, initial=len_data, dynamic_ncols=True, smoothing=0.01)

        self.current_epoch = epoch

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_meter = AverageMeter()
        loss_vgg_meter = AverageMeter()
        loss_refine_meter = AverageMeter()
        # switch to train mode
        # self.model.train()

        end = time.time()
        bar = Bar('Processing {} '.format(self.args.nets), max=len_data)
        for i in pbar:
            current_index = len_data * epoch + i
            done_iter = i + 1 + epoch

            inputs, mask, GT = next(self.train_loader)
            # mask_m = 1 - mask
            # inputs = inputs * mask_m
            inputs = inputs.float().to(self.device)
            mask = mask.float().to(self.device)
            GT = GT.float().to(self.device)


            outputs = self.model(self.norm(inputs), self.norm(mask))

            self.model.zero_grad_all()

            coarse_loss, refine_loss, style_loss = self.loss(
                inputs, outputs, self.norm(GT), mask)
            total_loss = self.args.lambda_l1 * (coarse_loss + refine_loss) + style_loss
            # compute gradient and do SGD step
            total_loss.backward()
            self.model.step_all()

            # measure accuracy and record loss
            losses_meter.update(coarse_loss.item(), inputs.size(0))
            if isinstance(refine_loss, int):
                loss_refine_meter.update(refine_loss, inputs.size(0))
            else:
                loss_refine_meter.update(refine_loss.item(), inputs.size(0))

            if self.args.lambda_content > 0 and not isinstance(style_loss, int):
                loss_vgg_meter.update(style_loss.item(), inputs.size(0))

            # measure elapsed timec
            batch_time.update(time.time() - end)
            # end = time.time()
            # plot progress
            pbar.set_description((
                f'loss L1: {losses_meter.avg:.4f} '
                f'loss Refine: {loss_refine_meter.avg:.4f}'
                f'loss VGG: {loss_vgg_meter.avg:.4f}'
            ))


            if self.args.freq > 0 and current_index % self.args.freq == 0:
                self.validate(current_index)
                self.flush()
                self.save_checkpoint()
            if i % 175 == 174:
                self.record('train/loss_L2', losses_meter.avg, current_index)
                self.record('train/loss_Refine', loss_refine_meter.avg, current_index)
                self.record('train/loss_VGG', loss_vgg_meter.avg, current_index)

                bg_pred = self.denorm(outputs[0] * mask + (1 - mask) * self.norm(inputs))
                show_size = 5 if inputs.shape[0] > 5 else inputs.shape[0]
                self.image_display = torch.cat([
                    inputs[0:show_size].detach().cpu(),  # input image
                    GT[0:show_size].detach().cpu(),  # ground truth
                    bg_pred[0:show_size].detach().cpu(),  # refine out
                    mask[0:show_size].detach().cpu().repeat(1, 3, 1, 1),

                ], dim=0)
                image_dis = torchvision.utils.make_grid(self.image_display, nrow=show_size)
                self.writer.add_image('Image', image_dis, current_index)
            if i > len_data:
                print('===================epoch Done===================')
            del outputs

    def validate(self, epoch, val_len_data):

        self.current_epoch = epoch

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_meter = AverageMeter()
        psnr_meter = AverageMeter()
        fpsnr_meter = AverageMeter()
        ssim_meter = AverageMeter()
        rmse_meter = AverageMeter()
        rmsew_meter = AverageMeter()

        coarse_psnr_meter = AverageMeter()
        coarse_rmsew_meter = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        print("===================val time====================")
        bar = Bar('Processing {} '.format(self.args.nets), max=val_len_data)
        with torch.no_grad():
            for i in range(val_len_data):

                current_index = val_len_data * epoch + i

                val_inputs, val_mask, val_GT = next(self.val_loader)
                # val_mask_m = 1 - val_mask
                # val_inputs = val_inputs * val_mask_m
                val_inputs = val_inputs.float().to(self.device)
                val_mask = val_mask.float().to(self.device)
                val_GT = val_GT.float().to(self.device)

                outputs = self.model(self.norm(val_inputs), self.norm(val_mask))
                imoutput = outputs

                if len(outputs) > 1:
                    imcoarse = imoutput[1]
                    imcoarse = imcoarse * val_mask + val_inputs * (1 - val_mask)
                else:
                    imcoarse = None
                imoutput = imoutput[0] if is_dic(imoutput) else imoutput

                imfinal = self.denorm(imoutput * val_mask + self.norm(val_inputs) * (1 - val_mask))

                eps = 1e-6

                psnr = 10 * log10(1 / F.mse_loss(imfinal, val_GT).item())

                fmse = F.mse_loss(imfinal * val_mask, val_GT * val_mask, reduction='none').sum(dim=[1, 2, 3]) / (
                        val_mask.sum(dim=[1, 2, 3]) * 3 + eps)
                fpsnr = 10 * torch.log10(1 / fmse).mean().item()
                ssim = pytorch_ssim.ssim(imfinal, val_GT)
                if imcoarse is not None:
                    psnr_coarse = 10 * log10(1 / F.mse_loss(imcoarse, val_GT).item())
                    rmsew_coarse = compute_RMSE(imcoarse, val_GT, val_mask, is_w=True)
                    coarse_psnr_meter.update(psnr_coarse, val_inputs.size(0))
                    coarse_rmsew_meter.update(rmsew_coarse, val_inputs.size(0))

                psnr_meter.update(psnr, val_inputs.size(0))
                fpsnr_meter.update(fpsnr, val_inputs.size(0))
                ssim_meter.update(ssim, val_inputs.size(0))
                rmse_meter.update(compute_RMSE(imfinal, val_GT, val_mask), val_inputs.size(0))
                rmsew_meter.update(compute_RMSE(imfinal, val_GT, val_mask, is_w=True), val_inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                if imcoarse is None:
                    suffix = '({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | PSNR: {psnr:.4f} | fPSNR: {fpsnr:.4f} | SSIM: {ssim:.4f} | RMSE: {rmse:.4f} | RMSEw: {rmsew:.4f}'.format(
                        batch=i + 1,
                        size=val_len_data,
                        data=data_time.val,
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        psnr=psnr_meter.avg,
                        fpsnr=fpsnr_meter.avg,
                        ssim=ssim_meter.avg,
                        rmse=rmse_meter.avg,
                        rmsew=rmsew_meter.avg
                    )
                else:
                    suffix = '({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | CPSNR: {cpsnr:.4f} | CRMSEw: {crmsew:.4f} | PSNR: {psnr:.4f} | fPSNR: {fpsnr:.4f} | RMSE: {rmse:.4f} | RMSEw: {rmsew:.4f} | SSIM: {ssim:.4f}'.format(
                        batch=i + 1,
                        size=val_len_data,
                        data=data_time.val,
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        cpsnr=coarse_psnr_meter.avg,
                        crmsew=coarse_rmsew_meter.avg,
                        psnr=psnr_meter.avg,
                        fpsnr=fpsnr_meter.avg,
                        ssim=ssim_meter.avg,
                        rmse=rmse_meter.avg,
                        rmsew=rmsew_meter.avg
                    )
                if i % 350 == 9:
                    print(suffix)
                # bar.next()
        print("Total:")
        print(suffix)
        bar.finish()

        print("Iter:%s,losses:%s,PSNR:%.4f,SSIM:%.4f" % (epoch, losses_meter.avg, psnr_meter.avg, ssim_meter.avg))
        self.record('val/loss_L2', losses_meter.avg, epoch)
        self.record('val/PSNR', psnr_meter.avg, epoch)
        self.record('val/SSIM', ssim_meter.avg, epoch)
        self.record('val/RMSEw', rmsew_meter.avg, epoch)
        show_size = 5 if val_inputs.shape[0] > 5 else val_inputs.shape[0]
        self.image_display = torch.cat([
            val_inputs[0:show_size].detach().cpu(),  # input image
            val_GT[0:show_size].detach().cpu(),  # ground truth
            imfinal[0:show_size].detach().cpu(),  # refine out
            val_mask[0:show_size].detach().cpu().repeat(1, 3, 1, 1),

        ], dim=0)
        image_dis = torchvision.utils.make_grid(self.image_display, nrow=show_size)
        self.writer.add_image('val_Image', image_dis, current_index)
        self.metric = psnr_meter.avg

        self.model.train()

    def test(self, val_len_data):
        self.model.eval()
        start = time.time()
        print("==> testing VM model ")
        psnr_meter = AverageMeter()
        fpsnr_meter = AverageMeter()
        LPIPS_meter = []
        ssim_meter = AverageMeter()
        rmse_meter = AverageMeter()
        rmsew_meter = AverageMeter()

        prediction_dir = os.path.join(self.args.checkpoint, 'np-r')
        if not os.path.exists(prediction_dir): os.makedirs(prediction_dir)

        # save_flag = True
        with torch.no_grad():
            for i in range(val_len_data):
                # val_inputs, val_mask= next(self.val_loader)
                val_inputs, val_mask, val_GT = next(self.val_loader)
                # val_mask_m = 1 - val_mask
                # val_inputs = val_inputs * val_mask_m
                val_inputs = val_inputs.float().to(self.device)
                val_mask = val_mask.float().to(self.device)
                val_GT = val_GT.float().to(self.device)
                # select the outputs by the giving arch
                # outputs = self.model(self.norm(val_inputs), self.norm(val_mask))
                outputs = self.model(self.norm(val_inputs))
                imoutput = outputs
                imoutput = imoutput[0] if is_dic(imoutput) else imoutput
                imfinal = self.denorm(imoutput * val_mask + self.norm(val_inputs) * (1 - val_mask))
                eps = 1e-6
                # LPIPS
                LPIPS = self.loss_fn.forward(val_GT, imfinal)
                LPIPS_meter.append(LPIPS.mean().item())


                psnr = 10 * log10(1 / F.mse_loss(imfinal, val_GT).item())


                fmse = F.mse_loss(imfinal * val_mask, val_GT * val_mask, reduction='none').sum(dim=[1, 2, 3]) / (
                        val_mask.sum(dim=[1, 2, 3]) * 3 + eps)
                fpsnr = 10 * torch.log10(1 / fmse).mean().item()
                ssim = pytorch_ssim.ssim(imfinal, val_GT)

                psnr_meter.update(psnr, val_inputs.size(0))
                fpsnr_meter.update(fpsnr, val_inputs.size(0))
                ssim_meter.update(ssim, val_inputs.size(0))
                rmse_meter.update(compute_RMSE(imfinal, val_GT, val_mask), val_inputs.size(0))
                rmsew_meter.update(compute_RMSE(imfinal, val_GT, val_mask, is_w=True), val_inputs.size(0))
                self.save_output(
                    inputs={'I': val_inputs, 'mask': val_mask, 'GT': val_GT},
                    preds={'bg': imfinal},
                    save_dir=prediction_dir,
                    num=i,
                    verbose=False)
        print("PSNR:%.4f,SSIM:%.4f,LPIPS:%.4f,RMSE:%.4f,RMSER:%.4f" % (
        psnr_meter.avg, ssim_meter.avg, sum(LPIPS_meter) / val_len_data, rmse_meter.avg, rmsew_meter.avg))
        self.record('test/PSNR', psnr_meter.avg, i)
        self.record('test/SSIM', ssim_meter.avg, i)
        self.record('test/RMSE', rmse_meter.avg, i)
        self.record('test/RMSEw', rmsew_meter.avg, i)
        print(time.time() - start)

    def resume(self, resume_path):
        # if isfile(resume_path):
        if not os.path.exists(resume_path):
            resume_path = os.path.join(self.args.checkpoint, 'checkpoint.pth.tar')
        if not os.path.exists(resume_path):
            raise Exception("=> no checkpoint found at '{}'".format(resume_path))
        print(resume_path)
        print("=> loading checkpoint '{}'".format(resume_path))
        current_checkpoint = torch.load(resume_path)
        if isinstance(current_checkpoint['state_dict'], torch.nn.DataParallel):
            current_checkpoint['state_dict'] = current_checkpoint['state_dict'].module

        if isinstance(current_checkpoint['optimizer'], torch.nn.DataParallel):
            current_checkpoint['optimizer'] = current_checkpoint['optimizer'].module

        # if self.args.start_epoch == 0:
        self.args.start_epoch = current_checkpoint['epoch']
        self.metric = current_checkpoint['best_acc']
        items = list(current_checkpoint['state_dict'].keys())
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_path, current_checkpoint['epoch']))
        ## restore the learning rate
        lr = self.args.lr
        # for epoch in self.args.schedule:
        #     if epoch <= self.args.start_epoch:
        #         lr *= self.args.gamma
        optimizers = [getattr(self.model, attr) for attr in dir(self.model) if
                      attr.startswith("optimizer") and getattr(self.model, attr) is not None]
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # ---------------- Load Model Weights --------------------------------------
        self.model.load_state_dict(current_checkpoint['state_dict'], strict=True)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_path, current_checkpoint['epoch']))

    def save_output(self, inputs, preds, save_dir, num, verbose=False, alpha=0.5):
        outs = []

        # GT model
        # image, bg_gt, mask_gt = inputs['I'], inputs['bg'], inputs['mask']
        # no GT model
        image, mask_gt, gt = inputs['I'], inputs['mask'], inputs['GT']

        image = cv2.cvtColor(self.tensor2np(image)[0], cv2.COLOR_RGB2BGR)
        # fg_gt = cv2.cvtColor(tensor2np(fg_gt)[0], cv2.COLOR_RGB2BGR)

        # no GT model will with "#"
        # bg_gt = cv2.cvtColor(self.tensor2np(bg_gt)[0], cv2.COLOR_RGB2BGR)
        mask_gt = self.tensor2np(mask_gt, isMask=True)[0]

        gt = cv2.cvtColor(self.tensor2np(gt)[0], cv2.COLOR_RGB2BGR)

        bg_pred = preds['bg']
        # fg_pred = cv2.cvtColor(tensor2np(fg_pred)[0], cv2.COLOR_RGB2BGR)
        bg_pred = cv2.cvtColor(self.tensor2np(bg_pred)[0], cv2.COLOR_RGB2BGR)
        # GT model
        # outs = [image, bg_gt, bg_pred, mask_gt]  # , main_mask]
        # no GT model
        outs = [image, bg_pred, gt, mask_gt]
        outimg = np.concatenate(outs, axis=1)

        if verbose == True:
            # print("show")
            cv2.imshow("out", outimg)
            cv2.waitKey(0)
        else:
            filrname = str(num) + ".jpg"
            out_fn = os.path.join(save_dir, filrname)
            cv2.imwrite(out_fn, outimg)

    def tensor2np(self, x, isMask=False):
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

    def save_checkpoint(self, filename='checkpoint.pth.tar', snapshot=None):
        is_best = True if self.best_acc < self.metric else False

        if is_best:
            self.best_acc = self.metric

        state = {
            'epoch': self.current_epoch + 1,
            'nets': self.args.nets,
            'state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
        }

        filepath = os.path.join(self.args.checkpoint, filename)
        torch.save(state, filepath)

        if snapshot and state['epoch'] % snapshot == 0:
            shutil.copyfile(filepath, os.path.join(self.args.checkpoint, 'checkpoint_{}.pth.tar'.format(state.epoch)))

        if is_best:
            self.best_acc = self.metric
            print('Saving Best Metric with PSNR:%s' % self.best_acc)
            if not os.path.exists(self.args.checkpoint): os.makedirs(self.args.checkpoint)
            shutil.copyfile(filepath, os.path.join(self.args.checkpoint, 'model_best.pth.tar'))

    def denorm(self, x):
        if self.args.gan_norm:
            return (x + 1.0) / 2.0
        else:
            return x

    def norm(self, x):
        if self.args.gan_norm:
            return x * 2.0 - 1.0
        else:
            return x

    def flush(self):
        self.writer.flush()
        sys.stdout.flush()

    def record(self, k, v, epoch):
        self.writer.add_scalar(k, v, epoch)

    def iterlen(self, x):
        n = 0
        try:
            while True:
                next(x)
                n += 1
        except StopIteration:
            pass
        return n