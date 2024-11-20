import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from progress.bar import Bar
import json
import numpy as np
from tensorboardX import SummaryWriter

import torch.optim
import sys,shutil,os
import time
import networks as nets

from utils.osutils import mkdir_p, isdir
from utils.losses import VGGLoss




class BasicModel(object):
    def __init__(self, datasets =(None,None), models = None, args = None, **kwargs):
        super(BasicModel, self).__init__()
        
        self.args = args
        
        # create model
        print("==> creating model ")
        self.model = nets.__dict__[self.args.nets](args=args)
        print("==> creating model [Finish]")
       
        self.train_loader, self.val_loader = datasets
        self.loss = torch.nn.MSELoss()
        
        self.title = args.name
        self.args.checkpoint = os.path.join(args.checkpoint, self.title)
        self.device = torch.device('cuda')
         # create checkpoint dir
        if not isdir(self.args.checkpoint):
            mkdir_p(self.args.checkpoint)

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                            lr=args.lr,
                            betas=(args.beta1,args.beta2),
                            weight_decay=args.weight_decay)  
        
        if not self.args.evaluate:
            self.writer = SummaryWriter(self.args.checkpoint+'/'+'ckpt')
        
        self.best_acc = 0
        self.is_best = False
        self.current_epoch = 0
        self.metric = -100000
        self.hl = 6 if self.args.hl else 1
        self.count_gpu = len(range(torch.cuda.device_count()))

        if self.args.lambda_style > 0:
            # init perception loss
            self.vggloss = VGGLoss(self.args.sltype).to(self.device)

        if self.count_gpu > 1 : # multiple
            # self.model = DataParallelModel(self.model, device_ids=range(torch.cuda.device_count()))
            # self.loss = DataParallelCriterion(self.loss, device_ids=range(torch.cuda.device_count()))
            self.model.multi_gpu()

        self.model.to(self.device)
        self.loss.to(self.device)

        print('==> Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters())/1000000.0))
        print('==> Total devices: %d' % (torch.cuda.device_count()))
        print('==> Current Checkpoint: %s' % (self.args.checkpoint))




