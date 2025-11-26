import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.multiprocessing as mp
import higher

from copy import deepcopy

import numpy as np
import random

import time


from framework.config import get_arch

def _weights_init(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        init.kaiming_normal_(m.weight)        

        
class Distill(nn.Module):
    def __init__(self, x_init, y_init, arch, window, lr, num_train_eval, img_pc, batch_pc,
                 num_classes=2, task_sampler_nc=2, train_y=False,
                 channel=3, im_size=(32, 32), inner_optim='SGD',
                 syn_intervention=None, real_intervention=None, cctype=0,
                 old_per_class=0, beta=1.0):
        super(Distill, self).__init__()

        # Basic config
        self.arch = arch
        self.lr = lr
        self.window = window
        self.num_train_eval = num_train_eval
        self.num_classes = num_classes
        self.channel = channel
        self.im_size = im_size
        self.batch_pc = batch_pc
        self.img_pc = img_pc
        self.task_sampler_nc = task_sampler_nc
        self.train_y = train_y
        self.inner_optim = inner_optim
        self.syn_intervention = syn_intervention
        self.real_intervention = real_intervention
        self.cctype = cctype
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.curriculum = window
        self.batch_id = 0

        # Synthetic data parameters (embedding table)
        self.data = nn.Embedding(img_pc * num_classes, int(channel * np.prod(im_size)))
        # Initialize from x_init (random or warm-started)
        self.data.weight.data = x_init.float().cuda()

        # Synthetic labels
        if train_y:
            self.label = nn.Embedding(img_pc * num_classes, num_classes)
            self.label.weight.data = y_init.float().cuda()
        else:
            # y_init is assumed to be a tensor of class indices or one-hot
            self.label = y_init

        # Student network architecture
        self.net = get_arch(arch, self.num_classes, self.channel, self.im_size)

        # Boost-DD specific attributes
        # First old_prefix_size synthetic samples belong to previous blocks
        # and will have their gradients rescaled by beta in the train loop.
        self.old_per_class = old_per_class
        self.beta = beta


    def _forward_net(self, net, x):
        out = net(x)
        # If the backbone returns (logits, features), keep the first
        if isinstance(out, tuple):
            return out[0]
        return out
        
    # shuffle the data 
    def shuffle(self):
        #True
        self.order_list = torch.randperm(self.img_pc)
        if self.img_pc >= self.batch_pc:
            self.order_list = torch.cat([self.order_list, self.order_list], dim=0)
    
    # randomly sample label sets from the full label set
    def get_task_indices(self):
        task_indices = list(range(self.num_classes))
        if self.task_sampler_nc < self.num_classes:
            random.shuffle(task_indices)
            task_indices = task_indices[:self.task_sampler_nc]
            task_indices.sort()
        return task_indices    
        
    def subsample(self):
        indices = []
    
        # Decide which classes / tasks we actually sample this batch
        if self.task_sampler_nc == self.num_classes:
            task_indices = list(range(self.num_classes))
        else:
            task_indices = self.get_task_indices()  # may be a subset of classes
    
        for i in task_indices:
            ind = torch.randperm(self.img_pc)[:self.batch_pc].sort()[0] + self.img_pc * i
            indices.append(ind)
    
        indices = torch.cat(indices).cuda()
    
        # How many tasks and how many examples per task we actually have
        n_tasks = len(task_indices)
        n_per_task = min(self.img_pc, self.batch_pc)
    
        imgs = self.data(indices)
        imgs = imgs.view(
            n_tasks * n_per_task,
            self.channel,
            self.im_size[0],
            self.im_size[1]
        ).contiguous()
    
        if self.train_y:
            labels = self.label(indices)
            labels = labels.view(
                n_tasks * n_per_task,
                self.num_classes
            ).contiguous()
        else:
            labels = self.label[indices]
    
        return imgs, labels


    def forward(self, x):
        # Re-initialize the net for inner-loop training
        self.net = get_arch(self.arch, self.num_classes, self.channel, self.im_size).cuda()
        self.net.train()
    
        if self.inner_optim == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        elif self.inner_optim == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
    
        if self.dd_type not in ['curriculum', 'standard']:
            print('The dataset distillation method is not implemented!')
            raise NotImplementedError()
    
        # Optional curriculum warmup
        if self.dd_type == 'curriculum':
            for i in range(self.curriculum):
                self.optimizer.zero_grad()
                imgs, label = self.subsample()
                imgs = self.syn_intervention(imgs, dtype='syn', seed=random.randint(0, 10000))
                logits = self._forward_net(self.net, imgs)
                loss = self.criterion(logits, label)
                loss.backward()
                self.optimizer.step()
    
        # Higher unroll
        with higher.innerloop_ctx(
                self.net, self.optimizer, copy_initial_weights=True
            ) as (fnet, diffopt):
            for i in range(self.window):
                imgs, label = self.subsample()
                imgs = self.syn_intervention(imgs, dtype='syn', seed=random.randint(0, 10000))
                logits = self._forward_net(fnet, imgs)
                loss = self.criterion(logits, label)
                diffopt.step(loss)
    
            x = self.real_intervention(x, dtype='real', seed=random.randint(0, 10000))
            logits_x = self._forward_net(fnet, x)
            # IMPORTANT: return a 2-tuple so `output, _ = model(inputs)` still works
            return logits_x, None

    
    def init_train(self, epoch, init=False, lim=True):
        if init:
            self.net = get_arch(self.arch, self.num_classes, self.channel, self.im_size).cuda()
    
            if self.inner_optim == 'SGD':
                self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            elif self.inner_optim == 'Adam':
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
    
        for i in range(epoch):
            self.optimizer.zero_grad()
            imgs, label = self.subsample()
            imgs = self.syn_intervention(imgs, dtype='syn', seed=random.randint(0, 10000))
            logits = self._forward_net(self.net, imgs)
            loss = self.criterion(logits, label)
            loss.backward()
            self.optimizer.step()

    
    # initialize the EMA with the currect data value
    def ema_init(self, ema_coef):
        self.shadow = -1e5
        self.ema_coef = ema_coef
    
    # update the EMA value
    def ema_update(self, grad_norm):
        if self.shadow == -1e5: 
            self.shadow = grad_norm
        else:
            self.shadow -= (1 - self.ema_coef) * (self.shadow - grad_norm)
        return self.shadow
    def test(self, x):
        with torch.no_grad():
            out = self.net(x)
            if isinstance(out, tuple):
                logits, feat = out
            else:
                logits, feat = out, None
        return logits, feat

        

def random_indices(y, nclass=10, intraclass=False, device='cuda'):
    n = len(y)
    if intraclass:
        index = torch.arange(n).to(device)
        for c in range(nclass):
            index_c = index[y == c]
            if len(index_c) > 0:
                randidx = torch.randperm(len(index_c))
                index[y == c] = index_c[randidx]
    else:
        index = torch.randperm(n).to(device)
    return index

    # def init_train(self, epoch, init=False, lim=True):
    #     if init:
    #         self.net = get_arch(self.arch, self.num_classes, self.channel, self.im_size).cuda()
                    
    #         if self.inner_optim == 'SGD':
    #             self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
    #         elif self.inner_optim == 'Adam':
    #             self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
    #         #self.shuffle()
    #     for i in range(epoch):
    #         self.optimizer.zero_grad()
    #         imgs, label = self.subsample()
    #         imgs = self.syn_intervention(imgs, dtype='syn', seed=random.randint(0, 10000))
    #         out, pres = self.net(imgs)
    #         loss = self.criterion(out, label)
    #         loss.backward()
    #         self.optimizer.step()

    # def forward(self, x):
    #     self.net = get_arch(self.arch, self.num_classes, self.channel, self.im_size).cuda()
    #     self.net.train()
            
    #     if self.inner_optim == 'SGD':
    #         self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
    #         # TODO: add decay rules for SGD
    #     elif self.inner_optim == 'Adam':
    #         self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
    #     if self.dd_type not in ['curriculum', 'standard']:
    #         print('The dataset distillation method is not implemented!')
    #         raise NotImplementedError()
        
    #     if self.dd_type == 'curriculum':
    #         for i in range(self.curriculum):
    #             self.optimizer.zero_grad()
    #             imgs, label = self.subsample()
    #             imgs = self.syn_intervention(imgs, dtype='syn', seed=random.randint(0, 10000))
    #             ratio = 0
    #             out, pres = self.net(imgs)
            
    #             loss = self.criterion(out, label)
    #             loss.backward()
    #             self.optimizer.step()
    #     with higher.innerloop_ctx(
    #             self.net, self.optimizer, copy_initial_weights=True
    #         ) as (fnet, diffopt):
    #         for i in range(self.window):
    #             imgs, label = self.subsample()
    #             imgs = self.syn_intervention(imgs, dtype='syn', seed=random.randint(0, 10000))
    #             ratio = 0
    #             out, pres = fnet(imgs)
            
    #             loss = self.criterion(out, label)
    #             diffopt.step(loss)
    #         x = self.real_intervention(x, dtype='real', seed=random.randint(0, 10000))
    #         return fnet(x)

    # def test(self, x):
    #     with torch.no_grad():
    #         out = self.net(x)
    #     return out