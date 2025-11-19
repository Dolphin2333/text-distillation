import torch
import torch.nn as nn
import torch.optim as optim
import higher
import random
import numpy as np
from framework.config import get_arch


class Distill(nn.Module):
    """
    ⚙️ NLP 数据蒸馏模块（只支持 SmallNLP）
    ----------------------------------------------------
    功能：
      - 存储并优化可学习的 synthetic text embeddings；
      - 在 inner-loop 中用合成数据训练一个临时 smallnlp；
      - 在 outer-loop 中用真实数据计算 meta loss。
    """

    def __init__(
        self,
        x_init, y_init,
        arch, window, lr, num_train_eval,
        img_pc, batch_pc,
        num_classes=4,  # AGNews 有 4 类
        task_sampler_nc=4,
        train_y=False,
        seq_len=128, embed_dim=128,
        inner_optim='SGD',
        syn_intervention=None, real_intervention=None,
        device=None,
    ):
        super(Distill, self).__init__()

        # ----- 固定 NLP 设定 -----
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.arch = arch
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.img_pc = img_pc
        self.batch_pc = batch_pc
        self.lr = lr
        self.window = window
        self.inner_optim = inner_optim
        self.syn_intervention = syn_intervention
        self.real_intervention = real_intervention
        self.task_sampler_nc = task_sampler_nc
        self.num_train_eval = num_train_eval

        # ----- 合成数据 -----
        flat_dim = seq_len * embed_dim
        self.data = nn.Embedding(img_pc * num_classes, flat_dim).to(self.device)

        # ----- 标签（soft label 可选）-----
        self.train_y = train_y
        if train_y:
            self.label = nn.Embedding(img_pc * num_classes, num_classes).to(self.device)
            self.label.weight.data = y_init.float().to(self.device)
        else:
            # 直接存 tensor
            self.label = y_init.to(self.device)

        # ----- 模型与损失函数 -----
        self.net = get_arch(arch, num_classes, 1, (seq_len, embed_dim)).to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    # =====================================================
    # 子任务采样：随机从所有类别采样若干个小任务
    # =====================================================
    def get_task_indices(self):
        task_indices = list(range(self.num_classes))
        if self.task_sampler_nc < self.num_classes:
            random.shuffle(task_indices)
            task_indices = task_indices[:self.task_sampler_nc]
            task_indices.sort()
        return task_indices

    # =====================================================
    # 从 self.data 中采样合成 batch
    # =====================================================
    def subsample(self):
        """
        从合成数据 self.data 中采样一个 batch：
          - 对每个类别采 batch_pc 个样本；
          - 返回 imgs 和 labels。
        自动处理 GPU / CPU 设备问题。
        """
        indices = []
        task_indices = self.get_task_indices()

        # ---- 按类别采样合成样本 ----
        for i in task_indices:
            ind = torch.randperm(self.img_pc, device=self.data.weight.device)[:self.batch_pc]
            ind = ind.sort()[0] + self.img_pc * i
            indices.append(ind)

        indices = torch.cat(indices).to(self.data.weight.device)
        batch = indices.shape[0]

        # ---- 从 embedding 取出合成句子 ----
        imgs = self.data(indices)  # [B, seq_len * embed_dim]
        imgs = imgs.view(batch, self.seq_len, self.embed_dim).contiguous()

        # ---- 取出对应标签 ----
        if self.train_y:
            labels = self.label(indices).view(batch, self.num_classes).contiguous()
        else:
            labels = self.label[indices]

        return imgs.to(self.device), labels.to(self.device)

    # =====================================================
    # inner-loop：可微的模型训练
    # =====================================================
    def forward(self, x):
      """
      x: [B, seq_len] → real data token ids (LongTensor)
      return: fnet(x) logits
      """
      # 初始化 student 模型
      self.net = get_arch(self.arch, self.num_classes, 1, (self.seq_len, self.embed_dim)).to(self.device)
      self.net.train()

      # inner optimizer
      if self.inner_optim == 'SGD':
          self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
      elif self.inner_optim == 'Adam':
          self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
      else:
          raise NotImplementedError(f"Inner optimizer {self.inner_optim} not supported")

      # 使用 higher 做元训练
      with higher.innerloop_ctx(
          self.net, self.optimizer, copy_initial_weights=True
      ) as (fnet, diffopt):
          for _ in range(self.window):
              imgs, labels = self.subsample()

              # NLP 没有数据增强，兼容接口
              try:
                  imgs = self.syn_intervention(imgs, dtype='syn', seed=random.randint(0, 10000))
              except TypeError:
                  imgs = self.syn_intervention(imgs)

              # ✅ 修复点：模型返回 tuple (logits, hidden)，这里只取 logits
              out, _ = fnet(imgs, is_embedding=True)
              loss = self.criterion(out, labels)
              diffopt.step(loss)

          # outer-loop：在真实数据 x 上计算输出（x 是 token ids）
          try:
              x = self.real_intervention(x, dtype='real', seed=random.randint(0, 10000))
          except TypeError:
              x = self.real_intervention(x)
          x = x.to(self.device)

          # ✅ 同样取出 logits
          out, hidden = fnet(x)
          return out, hidden


    # =====================================================
    # 评估阶段：直接在 synthetic data 上训练并测试
    # =====================================================
    def init_train(self, epochs, init=True):
        if init:
            self.net = get_arch(self.arch, self.num_classes, 1, (self.seq_len, self.embed_dim)).to(self.device)
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        for _ in range(epochs):
            self.optimizer.zero_grad()
            imgs, labels = self.subsample()
            try:
                imgs = self.syn_intervention(imgs, dtype='syn', seed=random.randint(0, 10000))
            except TypeError:
                imgs = self.syn_intervention(imgs)
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            out, _ = self.net(imgs, is_embedding=True)
            loss = self.criterion(out, labels)
            loss.backward()
            self.optimizer.step()

    # =====================================================
    # 测试接口
    # =====================================================
    def test(self, x):
        with torch.no_grad():
            logits, hidden = self.net(x.to(self.device))
        return logits, hidden


def random_indices(y, nclass=4, intraclass=False, device='cuda'):
    """随机打乱索引，用于重排标签。"""
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


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.init as init
# import torch.optim as optim
# import torch.multiprocessing as mp
# import higher

# from copy import deepcopy

# import numpy as np
# import random

# import time


# from framework.config import get_arch

# def _weights_init(m):
#     reset_parameters = getattr(m, "reset_parameters", None)
#     if callable(reset_parameters):
#         init.kaiming_normal_(m.weight)        

        
# class Distill(nn.Module):
#     def __init__(self, x_init, y_init, arch, window, lr, num_train_eval, img_pc, batch_pc, num_classes=2, task_sampler_nc=2, train_y=False, 
#                  channel=3, im_size=(32, 32), inner_optim='SGD', syn_intervention=None, real_intervention=None, cctype=0):
#         super(Distill, self).__init__()
#         self.data = nn.Embedding(img_pc*num_classes, int(channel*np.prod(im_size)))
#         self.train_y = train_y
#         if train_y:
#             self.label = nn.Embedding(img_pc*num_classes, num_classes)
#             self.label.weight.data = y_init.float().cuda()
#         else:
#             self.label = y_init
#         self.num_classes = num_classes
#         self.channel = channel
#         self.im_size = im_size
#         self.net = get_arch(arch, self.num_classes, self.channel, self.im_size)
#         self.img_pc = img_pc
#         self.batch_pc = batch_pc
#         self.arch = arch
#         self.lr = lr
#         self.window = window
#         self.criterion = nn.CrossEntropyLoss(reduction='mean')
#         self.num_train_eval = num_train_eval
#         self.curriculum = window
#         self.inner_optim = inner_optim
#         self.batch_id = 0
#         self.syn_intervention = syn_intervention
#         self.real_intervention = real_intervention
#         self.task_sampler_nc = task_sampler_nc
#         self.cctype = cctype
        
#     # shuffle the data 
#     def shuffle(self):
#         #True
#         self.order_list = torch.randperm(self.img_pc)
#         if self.img_pc >= self.batch_pc:
#             self.order_list = torch.cat([self.order_list, self.order_list], dim=0)
    
#     # randomly sample label sets from the full label set
#     def get_task_indices(self):
#         task_indices = list(range(self.num_classes))
#         if self.task_sampler_nc < self.num_classes:
#             random.shuffle(task_indices)
#             task_indices = task_indices[:self.task_sampler_nc]
#             task_indices.sort()
#         return task_indices    
        
#     def subsample(self):
#         indices = []
#         if self.task_sampler_nc == self.num_classes:
#             for i in range(self.num_classes):
#                 ind = torch.randperm(self.img_pc)[:self.batch_pc].sort()[0] + self.img_pc * i
#                 indices.append(ind)
#         else:
#             task_indices = self.get_task_indices()
#             for i in task_indices:
#                 ind = torch.randperm(self.img_pc)[:self.batch_pc].sort()[0] + self.img_pc * i
#                 indices.append(ind)
#         indices = torch.cat(indices).cuda()
#         imgs    = self.data(indices)
#         imgs = imgs.view(
#                    self.task_sampler_nc * min(self.img_pc, self.batch_pc),
#                    self.channel,
#                    self.im_size[0],
#                    self.im_size[1]
#                ).contiguous()
            
#         if self.train_y:
#             labels    = self.label(indices)
#             labels = labels.view(
#                        self.task_sampler_nc * min(self.img_pc, self.batch_pc),
#                        self.num_classes
#                    ).contiguous()
#         else:
#             labels = self.label[indices]
        
#         return imgs, labels

#     def forward(self, x):
#         self.net = get_arch(self.arch, self.num_classes, self.channel, self.im_size).cuda()
#         self.net.train()
            
#         if self.inner_optim == 'SGD':
#             self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
#             # TODO: add decay rules for SGD
#         elif self.inner_optim == 'Adam':
#             self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
#         if self.dd_type not in ['curriculum', 'standard']:
#             print('The dataset distillation method is not implemented!')
#             raise NotImplementedError()
        
#         if self.dd_type == 'curriculum':
#             for i in range(self.curriculum):
#                 self.optimizer.zero_grad()
#                 imgs, label = self.subsample()
#                 imgs = self.syn_intervention(imgs, dtype='syn', seed=random.randint(0, 10000))
#                 ratio = 0
#                 out, pres = self.net(imgs)
            
#                 loss = self.criterion(out, label)
#                 loss.backward()
#                 self.optimizer.step()
#         with higher.innerloop_ctx(
#                 self.net, self.optimizer, copy_initial_weights=True
#             ) as (fnet, diffopt):
#             for i in range(self.window):
#                 imgs, label = self.subsample()
#                 imgs = self.syn_intervention(imgs, dtype='syn', seed=random.randint(0, 10000))
#                 ratio = 0
#                 out, pres = fnet(imgs)
            
#                 loss = self.criterion(out, label)
#                 diffopt.step(loss)
#             x = self.real_intervention(x, dtype='real', seed=random.randint(0, 10000))
#             return fnet(x)

#     def init_train(self, epoch, init=False, lim=True):
#         if init:
#             self.net = get_arch(self.arch, self.num_classes, self.channel, self.im_size).cuda()
                    
#             if self.inner_optim == 'SGD':
#                 self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
#             elif self.inner_optim == 'Adam':
#                 self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
#             #self.shuffle()
#         for i in range(epoch):
#             self.optimizer.zero_grad()
#             imgs, label = self.subsample()
#             imgs = self.syn_intervention(imgs, dtype='syn', seed=random.randint(0, 10000))
#             out, pres = self.net(imgs)
#             loss = self.criterion(out, label)
#             loss.backward()
#             self.optimizer.step()
    
#     # initialize the EMA with the currect data value
#     def ema_init(self, ema_coef):
#         self.shadow = -1e5
#         self.ema_coef = ema_coef
    
#     # update the EMA value
#     def ema_update(self, grad_norm):
#         if self.shadow == -1e5: 
#             self.shadow = grad_norm
#         else:
#             self.shadow -= (1 - self.ema_coef) * (self.shadow - grad_norm)
#         return self.shadow
    
#     def test(self, x):
#         with torch.no_grad():
#             out = self.net(x)
#         return out
        

# def random_indices(y, nclass=10, intraclass=False, device='cuda'):
#     n = len(y)
#     if intraclass:
#         index = torch.arange(n).to(device)
#         for c in range(nclass):
#             index_c = index[y == c]
#             if len(index_c) > 0:
#                 randidx = torch.randperm(len(index_c))
#                 index[y == c] = index_c[randidx]
#     else:
#         index = torch.randperm(n).to(device)
#     return index
