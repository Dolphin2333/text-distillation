import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset
from torch.optim.lr_scheduler import MultiStepLR

import torch_optimizer


import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.nn.attention import sdpa_kernel, SDPBackend

from PIL import Image

import os
import sys
import time
import types
import numpy as np
import shutil
import warnings
import random
import math

import h5py

from framework.config import get_config, get_arch, get_dataset, get_transform, get_pin_memory, DistillDataset
from framework.distill_higher import Distill
from framework.util import Summary, AverageMeter, ProgressMeter, accuracy, accuracy_ind, ImageIntervention, init_gaussian

from statistics import mean
import wandb

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'


def main_worker(gpu, ngpus_per_node, args):
    # TODO: CHANGED IN STEP 5: ADDED THIS LINE
    with sdpa_kernel(SDPBackend.MATH):
        global best_acc1, best_loss1
        best_acc1 = 0
        best_loss1 = 1000
        args.gpu = gpu
        if args.gpu is not None:
            print("Use GPU: {} for training".format(args.gpu))
        if args.distributed:
            if args.dist_url == "env://" and args.rank == -1:
                args.rank = int(os.environ["RANK"])
            if args.mp_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                args.rank = args.rank * ngpus_per_node + gpu
            
            dist.init_process_group(backend=args.dist_backend,
                                    world_size=args.world_size, rank=args.rank)
    
        torch.manual_seed(args.rank + args.seed)
        np.random.seed(args.rank + args.seed)
        cudnn.benchmark = True
        cudnn.deterministic = True
        print("Torch Seed Specified with rank: %d"%(args.rank+args.seed))
        args.data_root = os.path.join(args.root, args.dataset)
        print("Dataset: %s"%args.dataset)
        print("Dataset Path: %s"%args.root)
        print(args)
        
        pathdir = './train_log/{}/{}_{}_Adam{}'.format(args.dataset, args.epochs, args.batch_size, int(args.lr*1000)) 
            
        # 0. Preprocess datasets
        print('==> Preparing data..')
        transform_train, transform_test = get_transform(args.dataset)
        print(transform_train, transform_test)
    
    
    # EDITED PART    # EDITED PART    # EDITED PART    # EDITED PART    # EDITED PART    # EDITED PART    # EDITED PART    
        train1, train2, testset, num_classes, shape, _ = get_dataset(
            args.dataset, args.data_root, transform_train, transform_test, zca=args.zca)
        print('Dataset: number of classes: {}'.format(num_classes))
        args.num_classes = num_classes
        
    
        if args.limit_train: train1 = torch.utils.data.Subset(train1, range(5000))
        print('Training set size: {}'.format(len(train1)))
        # Not finished! Use DDP for dataloader
        # Since difference processes are intialized with different random seeds, we do not need to use different data
        # We use the same data for all processes, without defining the sampler.
        if args.distributed:
            train_sampler = None #DistributedSampler(train1)
        #    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        #    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        else:
            train_sampler = None
        val_sampler = None
        
        train_loader = torch.utils.data.DataLoader(
                train1, batch_size=args.batch_size, shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(
                testset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True, sampler=val_sampler)
        
        # # Initialize Distilled Dataset
        # channel, height, width = shape
        # print('Image size: channel {}, height {}, width {}'.format(channel, height, width))
        # x_init = torch.randn(args.num_per_class * num_classes, channel*height*width)
        # y_list = list(range(0, num_classes)) * args.num_per_class
        # y_list.sort()
        # y_init = torch.tensor(y_list)
        # y_tmp = torch.zeros(num_classes)
        # # if train_y with soft label, change it to one_hot.   
        # if args.train_y:
        #     y_init = F.one_hot(y_init, num_classes)
        # Initialize Distilled Dataset
        channel, height, width = shape
        print('Image size: channel {}, height {}, width {}'.format(channel, height, width))
    
        # Start from random synthetic embeddings
        x_init = torch.randn(args.num_per_class * num_classes, channel * height * width)
    
        # Integer labels repeated per class
        y_list = list(range(0, num_classes)) * args.num_per_class
        y_list.sort()
        y_init = torch.tensor(y_list)
        y_tmp = torch.zeros(num_classes)
    
        # If train_y with soft label, change it to one_hot.
        if args.train_y:
            y_init = F.one_hot(y_init, num_classes)
    
    
        old_per_class = 0
        if hasattr(args, "boost_init_from") and args.boost_init_from != 'none':
            print(f'Boost-DD warm start from {args.boost_init_from}')
            db = h5py.File(args.boost_init_from, 'r')
    
            prev_data = torch.tensor(db['data'][:]).float()
            prev_n, prev_dim = prev_data.shape
            total_n, total_dim = x_init.shape
    
            if prev_dim != total_dim:
                raise ValueError(
                    f'Boost-DD warm-start: dimension mismatch between prev_data ({prev_dim}) and current x_init ({total_dim})'
                )
    
            # infer IPCs
            if prev_n % num_classes != 0:
                raise ValueError(
                    f'Boost-DD warm-start: prev_n={prev_n} is not divisible by num_classes={num_classes}'
                )
            prev_ipc = prev_n // num_classes
            curr_ipc = args.num_per_class
            if prev_ipc > curr_ipc:
                raise ValueError(
                    f'Boost-DD warm-start: prev_ipc={prev_ipc} > curr_ipc={curr_ipc}'
                )
    
            if total_n != curr_ipc * num_classes:
                raise ValueError(
                    f'Boost-DD: current synthetic table size {total_n} != curr_ipc*num_classes={curr_ipc*num_classes}'
                )
    
            # Copy per class:
            # for each class c, copy its [prev_ipc] rows into the first prev_ipc
            # positions of that class's block in the new table.
            for c in range(num_classes):
                src_start = c * prev_ipc
                src_end = src_start + prev_ipc
    
                dst_start = c * curr_ipc
                dst_end = dst_start + prev_ipc
    
                x_init[dst_start:dst_end] = prev_data[src_start:src_end]
    
            old_per_class = prev_ipc
    
            db.close()
    
            print(
                f'Boost-DD: warmed start prev_ipc={prev_ipc} per class; '
                f'curr_ipc={curr_ipc} per class; num_classes={num_classes}'
            )
    
        # store how many *per class* entries come from previous stages
        args.old_per_class = old_per_class
    
    
        
        # Data augmentations for synthetic data and the real data
        syn_intervention, real_intervention, interv_prob = set_up_interventions(args)
        print('Synthetic images, not_single {}, keys {}'.format(syn_intervention.not_single, syn_intervention.keys))
        args.syn_intervention = syn_intervention
        args.real_intervention = real_intervention
        args.shape = shape
        
        # 1. Initialize Distilled Dataset Module
        print('==> Building model..')
        print('Initialized data with size, x: {}, y:{}'.format(x_init.shape, y_init.shape))
        model = Distill(
            x_init,
            y_init,
            args.arch,
            args.window,
            args.inner_lr,
            args.num_train_eval,
            img_pc=args.num_per_class,
            batch_pc=args.batch_per_class,
            train_y=args.train_y,
            channel=shape[0],
            num_classes=num_classes,
            task_sampler_nc=args.task_sampler_nc,
            im_size=(shape[1], shape[2]),
            inner_optim=args.inner_optim,
            syn_intervention=syn_intervention,
            real_intervention=real_intervention,
            cctype=args.cctype,
            old_per_class=getattr(args, 'old_per_class', 0),
            beta=getattr(args, 'boost_beta', 1.0),
            width=args.width,
        )
    
        print(model.net)
        
        # The data is intialized with random values, not from real images to remove any bias
        # We project the data to a unit ball to control the norm.
        # After model is built, BEFORE training
        with torch.no_grad():
            # model might be wrapped (DDP/DP) or not
            m = model.module if hasattr(model, "module") else model
        
            w = m.data.weight.data  # shape: (curr_ipc * num_classes, dim)
        
            opc = getattr(args, "old_per_class", 0)
            if opc > 0:
                ipc = args.num_per_class
                C = args.num_classes
        
                mask = torch.ones(w.size(0), dtype=torch.bool, device=w.device)
                for c in range(C):
                    start = c * ipc
                    mask[start : start + opc] = False  # old rows → DO NOT PROJECT
        
                w[mask] = project(w[mask])            # project only new rows
                m.data.weight.data = w
            else:
                m.data.weight.data = project(w)
    
    
        
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
        elif args.distributed:
            if torch.cuda.is_available():
                if args.gpu is not None:
                    torch.cuda.set_device(args.gpu)
                    model.cuda(args.gpu)
                    if not args.train_y: model.label = model.label.cuda(args.gpu)
                    # we could use the same data across all gpus?
                    args.batch_size = int(args.batch_size / ngpus_per_node)
                    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
                else:
                    model.cuda()
                    if not args.train_y: model.label = model.label.cuda()
                    # DistributedDataParallel will divide and allocate batch_size to all
                    # available GPUs if device_ids are not set
                    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        elif args.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
            if not args.train_y: model.label = model.label.cuda(args.gpu)
            print('use direct cuda')
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            print('use data parallel only')
            if not args.train_y: model.label = model.label.cuda()
            model = torch.nn.DataParallel(model).cuda()
        
        if torch.cuda.is_available():
            if args.gpu:
                device = torch.device('cuda:{}'.format(args.gpu))
            else:
                device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
        model.module.dd_type = args.ddtype
        continue_training = False
        
        # if ckptname is given, load the data from the file as initialization
        if args.ckptname != 'none':
            db = h5py.File(args.ckptname, 'r')
            print(db['data'].shape[0], int(x_init.shape[0]/num_classes), args.num_per_class)
            base_data = torch.tensor(db['data'][:]).cuda()
            if args.train_y:
                label_data = torch.tensor(db['label'][:]).cuda()
                model.module.label.weight.data = label_data
            continue_training = True
            
        start_test_epoch = 0 if continue_training else 1
            
        if args.train_y:
            if args.outer_optim == 'Adam':
                optimizer = optim.Adam([{'params': model.module.data.weight}, 
                                       {'params': model.module.label.weight, 'lr': args.lr/args.label_lr_scale}], 
                                       lr=args.lr, betas=(0.9, 0.999), eps=args.eps, weight_decay=args.wd)
            else:
                raise NotImplementedError()
        else:
            #optimizer = optim.SGD([model.module.data], lr=args.lr, momentum=0.5, weight_decay=0)
            optimizer = optim.Adam([model.module.data.weight], lr=args.lr, betas=(0.9, 0.999), eps=args.eps, weight_decay=args.wd)
            
            
        # how to visualize in DDP?
        best_rec = {}
        grad_acc = []
        best_loss_ind = 0
        
        distill_steps = 0
        if args.ddtype == 'curriculum' and args.cctype != 2: 
            model.module.curriculum = [args.totwindow-args.window, args.minwindow, 0, 0, 0, 0][args.cctype]
        
        if model.module.data.weight.get_device() == 0 and args.wandb:
            wandb.init(
                    # set the wandb project where this run will be logged
                    project="Distributed-Distillation",
                    name = args.name,
                    config=vars(args),
                    settings=wandb.Settings(start_method="fork")) 
        
        # if args.ddtype == 'standard':
        #     fname = './grad_save_init_IPC_'+str(args.num_per_class)+'_no_curr_unroll_'+str(args.window)+args.fname+'.h5'
        if args.ddtype == 'standard':
            # fname = f'./out_step2_{args.dataset}_{args.arch}_ipc{args.num_per_class:02d}_s{args.stage}.h5'
            fname = f'./out_step5_{args.fname}.h5'
    
        else:
            fname = './save/{}/IPC_{}_{}_curr_unroll_{}_{}_{}_{}.h5'.format(str(args.dataset), str(args.num_per_class), str(args.cctype), str(args.window), str(args.totwindow), str(args.minwindow), args.fname)
        
        if args.load_ckpt:
            checkpoint = torch.load(fname[:-3]+'.pth')
            # Load the model and optimizer state_dicts
            base_data = checkpoint['model_state_dict']['module.data.weight'].cuda()
            model.module.data.weight.data = base_data
            if args.train_y:
                label_data = checkpoint['label_state_dict']['module.label.weight'].cuda()
                model.module.label.weight.data = label_data
            if not args.train_y: 
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                args.start_epoch = checkpoint['epoch']
                distill_steps = args.start_epoch * int(50000 / args.batch_size)
            print('=== Successfully loading the data from {} ==='.format(fname[:-3]+'.pth'))
            model.module.ema_init(args.clip_coef)
            ### TO DO: add EMA loading for distilled data
    
        for epoch in range(args.start_epoch, args.epochs):
            # initialize the EMA
            if epoch == 0: model.module.ema_init(args.clip_coef)  
            grad_tmp, losses_avg, distill_steps = train(train_loader, model, criterion, optimizer, epoch, device, distill_steps, args)
            grad_acc.append(grad_tmp)
            print('The current update step is {}'.format(distill_steps))
    
            # evaluate on validation set
            if (epoch - args.start_epoch + start_test_epoch) % args.test_freq == 0:
                if model.module.data.weight.get_device() == 0: print('The current seed is {}'.format(torch.seed()))
                if model.module.data.weight.get_device() == 0: print('The current lr is: {}'.format(model.module.lr))
                if model.module.data.weight.get_device() == 0: print('Testing Results:')
                test_acc, test_loss, scores = test([test_loader, train_loader], model, criterion, args)
                if model.module.data.weight.get_device() == 0: print(test_acc)
                tmp_index = test_acc[0].index(max(test_acc[0]))
                            
                if model.module.data.weight.get_device() == 0 and args.wandb:
                    wandb.log({"loss": test_loss, "epoch": int(epoch), 'distill_steps':distill_steps, "grad_norm": grad_tmp[-1],
                            "train_acc": test_acc[1][-1], "test_acc":test_acc[0][-1], "curr": model.module.curriculum})
                
            # remember best acc@1 and save checkpoint
                is_best = test_acc[0][tmp_index] > best_acc1
                if is_best:
                    best_acc1 = max(test_acc[0][tmp_index], best_acc1)
                    if model.module.data.weight.get_device() == 0:
                        best_rec['acc'] = test_acc[0][tmp_index]
                        best_rec['test'] = test_acc[0]
                        best_rec['train'] = test_acc[0]
                        best_rec['ind'] = tmp_index
                        best_rec['epoch'] = epoch + 1
                        best_rec['data'] = model.module.data.weight.data.cpu().clone().numpy()
                        if args.train_y:
                            best_rec['label'] = model.module.label.weight.data.cpu().clone().numpy()
                
                if model.module.data.weight.get_device() == 0:
                    with h5py.File(fname, 'w') as f:
                        f.create_dataset('data', data=best_rec['data'])
                        f.create_dataset('epoch', data=best_rec['epoch'])
                        if args.train_y:
                            f.create_dataset('label', data=best_rec['label'])
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': {'module.data.weight': model.state_dict()['module.data.weight'].cpu().clone()},
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    if args.train_y:
                        checkpoint['label_state_dict'] = {'module.label.weight': model.state_dict()['module.label.weight'].cpu().clone()}
                    torch.save(checkpoint, fname[:-3]+'.pth')
                # update curriculum
                
                if test_loss < best_loss1:
                    best_loss1 = test_loss
                    best_loss_ind = epoch
                else:
                    if epoch >= best_loss_ind + 60:
                        best_loss_ind = epoch
                        if args.ddtype == 'curriculum': 
                            if args.cctype == 0:
                                if model.module.curriculum == args.minwindow: break
                            elif args.cctype == 1:
                                if model.module.curriculum == args.totwindow-args.window: break
                                model.module.curriculum += args.window
                                model.module.curriculum = min(args.totwindow-args.window, model.module.curriculum)
                print('train loss {}, epoch {}, best loss {}, best_epoch {}'.format(test_loss, epoch, best_loss1, best_loss_ind))
                
    
        if model.module.data.weight.get_device() == 0: 
            print('=== Final results:')
            print(best_rec)
            if args.wandb: wandb.log({"best_acc": best_rec['acc']})
        grad_acc = np.concatenate(grad_acc)
        if model.module.data.weight.get_device() == 0:
            with h5py.File(fname, 'w') as f:
                f.create_dataset('grad', data=grad_acc)
                best_rec['data'] = model.module.data.weight.data.cpu().clone().numpy()
                f.create_dataset('data', data=best_rec['data'])
    
def train(train_loader, model, criterion, optimizer, epoch, device, distill_steps, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    model.module.net.train()

    end = time.time()
    grad_acc = []
    if model.module.cctype == 2:
        if args.rank == 0:
            shared_curriculum = torch.tensor(random.randint(args.minwindow, args.totwindow-args.window))
        else:
            shared_curriculum = torch.tensor(0)
        shared_curriculum = shared_curriculum.to(device)
        if args.mp_distributed:
            dist.broadcast(shared_curriculum, src=0)
        
        model.module.curriculum = shared_curriculum.item()

        
    print('GPU_{}_using curriculum {} with window {}'.format(args.rank, model.module.curriculum, model.module.window))
    
    
    for train1 in enumerate(train_loader):
        
        if args.complete_random:
            if model.module.cctype == 2:
                if args.rank == 0:
                    shared_curriculum = torch.tensor(random.randint(args.minwindow, args.totwindow-args.window))
                else:
                    shared_curriculum = torch.tensor(0)
                shared_curriculum = shared_curriculum.to(device)
                if args.mp_distributed:
                    dist.broadcast(shared_curriculum, src=0)

                model.module.curriculum = shared_curriculum.item()

        
            print('GPU_{}_using curriculum {} with window {}'.format(args.rank, model.module.curriculum, model.module.window))
            
        data_time.update(time.time() - end)
        
        i, (inputs, targets) = train1
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        output, _ = model(inputs)

        loss = criterion(output, targets)

        acc = accuracy(output, targets)
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc, inputs.size(0))

        # # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()
        # for clear_cache in range(5):
        #     torch.cuda.empty_cache()
                      
        # grad_norm = calculate_grad_norm(torch.norm(optimizer.param_groups[0]['params'][0].grad.clone().detach(), dim=1))

        # grad_acc.append(grad_norm)
        # # obtain the ema norm and perform gradient clipping
        # clip_coef = model.module.ema_update((torch.norm(optimizer.param_groups[0]['params'][0].grad.clone().detach(), dim=1)**2).sum().item() ** 0.5)

        # torch.nn.utils.clip_grad_norm_(
        #             model.module.data.weight, max_norm=clip_coef * 2)
        
        
        # optimizer.step()
        # TODO: RECOMMENDED CHANGES (2 LINES)
        m = model.module if hasattr(model, "module") else model
        w_before = m.data.weight.data.detach().clone()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # TODO: RECOMMENDED CHANGES: COMMMENTED OUT THIS IF
        # if hasattr(model.module, 'old_per_class') and model.module.old_per_class > 0:
        #     g = model.module.data.weight.grad
        #     if g is not None:
        #         old_ipc = model.module.old_per_class
        #         ipc = model.module.img_pc
        #         num_classes = model.module.num_classes
        #         beta = getattr(model.module, 'beta', 1.0)

        #         # For each class c, indices [c*ipc : c*ipc + old_ipc] are old points
        #         for c in range(num_classes):
        #             start = c * ipc
        #             end = start + old_ipc
        #             if beta == 0.0:
        #                 g[start:end] = 0
        #             elif beta != 1.0:
        #                 g[start:end] = g[start:end] * beta


        for clear_cache in range(5):
            torch.cuda.empty_cache()
                       
        grad_norm = calculate_grad_norm(
            torch.norm(optimizer.param_groups[0]['params'][0].grad.clone().detach(), dim=1)
        )

        grad_acc.append(grad_norm)
        # obtain the ema norm and perform gradient clipping
        clip_coef = model.module.ema_update(
            (torch.norm(optimizer.param_groups[0]['params'][0].grad.clone().detach(), dim=1) ** 2).sum().item() ** 0.5
        )

        torch.nn.utils.clip_grad_norm_(
            model.module.data.weight, max_norm=clip_coef * 2
        )
        
        optimizer.step()

        # TODO: RECOMMENDED CHANGE: ADD ADAM LR
        # Adam-faithful stale LR for old synthetic blocks (Boost-DD)
        m = model.module if hasattr(model, "module") else model
        old_ipc = getattr(m, "old_per_class", 0)
        beta = getattr(m, "beta", 1.0)

        if old_ipc > 0 and beta != 1.0:
            ipc = m.img_pc
            num_classes = m.num_classes
            with torch.no_grad():
                w_after = m.data.weight.data
                for c in range(num_classes):
                    start = c * ipc
                    end = start + old_ipc
                    # shrink the Adam step on old rows:
                    # w_old = w_before + beta * (w_after - w_before)
                    w_after[start:end] = (
                        w_before[start:end]
                        + beta * (w_after[start:end] - w_before[start:end])
                    )




        optimizer.zero_grad()
        model.module.net.zero_grad()
        
        if args.train_y:
             with torch.no_grad():
                model.module.label.weight.data = torch.clip(model.module.label.weight.data, min=0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        distill_steps += 1
        torch.cuda.empty_cache()
        
        # print the results
        if (i+1) % args.print_freq == 0 and model.module.data.weight.get_device() == 0:
            progress.display(i+1)
    return grad_acc, losses.avg, distill_steps

# use pair_aug with train will apply a deterministic augmentation for all the data
def set_up_interventions(args):
    syn_intervention  = ImageIntervention(
                             'syn_aug',
                             args.syn_strategy,
                             phase='test',
                             not_single=args.comp_aug
                         )
    real_intervention = ImageIntervention(
                             'real_aug',
                             args.real_strategy,
                             phase='test',
                             not_single=args.comp_aug_real
                         )
    # This is a customizable prob \in [0, 1]
    intervention_prob  = 1.0

    return syn_intervention, real_intervention, intervention_prob


def calculate_grad_norm(grad_norm):
    return grad_norm[grad_norm>1e-5].mean().item()

# testing function for train_y = True
def one_gpu_test(val_loader, model, args):
    cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
    def run_validate(loader, base_progress=0):
        acc_ind = []
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                    target = target.cuda(args.gpu, non_blocking=True)
                elif torch.cuda.is_available():
                    images = images.cuda()
                    target = target.cuda()

                output, _ = model.module.test(images)
                sample_wise_loss = cross_entropy_loss(output, target)
                acc_ind.append(sample_wise_loss)
        return torch.cat(acc_ind, 0)

    acc_ind = run_validate(val_loader)
    #print(acc_ind.shape)
    return acc_ind

# testing function for not training labels
def one_gpu_test_2(val_loader, model, args):
    def run_validate(loader, base_progress=0):
        acc_ind = []
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                    target = target.cuda(args.gpu, non_blocking=True)
                elif torch.cuda.is_available():
                    images = images.cuda()
                    target = target.cuda()

                output, _ = model.module.test(images)
                if len(target.shape) == 2: target = target.max(1)[1]
                acc_ind.append(accuracy_ind(output, target))
        return torch.cat(acc_ind, 0)

    acc_ind = run_validate(val_loader)
    return acc_ind.to(torch.int)
                                        
# project the data to a unit ball
def project(data, pgd_coef=1):
    coef_norm = 1 / math.sqrt(data.shape[1])
    data_norm = torch.reshape(
        torch.norm(torch.flatten(data, start_dim=1, end_dim=-1), dim=-1),
        [data.shape[0], *[1] * (data.dim() - 1)])
    return data / data_norm * pgd_coef

def test(data_loaders, model, criterion, args):
    epoch_list = [300, 600, 1000, 3000]
    acc = []
    for i in range(len(data_loaders)): acc.append([0] * (len(epoch_list)))
    loss = 0
    
    for train_ind in range(args.num_train_eval):
        model.module.init_train(0, init=True)
        start_epoch = 0
        for train_time in range(len(epoch_list)):
            model.train()
            model.module.net.train()
            model.module.init_train(epoch_list[train_time] - start_epoch)
            for loader_i in range(len(data_loaders)):
                tmp_acc, tmp_loss = default_test(data_loaders[loader_i], model, criterion, args)
                acc[loader_i][train_time] += tmp_acc
            start_epoch = epoch_list[train_time]
        loss += tmp_loss
        if train_ind == 0:
            acc_ind = one_gpu_test(data_loaders[0], model, args)
        else:
            acc_ind += one_gpu_test(data_loaders[0], model, args)
    acc_ind = args.num_train_eval - acc_ind
        
    for loader_i in range(len(data_loaders)):
        acc[loader_i] = [acc_id/args.num_train_eval for acc_id in acc[loader_i]]
        if model.module.data.weight.get_device() == 0:  
            for train_time in range(len(epoch_list)):
                if model.module.data.weight.get_device() == 0:  
                    print('Training for {} epoch: {}'.format(epoch_list[train_time], acc[loader_i][train_time]))
    return acc, tmp_loss / args.num_train_eval, acc_ind

def default_test(val_loader, model, criterion, args):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                    target = target.cuda(args.gpu, non_blocking=True)
                elif torch.cuda.is_available():
                    images = images.cuda()
                    target = target.cuda()

                output, _ = model.module.test(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                if len(target.shape) == 2: target = target.max(1)[1]
                acc = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc, images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                #if (i+1) % args.print_freq == 0 and model.module.data.get_device() == 0:
                #    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1],
        prefix='Test: ')

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        losses.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))
    if model.module.data.weight.get_device() == 0: progress.display_summary()

    return top1.avg, losses.avg 


