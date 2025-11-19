import argparse
import torch
import torch.multiprocessing as mp
from framework.base import main_worker

def parse_args():
    parser = argparse.ArgumentParser(description='NLP Dataset Distillation (AGNews Example)')

    # ===== 基础配置 =====
    parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducibility')
    parser.add_argument('--gpu', default=None, type=int, help='GPU ID to use for single-GPU mode')
    parser.add_argument('--mp_distributed', action='store_true', help='Enable multi-GPU (DDP) training')
    parser.add_argument('--world_size', default=1, type=int, help='Total number of processes (for DDP)')
    parser.add_argument('--rank', default=0, type=int, help='Rank of this process in distributed setup')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:29500', type=str, help='URL for process group init')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='Backend for DDP communication')

    # ===== 数据集与模型 =====
    parser.add_argument('--root', default='./dataset', type=str, help='Dataset root directory')
    parser.add_argument('--dataset', default='agnews', type=str, help='Dataset name (AGNews / IMDB / SST2 etc.)')
    parser.add_argument('--arch', default='smallnlp', type=str, help='Model architecture (SmallNLP)')

    # ===== 优化参数 =====
    parser.add_argument('--lr', default=0.01, type=float, help='Outer learning rate (for synthetic data)')
    parser.add_argument('--inner_lr', default=0.01, type=float, help='Inner learning rate (for model)')
    parser.add_argument('--inner_optim', default='SGD', type=str, help='Inner optimizer (SGD/Adam)')
    parser.add_argument('--epochs', default=3, type=int, help='Total training epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='Real data batch size')

    # ===== 蒸馏数据配置 =====
    parser.add_argument('--num_per_class', default=1, type=int, help='IPC: number of synthetic samples per class')
    parser.add_argument('--batch_per_class', default=1, type=int, help='Samples per class per batch')
    parser.add_argument('--task_sampler_nc', default=4, type=int, help='Number of classes per sampled task')
    parser.add_argument('--window', default=5, type=int, help='Number of inner-loop unrolling steps')
    parser.add_argument('--num_train_eval', default=3, type=int, help='How many times to train student for eval')
    parser.add_argument('--train_y', action='store_true', help='Whether to learn soft labels')

    # ===== 文本参数 =====
    parser.add_argument('--tokenizer', default='bert-base-uncased', type=str, help='Tokenizer name (HuggingFace)')
    parser.add_argument('--max_seq_len', default=128, type=int, help='Max sequence length')

    return parser.parse_args()


def main():
    args = parse_args()

    # ===== 环境设置 =====
    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1

    args.distributed = args.mp_distributed or args.world_size > 1

    print("========================================")
    print("🧠 NLP Dataset Distillation Launcher")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.arch}")
    print(f"  GPUs available: {ngpus_per_node}")
    print(f"  Mode: {'DDP (multi-GPU)' if args.distributed else 'Single GPU / CPU'}")
    print("========================================")

    # ===== 启动训练 =====
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        print(f"🚀 Launching {ngpus_per_node} parallel workers with DDP...")
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()


# import argparse
# import os
# import numpy as np
# import pandas as pd
# import torch
# import torch.multiprocessing as mp

# import wandb

# from framework.base import main_worker
# from framework.config import get_arch

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='Clean Train')
#     parser.add_argument('--seed', default=0, type=int)
#     # We have allowed distributed training on the data.
#     parser.add_argument('--mp_distributed', action='store_true', help='Use distributed training')
#     parser.add_argument('--world_size', default=1, type=int, help='Number of processes')
#     parser.add_argument('--rank', default=0, type=int)
#     parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                         help='url used to set up distributed training')
#     parser.add_argument('--dist-backend', default='nccl', type=str,
#                         help='distributed backend')
#     parser.add_argument('-j', '--workers', default=0, type=int, metavar='N')

#     # Specify the GPU for one GPU training
#     parser.add_argument('--gpu', default=None, type=int, help='GPU to use in non-distributed training')

#     # Training configs
#     parser.add_argument('--root', default='./dataset', type=str, help='Root directory for dataset')
#     parser.add_argument('--dataset', default='agnews', type=str, help='Dataset to use')
#     parser.add_argument('--arch', default='smallnlp', type=str,help='Model architecture for NLP, 在 get_arch 里实现 smallnlp')
#     parser.add_argument('--lr', default=0.01, type=float, help='learning rate for the distilled data')
#     parser.add_argument('--inner_optim', default='SGD', type=str, help='Inner optimizer for the neural network')
#     parser.add_argument('--outer_optim', default='Adam', type=str, help='Outer optimizer for the data')
#     parser.add_argument('--inner_lr', default=0.01, type=float, help='inner learning rate')
#     parser.add_argument('--label_lr_scale', default=1, type=float, help='scale the label lr')
#     parser.add_argument('--num_per_class', default=1, type=int, help='Number of samples per class (IPC)')
#     parser.add_argument('--batch_per_class', default=1, type=int, help='Number of samples per class per batch')
#     parser.add_argument('--task_sampler_nc', default=10, type=int, help='Number of tasks to sample per batch')
#     parser.add_argument('--window', default=20, type=int, help='Number of unrolling computing gradients')
#     parser.add_argument('--minwindow', default=0, type=int, help='Start unrolling from steps x')
#     parser.add_argument('--totwindow', default=20, type=int, help='Number of total unrolling computing gradients')
#     parser.add_argument('--num_train_eval', default=10, type=int, help='Num of training of network for evaluation')
#     parser.add_argument('--train_y', action='store_true', help='Train the label')
#     parser.add_argument('--batch_size', default=200, type=int, help='Batch size for sampling from the original distribution')
#     parser.add_argument('--eps', default=1e-8, type=float)
#     parser.add_argument('--wd', default=0, type=float)
#     parser.add_argument('--test_freq', default=5, type=int, help='Frequency of testing in epochs')
#     parser.add_argument('--print_freq', default=20, type=int, help='Frequency of printing in steps')
#     parser.add_argument('--start_epoch', default=0, type=int)
#     parser.add_argument('--epochs', default=100, type=int)

#     parser.add_argument('--tokenizer', default='bert-base-uncased', type=str,
#                     help='HuggingFace tokenizer 名称，用于 AGNews 文本分词')
#     parser.add_argument('--max_seq_len', default=128, type=int,
#                         help='文本最大序列长度，超过截断，不足补 pad')

#     parser.add_argument('--ddtype', default='standard', type=str, help='Data Distillation Type, in curriculum, ')
#     parser.add_argument('--cctype', default=0, type=int, help='Curriculum Type')
#     # if cctype == 0: use a fix window without moving
#     # if cctype == 1: use a window with moving forward
#     # if cctype == 2: use a randomly placed window. The random location is changed every epoch

#     # parser.add_argument('--zca', action='store_true')
#     parser.add_argument('--wandb', action='store_true')
#     parser.add_argument('--clip_coef', default=0.9, type=float, help='Clipping coefficient for the gradients in EMA')

#     parser.add_argument('--fname', default='_test', type=str, help='Filename for storing checkpoints')
#     parser.add_argument('--name', default='test', type=str, help='name of the experiment for wandb')
#     parser.add_argument('--comp_aug', action='store_true', help='Compose different augmentation methods, if not, use only one randomly')
#     parser.add_argument('--comp_aug_real', action='store_true', help='Compose different augmentation methods for the real data')
#     # parser.add_argument('--syn_strategy', default='flip_rotate', type=str, help='Synthetic data augmentation strategy')
#     parser.add_argument('--real_strategy', default='flip_rotate', type=str, help='Real data augmentation strategy')
#     parser.add_argument('--ckptname', default='none', type=str, help='Checkpoint name for initializing the distilled data')
#     parser.add_argument('--limit_train', action='store_true', help='Limit the training data')
#     parser.add_argument('--load_ckpt', action='store_true')
#     parser.add_argument('--complete_random', action='store_true')
#     parser.add_argument('--zca', action='store_true')
    
#     args = parser.parse_args()

#     args.distributed = args.world_size > 1 or args.mp_distributed
#     if torch.cuda.is_available():
#         ngpus_per_node = torch.cuda.device_count()
#     else:
#         ngpus_per_node = 1
        
#     args.num_train_eval = int(args.num_train_eval / ngpus_per_node)   
    
#     if args.mp_distributed:
#         args.world_size = ngpus_per_node * args.world_size
#         mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

#         for i in range(5):
#             torch.cuda.empty_cache()
#     else:
#         # Simply call main_worker function
#         main_worker(args.gpu, ngpus_per_node, args)
        
    
