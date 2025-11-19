'''Framework default config'''
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer


# =====================================================
# 🧩 SmallNLP 模型
# =====================================================
class SmallNLP(nn.Module):
    """
    轻量级 Transformer 文本分类器，兼容 token ids / 合成 embeddings。
    """

    def __init__(
        self,
        num_classes=4,
        vocab_size=30522,
        embed_dim=128,
        pad_idx=0,
        num_layers=2,
        num_heads=4,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len=128,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.uniform_(module.weight, -0.1, 0.1)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _build_positional(self, length, device):
        if length > self.max_seq_len:
            raise ValueError(f"Sequence length {length} exceeds max_seq_len {self.max_seq_len}.")
        return self.pos_embedding[:, :length, :].to(device)

    def forward(self, x, is_embedding=False, pad_mask=None):
        """
        x:
          - is_embedding=False → [B, L] token ids
          - is_embedding=True  → [B, L, D] synthetic embeddings
        """
        if not is_embedding:
            pad_mask = x.eq(self.pad_idx)
            x = self.embedding(x)
        elif pad_mask is None:
            pad_mask = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)

        pos = self._build_positional(x.size(1), x.device)
        x = x + pos
        encoded = self.encoder(x, src_key_padding_mask=pad_mask)
        pooled = encoded.mean(dim=1)
        pooled = self.dropout(self.norm(pooled))
        logits = self.fc(pooled)
        return logits, encoded


class TextCNN(nn.Module):
    """
    轻量级 TextCNN，支持 token ids / 合成 embeddings。
    """

    def __init__(
        self,
        num_classes=4,
        vocab_size=30522,
        embed_dim=128,
        pad_idx=0,
        num_filters=128,
        filter_sizes=(3, 4, 5),
        dropout=0.1,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=k // 2,
                )
                for k in filter_sizes
            ]
        )
        hidden_dim = num_filters * len(filter_sizes)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.uniform_(module.weight, -0.1, 0.1)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, is_embedding=False, pad_mask=None):
        if not is_embedding:
            pad_mask = x.eq(self.pad_idx)
            x = self.embedding(x)
        elif pad_mask is None:
            pad_mask = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)

        mask = (~pad_mask).float().unsqueeze(-1)
        x = x * mask
        x = x.transpose(1, 2)  # [B, D, L]

        feats = []
        for conv in self.convs:
            h = torch.relu(conv(x))
            h = torch.max(h, dim=2)[0]
            feats.append(h)

        feats = torch.cat(feats, dim=1)
        feats = self.dropout(self.norm(feats))
        logits = self.fc(feats)
        return logits, feats


# =====================================================
# 🏗️ 模型工厂
# =====================================================
def get_arch(arch, num_classes, channel=1, im_size=(128, 128), width=64):
    if arch == "smallnlp":
        seq_len, embed_dim = im_size
        return SmallNLP(num_classes=num_classes, embed_dim=embed_dim, max_seq_len=seq_len)
    elif arch == "textcnn":
        _, embed_dim = im_size
        return TextCNN(num_classes=num_classes, embed_dim=embed_dim)
    else:
        raise NotImplementedError(f"Architecture {arch} not supported for NLP.")


# =====================================================
# 🧰 NLP 数据加载（AGNews）
# =====================================================
def get_dataset(dataset, root, transform_train=None, transform_test=None, **kwargs):
    """
    加载 AGNews 数据集，返回 tokenized train/test 集。
    输出结构与原 CV pipeline 兼容。
    """
    if dataset != "agnews":
        raise NotImplementedError(f"Only AGNews is supported, got {dataset}.")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def encode_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    dataset = load_dataset("ag_news", cache_dir=root)
    dataset = dataset.map(encode_fn, batched=True)

    def to_torch(ds):
        input_ids = torch.tensor(ds["input_ids"], dtype=torch.long)
        labels = torch.tensor(ds["label"], dtype=torch.long)
        return torch.utils.data.TensorDataset(input_ids, labels)

    train_set = to_torch(dataset["train"])
    test_set = to_torch(dataset["test"])

    num_classes = 4
    shape = (128,)  # seq_len
    print(f"Loaded AGNews dataset: {len(train_set)} train, {len(test_set)} test, num_classes={num_classes}")
    return train_set, None, test_set, num_classes, shape, None


# =====================================================
# 🔤 Tokenizer transform 封装
# =====================================================
def get_transform(dataset):
    """
    返回 AGNews 用的 tokenizer 函数，供 dataset loader 调用。
    """
    if dataset != "agnews":
        raise NotImplementedError("Only AGNews NLP transform supported.")

    print("the dataset is agnews (NLP task)")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def encode_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    return encode_fn, encode_fn

# from framework.model import ResNet18
# from framework.vgg import VGG11, AlexNet
# from framework.convnet import ConvNet, ConvNet2
# import torchvision
# import torchvision.transforms as transforms
# from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageFolder

# import numpy as np

# import h5py

# import torch
# import os


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SmallNLP(nn.Module):
#     """
#     轻量级文本分类模型（AGNews 友好）。
#     输入自动识别：
#       - 真实文本:  x.shape = [B, L], dtype = torch.long  -> 通过 Embedding
#       - 合成样本:  x.shape = [B, L, D], dtype = float     -> 直接作为 embedding
#     输出： (logits, pooled)
#       - logits: [B, num_classes]
#       - pooled: [B, hidden_dim*2]  (用于与现有接口对齐的占位/可选特征)
#     """
#     def __init__(
#         self,
#         num_classes,
#         vocab_size=30522,
#         embed_dim=128,
#         hidden_dim=256,
#         num_layers=1,
#         pad_idx=0,
#         dropout=0.1
#     ):
#         super().__init__()
#         self.pad_idx = pad_idx
#         self.embed_dim = embed_dim
#         self.hidden_dim = hidden_dim

#         # 词嵌入：仅在输入是 token ids 时使用
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

#         # 编码器：双向 LSTM（batch_first=True）
#         self.encoder = nn.LSTM(
#             input_size=embed_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=True,
#         )

#         # 轻量正则化
#         self.dropout = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(hidden_dim * 2)

#         # 分类头
#         self.fc = nn.Linear(hidden_dim * 2, num_classes)

#         # 参数初始化
#         self.apply(self._init_weights)

#     # ---- 初始化策略：稳定&通用 ----
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             nn.init.xavier_uniform_(module.weight)
#             if module.bias is not None:
#                 nn.init.constant_(module.bias, 0)
#         elif isinstance(module, nn.Embedding):
#             # 注意：padding_idx 的向量会被框架保持为 0
#             nn.init.uniform_(module.weight, -0.1, 0.1)
#             if self.pad_idx is not None and 0 <= self.pad_idx < module.num_embeddings:
#                 with torch.no_grad():
#                     module.weight[self.pad_idx].fill_(0.0)
#         elif isinstance(module, nn.LSTM):
#             for name, param in module.named_parameters():
#                 if "weight" in name:
#                     nn.init.xavier_uniform_(param)
#                 elif "bias" in name:
#                     nn.init.constant_(param, 0)

#     # ---- mask mean pooling：对 pad 做掩码，更鲁棒 ----
#     def _masked_mean(self, x, mask):
#         """
#         x:    [B, L, D]
#         mask: [B, L]  (1=有效, 0=pad)
#         """
#         mask = mask.unsqueeze(-1).type_as(x)               # [B, L, 1]
#         summed = (x * mask).sum(dim=1)                     # [B, D]
#         denom = mask.sum(dim=1).clamp(min=1e-6)            # [B, 1]
#         return summed / denom

#     def forward(self, x):
#         """
#         x:
#           - [B, L]  (long)   -> token ids
#           - [B, L, D] (float)-> embeddings
#         返回:
#           logits: [B, num_classes]
#           pooled: [B, hidden_dim*2]
#         """
#         # ------- 自动识别输入类型 -------
#         if x.dim() == 2 and x.dtype == torch.long:
#             # 真实文本: token ids
#             input_ids = x
#             emb = self.embedding(input_ids)                # [B, L, E]
#             # 生成 attention mask（pad=0，其他=1）
#             attn_mask = (input_ids != self.pad_idx).to(x.dtype)  # [B, L], float 用于 _masked_mean
#         elif x.dim() == 3 and x.dtype in (torch.float16, torch.float32, torch.float64):
#             # 合成样本: 已经是 embeddings
#             emb = x                                        # [B, L, E]
#             # 合成数据默认无 pad，mask 全 1
#             attn_mask = torch.ones(emb.size(0), emb.size(1), device=emb.device, dtype=torch.float32)
#         else:
#             raise ValueError(
#                 f"SmallNLP.forward: unexpected input shape/dtype: shape={tuple(x.shape)}, dtype={x.dtype}"
#             )

#         # ------- 编码器 -------
#         enc_out, (h_n, c_n) = self.encoder(emb)            # enc_out: [B, L, 2*H]

#         # ------- 池化（对 pad 做掩码更稳）-------
#         pooled = self._masked_mean(enc_out, attn_mask)     # [B, 2*H]
#         pooled = self.norm(self.dropout(pooled))            # 轻量正则化

#         # ------- 分类 -------
#         logits = self.fc(pooled)                           # [B, num_classes]

#         # 与现有管线对齐：返回 (logits, pooled)
#         return logits, pooled

# class CIFAR10Dataset(CIFAR10):
#     def __getitem__(self, idx):
#         return self.data[idx], self.targets[idx]

# class CIFAR100Dataset(CIFAR100):
#     def __getitem__(self, idx):
#         return self.data[idx], self.targets[idx]

# class DistillDataset(torch.utils.data.Dataset):
#     def __init__(self, tensor_data, list_data):
#         assert len(tensor_data) == len(list_data), "Both inputs must have the same length"
#         self.tensor_data = tensor_data
#         self.list_data = list_data

#     def __len__(self):
#         return len(self.tensor_data)

#     def __getitem__(self, index):
#         return self.tensor_data[index].view(3, 32, 32), self.list_data[index]


# def get_config():
#     config = {
#             'root': '/home/fyz/dataset/',
#             'num_workers_mnist': 1,
#             'num_workers_cifar10': 4,
#             'num_workers_imagenet': 4
#     }
#     return config

# def get_arch(arch, num_classes, channel, im_size, width=64):
#     if arch == 'resnet18':
#         return ResNet18(channel=channel, num_classes=num_classes)
#     if arch == 'vgg':
#         return VGG11(channel=channel, num_classes=num_classes)
#     if arch == 'alexnet':
#         return AlexNet(channel=channel, num_classes=num_classes)
#     if arch == 'convnet':
#         net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
#         return ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = im_size)
#     if arch == 'convnet4':
#         net_width, net_depth, net_act, net_norm, net_pooling = 128, 4, 'relu', 'instancenorm', 'avgpooling'
#         return ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = im_size)

#     # ========= ✅ 新增 NLP 模型 =========
#     if arch == 'smallnlp':
#         # 注意：im_size=(seq_len,1) 来自 base.py
#         seq_len = im_size[0] if isinstance(im_size, (list, tuple)) else im_size
#         return SmallNLP(num_classes=num_classes, vocab_size=30522, embed_dim=128)

#     raise NotImplementedError

# # def get_arch(arch, num_classes, channel, im_size, width=64):
# #     if arch == 'resnet18':
# #         return ResNet18(channel=channel, num_classes=num_classes)
# #     if arch == 'vgg':
# #         return VGG11(channel=channel, num_classes=num_classes)
# #     if arch == 'alexnet':
# #         return AlexNet(channel=channel, num_classes=num_classes)
# #     if arch == 'convnet':
# #         net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
# #         return ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = im_size)
# #     if arch == 'convnet4':
# #         net_width, net_depth, net_act, net_norm, net_pooling = 128, 4, 'relu', 'instancenorm', 'avgpooling'
# #         return ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = im_size)    
# #     raise NotImplementedError


# def get_dataset(dataset, root, transform_train, transform_test, zca=False):
#     """
#     根据 dataset 名称加载对应数据集。
#     - 图像类数据：沿用原实现 (CIFAR, MNIST, ImageNet 等)
#     - 文本类数据（AGNews）：新增分支，返回经过 tokenizer 编码的文本序列
#     """
#     data_root = os.path.join(root, dataset)
#     process_config = None

#     # ==================== 1️⃣ 新增 NLP 任务: AGNews ====================
#     if dataset.lower() == 'agnews':
#         """
#         AGNews 是一个 4 类新闻分类数据集：
#         label ∈ {0, 1, 2, 3}
#         text: 新闻标题 + 内容
#         """

#         # 引入必要的库
#         from datasets import load_dataset
#         from transformers import AutoTokenizer
#         from torch.utils.data import Dataset
#         import torch

#         # 1. 加载数据集（会自动下载到 ~/.cache/huggingface/datasets）
#         raw_ds = load_dataset("ag_news")

#         # 2. 初始化 tokenizer
#         tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#         max_len = 128  # 统一序列长度（可与 args.max_seq_len 对齐）

#         # 3. 定义 Dataset 封装类
#         class AGNewsTorchDataset(Dataset):
#             def __init__(self, hf_split):
#                 self.hf_split = hf_split

#             def __len__(self):
#                 return len(self.hf_split)

#             def __getitem__(self, idx):
#                 sample = self.hf_split[idx]
#                 text, label = sample["text"], sample["label"]

#                 # Tokenize 并转成固定长度的 input_ids
#                 enc = tokenizer(
#                     text,
#                     truncation=True,
#                     padding="max_length",
#                     max_length=max_len
#                 )
#                 input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
#                 label = torch.tensor(label, dtype=torch.long)
#                 return input_ids, label

#         # 4. 分割训练 / 测试集
#         trainset = AGNewsTorchDataset(raw_ds["train"])
#         trainset_test = trainset  # 与图像版接口保持一致
#         testset = AGNewsTorchDataset(raw_ds["test"])

#         # 5. 基本信息
#         num_classes = 4
#         # shape 表示单条样本的形状，这里只有长度 (max_seq_len,)
#         shape = [max_len]

#         print(f"Loaded AGNews dataset: {len(trainset)} train, {len(testset)} test, num_classes={num_classes}")

#         return trainset, trainset_test, testset, num_classes, shape, process_config

#     # ==================== 2️⃣ 图像任务（原逻辑保持不变） ====================
#     if dataset == 'cifar10':
#         if zca:
#             print('Using ZCA')
#             trainset = CIFAR10Dataset(
#                     root=root, train=True, download=True, transform=None)
#             trainset_test = CIFAR10Dataset(
#                     root=root, train=True, download=True, transform=None)
#             testset = CIFAR10Dataset(
#                     root=root, train=False, download=True, transform=None)
#             trainset.data, testset.data, process_config = preprocess(trainset.data, testset.data, regularization=0.1)
#             trainset_test.data = trainset.data.clone()
#         else:
#             trainset = CIFAR10(
#                     root=root, train=True, download=True, transform=transform_train)
#             trainset_test = CIFAR10(
#                     root=root, train=True, download=True, transform=transform_test)
#             testset = CIFAR10(
#                     root=root, train=False, download=True, transform=transform_test)
#         num_classes = 10
#         shape = [3, 32, 32]
#     elif dataset == 'cifar100':
#         if zca:
#             print('Using ZCA')
#             trainset = CIFAR100Dataset(
#                     root=root, train=True, download=True, transform=None)
#             testset = CIFAR100Dataset(
#                     root=root, train=False, download=True, transform=None)
#             trainset.data, testset.data, process_config = preprocess(trainset.data, testset.data, regularization=0.1)
#             trainset_test = trainset
#         else:
#             trainset = CIFAR100(
#                     root=root, train=True, download=True, transform=transform_train)
#             trainset_test = CIFAR100(
#                     root=root, train=True, download=True, transform=transform_test)
#             testset = CIFAR100(
#                     root=root, train=False, download=True, transform=transform_test)
#         num_classes = 100
#         shape = [3, 32, 32]
#     elif dataset == 'tiny-imagenet-200':
#         shape = [3, 64, 64]
#         num_classes = 200
#         if zca:
#             print('Using ZCA')
#             db = h5py.File('./dataset/tiny-imagenet-200/zca_pro.h5', 'r')
#             train_data = torch.tensor(db['train'])
#             test_data = torch.tensor(db['test'])
#             train_label = torch.tensor(db['train_label'])
#             test_label = torch.tensor(db['test_label'])
#             trainset = TensorDataset(train_data, train_label)
#             trainset_test = trainset
#             testset = TensorDataset(test_data, test_label)
#         else:
#             raise NotImplementedError
#     elif dataset == 'cub-200':
#         shape = [3, 32, 32]
#         num_classes = 200
#         if zca:
#             print('Using ZCA')
#             db = h5py.File('./dataset/CUB_200_2011/zca_new.h5', 'r')
#             train_data = torch.tensor(db['train'])
#             test_data = torch.tensor(db['test'])
#             train_label = torch.tensor(db['train_label'])
#             test_label = torch.tensor(db['test_label'])
#             trainset = TensorDataset(train_data, train_label)
#             trainset_test = trainset
#             testset = TensorDataset(test_data, test_label)
#         else:
#             raise NotImplementedError
#     elif dataset == 'imagenet':
#         print('Using ImageNet')
#         shape = [3, 64, 64]
#         im_size = (64, 64)
#         num_classes = 1000
#         data_path = '/imagenet/'

#         mean = [0.485, 0.456, 0.406]
#         std = [0.229, 0.224, 0.225]

#         data_transforms = {
#             'train': transforms.Compose([
#                 transforms.Resize(im_size),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean, std)
#             ]),
#             'val': transforms.Compose([
#                 transforms.Resize(im_size),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean, std)
#             ]),
#         }

#         trainset = ImageFolder(os.path.join(data_path, "train"), transform=data_transforms['train'])
#         testset = ImageFolder(os.path.join(data_path, "val"), transform=data_transforms['val'])
#         trainset_test = trainset
        
#     elif dataset == 'mnist':
#         trainset = MNIST(
#                 root=root, train=True, download=True, transform=transform_train)
#         trainset_test = MNIST(
#                 root=root, train=True, download=True, transform=transform_test)
#         testset = MNIST(
#                 root=root, train=False, download=True, transform=transform_test)
#         num_classes = 10
#         shape = [1, 28, 28]
#     else:
#         raise NotImplementedError
        
#     return trainset, trainset_test, testset, num_classes, shape, process_config


# # remove all the ToTensor() for cifar10
# def get_transform(dataset):
#     print(dataset)

#     # ---- CIFAR10 ----
#     if dataset == 'cifar10':
#         default_transform_train = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                  (0.2023, 0.1994, 0.2010)),
#         ])
#         default_transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                  (0.2023, 0.1994, 0.2010)),
#         ])
#         print('the dataset is cifar10')

#     # ---- CIFAR100 ----
#     elif dataset == 'cifar100':
#         default_transform_train = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5071, 0.4867, 0.4408),
#                                  (0.2675, 0.2565, 0.2761)),
#         ])
#         default_transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5071, 0.4867, 0.4408),
#                                  (0.2675, 0.2565, 0.2761)),
#         ])
#         print('the dataset is cifar100')

#     # ---- Tiny ImageNet ----
#     elif dataset == 'tiny-imagenet-200':
#         default_transform_train = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406),
#                                  (0.229, 0.224, 0.225)),
#         ])
#         default_transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406),
#                                  (0.229, 0.224, 0.225)),
#         ])
#         print('the dataset is tiny-imagenet-200')

#     # ---- ImageNet ----
#     elif dataset == 'imagenet':
#         print('the dataset is imagenet')
#         default_transform_train = None
#         default_transform_test = None

#     # ---- CUB-200 ----
#     elif dataset == 'cub-200':
#         default_transform_train = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406),
#                                  (0.229, 0.224, 0.225)),
#         ])
#         default_transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406),
#                                  (0.229, 0.224, 0.225)),
#         ])
#         print('the dataset is cub-200-2011')

#     # ---- MNIST ----
#     elif dataset == 'mnist':
#         default_transform_train = transforms.Compose([
#             transforms.ToTensor(),
#         ])
#         default_transform_test = transforms.Compose([
#             transforms.ToTensor(),
#         ])

#     # ---- ✅ 新增：AGNEWS (NLP) ----
#     elif dataset.lower() == 'agnews':
#         print('the dataset is agnews (NLP task)')
#         # 对 NLP 来说，不需要图像增强，transform 可以为 None
#         # 你也可以使用 tokenizer 预处理文本
#         try:
#             from transformers import AutoTokenizer
#             tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#             def encode_fn(example):
#                 encoded = tokenizer(
#                     example["text"],
#                     truncation=True,
#                     padding="max_length",
#                     max_length=128,
#                     return_tensors="pt"
#                 )
#                 return encoded["input_ids"].squeeze(0), example["label"]

#             default_transform_train = encode_fn
#             default_transform_test = encode_fn
#         except ImportError:
#             print("⚠️ transformers 未安装或导入失败，transform 设为 None。")
#             default_transform_train = None
#             default_transform_test = None

#     else:
#         raise NotImplementedError(f"Transform not implemented for dataset: {dataset}")

#     return default_transform_train, default_transform_test



# def get_pin_memory(dataset):
#     return dataset == 'imagenet'

# import torch
# import numpy as np
# import os
# from PIL import Image, TarIO
# import pickle
# import tarfile

# class cub200(torch.utils.data.Dataset):
#     def __init__(self, root, train=True, transform=None):
#         super(cub200, self).__init__()

#         self.root = root
#         self.train = train
#         self.transform = transform


#         if self._check_processed():
#             print('Train file has been extracted' if self.train else 'Test file has been extracted')
#         else:
#             self._extract()

#         if self.train:
#             self.train_data, self.train_label = pickle.load(
#                 open(os.path.join(self.root, 'processed/train.pkl'), 'rb')
#             )
#         else:
#             self.test_data, self.test_label = pickle.load(
#                 open(os.path.join(self.root, 'processed/test.pkl'), 'rb')
#             )

#     def __len__(self):
#         return len(self.train_data) if self.train else len(self.test_data)

#     def __getitem__(self, idx):
#         if self.train:
#             img, label = self.train_data[idx], self.train_label[idx]
#         else:
#             img, label = self.test_data[idx], self.test_label[idx]
#         img = Image.fromarray(img)
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, label

#     def _check_processed(self):
#         assert os.path.isdir(self.root) == True
#         assert os.path.isfile(os.path.join(self.root, 'CUB_200_2011.tgz')) == True
#         return (os.path.isfile(os.path.join(self.root, 'processed/train.pkl')) and
#                 os.path.isfile(os.path.join(self.root, 'processed/test.pkl')))

#     def _extract(self):
#         processed_data_path = os.path.join(self.root, 'processed')
#         if not os.path.isdir(processed_data_path):
#             os.mkdir(processed_data_path)

#         cub_tgz_path = os.path.join(self.root, 'CUB_200_2011.tgz')
#         images_txt_path = 'CUB_200_2011/images.txt'
#         train_test_split_txt_path = 'CUB_200_2011/train_test_split.txt'

#         tar = tarfile.open(cub_tgz_path, 'r:gz')
#         images_txt = tar.extractfile(tar.getmember(images_txt_path))
#         train_test_split_txt = tar.extractfile(tar.getmember(train_test_split_txt_path))
#         if not (images_txt and train_test_split_txt):
#             print('Extract image.txt and train_test_split.txt Error!')
#             raise RuntimeError('cub-200-1011')

#         images_txt = images_txt.read().decode('utf-8').splitlines()
#         train_test_split_txt = train_test_split_txt.read().decode('utf-8').splitlines()

#         id2name = np.genfromtxt(images_txt, dtype=str)
#         id2train = np.genfromtxt(train_test_split_txt, dtype=int)
#         print('Finish loading images.txt and train_test_split.txt')
#         train_data = []
#         train_labels = []
#         test_data = []
#         test_labels = []
#         print('Start extract images..')
#         cnt = 0
#         train_cnt = 0
#         test_cnt = 0
#         for _id in range(id2name.shape[0]):
#             cnt += 1

#             image_path = 'CUB_200_2011/images/' + id2name[_id, 1]
#             image = tar.extractfile(tar.getmember(image_path))
#             if not image:
#                 print('get image: '+image_path + ' error')
#                 raise RuntimeError
#             image = Image.open(image)
#             label = int(id2name[_id, 1][:3]) - 1

#             if image.getbands()[0] == 'L':
#                 image = image.convert('RGB')
#             image_np = np.array(image)
#             image.close()

#             if id2train[_id, 1] == 1:
#                 train_cnt += 1
#                 train_data.append(image_np)
#                 train_labels.append(label)
#             else:
#                 test_cnt += 1
#                 test_data.append(image_np)
#                 test_labels.append(label)
#             if cnt%1000 == 0:
#                 print('{} images have been extracted'.format(cnt))
#         print('Total images: {}, training images: {}. testing images: {}'.format(cnt, train_cnt, test_cnt))
#         tar.close()
#         pickle.dump((train_data, train_labels),
#                     open(os.path.join(self.root, 'processed/train.pkl'), 'wb'))
#         pickle.dump((test_data, test_labels),
#                     open(os.path.join(self.root, 'processed/test.pkl'), 'wb'))

    
# class TensorDataset(torch.utils.data.Dataset):
#     def __init__(self, data_tensor, target_tensor):
#         assert data_tensor.size(0) == target_tensor.size(0), "Data and targets must have the same number of samples"
#         self.data_tensor = data_tensor
#         self.target_tensor = target_tensor

#     def __len__(self):
#         return self.data_tensor.size(0)

#     def __getitem__(self, index):
#         return self.data_tensor[index], self.target_tensor[index]

# # ZCA preprocess
# def preprocess(train, test, zca_bias=0, regularization=0, permute=True):
#     origTrainShape = train.shape
#     origTestShape = test.shape

#     train = np.ascontiguousarray(train, dtype=np.float32).reshape(train.shape[0], -1).astype('float64')
#     test = np.ascontiguousarray(test, dtype=np.float32).reshape(test.shape[0], -1).astype('float64')

#     nTrain = train.shape[0]
    
#     train_mean = np.mean(train, axis=1)[:,np.newaxis]
    
#     # Zero mean every feature
#     train = train - np.mean(train, axis=1)[:,np.newaxis]
#     test = test - np.mean(test, axis=1)[:,np.newaxis]

#     # Normalize
#     train_norms = np.linalg.norm(train, axis=1)
#     test_norms = np.linalg.norm(test, axis=1)

#     # Make features unit norm
#     train = train/train_norms[:,np.newaxis]
#     test = test/test_norms[:,np.newaxis]

#     trainCovMat = 1.0/nTrain * train.T.dot(train)

#     (E,V) = np.linalg.eig(trainCovMat)

#     E += zca_bias
#     sqrt_zca_eigs = np.sqrt(E + regularization * np.sum(E) / E.shape[0])
#     inv_sqrt_zca_eigs = np.diag(np.power(sqrt_zca_eigs, -1))
#     global_ZCA = V.dot(inv_sqrt_zca_eigs).dot(V.T)
#     inverse_ZCA = V.dot(np.diag(sqrt_zca_eigs)).dot(V.T)
    
#     train = (train).dot(global_ZCA)
#     test = (test).dot(global_ZCA)

#     train_tensor = torch.Tensor(train.reshape(origTrainShape).astype('float64'))
#     test_tensor  = torch.Tensor(test.reshape(origTestShape).astype('float64'))
#     if permute:
#         train_tensor = train_tensor.permute(0,3,1,2).contiguous()
#         test_tensor  = test_tensor.permute(0,3,1,2).contiguous()

#     return train_tensor, test_tensor, (inverse_ZCA, train_norms, train_mean)
