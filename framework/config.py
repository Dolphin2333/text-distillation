'''Framework default config'''
from .text_nlp import EmbeddingTensorDataset, TextMLP, TextTransformer
import torch

from framework.model import ResNet18
from framework.vgg import VGG11, AlexNet
from framework.convnet import ConvNet, ConvNet2
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageFolder

import numpy as np
import h5py
import os


class CIFAR10Dataset(CIFAR10):
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class CIFAR100Dataset(CIFAR100):
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class DistillDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_data, list_data):
        assert len(tensor_data) == len(list_data)
        self.tensor_data = tensor_data
        self.list_data = list_data
        self.shape = tensor_data.shape[1:]

    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, index):
        x = self.tensor_data[index].view(*self.shape)
        return x, self.list_data[index]


def get_config():
    return {
        'root': '/home/fyz/dataset/',
        'num_workers_mnist': 1,
        'num_workers_cifar10': 4,
        'num_workers_imagenet': 4,
    }


def get_arch(arch, num_classes, channel, im_size, width=64):
    if arch == 'text_mlp':
        d_in = channel * im_size[0] * im_size[1]
        return TextMLP(d_in=d_in, d_hidden=width, num_classes=num_classes)
    if arch == 'text_transformer':
        d_in = channel * im_size[0] * im_size[1]
        return TextTransformer(
            d_in=d_in,
            num_classes=num_classes,
            d_model=width,
            nhead=4,
            num_layers=2,
            seq_len=4,
        )
    raise NotImplementedError


def get_dataset(dataset, root, transform_train, transform_test, zca=False):
    data_root = os.path.join(root, dataset)
    process_config = None
    if dataset == 'cifar10':
        if zca:
            print('Using ZCA')
            trainset = CIFAR10Dataset(root=root, train=True, download=True, transform=None)
            trainset_test = CIFAR10Dataset(root=root, train=True, download=True, transform=None)
            testset = CIFAR10Dataset(root=root, train=False, download=True, transform=None)
            trainset.data, testset.data, process_config = preprocess(trainset.data, testset.data, regularization=0.1)
            trainset_test.data = trainset.data.clone()
        else:
            trainset = CIFAR10(root=root, train=True, download=True, transform=transform_train)
            trainset_test = CIFAR10(root=root, train=True, download=True, transform=transform_test)
            testset = CIFAR10(root=root, train=False, download=True, transform=transform_test)
        num_classes = 10
        shape = [3, 32, 32]
    elif dataset == 'mrpc_emb':
        train_emb_path = os.path.join(root, 'mrpc_train_emb.pt')
        train_label_path = os.path.join(root, 'mrpc_train_labels.pt')
        val_emb_path = os.path.join(root, 'mrpc_val_emb.pt')
        val_label_path = os.path.join(root, 'mrpc_val_labels.pt')

        trainset = EmbeddingTensorDataset(train_emb_path, train_label_path)
        trainset_test = EmbeddingTensorDataset(val_emb_path, val_label_path)
        testset = trainset_test
        num_classes = int(torch.max(trainset.y).item()) + 1
        d_emb = trainset.d_emb
        shape = [1, d_emb, 1]
        process_config = None
        return trainset, trainset_test, testset, num_classes, shape, process_config
    elif dataset == 'agnews_emb':
        train_emb_path = os.path.join(root, 'agnews_train_emb.pt')
        train_label_path = os.path.join(root, 'agnews_train_labels.pt')
        val_emb_path = os.path.join(root, 'agnews_val_emb.pt')
        val_label_path = os.path.join(root, 'agnews_val_labels.pt')

        trainset = EmbeddingTensorDataset(train_emb_path, train_label_path)
        trainset_test = EmbeddingTensorDataset(val_emb_path, val_label_path)
        testset = trainset_test
        num_classes = int(torch.max(trainset.y).item()) + 1
        d_emb = trainset.d_emb
        shape = [1, d_emb, 1]
        process_config = None
        return trainset, trainset_test, testset, num_classes, shape, process_config
    elif dataset == 'cifar100':
        if zca:
            print('Using ZCA')
            trainset = CIFAR100Dataset(root=root, train=True, download=True, transform=None)
            testset = CIFAR100Dataset(root=root, train=False, download=True, transform=None)
            trainset.data, testset.data, process_config = preprocess(trainset.data, testset.data, regularization=0.1)
            trainset_test = trainset
        else:
            trainset = CIFAR100(root=root, train=True, download=True, transform=transform_train)
            trainset_test = CIFAR100(root=root, train=True, download=True, transform=transform_test)
            testset = CIFAR100(root=root, train=False, download=True, transform=transform_test)
        num_classes = 100
        shape = [3, 32, 32]
    elif dataset == 'tiny-imagenet-200':
        shape = [3, 64, 64]
        num_classes = 200
        if zca:
            print('Using ZCA')
            db = h5py.File('./dataset/tiny-imagenet-200/zca_pro.h5', 'r')
            train_data = torch.tensor(db['train'])
            test_data = torch.tensor(db['test'])
            train_label = torch.tensor(db['train_label'])
            test_label = torch.tensor(db['test_label'])
            trainset = TensorDataset(train_data, train_label)
            trainset_test = trainset
            testset = TensorDataset(test_data, test_label)
        else:
            raise NotImplementedError
    elif dataset == 'cub-200':
        shape = [3, 32, 32]
        num_classes = 200
        if zca:
            print('Using ZCA')
            db = h5py.File('./dataset/CUB_200_2011/zca_new.h5', 'r')
            train_data = torch.tensor(db['train'])
            test_data = torch.tensor(db['test'])
            train_label = torch.tensor(db['train_label'])
            test_label = torch.tensor(db['test_label'])
            trainset = TensorDataset(train_data, train_label)
            trainset_test = trainset
            testset = TensorDataset(test_data, test_label)
        else:
            raise NotImplementedError
    elif dataset == 'imagenet':
        print('Using ImageNet')
        shape = [3, 64, 64]
        num_classes = 1000
        data_path = '/imagenet/'
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        trainset = ImageFolder(os.path.join(data_path, 'train'), transform=data_transforms['train'])
        trainset_test = trainset
        testset = ImageFolder(os.path.join(data_path, 'val'), transform=data_transforms['val'])
    elif dataset == 'mnist':
        trainset = MNIST(root=root, train=True, download=True, transform=transform_train)
        trainset_test = MNIST(root=root, train=True, download=True, transform=transform_test)
        testset = MNIST(root=root, train=False, download=True, transform=transform_test)
        num_classes = 10
        shape = [1, 28, 28]
    else:
        raise NotImplementedError

    return trainset, trainset_test, testset, num_classes, shape, process_config


def get_transform(dataset):
    if dataset.endswith('_emb'):
        return None, None
    print(dataset)
    if dataset == 'mrpc_emb':
        default_transform_train = None
        default_transform_test = None
    else:
        raise NotImplementedError
    return default_transform_train, default_transform_test


def get_pin_memory(dataset):
    return dataset == 'imagenet'

# Remaining CUB-200 helper classes kept identical to Tim's repo (omitted for brevity)

import torch
import numpy as np
from PIL import Image, TarIO
import pickle
import tarfile

class cub200(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        super(cub200, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform
        if self._check_processed():
            print('Train file has been extracted' if self.train else 'Test file has been extracted')
        else:
            self._extract()
        if self.train:
            self.train_data, self.train_label = pickle.load(
                open(os.path.join(self.root, 'processed/train.pkl'), 'rb')
            )
        else:
            self.test_data, self.test_label = pickle.load(
                open(os.path.join(self.root, 'processed/test.pkl'), 'rb')
            )

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def __getitem__(self, idx):
        if self.train:
            img, label = self.train_data[idx], self.train_label[idx]
        else:
            img, label = self.test_data[idx], self.test_label[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def _check_processed(self):
        assert os.path.isdir(self.root)
        assert os.path.isfile(os.path.join(self.root, 'CUB_200_2011.tgz'))
        return (
            os.path.isfile(os.path.join(self.root, 'processed/train.pkl')) and
            os.path.isfile(os.path.join(self.root, 'processed/test.pkl'))
        )

    def _extract(self):
        processed_data_path = os.path.join(self.root, 'processed')
        if not os.path.isdir(processed_data_path):
            os.mkdir(processed_data_path)

        cub_tgz_path = os.path.join(self.root, 'CUB_200_2011.tgz')
        images_txt_path = 'CUB_200_2011/images.txt'
        train_test_split_txt_path = 'CUB_200_2011/train_test_split.txt'

        tar = tarfile.open(cub_tgz_path, 'r:gz')
        images_txt = tar.extractfile(tar.getmember(images_txt_path))
        train_test_split_txt = tar.extractfile(tar.getmember(train_test_split_txt_path))
        if not (images_txt and train_test_split_txt):
            raise RuntimeError('cub-200-1011')

        images_txt = images_txt.read().decode('utf-8').splitlines()
        train_test_split_txt = train_test_split_txt.read().decode('utf-8').splitlines()

        id2name = np.genfromtxt(images_txt, dtype=str)
        id2train = np.genfromtxt(train_test_split_txt, dtype=int)
        train_data, train_labels = [], []
        test_data, test_labels = [], []
        for _id in range(id2name.shape[0]):
            image_path = 'CUB_200_2011/images/' + id2name[_id, 1]
            image = tar.extractfile(tar.getmember(image_path))
            if not image:
                raise RuntimeError('image not found')
            image = Image.open(image)
            label = int(id2name[_id, 1][:3]) - 1
            if image.getbands()[0] == 'L':
                image = image.convert('RGB')
            image_np = np.array(image)
            image.close()
            if id2train[_id, 1] == 1:
                train_data.append(image_np)
                train_labels.append(label)
            else:
                test_data.append(image_np)
                test_labels.append(label)
        tar.close()
        pickle.dump((train_data, train_labels),
                    open(os.path.join(self.root, 'processed/train.pkl'), 'wb'))
        pickle.dump((test_data, test_labels),
                    open(os.path.join(self.root, 'processed/test.pkl'), 'wb'))


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]


def preprocess(train, test, zca_bias=0, regularization=0, permute=True):
    origTrainShape = train.shape
    origTestShape = test.shape

    train = np.ascontiguousarray(train, dtype=np.float32).reshape(train.shape[0], -1).astype('float64')
    test = np.ascontiguousarray(test, dtype=np.float32).reshape(test.shape[0], -1).astype('float64')

    nTrain = train.shape[0]
    train_mean = np.mean(train, axis=1)[:, np.newaxis]

    train = train - np.mean(train, axis=1)[:, np.newaxis]
    test = test - np.mean(test, axis=1)[:, np.newaxis]

    train_norms = np.linalg.norm(train, axis=1)
    test_norms = np.linalg.norm(test, axis=1)

    train = train / train_norms[:, np.newaxis]
    test = test / test_norms[:, np.newaxis]

    trainCovMat = 1.0 / nTrain * train.T.dot(train)

    (E, V) = np.linalg.eig(trainCovMat)

    E += zca_bias
    sqrt_zca_eigs = np.sqrt(E + regularization * np.sum(E) / E.shape[0])
    inv_sqrt_zca_eigs = np.diag(np.power(sqrt_zca_eigs, -1))
    global_ZCA = V.dot(inv_sqrt_zca_eigs).dot(V.T)
    inverse_ZCA = V.dot(np.diag(sqrt_zca_eigs)).dot(V.T)

    train = (train).dot(global_ZCA)
    test = (test).dot(global_ZCA)

    train_tensor = torch.Tensor(train.reshape(origTrainShape).astype('float64'))
    test_tensor = torch.Tensor(test.reshape(origTestShape).astype('float64'))
    if permute:
        train_tensor = train_tensor.permute(0, 3, 1, 2).contiguous()
        test_tensor = test_tensor.permute(0, 3, 1, 2).contiguous()

    return train_tensor, test_tensor, (inverse_ZCA, train_norms, train_mean)

def describe_boostdd_schedule(num_classes, block_ipc, num_blocks):
    if num_classes <= 0:
        raise ValueError("describe_boostdd_schedule: num_classes must be positive.")
    if block_ipc <= 0:
        raise ValueError("describe_boostdd_schedule: block_ipc must be positive.")
    if num_blocks <= 0:
        raise ValueError("describe_boostdd_schedule: num_blocks must be positive.")

    total_ipc = block_ipc * num_blocks
    stage_ipc = [block_ipc * (j + 1) for j in range(num_blocks)]
    stage_total = [num_classes * ipc for ipc in stage_ipc]

    return {
        "num_classes": num_classes,
        "block_ipc": block_ipc,
        "num_blocks": num_blocks,
        "total_ipc": total_ipc,
        "total_synthetic": total_ipc * num_classes,
        "stage_ipc": stage_ipc,
        "stage_total_synthetic": stage_total,
    }
