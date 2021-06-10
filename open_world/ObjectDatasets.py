import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.core._multiarray_umath import ndarray
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torch.utils.data as data_utils

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

import abc


class MetaDataset(data_utils.Dataset):


    def __init__(self, data_path, top_n=9, top_k=5, n_cls=80, n_smpl=100, train=True, same_class_reverse=False,
                 same_class_extend_entries=False):

        self.n_cls = n_cls
        self.n_smpl = n_smpl
        self.same_class_reverse = same_class_reverse
        self.same_class_extend_entries = same_class_extend_entries
        self.data_path = data_path + '/memory_' + str(n_cls) + '_' + str(n_smpl) + '.npz'
        self.top_n = top_n
        self.top_k = top_k
        self.train = train
        self.memory, self.true_labels = self.load_memory(self.data_path)
        self.load_train_idx(self.data_path)
        self.load_valid_idx(self.data_path)
        self.classes = [i for i in range(100)]

    #
    def __len__(self):
        if self.train:
            return self.train_X0.shape[0]
        else:
            return self.valid_X0.shape[0]

    def __getitem__(self, idx):
        # idx_X0 = np.where(self.train_X0 == idx)
        if self.train:
            idx_X0 = self.train_X0[idx]
            idx_X1 = self.train_X1[idx,:]
            x0_rep = self.memory[idx_X0,:]
            x1_rep = self.memory[idx_X1,:]
            y = torch.tensor(self.train_Y[idx], dtype=torch.float)
            true_label_X0 = self.true_labels[idx_X0]
            true_label_X1 = self.true_labels[idx_X1]
        else:
            idx_X0 = self.valid_X0[idx]
            idx_X1 = self.valid_X1[idx,:]
            x0_rep = self.memory[idx_X0, :]
            x1_rep = self.memory[idx_X1, :]
            y = torch.tensor(self.valid_Y[idx], dtype=torch.float)
            true_label_X0 = self.true_labels[idx_X0]
            true_label_X1 = self.true_labels[idx_X1]

        return [x0_rep, x1_rep], y, [true_label_X0, true_label_X1]


    def load_memory(self, data_path):
        features = np.load(data_path)['train_rep']
        try:
            labels = np.load(data_path)['labels_rep']
        except KeyError:
            labels = np.zeros(features.shape[0])
        return features, labels

    def load_train_idx(self, data_path):
        data = np.load(data_path)
        self.train_X0 = np.repeat(data['train_X0'], self.top_n, axis=0)
        # Get indices non-similar classes
        train_X1_nonsim = data['train_X1'][:, -(self.top_n-1):, -self.top_k:].reshape(-1, self.top_k)

        if self.same_class_reverse:
            train_X1_sim = data['train_X1'][:, -1, :self.top_k].reshape(-1,self.top_k)
        self.train_Y = data['train_Y'][:, -self.top_n:].reshape(-1, )

    def load_valid_idx(self, data_path):
        data = np.load(data_path)
        self.valid_X0 = np.repeat(data['valid_X0'], 2, axis=0)  # the validation data is balanced.
        self.valid_X1 = data['valid_X1'][:, -2:, -self.top_k:].reshape(-1, self.top_k)
        self.valid_Y = data['valid_Y'][:, -2:].reshape(-1, )

class ObjectDatasetBase(abc.ABC):

    @abc.abstractmethod
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        pass

    @abc.abstractmethod
    def getImage(self, x):
        pass

    def getData(self):
        return (self.train_data, self.test_data)

    def getDataloaders(self, batch_size):
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

        return (self.train_loader, self.test_loader)



# MNIST
class MNISTDataset(ObjectDatasetBase):

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

        self.transform = transforms.ToTensor()

        self.train_data = datasets.MNIST(root='../datasets', train=True, download=True, transform=self.transform)
        self.test_data = datasets.MNIST(root='../datasets', train=False, download=True, transform=self.transform)
        self.image_shape = [1, 1, 28, 28]
        self.classes = 	[int(cls.split(' -')[0]) for cls in self.train_data.classes]


    def getImage(self, x):
        return self.test_data[x][0].reshape(self.image_shape[-2:])

# CIFAR100
class CIFAR100Dataset(ObjectDatasetBase):

    def __init__(self, dataset_path, top_n=9, top_k=5, n_cls=100, n_smpl=80, enable_training=True,
                 same_class_reverse=False, same_class_extend_entries=False):
        super().__init__(dataset_path)

        mean_pixel = [x / 255.0 for x in [125.3, 123.0, 113.9]]
        std_pixel = [x / 255.0 for x in [63.0, 62.1, 66.7]]

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pixel, std=std_pixel)
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pixel, std=std_pixel)
        ])

        self.train_data = datasets.CIFAR100(root='datasets', train=True, download=True, transform=self.transform_train)
        self.test_data = datasets.CIFAR100(root='datasets', train=False, download=True, transform=self.transform_test)
        self.image_shape = [1, 3, 32, 32]
        self.classes = [i for i in range(len(self.train_data.classes))]

    def getImage(self, x):

        # inv_normalize = transforms.Normalize(
        #     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        #     std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        # )
        # im = inv_normalize(self.test_data[x][0])
        im =self.test_data[x][0]
        im = np.transpose(im.numpy(), (1, 2, 0))
        return im





# MNIST FASHION
class FashionMNISTDataset(ObjectDatasetBase):

    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.transform = transforms.ToTensor()
        self.train_data = datasets.FashionMNIST(root='../datasets', train=True, download=True, transform=self.transform)
        self.test_data = datasets.FashionMNIST(root='../datasets', train=False, download=True, transform=self.transform)
        self.image_shape = self.image_shape = [1, 1, 28, 28]
        self.classes = ['T-shirt', 'Trouser', 'Sweater', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']


    def getImage(self, x):
        return self.test_data[x][0].reshape(self.image_shape[-2:])

# CATS_DOGS
class CatDogDataset(ObjectDatasetBase):

    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(10),  # rotate +/- 10 degrees
            transforms.RandomHorizontalFlip(),  # reverse 50% of images
            transforms.Resize(224),  # resize shortest side to 224 pixels
            transforms.CenterCrop(224),  # crop longest side to 224 pixels at center
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.train_data = datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=self.train_transform)
        self.test_data = datasets.ImageFolder(os.path.join(dataset_path, 'test'), transform=self.test_transform)
        self.image_shape = [1, 3, 224, 224]
        self.classes = ["Cat", "Dog"]


    def getImage(self, x):

        inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        im = inv_normalize(self.test_data[x][0])
        im = np.transpose(im.numpy(), (1, 2, 0))
        return im