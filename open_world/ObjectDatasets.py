import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

import abc

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
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def getImage(self, x):
        return self.test_data[x][0].reshape(self.image_shape[-2:])


# MNIST FASHION
class FashionMNISTDataset(ObjectDatasetBase):

    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.transform = transforms.ToTensor()
        self.train_data = datasets.FashionMNIST(root='../datasets', train=True, download=True, transform=self.transform)
        self.test_data = datasets.FashionMNIST(root='../datasets', train=False, download=True, transform=self.transform)
        self.image_shape = self.image_shape = [1, 1, 28, 28]
        self.class_names = ['T-shirt', 'Trouser', 'Sweater', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']


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
        self.class_names = ["Cat", "Dog"]


    def getImage(self, x):

        inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        im = inv_normalize(self.test_data[x][0])
        im = np.transpose(im.numpy(), (1, 2, 0))
        return im