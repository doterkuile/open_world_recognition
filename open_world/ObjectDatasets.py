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

import abc

class ObjectDatasetBase(abc.ABC):

    @abc.abstractmethod
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        pass

    @abc.abstractmethod
    def getData(self):
        pass

    @abc.abstractmethod
    def getDataloaders(self, batch_size):
        pass



# MNIST
class MNISTDataset(ObjectDatasetBase):

    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.transform = transforms.ToTensor()


    def getData(self):
        self.train_data = datasets.MNIST(root='../datasets', train=True, download=True, transform=self.transform)
        self.test_data = datasets.MNIST(root='../datasets', train=False, download=True, transform=self.transform)

        return (self.train_data, self.test_data)

    def getDataloaders(self, batch_size):
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

        return (self.train_loader, self.test_loader)


# MNIST FASHION
class MNISTFashionDataset(ObjectDatasetBase):

    def __init__(self):
        super.__init__()


# CATS_DOGS
class CatDogDataset(ObjectDatasetBase):

    def __init__(self):
        super.__init__()