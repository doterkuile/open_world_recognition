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
        self.data_path = data_path
        self.top_n = top_n
        self.top_k = top_k
        self.train = train
        self.trn_memory, self.trn_true_labels = self.load_trn_memory(self.data_path)
        self.tst_memory, self.tst_true_labels = self.load_tst_memory(self.data_path)

        if same_class_extend_entries:
            self.load_balanced_train_idx(self.data_path)
        else:
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
            x0_rep = self.trn_memory[idx_X0,:]
            x1_rep = self.trn_memory[idx_X1,:]
            y = torch.tensor(self.train_Y[idx], dtype=torch.float)
            true_label_X0 = self.trn_true_labels[idx_X0]
            true_label_X1 = self.trn_true_labels[idx_X1]
        else:
            idx_X0 = self.valid_X0[idx]
            idx_X1 = self.valid_X1[idx,:]
            x0_rep = self.tst_memory[idx_X0, :]
            x1_rep = self.tst_memory[idx_X1, :]
            y = torch.tensor(self.valid_Y[idx], dtype=torch.float)
            true_label_X0 = self.tst_true_labels[idx_X0]
            true_label_X1 = self.tst_true_labels[idx_X1]

        return [x0_rep, x1_rep], y, [true_label_X0, true_label_X1]


    def load_trn_memory(self, data_path):
        features = np.load(data_path)['train_rep']
        try:
            labels = np.load(data_path)['trn_labels_rep']
        except KeyError:
            labels = np.zeros(features.shape[0])
        return features, labels

    def load_tst_memory(self, data_path):
        features = np.load(data_path)['test_rep']
        try:
            labels = np.load(data_path)['tst_labels_rep']
        except KeyError:
            labels = np.zeros(features.shape[0])
        return features, labels

    def load_balanced_train_idx(self, data_path):
        data = np.load(data_path)

        y_train = data['train_Y'][:, -(self.top_n+1):]
        y_extra = np.ones([len(y_train), self.top_n-1, ])
        self.train_Y = np.concatenate([y_train, y_extra], axis=1).reshape(-1, )

        x1 = data['train_X1'][:, :self.top_n, -self.top_k:]
        x1_extra= data['train_X1'][:, -1, -self.top_n*self.top_k:].reshape(x1.shape[0],-1,self.top_k)
        x1_test = x1_extra
        self.train_X1 = np.concatenate([x1, x1_extra], axis=1).reshape(-1,self.top_k)

        self.train_X0 = np.repeat(data['train_X0'], 2 *self.top_n, axis=0)

        return

    def load_train_idx(self, data_path):
        data = np.load(data_path)
        

        self.train_X0 = np.repeat(data['train_X0'], self.top_n+1, axis=0)
        # Get indices non-similar classes (the above top_n in the second dim)
        self.train_X1 = data['train_X1'][:, :self.top_n, -self.top_k:]

        # Get indices of the same class samples
        if self.same_class_reverse:
            train_X1_sim = data['train_X1'][:, -1, :self.top_k].reshape(-1, 1, self.top_k)

        else:
            train_X1_sim = data['train_X1'][:, -1, -self.top_k:].reshape(-1, 1, self.top_k)

        # Add same class to the non-similar classes
        self.train_X1 = np.concatenate([self.train_X1, train_X1_sim], axis=1).reshape(-1, self.top_k)

        # Get the labels (n+1 to account for the extra same class samples)
        self.train_Y = data['train_Y'][:, -(self.top_n+1):].reshape(-1, )

    def load_valid_idx(self, data_path):
        data = np.load(data_path)
        self.valid_X0 = np.repeat(data['valid_X0'], 2, axis=0)  # the validation data is balanced.
        self.valid_X1 = data['valid_X1'][:, 0, -self.top_k:].reshape(-1, 1, self.top_k)

        if self.same_class_reverse:
            valid_X1_sim = data['valid_X1'][:, -1, :self.top_k].reshape(-1, 1, self.top_k)

        else:
            valid_X1_sim = data['valid_X1'][:, -1, -self.top_k:].reshape(-1, 1, self.top_k)

            # Add same class to the non-similar classes
        self.valid_X1 = np.concatenate([self.valid_X1, valid_X1_sim], axis=1).reshape(-1, self.top_k)



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
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=True, pin_memory=True)

        return (self.train_loader, self.test_loader)




# CIFAR100
class CIFAR100Dataset(ObjectDatasetBase):

    def __init__(self, dataset_path, image_resize=64):
        super().__init__(dataset_path)

        self.trn_mean_pixel = [x / 255.0 for x in [129.3, 124.1,  112.4]]
        self.trn_std_pixel = [x / 255.0 for x in [68.2, 65.4, 70.4]]

        self.tst_mean_pixel = [x / 255.0 for x in [129.7, 124.3, 112.7]]
        self.tst_std_pixel = [x / 255.0 for x in [68.4, 65.6, 70.7]]

        self.transform_train = transforms.Compose([
                transforms.Resize(image_resize),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Normalize(mean=self.trn_mean_pixel, std=self.trn_std_pixel),
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize(image_resize),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=self.tst_mean_pixel, std=self.tst_std_pixel),

        ])

        self.train_data = datasets.CIFAR100(root='datasets', train=True, download=True, transform=self.transform_train)
        self.test_data = datasets.CIFAR100(root='datasets', train=False, download=True, transform=self.transform_test)
        self.image_shape = [1, 3, 32, 32]
        self.classes = [i for i in range(len(self.train_data.classes))]

    def getImage(self, x):

        inv_normalize = transforms.Normalize(
            mean=[-self.trn_mean_pixel[0] / self.trn_std_pixel[0], -self.trn_mean_pixel[1] / self.trn_std_pixel[1], -self.trn_mean_pixel[1] / self.trn_std_pixel[1]],
            std=[1 / self.trn_std_pixel[0], 1 / self.trn_std_pixel[1], 1 / self.trn_std_pixel[2]]
        )
        im = inv_normalize(self.test_data[x][0])
        # im =self.test_data[x][0]
        label = self.test_data[x][1]
        im = np.transpose(im.numpy(), (1, 2, 0))
        plt.imshow(np.transpose(im, (0, 1, 2)))
        plt.title(self.train_data.classes[label])
        plt.show()

        return im, label


class TinyImageNetDataset(ObjectDatasetBase):


    def __init__(self, dataset_path, image_resize=64):
        super().__init__(dataset_path)

        self.trn_mean_pixel = [0.4914, 0.4822, 0.4465]
        self.trn_std_pixel = [0.2023, 0.1994, 0.2010]

        self.tst_mean_pixel = [0.4914, 0.4822, 0.4465]
        self.tst_std_pixel = [0.2023, 0.1994, 0.2010]

        self.transform_train = transforms.Compose([
                transforms.Resize(image_resize),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Normalize(mean=self.trn_mean_pixel, std=self.trn_std_pixel),
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize(image_resize),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=self.tst_mean_pixel, std=self.tst_std_pixel),

        ])

        self.train_data = datasets.ImageFolder(root=f'{dataset_path}/train', transform=self.transform_train)
        self.test_data = datasets.ImageFolder(root=f'{dataset_path}/test', transform=self.transform_test)
        # self.image_shape = [1, 3, 32, 32]
        self.classes = [i for i in range(len(self.train_data.classes))]

    def getImage(self, x):

        inv_normalize = transforms.Normalize(
            mean=[-self.trn_mean_pixel[0] / self.trn_std_pixel[0], -self.trn_mean_pixel[1] / self.trn_std_pixel[1], -self.trn_mean_pixel[1] / self.trn_std_pixel[1]],
            std=[1 / self.trn_std_pixel[0], 1 / self.trn_std_pixel[1], 1 / self.trn_std_pixel[2]]
        )
        im = inv_normalize(self.test_data[x][0])
        # im =self.test_data[x][0]
        label = self.test_data[x][1]
        im = np.transpose(im.numpy(), (1, 2, 0))
        plt.imshow(np.transpose(im, (0, 1, 2)))
        plt.title(self.train_data.classes[label])
        plt.show()

        return im, label


def main():


    data_path = "datasets/CIFAR100"
    top_n = 9
    top_k = 10
    n_cls=80
    n_smpl=500
    train=True
    same_class_reverse=False
    same_class_extend_entries=True


    dataset = MetaDataset(data_path, top_n, top_k, n_cls, n_smpl, train, same_class_reverse,
             same_class_extend_entries)



if __name__ == "__main__":
    main()