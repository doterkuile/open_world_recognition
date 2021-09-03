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


    def __init__(self, data_path, top_n=9, top_k=5, n_cls=80, n_smpl=100, train_phase='trn', same_class_reverse=False,
                 same_class_extend_entries=False):

        self.n_cls = n_cls
        self.n_smpl = n_smpl
        self.same_class_reverse = same_class_reverse
        self.same_class_extend_entries = same_class_extend_entries
        self.data_path = data_path
        self.top_n = top_n
        self.top_k = top_k
        self.train_phase = train_phase
        self.memory, self.true_labels = self.load_memory(self.data_path)
        if self.train_phase == 'tst':
            self.load_test_idx(self.data_path)
        else:
            if same_class_extend_entries:
                self.load_balanced_train_idx(self.data_path)
            else:
                self.load_train_idx(self.data_path)

            self.load_valid_idx(self.data_path)

        self.classes = [i for i in range(100)]

    #
    def __len__(self):
        if self.train_phase == 'trn':
            return self.train_X0.shape[0]
        elif self.train_phase == 'val':
            return self.valid_X0.shape[0]
        elif self.train_phase == 'tst':
            return self.test_X0.shape[0]

    def __getitem__(self, idx):
        # idx_X0 = np.where(self.train_X0 == idx)
        if self.train_phase == 'trn':
            idx_X0 = self.train_X0[idx]
            idx_X1 = self.train_X1[idx,:]
            y = torch.tensor(self.train_Y[idx], dtype=torch.float)
        elif self.train_phase == 'val':
            idx_X0 = self.valid_X0[idx]
            idx_X1 = self.valid_X1[idx,:]
            y = torch.tensor(self.valid_Y[idx], dtype=torch.float)
        else:
            idx_X0 = self.test_X0[idx]
            idx_X1 = self.test_X1[idx,:]
            y = torch.tensor(self.test_Y[idx], dtype=torch.float)

        x0_rep = self.memory[idx_X0,:]
        x1_rep = self.memory[idx_X1,:]
        true_label_X0 = self.true_labels[idx_X0]
        true_label_X1 = self.true_labels[idx_X1]
        return [x0_rep, x1_rep], y, [true_label_X0, true_label_X1]


    def load_memory(self, data_path):
        features = np.load(data_path)['data_rep']
        try:
            labels = np.load(data_path)['data_labels']
        except KeyError:
            labels = np.zeros(features.shape[0])
        return features, labels

    def load_balanced_train_idx(self, data_path):
        data = np.load(data_path)

        y_train = data['train_Y'][:, -(self.top_n+1):]
        y_extra = np.ones([len(y_train), self.top_n-1, ])
        self.train_Y = np.concatenate([y_train, y_extra], axis=1).reshape(-1, )

        x1 = data['train_X1'][:,-(self.top_n+1):-1, -self.top_k:]
        x1_extra= data['train_X1'][:, -1, -self.top_n*self.top_k:].reshape(x1.shape[0],-1,self.top_k)
        self.train_X1 = np.concatenate([x1, x1_extra], axis=1).reshape(-1,self.top_k)

        self.train_X0 = np.repeat(data['train_X0'], 2 *self.top_n, axis=0)

        return

    def load_train_idx(self, data_path):
        data = np.load(data_path)

        self.train_X0 = np.repeat(data['train_X0'], self.top_n+1, axis=0)
        # Get indices non-similar classes (the above top_n in the second dim)
        self.train_X1 = data['train_X1'][:, -(self.top_n+1):-1, -self.top_k:]

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
        self.valid_X1 = data['valid_X1'][:, -2, -self.top_k:].reshape(-1, 1, self.top_k)

        if self.same_class_reverse:
            valid_X1_sim = data['valid_X1'][:, -1, :self.top_k].reshape(-1, 1, self.top_k)

        else:
            valid_X1_sim = data['valid_X1'][:, -1, -self.top_k:].reshape(-1, 1, self.top_k)

            # Add same class to the non-similar classes
        self.valid_X1 = np.concatenate([self.valid_X1, valid_X1_sim], axis=1).reshape(-1, self.top_k)
        self.valid_Y = data['valid_Y'][:, -2:].reshape(-1, )

    def load_test_idx(self, data_path):
        data = np.load(data_path)
        self.test_X0 = np.repeat(data['test_X0'], self.top_n, axis=0)  # the validation data is balanced.
        self.test_X1 = data['test_X1'][:, -self.top_n:, -self.top_k:].reshape(-1,self.top_k)
        self.test_Y = data['test_Y'][:, -self.top_n:].reshape(-1, )

class ObjectDatasetBase(abc.ABC):

    @abc.abstractmethod
    def __init__(self, dataset_path: str, class_ratio, train_phase):
        self.dataset_path = dataset_path
        self.setTrainingPhaseIdx(class_ratio, train_phase)

        pass

    @abc.abstractmethod
    def getImage(self, x):
        pass

    def setupDataSplit(self, train_data, test_data, val_data, class_ratio, train_phase):
        if train_phase == "encoder_train":
            self.encoderDataSplit(train_data, test_data, val_data, class_ratio, train_phase)
        else:
            self.metaDataSplit(train_data, test_data, val_data, class_ratio, train_phase)

    def getData(self):
        return (self.train_data, self.val_data, self.test_data)

    def getDataloaders(self, batch_size):
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True, pin_memory=True)#, num_workers=4)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=True, pin_memory=True)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=True, pin_memory=True)#, num_workers=4)

        return (self.train_loader, self.val_loader, self.test_loader)

    def setTrainingPhaseIdx(self, class_ratio, train_phase):
        self.class_idx = {key: [] for key in class_ratio.keys()}
        self.class_idx['encoder_train'] = np.arange(0, class_ratio['encoder_train'])

        if self.class_idx['encoder_train'].size == 0:
            l2ac_first_idx = 0

            self.class_idx['l2ac_train'] = np.arange(l2ac_first_idx,
                                                     l2ac_first_idx + class_ratio['l2ac_train'])

        else:
            l2ac_first_idx = self.class_idx['encoder_train'].max()

            self.class_idx['l2ac_train'] = np.arange(l2ac_first_idx + 1,
                                                  l2ac_first_idx + class_ratio['l2ac_train'] + 1)

        self.class_idx['l2ac_val'] = np.arange(self.class_idx['l2ac_train'].max() + 1,
                                                  self.class_idx['l2ac_train'].max() + class_ratio['l2ac_val'] + 1)
        self.class_idx['l2ac_test'] = np.arange(self.class_idx['l2ac_val'].max() + 1,
                                                  self.class_idx['l2ac_val'].max() + class_ratio['l2ac_test'] + 1)
        self.train_phase = train_phase

    def setDataTransforms(self, image_resize):

        self.transform_test = transforms.Compose([
            transforms.Resize(image_resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.tst_mean_pixel, std=self.tst_std_pixel),

        ])
        if self.train_phase == 'encoder_train':
            self.transform_train = transforms.Compose([
                transforms.Resize(image_resize),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.trn_mean_pixel, std=self.trn_std_pixel),
            ])
        else:
            self.transform_train = self.transform_test

        return




# CIFAR100
class CIFAR100Dataset(ObjectDatasetBase):

    def __init__(self, dataset_path, class_ratio, train_phase='encoder', image_resize=64):
        super().__init__(dataset_path, class_ratio, train_phase)

        self.trn_mean_pixel = [x / 255.0 for x in [129.3, 124.1,  112.4]]
        self.trn_std_pixel = [x / 255.0 for x in [68.2, 65.4, 70.4]]

        self.tst_mean_pixel = [x / 255.0 for x in [129.7, 124.3, 112.7]]
        self.tst_std_pixel = [x / 255.0 for x in [68.4, 65.6, 70.7]]

        self.setDataTransforms(image_resize)

        train_data = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=self.transform_train)
        test_data = datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=self.transform_test)
        self.setupDataSplit(train_data, test_data, class_ratio, train_phase)

        self.image_shape = [1, 3, image_resize, image_resize]
        self.classes = [i for i in range(len(self.train_data.classes))]

    def setupDataSplit(self,train_data, test_data, class_ratio, train_phase):

        train_data.classes = [train_data.classes[i] for i in self.class_idx[train_phase]]
        train_data.class_to_idx = {i: train_data.class_to_idx[i] for i in train_data.classes}
        idx = [i for i in range(0,len(train_data.targets)) if train_data.targets[i] in self.class_idx[train_phase]]
        train_data.targets = [train_data.targets[i] for i in idx]
        train_data.data = train_data.data[idx]


        self.train_data = train_data

        test_data.classes = [test_data.classes[i] for i in self.class_idx[train_phase]]
        test_data.class_to_idx = {i: test_data.class_to_idx[i] for i in test_data.classes}

        idx = [i for i in range(0,len(test_data.targets)) if test_data.targets[i] in self.class_idx[train_phase]]
        test_data.targets = [test_data.targets[i] for i in idx]
        test_data.data = test_data.data[idx]
        self.test_data = test_data


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


    def __init__(self, dataset_path, class_ratio, train_phase='encoder', image_resize=64):
        super().__init__(dataset_path, class_ratio, train_phase)

        self.trn_mean_pixel = [0.4802, 0.4481, 0.3975]
        self.trn_std_pixel = [0.2302, 0.2265, 0.2262]

        self.tst_mean_pixel = [0.4802, 0.4481, 0.3975]
        self.tst_std_pixel = [0.2302, 0.2265, 0.2262]

        self.setDataTransforms(image_resize)

        train_data = datasets.ImageFolder(root=f'{dataset_path}/train', transform=self.transform_train)
        test_data = datasets.ImageFolder(root=f'{dataset_path}/test', transform=self.transform_test)
        val_data = datasets.ImageFolder(root=f'{dataset_path}/val', transform=self.transform_test)


        self.setupDataSplit(train_data, test_data, val_data, class_ratio, train_phase)

        self.image_shape = [1, 3, image_resize, image_resize]
        self.classes = [self.class_idx[train_phase]]


    def encoderDataSplit(self, train_data, test_data, val_data, class_ratio, train_phase):

        train_data.classes = [train_data.classes[i] for i in self.class_idx[train_phase]]
        train_data.class_to_idx = {i: train_data.class_to_idx[i] for i in train_data.classes}
        train_data.imgs = [(img[0], img[1]) for img in train_data.imgs if img[1] in self.class_idx[train_phase]]
        train_data.samples = [(s[0], s[1]) for s in train_data.imgs if s[1] in self.class_idx[train_phase]]
        train_data.targets = [t for t in train_data.targets if t in self.class_idx[train_phase]]

        self.train_data = train_data

        test_data.classes = [test_data.classes[i] for i in self.class_idx[train_phase]]
        test_data.class_to_idx = {i: test_data.class_to_idx[i] for i in test_data.classes}
        test_data.imgs = [(img[0], img[1]) for img in test_data.imgs if img[1] in self.class_idx[train_phase]]
        test_data.samples = [(s[0], s[1]) for s in test_data.imgs if s[1] in self.class_idx[train_phase]]
        test_data.targets = [t for t in test_data.targets if t in self.class_idx[train_phase]]

        self.test_data = test_data

        val_data.classes = [val_data.classes[i] for i in self.class_idx[train_phase]]
        val_data.class_to_idx = {i: val_data.class_to_idx[i] for i in val_data.classes}
        val_data.imgs = [(img[0], img[1]) for img in val_data.imgs if img[1] in self.class_idx[train_phase]]
        val_data.samples = [(s[0], s[1]) for s in val_data.imgs if s[1] in self.class_idx[train_phase]]
        val_data.targets = [t for t in val_data.targets if t in self.class_idx[train_phase]]

        self.val_data = val_data

        return

    def metaDataSplit(self, train_data, test_data, val_data, class_ratio, train_phase):
        # Datasplitter for L2AC algorithm
        train_data.classes = [train_data.classes[i] for i in self.class_idx[train_phase]]
        train_data.class_to_idx = {i: train_data.class_to_idx[i] for i in train_data.classes}
        train_data.imgs = [(img[0], img[1]) for img in train_data.imgs if img[1] in self.class_idx[train_phase]]
        train_data.samples = [(s[0], s[1]) for s in train_data.imgs if s[1] in self.class_idx[train_phase]]
        train_data.targets = [t for t in train_data.targets if t in self.class_idx[train_phase]]

        self.train_data = train_data

        test_data.classes = [test_data.classes[i] for i in self.class_idx[train_phase]]
        test_data.class_to_idx = {i: test_data.class_to_idx[i] for i in test_data.classes}
        test_data.imgs = [(img[0], img[1]) for img in test_data.imgs if img[1] in self.class_idx[train_phase]]
        test_data.samples = [(s[0], s[1]) for s in test_data.imgs if s[1] in self.class_idx[train_phase]]
        test_data.targets = [t for t in test_data.targets if t in self.class_idx[train_phase]]

        self.test_data = test_data

        val_data.classes = [val_data.classes[i] for i in self.class_idx[train_phase]]
        val_data.class_to_idx = {i: val_data.class_to_idx[i] for i in val_data.classes}
        val_data.imgs = [(img[0], img[1]) for img in val_data.imgs if img[1] in self.class_idx[train_phase]]
        val_data.samples = [(s[0], s[1]) for s in val_data.imgs if s[1] in self.class_idx[train_phase]]
        val_data.targets = [t for t in val_data.targets if t in self.class_idx[train_phase]]

        self.val_data = val_data


        return


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

    def getImageFromClass(self, class_idx):

        inv_normalize = transforms.Normalize(
            mean=[-self.trn_mean_pixel[0] / self.trn_std_pixel[0], -self.trn_mean_pixel[1] / self.trn_std_pixel[1], -self.trn_mean_pixel[1] / self.trn_std_pixel[1]],
            std=[1 / self.trn_std_pixel[0], 1 / self.trn_std_pixel[1], 1 / self.trn_std_pixel[2]]
        )
        image_idx = np.where(np.array(self.train_data.targets) == class_idx)[0][0]
        im = self.train_data[image_idx][0]
        im = inv_normalize(im)
        # im =self.test_data[x][0]
        label = self.train_data[image_idx][1]
        im = np.transpose(im.numpy(), (1, 2, 0))


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
