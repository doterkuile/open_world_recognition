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
from enum import Enum
import os

import abc



class TrainPhase(Enum):
    ENCODER_TRN = 'encoder_trn'
    ENCODER_VAL = 'encoder_val'
    META_TRN = 'meta_trn'
    META_VAL = 'meta_val'
    META_TST = 'meta_tst'


class MetaDataset(data_utils.Dataset):


    def __init__(self, data_path, top_n=9, top_k=5, n_cls=80, n_smpl=100, train_phase=TrainPhase.META_TRN, same_class_reverse=False,
                 same_class_extend_entries=False, unknown_classes=0, memory_classes=1):

        self.n_cls = n_cls
        self.n_smpl = n_smpl
        self.same_class_reverse = same_class_reverse
        self.same_class_extend_entries = same_class_extend_entries
        self.data_path = data_path
        self.top_n = top_n
        self.top_k = top_k
        self.train_phase = train_phase
        self.unknown_classes = unknown_classes
        self.memory_classes = memory_classes
        self.memory, self.true_labels = self.load_memory(self.data_path)
        if self.train_phase == TrainPhase.META_TST:
            self.load_test_idx(self.data_path, unknown_classes, memory_classes)
        else:
            if same_class_extend_entries:
                self.load_balanced_train_idx(self.data_path)
            else:
                self.load_train_idx(self.data_path)

            self.load_valid_idx(self.data_path)

        self.classes = [i for i in range(100)]

    #
    def __len__(self):
        if self.train_phase == TrainPhase.META_TRN:
            return self.train_X0.shape[0]
        elif self.train_phase == TrainPhase.META_VAL:
            return self.valid_X0.shape[0]
        elif self.train_phase == TrainPhase.META_TST:
            return self.test_X0.shape[0]

    def __getitem__(self, idx):
        # idx_X0 = np.where(self.train_X0 == idx)
        if self.train_phase == TrainPhase.META_TRN:
            idx_X0 = self.train_X0[idx]
            idx_X1 = self.train_X1[idx,:]
            y = torch.tensor(self.train_Y[idx], dtype=torch.float)
        elif self.train_phase == TrainPhase.META_VAL:
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

    def load_test_idx(self, data_path, unknown_classes, memory_classes):
        data = np.load(data_path)

        # Complete data
        X0_all = data['test_X0']
        X1_all = data['test_X1']
        Y_all = data['test_Y']


        if np.isin(self.true_labels[data['train_X0']], self.true_labels[data['test_X0']]).any():
            # complete set of classes of memory and input
            memory_label_set = np.unique(self.true_labels[data['train_X0']])
            input_label_set = np.unique(self.true_labels[data['test_X0']])

            # Find unknown classes in input
            input_unknown_labels_set = input_label_set[(~np.isin(input_label_set, memory_label_set)).nonzero()[0]]

            # Randomly select a subset of indices of known (memory) and unknown (input) classes
            shuffle_idx_known = np.random.permutation(memory_label_set.shape[0])[:memory_classes]
            shuffle_idx_unknown = input_unknown_labels_set[np.random.permutation(input_unknown_labels_set.shape[0])[:unknown_classes]]
            shuffle_idx_unknown = np.where(input_label_set.reshape(-1,1) == shuffle_idx_unknown)[0]

        else:
            memory_label_set = np.unique(self.true_labels[data['test_X0']])
            input_label_set = np.unique(self.true_labels[data['test_X0']])
            shuffle_idx = np.random.permutation(memory_label_set.shape[0])
            shuffle_idx_known = shuffle_idx[:memory_classes]
            shuffle_idx_unknown = shuffle_idx[memory_classes:(memory_classes + unknown_classes)]

        # Select set of classes for known and unknown
        known_labels_input = memory_label_set[shuffle_idx_known]
        unknown_labels_input = input_label_set[shuffle_idx_unknown]

        labels_input = np.concatenate([known_labels_input, unknown_labels_input])

        # Find indices of all samples that have input lables
        idx_X0 = np.where(self.true_labels[X0_all] == labels_input)[0]

        # Filter out right input samples with corresponding memory samples and binary classification label
        X0 = X0_all[idx_X0]
        X1_subset = X1_all[idx_X0]
        Y_subset = Y_all[idx_X0]

        memory_true_labels = self.true_labels[X1_subset]

        memory_idx = np.isin(memory_true_labels[:, :, 0], input_label_set[shuffle_idx_known]).nonzero()
        X1 = X1_subset[memory_idx[0], memory_idx[1], :].reshape(
            X0.shape[0], memory_classes, -1)
        Y = Y_subset[memory_idx[0], memory_idx[1]].reshape(X0.shape[0], memory_classes)

        self.test_X0 = np.repeat(X0, self.top_n, axis=0)  # the validation data is balanced.
        self.test_X1 = X1[:, -self.top_n:, -self.top_k:].reshape(-1,self.top_k)
        self.test_Y = Y[:, -self.top_n:].reshape(-1, )

        return


    def replace_test_data(self, data_rep, cls_rep, labels, X0_new, X1_new, Y_new):


        self.memory = data_rep
        self.true_labels = labels
        self.test_X0 = X0_new
        self.test_X1 = X1_new
        self.test_y = Y_new
        return

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
        if train_phase == TrainPhase.ENCODER_TRN or train_phase == TrainPhase.ENCODER_VAL:
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
        self.class_idx[TrainPhase.ENCODER_TRN.value] = np.arange(0, class_ratio[TrainPhase.ENCODER_TRN.value])

        if self.class_idx[TrainPhase.ENCODER_TRN.value].size == 0:
            l2ac_first_idx = 0

            self.class_idx[TrainPhase.META_TRN.value] = np.arange(l2ac_first_idx,
                                                     l2ac_first_idx + class_ratio[TrainPhase.META_TRN.value])

        else:
            l2ac_first_idx = self.class_idx[TrainPhase.ENCODER_TRN.value].max()

            self.class_idx[TrainPhase.META_TRN.value] = np.arange(l2ac_first_idx + 1,
                                                  l2ac_first_idx + class_ratio[TrainPhase.META_TRN.value] + 1)

        self.class_idx[TrainPhase.META_VAL.value] = np.arange(self.class_idx[TrainPhase.META_TRN.value].max() + 1,
                                                  self.class_idx[TrainPhase.META_TRN.value].max() + class_ratio[TrainPhase.META_VAL.value] + 1)
        self.class_idx[TrainPhase.META_TST.value] = np.arange(self.class_idx[TrainPhase.META_VAL.value].max() + 1,
                                                  self.class_idx[TrainPhase.META_VAL.value].max() + class_ratio[TrainPhase.META_TST.value] + 1)
        self.train_phase = train_phase

    def setDataTransforms(self, image_resize):

        self.transform_test = transforms.Compose([
            transforms.Resize(image_resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.tst_mean_pixel, std=self.tst_std_pixel),

        ])
        if self.train_phase == TrainPhase.ENCODER_TRN:
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

    def __init__(self, dataset_path, class_ratio, train_phase=TrainPhase.ENCODER_TRN, image_resize=64):
        super().__init__(dataset_path, class_ratio, train_phase)

        self.trn_mean_pixel = [x / 255.0 for x in [129.3, 124.1,  112.4]]
        self.trn_std_pixel = [x / 255.0 for x in [68.2, 65.4, 70.4]]

        self.tst_mean_pixel = [x / 255.0 for x in [129.7, 124.3, 112.7]]
        self.tst_std_pixel = [x / 255.0 for x in [68.4, 65.6, 70.7]]

        self.setDataTransforms(image_resize)

        train_data = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=self.transform_train)
        val_data = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=self.transform_test)
        test_data = datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=self.transform_test)
        self.setupDataSplit(train_data, test_data, val_data, class_ratio, train_phase)

        self.image_shape = [1, 3, image_resize, image_resize]
        self.classes = [i for i in range(len(self.train_data.classes))]

    def setupDataSplit(self, train_data, test_data, val_data, class_ratio, train_phase):

        train_data.classes = [train_data.classes[i] for i in self.class_idx[train_phase.value]]
        train_data.class_to_idx = {i: train_data.class_to_idx[i] for i in train_data.classes}
        idx = [i for i in range(0,len(train_data.targets)) if train_data.targets[i] in self.class_idx[train_phase.value]]
        train_data.targets = [train_data.targets[i] for i in idx]
        train_data.data = train_data.data[idx]

        self.train_data = train_data

        val_data.classes = [val_data.classes[i] for i in self.class_idx[train_phase.value]]
        val_data.class_to_idx = {i: val_data.class_to_idx[i] for i in val_data.classes}
        idx = [i for i in range(0,len(val_data.targets)) if val_data.targets[i] in self.class_idx[train_phase.value]]
        val_data.targets = [val_data.targets[i] for i in idx]
        val_data.data = val_data.data[idx]

        self.val_data = val_data

        test_data.classes = [test_data.classes[i] for i in self.class_idx[train_phase.value]]
        test_data.class_to_idx = {i: test_data.class_to_idx[i] for i in test_data.classes}

        idx = [i for i in range(0,len(test_data.targets)) if test_data.targets[i] in self.class_idx[train_phase.value]]
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


    def __init__(self, dataset_path, class_ratio, train_phase=TrainPhase.ENCODER_TRN, image_resize=64):
        super().__init__(dataset_path,   class_ratio, train_phase)

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
        self.classes = [self.class_idx[train_phase.value]]


    def encoderDataSplit(self, train_data, test_data, val_data, class_ratio, train_phase):

        train_data.classes = [train_data.classes[i] for i in self.class_idx[train_phase.value]]
        train_data.class_to_idx = {i: train_data.class_to_idx[i] for i in train_data.classes}
        train_data.imgs = [(img[0], img[1]) for img in train_data.imgs if img[1] in self.class_idx[train_phase.value]]
        train_data.samples = [(s[0], s[1]) for s in train_data.imgs if s[1] in self.class_idx[train_phase.value]]
        train_data.targets = [t for t in train_data.targets if t in self.class_idx[train_phase.value]]

        self.train_data = train_data

        test_data.classes = [test_data.classes[i] for i in self.class_idx[train_phase.value]]
        test_data.class_to_idx = {i: test_data.class_to_idx[i] for i in test_data.classes}
        test_data.imgs = [(img[0], img[1]) for img in test_data.imgs if img[1] in self.class_idx[train_phase.value]]
        test_data.samples = [(s[0], s[1]) for s in test_data.imgs if s[1] in self.class_idx[train_phase.value]]
        test_data.targets = [t for t in test_data.targets if t in self.class_idx[train_phase.value]]

        self.test_data = test_data

        val_data.classes = [val_data.classes[i] for i in self.class_idx[train_phase.value]]
        val_data.class_to_idx = {i: val_data.class_to_idx[i] for i in val_data.classes}
        val_data.imgs = [(img[0], img[1]) for img in val_data.imgs if img[1] in self.class_idx[train_phase.value]]
        val_data.samples = [(s[0], s[1]) for s in val_data.imgs if s[1] in self.class_idx[train_phase.value]]
        val_data.targets = [t for t in val_data.targets if t in self.class_idx[train_phase.value]]

        self.val_data = val_data

        return

    def metaDataSplit(self, train_data, test_data, val_data, class_ratio, train_phase):
        # Datasplitter for L2AC algorithm
        train_data.classes = [train_data.classes[i] for i in self.class_idx[train_phase.value]]
        train_data.class_to_idx = {i: train_data.class_to_idx[i] for i in train_data.classes}
        train_data.imgs = [(img[0], img[1]) for img in train_data.imgs if img[1] in self.class_idx[train_phase.value]]
        train_data.samples = [(s[0], s[1]) for s in train_data.imgs if s[1] in self.class_idx[train_phase.value]]
        train_data.targets = [t for t in train_data.targets if t in self.class_idx[train_phase.value]]

        self.train_data = train_data

        test_data.classes = [test_data.classes[i] for i in self.class_idx[train_phase.value]]
        test_data.class_to_idx = {i: test_data.class_to_idx[i] for i in test_data.classes}
        test_data.imgs = [(img[0], img[1]) for img in test_data.imgs if img[1] in self.class_idx[train_phase.value]]
        test_data.samples = [(s[0], s[1]) for s in test_data.imgs if s[1] in self.class_idx[train_phase.value]]
        test_data.targets = [t for t in test_data.targets if t in self.class_idx[train_phase.value]]

        self.test_data = test_data

        val_data.classes = [val_data.classes[i] for i in self.class_idx[train_phase.value]]
        val_data.class_to_idx = {i: val_data.class_to_idx[i] for i in val_data.classes}
        val_data.imgs = [(img[0], img[1]) for img in val_data.imgs if img[1] in self.class_idx[train_phase.value]]
        val_data.samples = [(s[0], s[1]) for s in val_data.imgs if s[1] in self.class_idx[train_phase.value]]
        val_data.targets = [t for t in val_data.targets if t in self.class_idx[train_phase.value]]

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
