import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import torchvision.models.resnet as resnet
from torchvision.utils import make_grid
import torch.utils.model_zoo as model_zoo

from torchvision.models import resnet50
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import abc


class MNISTNetwork(nn.Module):
    def __init__(self, model_path, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,num_classes)

        self.model_path = model_path

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 5*5*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

    def getFeatureExtractor(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 5 * 5 * 16)

        return X

class FashionMNISTNetwork(nn.Module):
    def __init__(self, model_path, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5*5*16, 100)
        self.fc2 = nn.Linear(100, num_classes)

        self.model_path = model_path

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 5*5*16)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)

class CATDOGNetwork(nn.Module):
    def __init__(self,  model_path, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(54 * 54 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,num_classes)

        self.model_path = model_path


    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54 * 54 * 16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)


class ResNet50(models.ResNet):
    def __init__(self, model_path, num_classes):

        super().__init__(resnet.Bottleneck, [3, 4, 6, 3])
        self.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet50']))
        for param in self.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(512 , num_classes)
        self.model_path = model_path


class ResNet50Features(models.ResNet):
    def __init__(self, model_path, num_classes):
        super().__init__(resnet.Bottleneck, [3, 4, 6, 3])
        self.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet50']))
        for param in self.parameters():
            param.requires_grad = False
        self.fc = Identity()

        self.model_path = model_path


class Identity(torch.nn.Module):

    def __init__(self):
        super(Identity,self).__init__()

    def forward(self, x):
        return x

class L2AC(torch.nn.Module):

    def __init__(self, model_path, num_classes, top_k=5):
        super(L2AC, self).__init__()
        self.batch_size = 1
        self.hidden_size = 1
        self.fc1 = nn.Linear(2 * 2048, 2048)
        self.fc2 = nn.Linear(2048, 1)
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size, bidirectional=True, batch_first=False)
        self.fc3 = nn.Linear(10, 1)
        self.reset_hidden()




    def forward(self,x0, x1):

        x0 = x0.repeat_interleave(x1.shape[1],dim=1)
        x = self.similarity_function(x0, x1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = torch.sigmoid(self.fc2(x))
        # x_rep = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(1))(x_rep)
        x, cell_state = self.lstm(x.view(x1.shape[1], self.batch_size, -1), self.hidden)
        x = torch.sigmoid(self.fc3(x.view(self.batch_size, -1, 10)))


        return x.view(self.batch_size,-1)

    def similarity_function(self, x0, x1):
        x_abssub = x0.sub(x1)
        x_abssub.abs_()
        x_add = x0.add(x1)
        x0 = torch.cat((x_abssub, x_add), dim=2)
        return x0

    def reset_hidden(self):
        self.hidden = (torch.zeros(2, self.batch_size, self.hidden_size).to('cuda'),
                       torch.zeros(2, self.batch_size, self.hidden_size).to('cuda'))