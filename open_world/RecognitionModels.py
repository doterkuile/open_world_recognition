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
from sklearn import metrics
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
    def __init__(self, model_path, num_classes, batch_size, top_k):
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

    def __init__(self, model_path, num_classes,  batch_size=10, top_k=5):
        super(L2AC, self).__init__()
        # self.feature_size = 2048
        # self.input_size = 2048

        self.feature_size = 512
        self.input_size = 512
        self.batch_size = batch_size
        self.hidden_size = 1
        self.fc1 = nn.Linear(2 * self.feature_size, self.input_size)
        self.fc2 = nn.Linear(self.input_size, 1)
        self.lstm = nn.LSTM(input_size=top_k, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.fc3 = nn.Linear(2 * self.hidden_size, 1)
        self.reset_hidden()

    def forward(self,x0, x1):
        self.reset_hidden()
        x0 = x0.repeat_interleave(x1.shape[1], dim=1)
        x = self.similarity_function(x0, x1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        x = x.sigmoid()
        x, cell_state = self.lstm(x.view(x.shape[0],-1, x.shape[1]), self.hidden)
        x = self.fc3(x.reshape(x.shape[0], -1))

        return x

    def similarity_function(self, x0, x1):
        x_abssub = x0.sub(x1)
        x_abssub.abs_()
        x_add = x0.add(x1)
        x0 = torch.cat((x_abssub, x_add), dim=2)
        return x0

    def reset_hidden(self):
        self.hidden = (torch.zeros(2, self.batch_size, self.hidden_size).to('cuda'),
                       torch.zeros(2, self.batch_size, self.hidden_size).to('cuda'))

class L2AC_cosine(torch.nn.Module):

    def __init__(self, model_path, num_classes,  batch_size=10, top_k=5):
        super(L2AC_cosine, self).__init__()
        self.feature_size = 2048
        self.input_size = 2048
        self.batch_size = batch_size
        self.hidden_size = 1
        self.fc1 = nn.Linear(2 * self.feature_size, self.input_size)
        self.fc2 = nn.Linear(self.input_size, 1)
        self.lstm = nn.LSTM(input_size=top_k, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.fc3 = nn.Linear(2 * self.hidden_size, 1)
        self.reset_hidden()

    def forward(self,x0, x1):
        self.reset_hidden()
        x0 = x0.repeat_interleave(x1.shape[1], dim=1)
        x = torch.cosine_similarity(x0, x1, dim=2).reshape(x0.shape[0],-1,1)
        x = x.sigmoid()
        x, cell_state = self.lstm(x.view(x.shape[0],-1, x.shape[1]), self.hidden)
        x = self.fc3(x.reshape(x.shape[0], -1))
        return x

    def reset_hidden(self):
        self.hidden = (torch.zeros(2, self.batch_size, self.hidden_size).to('cuda'),
                       torch.zeros(2, self.batch_size, self.hidden_size).to('cuda'))


class L2AC_no_lstm(torch.nn.Module):

    def __init__(self, model_path, num_classes,  batch_size=10, top_k=5):
        super(L2AC_no_lstm, self).__init__()
        self.feature_size = 2048
        self.input_size = 2048
        self.batch_size = batch_size
        self.top_k = top_k
        self.hidden_size = 1
        self.fc1 = nn.Linear(2 * self.feature_size, self.input_size)
        self.fc2 = nn.Linear(self.input_size, 1)
        self.fc3 = nn.Linear(self.top_k, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 64)
        self.fc6 = nn.Linear(64, 1)

    def forward(self,x0, x1):
        x0 = x0.repeat_interleave(x1.shape[1], dim=1)
        x = self.similarity_function(x0, x1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        x = x.sigmoid()
        x = F.relu(self.fc3(x.reshape(x.shape[0],-1)))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, p=0.25)
        x = F.relu(self.fc5(x))
        x = F.dropout(x, p=0.1)
        x = self.fc6(x.reshape(x.shape[0], -1))

        return x

    def similarity_function(self, x0, x1):
        x_abssub = x0.sub(x1)
        x_abssub.abs_()
        x_add = x0.add(x1)
        x0 = torch.cat((x_abssub, x_add), dim=2)
        return x0

    def reset_hidden(self):
        pass


class L2AC_extended_similarity(torch.nn.Module):

    def __init__(self, model_path, num_classes,  batch_size=10, top_k=5):
        super(L2AC_extended_similarity, self).__init__()
        self.feature_size = 2048
        self.input_size = 2048
        self.batch_size = batch_size
        self.hidden_size = 1
        self.fc1 = nn.Linear(2 * self.feature_size, self.input_size)
        self.fc2 = nn.Linear(self.input_size, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 1)
        self.lstm = nn.LSTM(input_size=top_k, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.fc6 = nn.Linear(2 * self.hidden_size, 1)
        self.reset_hidden()

    def forward(self,x0, x1):
        self.reset_hidden()
        x0 = x0.repeat_interleave(x1.shape[1], dim=1)
        x = self.similarity_function(x0, x1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.25)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, p=0.1)
        x = F.relu(self.fc5(x))
        x = x.sigmoid()
        x, cell_state = self.lstm(x.view(x.shape[0],-1, x.shape[1]), self.hidden)
        x = self.fc6(x.reshape(x.shape[0], -1))

        return x

    def similarity_function(self, x0, x1):
        x_abssub = x0.sub(x1)
        x_abssub.abs_()
        x_add = x0.add(x1)
        x0 = torch.cat((x_abssub, x_add), dim=2)
        return x0

    def reset_hidden(self):
        self.hidden = (torch.zeros(2, self.batch_size, self.hidden_size).to('cuda'),
                       torch.zeros(2, self.batch_size, self.hidden_size).to('cuda'))


