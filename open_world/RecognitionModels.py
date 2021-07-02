import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import efficientnet_pytorch as efficientnetPy



from torchvision.utils import make_grid
import torch.utils.model_zoo as model_zoo
import os
from open_world import OpenWorldUtils
from open_world import ObjectDatasets

from torchvision.models import resnet50
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import abc


class EncoderBase(nn.Module):

    def __init__(self, model_class, model_path, train_classes, feature_layer, pretrained=True):
        super().__init__()
        self.model_path = model_path
        self.model_class = model_class
        self.pretrained = pretrained
        self.feature_layer = feature_layer
        self.selected_out = OrderedDict()
        self.model = self.getModel(pretrained)
        self.output_classes = train_classes

        # if pretrained:
        #     for param in self.model.parameters():
        #         param.requires_grad = False

        self.reset_final_layer(train_classes)
        self.hook = getattr(self.model, feature_layer).register_forward_hook(self.feature_hook(feature_layer))

        pass

    @abc.abstractmethod
    def reset_final_layer(self):
        pass

    @abc.abstractmethod
    def getModel(self, pretrained):
        pass

    def feature_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output.reshape(input[0].shape[0], -1)
        return hook

    def forward(self, x):

        x = self.model(x)

        return x, self.selected_out[self.feature_layer]

class Resnet50(EncoderBase):

    def __init__(self, model_class, model_path,train_classes, feature_layer, pretrained=True):
        super().__init__(model_class, model_path,train_classes, feature_layer, pretrained)

        self.hook = getattr(self.model, feature_layer).register_forward_hook(self.feature_hook(feature_layer))


    def getModel(self, pretrained):
        model = models.resnet50(pretrained=pretrained)
        return model

    def reset_final_layer(self, output_classes):
        self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=output_classes)
        return

class Resnet152(EncoderBase):

    def __init__(self, model_class, model_path,train_classes, feature_layer, pretrained=True):
        super().__init__(model_class, model_path,train_classes, feature_layer, pretrained)
        self.hook = getattr(self.model, feature_layer).register_forward_hook(self.feature_hook(feature_layer))


    def getModel(self, pretrained):
        model = models.resnet152(pretrained=pretrained)
        return model

    def reset_final_layer(self, output_classes):
        self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=output_classes)
        return

class AlexNet(EncoderBase):

    def __init__(self, model_class, model_path,train_classes, feature_layer, pretrained=True):

        super().__init__(model_class, model_path,train_classes, feature_layer, pretrained)

        self.hook = getattr(self.model, feature_layer).register_forward_hook(self.feature_hook(feature_layer))


    def getModel(self, pretrained):
        model = alexnet(pretrained=pretrained, out_classes=self.output_classes)
        return model

    def reset_final_layer(self, output_classes):
        self.model.classifier[-1] = torch.nn.Linear(in_features=self.model.classifier[-1].in_features, out_features=output_classes)
        return

class EfficientNet(EncoderBase):

    def __init__(self, model_class, model_path, train_classes, feature_layer, pretrained=True):

        super().__init__(model_class, model_path, train_classes, feature_layer, pretrained)

        self.hook = getattr(self.model, feature_layer).register_forward_hook(self.feature_hook(feature_layer))


    def getModel(self, pretrained):
        model = efficientnetPy.EfficientNet.from_pretrained('efficientnet-b0')

        return model

    def reset_final_layer(self, output_classes):
        self.model._fc = torch.nn.Linear(in_features=self.model._fc.in_features, out_features=output_classes)
        return



class ResNet50old(nn.Module):
    def __init__(self, model_path, train_classes):

        super().__init__()
        self.resnet = models.resnet50(pretrained=False)
        # self.load_state_dict(torch.hub.load('pytorch/vision:v0.2.2', 'resnet50', pretrained=False))
        # for param in self.parameters():
        #     param.requires_grad = False
        self.resnet.fc = nn.Linear(2048, train_classes)
        self.model_path = model_path
        self.selected_out = OrderedDict()

    def feature_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output.reshape(input[0].shape[0], -1)
        return hook


    def forward(self, x):
        x = self.resnet(x)
        return x



class ResNet50Features(models.ResNet):
    def __init__(self, model_path, feature_layer):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3])

        self.pretrained = torch.hub.load('pytorch/vision:v0.2.2', 'resnet50', pretrained=True)
        torch.save(self.pretrained.state_dict(), model_path)

        for param in self.pretrained.parameters():
            param.requires_grad = False

        self.model_path = model_path
        self.feature_layer = feature_layer

        self.selected_out = OrderedDict()
        self.hook = getattr(self.pretrained, feature_layer).register_forward_hook(self.feature_hook(feature_layer))

    def feature_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output.reshape(input[0].shape[0], -1)
        return hook

    def forward(self, x):
        x = self.pretrained(x)

        return x, self.selected_out[self.feature_layer]

__all__ = ['AlexNet', 'alexnet']



class AlexNet_modified(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet_modified, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 1 * 1)
        x = self.classifier(x)
        return x

def alexnet(pretrained=False, out_classes=200, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    }

    model = AlexNet_modified(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    model.classifier[1] = nn.Linear(256 * 1 * 1, 4096)
    model.classifier[6] = nn.Linear(4096, out_classes)
    return model


class Identity(torch.nn.Module):

    def __init__(self):
        super(Identity,self).__init__()

    def forward(self, x):
        return x


class L2AC_base(torch.nn.Module):
    def __init__(self, model_path, num_classes, feature_size=2048, batch_size=10, top_k=5):
        super().__init__()
        self.has_lstm = True
        self.feature_size = feature_size
        self.input_size = 2048
        self.batch_size = batch_size
        self.hidden_size = 1
        self.top_k = top_k

        self.matching_layer = self.setMatchingLayer()
        self.aggregation_layer = self.setAggregationLayer()

        # Create hook for similarity function
        self.selected_out = OrderedDict()
        self.hook = getattr(self, 'matching_layer').register_forward_hook(self.sim_func_hook('matching_layer'))

        pass

    @abc.abstractmethod
    def similarity_function(self):
        pass

    @abc.abstractmethod
    def setMatchingLayer(self):
        pass

    @abc.abstractmethod
    def setAggregationLayer(self):
        pass

    def reset_hidden(self, batch_size):
        self.hidden = (torch.zeros(2, batch_size, self.hidden_size).to('cuda'),
                       torch.zeros(2, batch_size, self.hidden_size).to('cuda'))

    def sim_func_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x0, x1):
        x = self.similarity_function(x0, x1)
        x = self.matching_layer(x)
        if self.has_lstm:

            self.reset_hidden(x0.shape[0])
            x, (self.hidden, cell_state) = self.lstm(x.view(x.shape[0], -1, x.shape[1]), self.hidden)

        x = x.reshape(x.shape[0], -1)

        x = self.aggregation_layer(x)
        return x, self.selected_out['matching_layer']


class L2AC(L2AC_base):

    def __init__(self, model_path, num_classes, feature_size=2048, batch_size=10, top_k=5):

        super().__init__(model_path, num_classes, feature_size, batch_size, top_k)
        self.lstm = nn.LSTM(input_size=top_k, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)

    def setMatchingLayer(self):
        matching_layer = nn.Sequential(nn.Linear(2 * self.feature_size, self.input_size),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(self.input_size, 1),
                                       nn.Sigmoid(),
                                       )
        return matching_layer

    def setAggregationLayer(self):
        aggregation_layer = nn.Sequential(nn.Linear(2 * self.hidden_size, 1))

        return aggregation_layer


    def similarity_function(self, x0, x1):
        x = x0.repeat_interleave(x1.shape[1], dim=1)
        x_abssub = x.sub(x1)
        x_abssub.abs_()
        x_add = x.add(x1)
        x = torch.cat((x_abssub, x_add), dim=2)
        return x


class L2AC_cosine(L2AC_base):

    def __init__(self, model_path, num_classes, feature_size=2048,  batch_size=10, top_k=5):
        super().__init__(model_path, num_classes, feature_size, batch_size, top_k)

        self.lstm = nn.LSTM(input_size=top_k, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)

        return

    def setMatchingLayer(self):
        matching_layer = nn.Sequential(nn.Sigmoid()
                                       )
        return matching_layer

    def setAggregationLayer(self):
        aggregation_layer = nn.Sequential(nn.Linear(2 * self.hidden_size, 1))

        return aggregation_layer


    def similarity_function(self, x0, x1):
        x = x0.repeat_interleave(x1.shape[1], dim=1)
        x = torch.cosine_similarity(x, x1, dim=2).reshape(x.shape[0], -1, 1)
        return x


class L2AC_no_lstm(L2AC_base):

    def __init__(self, model_path, num_classes, feature_size=2048,  batch_size=10, top_k=5):
        super().__init__(model_path, num_classes, feature_size, batch_size, top_k)

        self.has_lstm = False

        return


    def setMatchingLayer(self):

        matching_layer = nn.Sequential(nn.Linear(2 * self.feature_size, self.input_size),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(self.input_size, 1),
                                       nn.Sigmoid(),
                                       )
        return matching_layer

    def setAggregationLayer(self):

        aggregation_layer = nn.Sequential(nn.Linear(self.top_k, 256),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.5),
                                          nn.Linear(256, 256),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.25),
                                          nn.Linear(256, 64),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.1),
                                          nn.Linear(64, 1))

        return aggregation_layer


    def similarity_function(self, x0, x1):
        x = x0.repeat_interleave(x1.shape[1], dim=1)
        x_abssub = x.sub(x1)
        x_abssub.abs_()
        x_add = x.add(x1)
        x = torch.cat((x_abssub, x_add), dim=2)
        return x


class L2AC_extended_similarity(L2AC_base):

    def __init__(self, model_path, num_classes, feature_size=2048,  batch_size=10, top_k=5):
        super().__init__(model_path, num_classes, feature_size, batch_size, top_k)

        self.lstm = nn.LSTM(input_size=top_k, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)

        return

    def setMatchingLayer(self):
        matching_layer = nn.Sequential(nn.Linear(2 * self.feature_size, self.input_size),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(self.input_size, 1024),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(1024, 512),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.25),
                                       nn.Linear(512, 128),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.1),
                                       nn.Linear(128, 1),
                                       nn.Sigmoid())
        return matching_layer

    def setAggregationLayer(self):
        aggregation_layer = nn.Sequential(nn.Linear(2 * self.hidden_size, 1))

        return aggregation_layer


    def similarity_function(self, x0, x1):
        x = x0.repeat_interleave(x1.shape[1], dim=1)
        x_abssub = x.sub(x1)
        x_abssub.abs_()
        x_add = x.add(x1)
        x = torch.cat((x_abssub, x_add), dim=2)
        return x


class L2AC_smaller_fc(L2AC_base):

    def __init__(self, model_path, num_classes, feature_size=2048, batch_size=10, top_k=5):
        super().__init__(model_path, num_classes, feature_size, batch_size, top_k)
        # Override default input size
        self.input_size = 512
        self.matching_layer = self.setMatchingLayer()
        self.hook = getattr(self, 'matching_layer').register_forward_hook(self.sim_func_hook('matching_layer'))

        self.lstm = nn.LSTM(input_size=top_k, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.fc_reduce = nn.Linear(self.feature_size, self.input_size)


    def setMatchingLayer(self):
        matching_layer = nn.Sequential(nn.Linear(2 * self.input_size, self.input_size),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(self.input_size, 1),
                                       nn.Sigmoid(),
                                       )
        return matching_layer

    def setAggregationLayer(self):
        aggregation_layer = nn.Sequential(nn.Linear(2 * self.hidden_size, 1))

        return aggregation_layer


    def similarity_function(self, x0, x1):

        x0 = x0.repeat_interleave(x1.shape[1], dim=1)
        x = F.relu(self.fc_reduce(x0))
        x1 = F.relu(self.fc_reduce(x1))
        x_abssub = x.sub(x1)
        x_abssub.abs_()
        x_add = x.add(x1)
        x = torch.cat((x_abssub, x_add), dim=2)
        return x


class L2AC_abssub(L2AC_base):

    def __init__(self, model_path, num_classes, feature_size=2048, batch_size=10, top_k=5):
        super().__init__(model_path, num_classes, feature_size, batch_size, top_k)
        self.lstm = nn.LSTM(input_size=top_k, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)

    def setMatchingLayer(self):
        matching_layer = nn.Sequential(nn.Linear(self.feature_size, self.input_size),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(self.input_size, 1),
                                       nn.Sigmoid(),
                                       )
        return matching_layer

    def setAggregationLayer(self):
        aggregation_layer = nn.Sequential(nn.Linear(2 * self.hidden_size, 1))

        return aggregation_layer

    def similarity_function(self, x0, x1):
        x = x0.repeat_interleave(x1.shape[1], dim=1)
        x_abssub = x.sub(x1)
        x = x_abssub.abs()
        return x

class L2AC_concat(L2AC_base):

    def __init__(self, model_path, num_classes, feature_size=2048, batch_size=10, top_k=5):
        super().__init__(model_path, num_classes, feature_size, batch_size, top_k)
        self.lstm = nn.LSTM(input_size=top_k, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)

    def setMatchingLayer(self):
        matching_layer = nn.Sequential(nn.Linear(2 * self.feature_size, self.input_size),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(self.input_size, 1),
                                       nn.Sigmoid(),
                                       )
        return matching_layer

    def setAggregationLayer(self):
        aggregation_layer = nn.Sequential(nn.Linear(2 * self.hidden_size, 1))

        return aggregation_layer


    def similarity_function(self, x0, x1):
        x = x0.repeat_interleave(x1.shape[1], dim=1)
        x = torch.cat((x, x1), dim=2)
        return x


def main():
    model = resnet50(pretrained=True, num_classes=1000)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(2048, 100)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # dataset_path =
    # dataset = ObjectDatasets.CIFAR100Dataset(dataset_path)
    # train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # OpenWorldUtils.OpentrainModel(model, train_loader, test_loader, epochs, criterion, optimizer)

    return





if __name__ == "__main__":
    main()