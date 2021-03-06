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

    def __init__(self, model_class, train_classes, feature_layer, unfreeze_layers,feature_scaling, pretrained=True):
        super().__init__()
        self.model_class = model_class
        self.pretrained = pretrained
        self.feature_layer = feature_layer
        self.selected_out = OrderedDict()
        self.output_classes = train_classes
        self.feature_scaling = feature_scaling

        self.model = self.getModel(pretrained)

        self.freeze_feature_layers(unfreeze_layers)



        self.reset_final_layer(train_classes)
        self.hook = getattr(self.model, feature_layer).register_forward_hook(self.feature_hook(feature_layer))

        pass

    @abc.abstractmethod
    def reset_final_layer(self):
        pass

    @abc.abstractmethod
    def getModel(self, pretrained):
        pass

    def scale_features(self, feature_vector):
        if self.feature_scaling == 'sigmoid':
            out_features = feature_vector.sigmoid().reshape(feature_vector.shape[0], -1)
        elif self.feature_scaling == 'max_value':
            out_features = feature_vector.reshape(feature_vector.shape[0], -1)
            out_features = torch.transpose(out_features, 1, 0)
            max_values = (out_features.max(dim=0)).values
            out_features = torch.div(out_features, max_values)
            out_features = torch.transpose(out_features, 1, 0)
        else:
            out_features = feature_vector.reshape(feature_vector.shape[0], -1)

        return out_features

    def feature_hook(self, layer_name):
        def hook(module, input, output):

            out_features = self.scale_features(output)
            self.selected_out[layer_name] = out_features

        return hook

    def freeze_feature_layers(self, feature_depth):
        param_keys = list(self.model.state_dict())[:-feature_depth]
        for name , param in self.model.named_parameters():
            if name in param_keys:
                param.requires_grad = False

    def forward(self, x):

        x = self.model(x)

        return x, self.selected_out[self.feature_layer]

class ResNet50(EncoderBase):

    def __init__(self, model_class, train_classes, feature_layer, unfreeze_layers,feature_scaling, pretrained=True):
        super().__init__(model_class,train_classes, feature_layer, unfreeze_layers,feature_scaling, pretrained)

        self.hook = getattr(self.model, feature_layer).register_forward_hook(self.feature_hook(feature_layer))

        # Freeze all layers until 62: conv 5 and up trained

    def getModel(self, pretrained):
        model = models.resnet50(pretrained=pretrained)
        return model

    def reset_final_layer(self, output_classes):
        self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=output_classes)
        return

class ResNet152(EncoderBase):

    def __init__(self, model_class, train_classes, feature_layer, unfreeze_layers,feature_scaling, pretrained=True):
        super().__init__(model_class,train_classes, feature_layer, unfreeze_layers,feature_scaling, pretrained)

        self.hook = getattr(self.model, feature_layer).register_forward_hook(self.feature_hook(feature_layer))
        # Freeze all layers until 62: conv 5 and up trained


    def getModel(self, pretrained):
        model = models.resnet152(pretrained=pretrained)
        return model

    def reset_final_layer(self, output_classes):
        self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=output_classes)
        return

class AlexNet(EncoderBase):

    def __init__(self, model_class, train_classes, feature_layer, unfreeze_layers,feature_scaling, pretrained=True):
        super().__init__(model_class,train_classes, feature_layer, unfreeze_layers,feature_scaling, pretrained)

        self.hook = getattr(self.model, feature_layer).register_forward_hook(self.feature_hook(feature_layer))
        # Freeze all layers until 8: conv 5 and up trained


    def getModel(self, pretrained):
        model = alexnet(pretrained=pretrained, out_classes=self.output_classes)
        return model

    def reset_final_layer(self, output_classes):

        self.model.classifier = torch.nn.Linear(in_features=self.model.classifier.in_features, out_features=output_classes)
        
        return

class EfficientNet(EncoderBase):

    def __init__(self, model_class, train_classes, feature_layer, unfreeze_layers,feature_scaling, pretrained=True):
        super().__init__(model_class,train_classes, feature_layer, unfreeze_layers,feature_scaling, pretrained)

        self.hook = getattr(self.model, feature_layer).register_forward_hook(self.feature_hook(feature_layer))
         # Freeze all layers until 74: conv 13, 14 15 and up trained


    def getModel(self, pretrained):
        model = efficientnetPy.EfficientNet.from_pretrained('efficientnet-b0')

        return model

    def reset_final_layer(self, output_classes):
        self.model._fc = torch.nn.Linear(in_features=self.model._fc.in_features, out_features=output_classes)
        return



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
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.fc7(x)
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
    # model.classifier[1] = nn.Linear(256 * 6 * 6, 4096)
    model.fc7 = model.classifier[:-1]
    del model.classifier

    model.classifier = nn.Linear(4096, out_classes)
    return model


class Identity(torch.nn.Module):

    def __init__(self):
        super(Identity,self).__init__()

    def forward(self, x):
        return x


class L2AC_base(torch.nn.Module):
    def __init__(self, num_classes, feature_size=2048, batch_size=10, top_k=5):
        super().__init__()
        self.has_lstm = True
        self.feature_size = feature_size
        self.input_size = 2048
        self.batch_size = batch_size
        self.hidden_size = top_k
        self.top_k = top_k
        # self.embedding_layer = self.setEmbeddingLayer(data_path)
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

    def setEmbeddingLayer(self, dataset_path):

        features = torch.tensor(np.load(dataset_path)['train_rep'],dtype=torch.float)
        features.requires_grad = False
        # embed_layer = nn.Embedding(features.shape[0], features.shape[1],_weight=features)
        embed_layer = nn.Embedding.from_pretrained(features, freeze=True)
        return embed_layer



    def initialize_weights(self):


        for m in self.modules():
            if isinstance(m, nn.LSTM):
                nn.init.orthogonal_(self.lstm.weight_ih_l0)
                nn.init.orthogonal_(self.lstm.weight_hh_l0)
                nn.init.orthogonal_(self.lstm.weight_ih_l0_reverse)
                nn.init.orthogonal_(self.lstm.weight_hh_l0_reverse)

                nn.init.zeros_(self.lstm.bias_ih_l0)
                nn.init.ones_(self.lstm.bias_ih_l0[self.hidden_size:self.hidden_size * 2])
                nn.init.zeros_(self.lstm.bias_hh_l0)
                nn.init.ones_(self.lstm.bias_hh_l0[self.hidden_size:self.hidden_size * 2])
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.0)


    def forward(self, x0, x1):

        # x0 = self.embedding_layer(x0)
        # x1 = self.embedding_layer(x1)

        x = self.similarity_function(x0, x1)
        x = self.matching_layer(x)
        x = x.sigmoid()
        if self.has_lstm:

            self.reset_hidden(x0.shape[0])
            x, (self.hidden, cell_state) = self.lstm(x.view(x.shape[0], -1, x.shape[1]), self.hidden)

        x = x.reshape(x.shape[0], -1)

        x = self.aggregation_layer(x)
        return x, self.selected_out['matching_layer']


class L2AC(L2AC_base):

    def __init__(self, num_classes, feature_size=2048, batch_size=10, top_k=5):

        super().__init__(num_classes, feature_size, batch_size, top_k)
        self.lstm = nn.LSTM(input_size=top_k, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.initialize_weights()

    def setMatchingLayer(self):
        matching_layer = nn.Sequential(
                                       nn.Dropout(p=0.5),
                                       nn.Linear(2 * self.feature_size, self.input_size),
                                       nn.LeakyReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(self.input_size, 1),
                                       # nn.Sigmoid(),
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

    def __init__(self, num_classes, feature_size=2048,  batch_size=10, top_k=5):
        super().__init__(num_classes, feature_size, batch_size, top_k)

        self.lstm = nn.LSTM(input_size=top_k, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.initialize_weights()

        return

    def setMatchingLayer(self):
        matching_layer = nn.Sequential(nn.Dropout(p=0.5),
                                       # nn.Sigmoid()
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

    def __init__(self, num_classes, feature_size=2048,  batch_size=10, top_k=5):
        super().__init__(num_classes, feature_size, batch_size, top_k)

        self.has_lstm = False
        self.initialize_weights()

        return


    def setMatchingLayer(self):

        matching_layer = nn.Sequential(nn.Dropout(p=0.5),
                                       nn.Linear(2 * self.feature_size, self.input_size),
                                       nn.LeakyReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(self.input_size, 1),
                                       # nn.Sigmoid(),
                                       )
        return matching_layer

    def setAggregationLayer(self):

        aggregation_layer = nn.Sequential(nn.Linear(self.top_k, 256),
                                          nn.LeakyReLU(),
                                          nn.Dropout(p=0.5),
                                          nn.Linear(256, 256),
                                          nn.LeakyReLU(),
                                          nn.Dropout(p=0.5),
                                          nn.Linear(256, 64),
                                          nn.LeakyReLU(),
                                          nn.Dropout(p=0.5),
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

    def __init__(self, num_classes, feature_size=2048,  batch_size=10, top_k=5):
        super().__init__(num_classes, feature_size, batch_size, top_k)
        # self.input_size = 512
        # self.matching_layer = self.setMatchingLayer()
        # self.hook = getattr(self, 'matching_layer').register_forward_hook(self.sim_func_hook('matching_layer'))

        self.lstm = nn.LSTM(input_size=top_k, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.initialize_weights()
        # self.fc_reduce = nn.Linear(self.feature_size, self.input_size)


        return

    def setMatchingLayer(self):
        matching_layer = nn.Sequential(nn.Dropout(p=0.5),
                                       nn.Linear(2 * self.feature_size, self.input_size),
                                       nn.LeakyReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(self.input_size, 1024),
                                       nn.LeakyReLU(),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(1024, 128),
                                       nn.LeakyReLU(),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(128, 1),
                                       # nn.Sigmoid()
                                       )
        return matching_layer

    def setAggregationLayer(self):
        aggregation_layer = nn.Sequential(nn.Linear(2 * self.hidden_size, 1))

        return aggregation_layer


    def similarity_function(self, x0, x1):
        x = x0.repeat_interleave(x1.shape[1], dim=1)
        # x = F.relu(self.fc_reduce(x))
        # x1 = F.relu(self.fc_reduce(x1))
        x = torch.cat((x, x1), dim=2)
        return x


class L2AC_smaller_fc(L2AC_base):

    def __init__(self, num_classes, feature_size=2048, batch_size=10, top_k=5):
        super().__init__(num_classes, feature_size, batch_size, top_k)
        # Override default input size
        self.input_size = 512
        self.matching_layer = self.setMatchingLayer()
        self.hook = getattr(self, 'matching_layer').register_forward_hook(self.sim_func_hook('matching_layer'))

        self.lstm = nn.LSTM(input_size=top_k, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.fc_reduce = nn.Linear(self.feature_size, self.input_size)
        self.initialize_weights()


    def setMatchingLayer(self):
        matching_layer = nn.Sequential(nn.Dropout(p=0.5),
                                       nn.Linear(2 * self.input_size, self.input_size),
                                       nn.LeakyReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(self.input_size, 1),
                                       # nn.Sigmoid(),
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

    def __init__(self, num_classes, feature_size=2048, batch_size=10, top_k=5):
        super().__init__(num_classes, feature_size, batch_size, top_k)
        self.lstm = nn.LSTM(input_size=top_k, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.initialize_weights()

    def setMatchingLayer(self):
        matching_layer = nn.Sequential(nn.Dropout(p=0.5),
                                       nn.Linear(self.feature_size, self.input_size),
                                       nn.LeakyReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(self.input_size, 1),
                                       # nn.Sigmoid(),
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

    def __init__(self, num_classes, feature_size=2048, batch_size=10, top_k=5):
        super().__init__(num_classes, feature_size, batch_size, top_k)
        # self.input_size = 512
        # self.matching_layer = self.setMatchingLayer()
        # self.hook = getattr(self, 'matching_layer').register_forward_hook(self.sim_func_hook('matching_layer'))

        self.lstm = nn.LSTM(input_size=top_k, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.initialize_weights()
        # self.fc_reduce = nn.Linear(self.feature_size, self.input_size)



    def setMatchingLayer(self):
        matching_layer = nn.Sequential(nn.Dropout(p=0.5),
                                       nn.Linear(2 * self.feature_size, self.input_size),
                                       nn.LeakyReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(self.input_size, 1),
                                       # nn.Sigmoid(),
                                       )
        return matching_layer

    def setAggregationLayer(self):
        aggregation_layer = nn.Sequential(nn.Linear(2 * self.hidden_size, 1))

        return aggregation_layer


    def similarity_function(self, x0, x1):
        x = x0.repeat_interleave(x1.shape[1], dim=1)
        # x = F.relu(self.fc_reduce(x))
        # x1 = F.relu(self.fc_reduce(x1))
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