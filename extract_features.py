from open_world import ObjectDatasets
from open_world import OpenWorldUtils
from open_world import RecognitionModels

import torch
import torchvision
import sklearn

import os
import torch.nn as nn
import time
import open_world.meta_learner.meta_learner_utils as meta_utils
import yaml

import numpy as np

ENABLE_TRAINING = True
SAVE_IMAGES = True


def main():

    if not torch.cuda.is_available():
        print("Cuda device not available make sure CUDA has been installed")
        return
    torch.manual_seed(42)
    load_data = False
    config_file = 'config/CIFAR100_features.yaml'

    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    print(config['dataset_path'])
    if not os.path.exists(config['dataset_path']):
        os.makedirs(config['dataset_path'])
    memory_path = config['dataset_path'] + '/memory.npz'

    # Parse config file
    (dataset, model, criterion, optimizer, epochs, batch_size, learning_rate, config) = OpenWorldUtils.parseConfigFile(config_file, ENABLE_TRAINING)

    # Setup dataset
    (train_data, test_data) = dataset.getData()
    classes = dataset.classes

    # Extract features
    print('Extract features')
    data_rep, data_cls_rep, labels_rep = meta_utils.extract_features(train_data, model, classes, memory_path, load_data)

    print('Save features at: ' + memory_path)

    np.savez(memory_path, data_rep=data_rep, train_cls_rep=data_cls_rep, labels_rep=labels_rep)

    return


if __name__ == "__main__":
    main()
