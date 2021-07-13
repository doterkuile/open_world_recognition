from open_world import ObjectDatasets
from open_world import OpenWorldUtils
from open_world import RecognitionModels

import torch
import torchvision
import sklearn

import os
import torch.nn as nn
import time
import argparse

import open_world.meta_learner.meta_learner_utils as meta_utils
import yaml

import numpy as np
import sys

def main():

    # set random seed
    torch.manual_seed(42)

    # Main gpu checks
    multiple_gpu = True if torch.cuda.device_count() > 1 else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Cuda device not available make sure CUDA has been installed")
        return
    else:
        print(f'Running with {torch.cuda.device_count()} GPUs')

    # Get config file argument
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()
    config_file = args.config_file

    # Overwrite terminal argument if necessary
    # config_file = 'config/CIFAR100_features.yaml'

    # Parse config file
    (dataset, model, criterion, optimizer, epochs, batch_size, learning_rate, config) = OpenWorldUtils.parseConfigFile(config_file, device, multiple_gpu)

    # Setup dataset
    (train_data, test_data) = dataset.getData()
    classes = dataset.classes

    # Save features in memory.npz. If folder does not exist make folder
    if not os.path.exists(config['dataset_path']):
        os.makedirs(config['dataset_path'])

    n_cls = config['train_classes']
    n_smpl = config['train_samples_per_cls']
    memory_path = config['dataset_path'] + '/memory_' + str(n_cls) + '_' + str(n_smpl) + '.npz'

    # Extract features
    print('Extract features')
    data_rep, data_cls_rep, labels_rep = meta_utils.extract_features(train_data, model, classes)

    print('Save features at: ' + memory_path)
    np.savez(memory_path, data_rep=data_rep, train_cls_rep=data_cls_rep, labels_rep=labels_rep)
    return


if __name__ == "__main__":
    main()
