import torch
from open_world import OpenWorldUtils
import open_world.meta_learner.meta_learner_utils as meta_utils
import yaml
import numpy as np
import argparse
import os

ENABLE_TRAINING = True


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

    # If dataset folder does not exist make folder
    if not os.path.exists(config['dataset_path']):
        os.makedirs(config['dataset_path'])

    load_memory = config['load_memory']
    memory_path = config['dataset_path'] + '/memory.npz'

    print('Extract features')
    data_rep, data_cls_rep, labels_rep = meta_utils.extract_features(train_data, model, classes, memory_path, load_memory)
    top_n = config['top_n']  # Top similar classes
    train_samples_per_cls = config['train_samples_per_cls'] # Number of samples per class

    class_set = classes

    num_train = config['train_classes']  # Classes used for training
    num_valid = len(classes) - num_train  # Classes used for validation

    randomize_samples = True

    print(f'Rank training samples with {num_train} classes, {train_samples_per_cls} samples per class')
    train_X0, train_X1, train_Y = meta_utils.rank_samples_from_memory(class_set[:num_train], data_rep, data_cls_rep,
                                                                      labels_rep, classes, train_samples_per_cls, top_n,
                                                                      randomize_samples)

    print(f'Rank validation samples with {num_valid} classes, {train_samples_per_cls} samples per class')
    valid_X0, valid_X1, valid_Y = meta_utils.rank_samples_from_memory(class_set[-num_valid:], data_rep, data_cls_rep,
                                                                      labels_rep, classes, train_samples_per_cls, top_n,
                                                                      randomize_samples)


    memory_path = config['dataset_path'] + '/memory_' + str(num_train) + '_' + str(train_samples_per_cls) + '.npz'
    print(f'Save results to {memory_path}')
    np.savez(memory_path,
             train_rep=data_rep, labels_rep=labels_rep,  # including all validation examples.
             train_X0=train_X0, train_X1=train_X1, train_Y=train_Y,
             valid_X0=valid_X0, valid_X1=valid_X1, valid_Y=valid_Y)
    return


if __name__ == "__main__":
    main()
