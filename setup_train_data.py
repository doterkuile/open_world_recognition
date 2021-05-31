import torch
from open_world import OpenWorldUtils
import open_world.meta_learner.meta_learner_utils as meta_utils
import yaml
import numpy as np

ENABLE_TRAINING = True


def main():
    if not torch.cuda.is_available():
        print("Cuda device not available make sure CUDA has been installed")
        return
    torch.manual_seed(42)
    load_data = True
    config_file = 'config/CIFAR100_features.yaml'

    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    memory_path = config['dataset_path'] + '/memory.npz'

    # Parse config file
    (dataset, model, criterion, optimizer, epochs, batch_size, learning_rate, config) = OpenWorldUtils.parseConfigFile(
		config_file, ENABLE_TRAINING)

    # Setup dataset
    (train_data, test_data) = dataset.getData()
    classes = dataset.classes

    data_rep, data_cls_rep, labels_rep = meta_utils.extract_features(train_data, model, classes, memory_path, load_data)
    top_n = 9  # Top similar classes
    train_per_cls = 100  # Number of samples per class

    class_set = classes

    num_train = 80  # Classes used for training
    num_valid = 20  # Classes used for validation

    randomize_samples = True

    train_X0, train_X1, train_Y = meta_utils.rank_samples_from_memory(class_set[:num_train], data_rep, data_cls_rep,
                                                                      labels_rep, classes, train_per_cls, top_n,
                                                                      randomize_samples)
    valid_X0, valid_X1, valid_Y = meta_utils.rank_samples_from_memory(class_set[-num_valid:], data_rep, data_cls_rep,
                                                                      labels_rep, classes, train_per_cls, top_n,
                                                                      randomize_samples)

    np.savez(config['dataset_path'] + "/train_idx.npz",
             train_rep=data_rep, labels_rep=labels_rep,  # including all validation examples.
             train_X0=train_X0, train_X1=train_X1, train_Y=train_Y,
             valid_X0=valid_X0, valid_X1=valid_X1, valid_Y=valid_Y)
    return


if __name__ == "__main__":
    main()
