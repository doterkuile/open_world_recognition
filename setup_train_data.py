import torch
from open_world import ObjectDatasets
from open_world import RecognitionModels
import open_world.meta_learner.meta_learner_utils as meta_utils
import yaml
import numpy as np
import argparse
import os


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

    train_phase = 'l2ac_train'


    # Parse config file
    dataset, model, top_n, train_classes, test_classes, train_samples_per_cls, randomize_samples, config = parseConfigFile(device, multiple_gpu, train_phase)



    # Setup dataset
    (train_data, _) = dataset.getData()



    feature_layer = config['feature_layer']
    model_class = config['model_class']
    dataset_path = f"datasets/{config['dataset_path']}/{model_class}"
    memory_path = f'{dataset_path}/{feature_layer}_{train_classes}_{train_samples_per_cls}_{top_n}_train.npz'
    model_path = config['model_path']
    # If dataset folder does not exist make folder
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    load_memory = config['load_memory']
    feature_memory_path = dataset_path + '/memory.npz'
    #
    # if feature_layer not in model.state_dict().keys():
    #     print("Feature layer not present in state dict, make sure they are correct:\n")
    #     print(f'Model:{model_path}\nFeature layer: {feature_layer}')
    #     return

    train_classes_idx = dataset.class_idx[train_phase]






    print('Extract features')
    trn_data_rep, trn_data_cls_rep, trn_labels_rep = meta_utils.extract_features(train_data, model, train_classes_idx, device, feature_memory_path, load_memory)

    print(f'Rank training samples with {train_classes} classes, {train_samples_per_cls} samples per class')
    train_X0, train_X1, train_Y = meta_utils.rank_samples_from_memory(train_classes_idx, trn_data_rep, trn_data_cls_rep,
                                                                      trn_labels_rep, train_classes_idx, train_samples_per_cls, top_n,
                                                                      randomize_samples)




    print(f'Save results to {memory_path}')
    np.savez(memory_path,
             train_rep=trn_data_rep, trn_labels_rep=trn_labels_rep,  # including all validation examples.
             train_X0=train_X0, train_X1=train_X1, train_Y=train_Y)
    return


def parseConfigFile(device, multiple_gpu, train_phase):

    # Get config file argument
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()
    config_file = args.config_file


    with open('config/' + config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # L2AC Parameters
    top_n = int(config['top_n'])
    train_samples_per_cls = config['train_samples_per_cls'] # Number of samples per class
    class_ratio = config['class_ratio']

    train_classes = class_ratio['l2ac_train'] # Classes used for training
    test_classes = class_ratio['l2ac_test'] # Classes used for validation
    randomize_samples = config['randomize_samples']

    # Load dataset
    dataset_path = f"datasets/{config['dataset_path']}"
    dataset_class = config['dataset_class']
    figure_size = config['image_resize']
    unfreeze_layer = config['unfreeze_layer']


    train_dataset = eval('ObjectDatasets.' + dataset_class)(dataset_path, class_ratio, train_phase, figure_size)

    # Load model
    model_path = config['model_path']
    model_class = config['model_class']
    pretrained = config['pretrained']
    feature_layer = config['feature_layer']
    num_classes = config['class_ratio']['encoder']
    model = eval('RecognitionModels.' + model_class)(model_class, model_path, num_classes, feature_layer, pretrained).to(device)
    encoder_file_path = f'{dataset_path}/{config["model_class"]}/feature_encoder_{figure_size}_{unfreeze_layer}.pt'

    model.load_state_dict(torch.load(encoder_file_path))

    return train_dataset, model, top_n, train_classes, test_classes, train_samples_per_cls, randomize_samples, config

if __name__ == "__main__":
    main()
