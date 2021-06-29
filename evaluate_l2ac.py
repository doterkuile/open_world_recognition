import torch
import torch.utils.data as data_utils
from open_world import ObjectDatasets
from open_world import RecognitionModels
import open_world.meta_learner.meta_learner_utils as meta_utils
import open_world.plot_utils as plot_utils
import yaml
import numpy as np
from torch.utils.data import DataLoader
import argparse
import os
import shutil
import matplotlib.pyplot as plt
import torch.nn as nn
from open_world import OpenWorldUtils
import open_world.meta_learner.meta_learner_utils as meta_utils
import time



def parseConfigFile(config_file, device, multiple_gpu):

    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)



    ## Training hyperparameters
    batch_size = config['batch_size']

    ## L2AC Parameters
    top_k = int(config['top_k'])
    top_n = int(config['top_n'])
    train_classes = config['train_classes']
    test_classes = config['test_classes']
    train_samples_per_cls = config['train_samples_per_cls']

    ## Dataset preparation parameters:
    same_class_reverse = config['same_class_reverse']
    same_class_extend_entries = config['same_class_extend_entries']

    # If same class extend entries is true then dataset is already balanced
    if not same_class_extend_entries:
        pos_weight = torch.tensor([top_n]).to(device).to(dtype=torch.float)
    else:
        pos_weight = torch.tensor([1.0]).to(device).to(dtype=torch.float)

    criterion = eval('nn.' + config['criterion'])(pos_weight, reduction='mean')

    ## Classes
    # Load dataset
    encoder = config['encoder']
    feature_layer = config['feature_layer']
    dataset_path = config['dataset_path'] + f'/{encoder}/{feature_layer}_{train_classes}_{train_samples_per_cls}_{top_n}.npz'
    dataset_class = config['dataset_class']


    test_dataset = eval('ObjectDatasets.' + dataset_class)(dataset_path, top_n, top_k, train_classes,
                                                            train_samples_per_cls
                                                            , False, same_class_reverse, same_class_extend_entries)

    # Load model
    features_size = len(test_dataset.memory[0])


    model_path = 'output/' + str(config['name']) + '/' + str(config['name']) + '_model.pt'
    state_path = 'output/' + str(config['name']) + '/' + str(config['name']) + '_best_state.pth'
    best_state = torch.load(state_path)
    model_class = config['model_class']
    model = eval('RecognitionModels.' + model_class)(model_path, train_classes,features_size, batch_size, top_k).to(device)
    print('Load model ' + model_path)
    # OpenWorldUtils.loadModel(model, model_path)
    model.load_state_dict(best_state['model'])

    return test_dataset, model, config, criterion

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
        print(f"Running with {torch.cuda.device_count()} GPUs")
    # Get config file argument

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()
    evaluation_config_file = args.config_file

    with open('config/' + evaluation_config_file) as file:
        config_evaluate = yaml.load(file, Loader=yaml.FullLoader)

    exp_nrs = config_evaluate['name']
    loop_variable_name = config_evaluate['variable']
    loop_variable = {loop_variable_name: []}
    figure_path = config_evaluate['figure_path']

    metrics_dict = {'loss': [],
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'F1': [],
                    'mean_pred': [],
                    'mean_true': []}

    for exp in exp_nrs:

        exp_folder = f'output/{exp}'
        train_config_file = f'{exp_folder}/{exp}_config.yaml'

        # Overwrite terminal argument if necessary
        # config_file = 'config/L2AC_train.yaml'

        # Parse config file
        test_dataset, model, config, criterion = parseConfigFile(
            train_config_file, device, multiple_gpu)

        loop_variable[loop_variable_name].append(config[loop_variable_name])
        # Get hyperparameters
        train_classes = config['train_classes']
        train_samples_per_cls = config['train_samples_per_cls']
        probability_treshold = config['probability_threshold']

        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True)

        # meta_utils.validate_similarity_scores(trn_similarity_scores, model, train_loader, device)
        # meta_utils.validate_similarity_scores(tst_similarity_scores, model, test_loader, device)
        tst_y_pred, tst_y_true, tst_loss, tst_sim_scores, tst_y_pred_raw= meta_utils.validate_model(test_loader, model, criterion, device, probability_treshold)

        meta_utils.calculate_metrics(metrics_dict, tst_y_true, tst_y_pred, tst_loss)


    plot_utils.plot_best_F1(metrics_dict['F1'], loop_variable, f'{figure_path}_F1')
    plot_utils.plot_best_loss(metrics_dict['loss'], loop_variable, f'{figure_path}_loss')



    return


if __name__ == "__main__":
    main()
