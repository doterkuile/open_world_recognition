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
from open_world import loss_functions
import sklearn

import time





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

    with open(evaluation_config_file) as file:
        config_evaluate = yaml.load(file, Loader=yaml.FullLoader)

    exp_nrs = config_evaluate['name']
    loop_variable_name = config_evaluate['variable']
    loop_variable = {loop_variable_name: []}
    figure_path = config_evaluate['figure_path']
    unknown_classes = config_evaluate['unknown_classes']
    tst_memory_cls = config_evaluate['tst_memory_cls']

    metrics_dict = {'loss': [],
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'F1': [],
                    'mean_pred': [],
                    'mean_true': []}
    results = {key: [] for key in exp_nrs}
    for exp in exp_nrs:

        exp_folder = f'output/{exp}'
        train_config_file = f'{exp_folder}/{exp}_config.yaml'

        # Parse config file
        dataset, model, config, dataset_path,  criterion = parseConfigFile(
            train_config_file, device, multiple_gpu)


        tst_cls_selection = config['test_class_selection']
        same_cls_reverse = config['same_class_reverse']
        same_cls_extend_entries = config['same_class_extend_entries']
        top_n = config['top_n']
        top_k = config['top_k']
        train_classes = config['class_ratio']['l2ac_train']
        class_ratio = config['class_ratio']
        sample_ratio = config['sample_ratio']
        probability_treshold = config['probability_threshold']


        data = np.load(dataset_path)
        data_rep = data['data_rep']
        labels = data['data_labels']
        cls_rep = data['cls_rep']

        load_features = True
        results[exp] = {'macro_f1': [],
                        'weighted_f1': [],
                        'accuracy': [],
                        'open_world_error': [],
                        'unknown_classes': [],
                        'true_labels': [],
                        'final_labels': []
                        }

        for unknown_class in unknown_classes:
            print(f'Start experiment {exp} with {unknown_class} unknown classes')
            input_classes, input_samples, memory_classes, memory_samples, complete_cls_set = getTestIdxSelection(
                                                                                                    tst_cls_selection,
                                                                                                    class_ratio,
                                                                                                    sample_ratio,
                                                                                                    unknown_class,
                                                                                                    tst_memory_cls)
            tst_data_path = getTestDataPath(config, unknown_class)
            if not os.path.exists(tst_data_path):
                X0, X1, Y = meta_utils.rank_test_data(data_rep, labels, data_rep, labels, cls_rep, input_samples,
                                          memory_samples, input_classes, memory_classes, complete_cls_set, top_n)


                np.savez(f'{tst_data_path}',
                         data_rep=data_rep, data_labels=labels, cls_rep=cls_rep,
                         test_X0=X0, test_X1=X1, test_Y=Y)

            train_phase = 'tst'
            test_dataset = ObjectDatasets.MetaDataset(tst_data_path, top_n, top_k, train_classes,
                                                                sample_ratio['l2ac_test_samples'],
                                                                train_phase, same_cls_reverse, same_cls_extend_entries)


            test_loader = DataLoader(test_dataset, batch_size=top_n, shuffle=False, pin_memory=True)

            y_score, memory_labels, true_labels = meta_utils.test_model(test_loader, model, criterion, device, probability_treshold)

            # Unknown label is set to max idx + 1
            unknown_label = complete_cls_set.max() + 1

            # retrieve final label from top n X1 by getting the max prediction score
            final_label = memory_labels[np.arange(len(memory_labels)), y_score.argmax(axis=1)]

            # Set all final labels lower than threshold to unknown
            final_label[np.where(y_score.max(axis=1) < probability_treshold)] = unknown_label


            # Find all classes that are not in memory but only in input
            unknown_class_labels = input_classes[~np.isin(input_classes, memory_classes)]
            # Set all true input labels not in memory to unknown
            true_labels[np.isin(true_labels, unknown_class_labels)] = unknown_label

            macro_f1, weighted_f1, accuracy, open_world_error = calculateMetrics(true_labels, final_label, unknown_label)

            results[exp]['macro_f1'].append(macro_f1)
            results[exp]['weighted_f1'].append(weighted_f1)
            results[exp]['accuracy'].append(accuracy)
            results[exp]['open_world_error'].append(open_world_error)
            results[exp]['unknown_classes'].append(unknown_class)
            results[exp]['true_labels'].append(true_labels)
            results[exp]['final_labels'].append(final_label)

        # TODO Calculate wilderness impact and wilderness
        # Adjust from paper, because paper does not take rejection into account




    plot_utils.plot_best_F1(metrics_dict['F1'], loop_variable, f'{figure_path}_F1')
    plot_utils.plot_best_loss(metrics_dict['loss'], loop_variable, f'{figure_path}_loss')
    plot_utils.plot_best_accuracy(metrics_dict['accuracy'], loop_variable, f'{figure_path}_accuracy')



    return


def getTestIdxSelection(tst_cls_selection, class_ratio, sample_ratio, unknown_classes, tst_memory_cls):
    if tst_cls_selection == 'same_cls':
        input_classes = np.arange(class_ratio['encoder_train'],
                                  class_ratio['encoder_train'] + class_ratio['l2ac_train'] + unknown_classes)
        input_samples = np.arange(sample_ratio['l2ac_train_samples'] + sample_ratio['l2ac_val_samples'],
                                  sample_ratio['l2ac_train_samples'] + sample_ratio['l2ac_val_samples'] +
                                  sample_ratio['l2ac_test_samples'])
        memory_classes = np.arange(class_ratio['encoder_train'],
                                   class_ratio['encoder_train'] + class_ratio['l2ac_train'])
        memory_samples = np.arange(0, sample_ratio['l2ac_train_samples'])
    else:
        input_classes = np.arange(class_ratio['encoder_train'] + class_ratio['l2ac_train'] + class_ratio['l2ac_val'],
                                  class_ratio['encoder_train'] + class_ratio['l2ac_train'] + class_ratio['l2ac_val'] +
                                  tst_memory_cls + unknown_classes)
        input_samples = np.arange(sample_ratio['l2ac_train_samples'] + sample_ratio['l2ac_val_samples'],
                                  sample_ratio['l2ac_train_samples'] + sample_ratio['l2ac_val_samples'] +
                                  sample_ratio['l2ac_test_samples'])
        memory_classes = np.arange(class_ratio['encoder_train'] + class_ratio['l2ac_train'] + class_ratio['l2ac_val'],
                                  class_ratio['encoder_train'] + class_ratio['l2ac_train'] + class_ratio['l2ac_val'] +
                                  tst_memory_cls)
        memory_samples = np.arange(0, sample_ratio['l2ac_train_samples'])

    complete_cls_set = np.arange(class_ratio['encoder_train'],
                                 class_ratio['encoder_train'] + class_ratio['l2ac_train'] + class_ratio['l2ac_val'] +
                                 class_ratio['l2ac_test'])

    return input_classes, input_samples, memory_classes, memory_samples, complete_cls_set

def calculateMetrics(true_labels, final_label, unknown_cls_label):
    macro_f1 = sklearn.metrics.f1_score(true_labels, final_label, average='macro')
    weighted_f1 = sklearn.metrics.f1_score(true_labels, final_label, average='weighted')
    accuracy = sklearn.metrics.accuracy_score(true_labels, final_label)
    open_world_error = openWorldError(true_labels, final_label, unknown_cls_label)
    return macro_f1, weighted_f1, accuracy, open_world_error

def openWorldError(true_labels, final_labels, unknown_cls_label):

    unknown_idx = np.where(true_labels == unknown_cls_label)[0]
    known_idx = np.where(true_labels != unknown_cls_label)[0]
    unknown_samples = unknown_idx.shape[0]
    known_samples = known_idx.shape[0]
    try:
        unknown_error = 1/unknown_samples * np.where(final_labels[unknown_idx] != unknown_cls_label)[0].shape[0]
    except ZeroDivisionError:
        unknown_error = 0
    try:
        known_error = 1/known_samples * (true_labels[known_idx] != final_labels[known_idx])[:, 0].sum()
    except ZeroDivisionError:
        known_error = 0
    open_world_error = known_error + unknown_error
    return open_world_error


def getTestDataPath(config, unknown_classes):
    encoder = config['encoder']
    feature_layer = config['feature_layer']
    image_resize = config['image_resize']
    unfreeze_layer = config['unfreeze_layer']
    tst_cls_selection = config['test_class_selection']
    feature_scaling = config['feature_scaling']
    top_n = config['top_n']
    train_samples = config['sample_ratio']['l2ac_train_samples']
    train_classes = config['class_ratio']['l2ac_train']

    test_data_path = f"datasets/{config['dataset_path']}" + f'/{encoder}/{feature_layer}_{feature_scaling}_{image_resize}_{unfreeze_layer}_{train_classes}_{train_samples}_{top_n}_{tst_cls_selection}_{unknown_classes}_tst.npz'
    return test_data_path

def parseConfigFile(config_file, device, multiple_gpu):

    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    ## Training hyperparameters
    batch_size = config['batch_size']

    ## L2AC Parameters
    top_k = int(config['top_k'])
    train_classes = config['class_ratio']['l2ac_train']

    ## Dataset preparation parameters:
    same_class_reverse = config['same_class_reverse']
    same_class_extend_entries = config['same_class_extend_entries']

    ## Classes
    # Load dataset

    encoder = config['encoder']
    feature_layer = config['feature_layer']
    image_resize = config['image_resize']
    unfreeze_layer = config['unfreeze_layer']
    tst_cls_selection = config['test_class_selection']
    feature_scaling = config['feature_scaling']
    top_n = config['top_n']
    train_samples = config['sample_ratio']['l2ac_train_samples']

    dataset_path = f"datasets/{config['dataset_path']}" + f'/{encoder}/{feature_layer}_{feature_scaling}_{image_resize}_{unfreeze_layer}_{train_classes}_{train_samples}_{top_n}_{tst_cls_selection}.npz'
    dataset_class = config['dataset_class']

    trn_phase = 'trn'
    test_dataset = eval('ObjectDatasets.' + dataset_class)(dataset_path, top_n, top_k, train_classes,
                                                           train_samples,
                                                           trn_phase, same_class_reverse, same_class_extend_entries)
    # Load model
    features_size = len(test_dataset.memory[0])

    model_path = 'output/' + str(config['name']) + '/' + str(config['name']) + '_model.pt'
    state_path = 'output/' + str(config['name']) + '/' + str(config['name']) + '_best_state.pth'
    best_state = torch.load(state_path)
    model_class = config['model_class']
    model = eval('RecognitionModels.' + model_class)(train_classes, features_size, batch_size, top_k).to(device)
    print('Load model ' + model_path)
    # OpenWorldUtils.loadModel(model, model_path)
    model.load_state_dict(best_state['model'])

    criterion = eval(f'loss_functions.{config["criterion"]}')()

    return test_dataset, model, config, dataset_path, criterion


if __name__ == "__main__":
    main()
