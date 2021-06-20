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
import torch.nn as nn
from open_world import OpenWorldUtils
import open_world.meta_learner.meta_learner_utils as meta_utils
import time



def parseConfigFile(config_file, device, multiple_gpu):

    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    ## Training/laoding parameters:
    load_memory = config['load_memory']
    save_images = config['save_images']
    enable_training = config['enable_training']

    ## Training hyperparameters
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    epochs = config['epochs']

    ## L2AC Parameters
    top_k = int(config['top_k'])
    top_n = int(config['top_n'])
    train_classes = config['train_classes']
    test_classes = config['test_classes']
    train_samples_per_cls = config['train_samples_per_cls']

    ## Dataset preparation parameters:
    same_class_reverse = config['same_class_reverse']
    same_class_extend_entries = config['same_class_extend_entries']


    ## Classes
    # Load dataset
    dataset_path = config['dataset_path']
    dataset_class = config['dataset_class']

    train_dataset = eval('ObjectDatasets.' + dataset_class)(dataset_path, top_n, top_k, train_classes, train_samples_per_cls
                                                      ,True,  same_class_reverse, same_class_extend_entries)
    test_dataset = eval('ObjectDatasets.' + dataset_class)(dataset_path, top_n, top_k, train_classes,
                                                            train_samples_per_cls
                                                            , False, same_class_reverse, same_class_extend_entries)

    # Load model
    features_size = len(train_dataset.memory[0])


    model_path = 'output/' + str(config['name']) + '/' + str(config['name']) + '_model.pt'
    model_class = config['model_class']
    model = eval('RecognitionModels.' + model_class)(model_path, train_classes,features_size, batch_size, top_k).to(device)

    # If multiple gpu's available
    if multiple_gpu:
        print(f'The use of multiple gpus is enabled: using {torch.cuda.device_count()} gpus')
        model = nn.DataParallel(model)


    print('Load model ' + model_path)
    OpenWorldUtils.loadModel(model, model_path)

    return train_dataset, test_dataset, model, config

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

    exp_name = config_evaluate['name']
    batch_size = config_evaluate['batch_size']

    exp_folder = 'output/' + exp_name
    train_config_file = exp_folder + '/' + exp_name + '_config.yaml'
    figure_path = exp_folder + '/' + exp_name

    # Overwrite terminal argument if necessary
    # config_file = 'config/L2AC_train.yaml'

    # Parse config file
    train_dataset, test_dataset, model, config= parseConfigFile(
        train_config_file, device, multiple_gpu)


    # Get hyperparameters
    train_classes = config['train_classes']
    train_samples_per_cls = config['train_samples_per_cls']
    probability_treshold = config['probability_threshold']
    criterion = eval('nn.' + config['criterion'])()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    trn_similarity_scores = {'final_same_cls': [],
                             'intermediate_same_cls': [],
                             'final_diff_cls': [],
                             'intermediate_diff_cls': [],
                             }
    tst_similarity_scores = {'final_same_cls': [],
                             'intermediate_same_cls': [],
                             'final_diff_cls': [],
                             'intermediate_diff_cls': [],
                             }

    meta_utils.validate_similarity_scores(trn_similarity_scores, model, train_loader, device)
    # meta_utils.validate_similarity_scores(tst_similarity_scores, model, test_loader, device)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    trn_y_pred, trn_y_true, trn_loss, trn_sim_scores = meta_utils.validate_model(train_loader, model, criterion, device, probability_treshold)
    tst_y_pred, tst_y_true, tst_loss, tst_sim_scores = meta_utils.validate_model(test_loader, model, criterion, device, probability_treshold)

    trn_y_pred = np.array(torch.cat(trn_y_pred))
    trn_y_true = np.array(torch.cat(trn_y_true))
    trn_sim_scores = np.array(torch.cat(trn_sim_scores, dim=1).detach()).transpose(1, 0)

    tst_y_pred = np.array(torch.cat(tst_y_pred))
    tst_y_true = np.array(torch.cat(tst_y_true))
    tst_sim_scores = np.array(torch.cat(tst_sim_scores, dim=1).detach()).transpose(1, 0)

    start = time.time()

    title = 'Intermediate similarity_score'
    plot_utils.plot_prob_density(trn_sim_scores, trn_y_true,tst_sim_scores, tst_y_true, title, figure_path + '_intermediate_sim_pbd')

    print("Trn scores took " + str(time.time() - start) + 's')
    start = time.time()

    title = 'Final similarity_score'
    plot_utils.plot_prob_density(trn_y_pred, trn_y_true,tst_y_pred, tst_y_true, title, figure_path + '_final_sim_pbd')

    print("Tst scores took " + str(time.time() - start) + 's')

    #
    # plot_utils.plot_intermediate_similarity(trn_intermediate_same_cls, trn_intermediate_diff_cls,
    #                                         tst_intermediate_same_cls, tst_intermediate_diff_cls, figure_path)
    # plot_utils.plot_final_similarity(trn_final_same_cls, trn_final_diff_cls, tst_final_same_cls, tst_final_diff_cls,
    #                                  figure_path)
    #
    # OpenWorldUtils.saveModel(model, model_path)
    #
    # np.savez(results_path, train_loss=trn_loss, test_loss=tst_loss, train_acc=trn_acc, test_acc=tst_acc,
    #          train_precision=trn_precision, test_precision=tst_precision, train_recall=trn_recall,
    #          test_recall=tst_recall, train_F1=trn_F1, test_F1=tst_F1)
    #
    # np.savez(sim_path,
    #          trn_final_same_cls=trn_similarity_scores['final_same_cls'],
    #          trn_intermediate_same_cls=trn_similarity_scores['intermediate_same_cls'],
    #          trn_final_diff_cls=trn_similarity_scores['final_diff_cls'],
    #          trn_intermediate_diff_cls=trn_similarity_scores['intermediate_diff_cls'],
    #          tst_final_same_cls=tst_similarity_scores['final_same_cls'],
    #          tst_intermediate_same_cls=tst_similarity_scores['intermediate_same_cls'],
    #          tst_final_diff_cls=tst_similarity_scores['final_diff_cls'],
    #          tst_intermediate_diff_cls=tst_similarity_scores['intermediate_diff_cls'],
    #          )

    return


if __name__ == "__main__":
    main()
