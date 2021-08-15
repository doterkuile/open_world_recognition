import torch
import torch.utils.data as data_utils
from open_world import OpenWorldUtils
from open_world import ObjectDatasets
from open_world import loss_functions
import open_world.meta_learner.meta_learner_utils as meta_utils
import open_world.plot_utils as plot_utils
import yaml
import numpy as np
from torch.utils.data import DataLoader
import argparse
import os
import shutil
import matplotlib.pyplot as plt
import time


def main():
    start_time = time.time()
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

    # Parse config file
    (train_dataset, model, criterion, optimizer, epochs, batch_size, learning_rate,
     config) = OpenWorldUtils.parseConfigFile(
        device, multiple_gpu)

    ## Create new entry folder for results of experiment

    exp_name = str(config['name'])
    exp_folder = 'output/' + exp_name
    create_similarity_gif = config['create_similarity_gif']

    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    gif_path = None

    if create_similarity_gif:
        gif_path = exp_folder + '/' + exp_name

    if not os.path.exists(gif_path):
        os.makedirs(gif_path)

    if not os.path.exists(exp_folder + '/matching_layer'):
        os.makedirs(exp_folder + '/matching_layer')

    figure_path = exp_folder + '/' + exp_name
    ml_figure_path = exp_folder + '/matching_layer/' + exp_name
    results_path = exp_folder + '/' + exp_name + '_results.npz'
    sim_path = exp_folder + '/' + exp_name + '_similarities.npz'
    model_path = exp_folder + '/' + exp_name + '_model.pt'
    dataset_path = train_dataset.data_path

    # Get hyperparameters
    test_classes = config['class_ratio']['l2ac_test']
    train_samples_per_cls = config['train_samples_per_cls']
    probability_threshold = config['probability_threshold']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)#, num_workers=2)

    test_dataset = ObjectDatasets.MetaDataset(dataset_path, config['top_n'], config['top_k'],
                                              test_classes, train_samples_per_cls, train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)#, num_workers=2)
    pos_weight = torch.tensor(1.0).to(device).to(dtype=torch.float)
    ml_weight = torch.tensor(config['top_n']).to(device).to(dtype=torch.float)
    test_criterion = eval('loss_functions.' + config['criterion'])(pos_weight)
    ml_criterion = loss_functions.bce_loss_matching_layer(ml_weight)
    ml_criterion_test = loss_functions.bce_loss_matching_layer(pos_weight)
    two_step_training = config['two_step_training']
    freeze_matching_layer = config['freeze_matching_layer']

    if two_step_training or freeze_matching_layer:

        param_keys = 'matching_layer'
        for name, param in model.named_parameters():
            if 'matching_layer' not in name:
                param.requires_grad = False
        optimizer = eval('torch.optim.' + config['optimizer'])(filter(lambda p: p.requires_grad, model.parameters()),
                                                               lr=learning_rate, weight_decay=1e-5)

        trn_metrics_ml, tst_metrics_ml, best_state_ml = meta_utils.trainMatchingLayer(
            model,
            train_loader,
            test_loader,
            epochs,
            ml_criterion,
            ml_criterion_test,
            optimizer,
            device,
            probability_threshold,
            gif_path)

        print(f'Choosing the matching layer model of epoch {best_state_ml["epoch"]}, with F1 score: {best_state_ml["F1"]} ')
        model.load_state_dict(best_state_ml['model'])

        for name, param in model.named_parameters():

            if freeze_matching_layer:
                if 'matching_layer' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = True

        optimizer = eval('torch.optim.' + config['optimizer'])(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate, weight_decay=1e-5)

        # Plot metrics
        plot_utils.plot_losses(trn_metrics_ml['loss'], tst_metrics_ml['loss'], ml_figure_path)
        plot_utils.plot_accuracy(trn_metrics_ml['accuracy'], tst_metrics_ml['accuracy'], ml_figure_path)
        plot_utils.plot_precision(trn_metrics_ml['precision'], tst_metrics_ml['precision'], ml_figure_path)
        plot_utils.plot_recall(trn_metrics_ml['recall'], tst_metrics_ml['recall'], ml_figure_path)
        plot_utils.plot_F1(trn_metrics_ml['F1'], tst_metrics_ml['F1'], ml_figure_path)
        plot_utils.plot_mean_prediction(trn_metrics_ml['mean_pred'], trn_metrics_ml['mean_true'],
                                        tst_metrics_ml['mean_pred'],
                                        tst_metrics_ml['mean_true'], ml_figure_path)


    trn_metrics, tst_metrics, best_state = meta_utils.trainMetaModel(model,
                                                                      train_loader,
                                                                      test_loader,
                                                                      epochs,
                                                                      criterion,
                                                                      test_criterion,
                                                                      optimizer,
                                                                      device,
                                                                      probability_threshold,
                                                                      gif_path)

    model.load_state_dict(best_state['model'])

    # Plot metrics
    plot_utils.plot_losses(trn_metrics['loss'], tst_metrics['loss'], figure_path)
    plot_utils.plot_accuracy(trn_metrics['accuracy'], tst_metrics['accuracy'], figure_path)
    plot_utils.plot_precision(trn_metrics['precision'], tst_metrics['precision'], figure_path)
    plot_utils.plot_recall(trn_metrics['recall'], tst_metrics['recall'], figure_path)
    plot_utils.plot_F1(trn_metrics['F1'], tst_metrics['F1'], figure_path)
    plot_utils.plot_mean_prediction(trn_metrics['mean_pred'], trn_metrics['mean_true'], tst_metrics['mean_pred'],
                                    tst_metrics['mean_true'], figure_path)

    trn_y_pred, trn_y_true, trn_losses, trn_sim_scores, trn_y_pred_raw = meta_utils.validate_model(
        train_loader, model,
        criterion, device,
        probability_threshold)

    tst_y_pred, tst_y_true, tst_losses, tst_sim_scores, tst_y_pred_raw = meta_utils.validate_model(
        test_loader, model,
        test_criterion, device,
        probability_threshold)

    title = 'Intermediate similarity score'
    fig_sim, axs_sim = plt.subplots(2, 1, figsize=(15, 10))

    ml_trn_same_idx = (trn_y_true == 1).nonzero()[0].squeeze()
    ml_trn_diff_idx = (trn_y_true == 0).nonzero()[0].squeeze()
    ml_tst_same_idx = (tst_y_true == 1).nonzero()[0].squeeze()
    ml_tst_diff_idx = (tst_y_true == 0).nonzero()[0].squeeze()

    plot_utils.plot_prob_density(fig_sim, axs_sim, trn_sim_scores, ml_trn_same_idx, ml_trn_diff_idx, tst_sim_scores, ml_tst_same_idx,
                      ml_tst_diff_idx, title, figure_path + '_intermediate_similarity')

    title = 'Final similarity score'
    fig_final, axs_final = plt.subplots(2, 1, figsize=(15, 10))

    al_trn_same_idx = (trn_y_true == 1).nonzero()[0].squeeze()
    al_trn_diff_idx = (trn_y_true == 0).nonzero()[0].squeeze()
    al_tst_same_idx = (tst_y_true == 1).nonzero()[0].squeeze()
    al_tst_diff_idx = (tst_y_true == 0).nonzero()[0].squeeze()

    plot_utils.plot_prob_density(fig_final, axs_final, trn_y_pred_raw, al_trn_same_idx, al_trn_diff_idx, tst_y_pred_raw,
                                 al_tst_same_idx, al_tst_diff_idx, title, figure_path + '_final_similarity')

    OpenWorldUtils.saveModel(model, model_path)
    print(f'Saving the final model of epoch {best_state["epoch"]}, with F1 score: {best_state["F1"]} ')
    best_state['model_class'] = config['model_class']
    torch.save(best_state, f'{exp_folder}/{exp_name}_best_state.pth')

    np.savez(results_path,
             train_loss=trn_metrics['loss'],
             test_loss=tst_metrics['loss'],
             train_acc=trn_metrics['accuracy'],
             test_acc=tst_metrics['accuracy'],
             train_precision=trn_metrics['precision'],
             test_precision=tst_metrics['precision'],
             train_recall=trn_metrics['recall'],
             test_recall=tst_metrics['recall'],
             train_F1=trn_metrics['F1'],
             test_F1=tst_metrics['F1'])

    print(f'\nTotal duration: {time.time() - start_time:.0f} seconds')
    return


if __name__ == "__main__":
    main()
