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
    (train_dataset, model, criterion, optimizer, epochs, batch_size, learning_rate, config) = OpenWorldUtils.parseConfigFile(
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)#, num_workers=4)

    test_dataset = ObjectDatasets.MetaDataset(dataset_path, config['top_n'], config['top_k'],
                                              test_classes, train_samples_per_cls, train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)#, num_workers=4)

    test_criterion = eval('torch.nn.' + config['criterion'])(reduction='mean')

    two_step_training = config['two_step_training']
    train_matching_layer_only = config['train_matching_layer_only']

    if two_step_training or train_matching_layer_only:

        param_keys = 'matching_layer'
        for name , param in model.named_parameters():
            if 'matching_layer' not in name:
                param.requires_grad = False
        optimizer = eval('torch.optim.' + config['optimizer'])(filter(lambda p: p.requires_grad, model.parameters()),
                                                               lr=learning_rate, weight_decay=1e-5)

        trn_metrics_ml, trn_similarity_scores_ml, tst_metrics_ml, tst_similarity_scores_ml, best_state_ml, = meta_utils.trainMatchingLayer(
            model,
            train_loader,
            test_loader,
            epochs,
            criterion,
            test_criterion,
            optimizer,
            device,
            probability_threshold,
            gif_path)

        for name, param in model.named_parameters():
            if 'matching_layer' in name:
                param.requires_grad = False
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

        if train_matching_layer_only:
            print('Trained only matching layer')
            print(f'\nTotal duration: {time.time() - start_time:.0f} seconds')
            return



    trn_metrics, trn_similarity_scores, tst_metrics, tst_similarity_scores, best_state, = meta_utils.trainMetaModel(model,
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

    plot_utils.plot_intermediate_similarity(
        trn_similarity_scores['intermediate_same_cls'],
        trn_similarity_scores['intermediate_diff_cls'],
        tst_similarity_scores['intermediate_same_cls'],
        tst_similarity_scores['intermediate_diff_cls'],
        figure_path)

    plot_utils.plot_final_similarity(
        trn_similarity_scores['final_same_cls'],
        trn_similarity_scores['final_diff_cls'],
        tst_similarity_scores['final_same_cls'],
        tst_similarity_scores['final_diff_cls'],
        figure_path)


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
    plot_utils.plot_prob_density(fig_sim, axs_sim, trn_sim_scores, trn_y_true, tst_sim_scores, tst_y_true, title,
                                 figure_path + '_intermediate_similarity')


    title = 'Final similarity score'
    fig_final, axs_final = plt.subplots(2, 1, figsize=(15, 10))
    plot_utils.plot_prob_density(fig_final, axs_final, trn_y_pred_raw, trn_y_true, tst_y_pred_raw, tst_y_true, title,
                                 figure_path + '_final_similarity')


    OpenWorldUtils.saveModel(model, model_path)
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

    np.savez(sim_path,
             trn_final_same_cls=trn_similarity_scores['final_same_cls'],
             trn_intermediate_same_cls=trn_similarity_scores['intermediate_same_cls'],
             trn_final_diff_cls=trn_similarity_scores['final_diff_cls'],
             trn_intermediate_diff_cls=trn_similarity_scores['intermediate_diff_cls'],
             tst_final_same_cls=tst_similarity_scores['final_same_cls'],
             tst_intermediate_same_cls=tst_similarity_scores['intermediate_same_cls'],
             tst_final_diff_cls=tst_similarity_scores['final_diff_cls'],
             tst_intermediate_diff_cls=tst_similarity_scores['intermediate_diff_cls'],
             )

    print(f'\nTotal duration: {time.time() - start_time:.0f} seconds')
    return


if __name__ == "__main__":
    main()
