import torch
import torch.utils.data as data_utils
from open_world import OpenWorldUtils
from open_world import ObjectDatasets
from open_world import RecognitionModels
from open_world import loss_functions
import open_world.meta_learner.meta_learner_utils as meta_utils
import open_world.plot_utils as plot_utils
import yaml
import json
import torchvision
from torchvision import transforms
from open_world.ObjectDatasets import TrainPhase
from evaluate_l2ac import calculateMetrics
import sklearn

import numpy as np
from torch.utils.data import DataLoader
import argparse
import os
import shutil
import matplotlib.pyplot as plt

import time


def main():
    torch.manual_seed(42)

    # Main gpu checks
    multiple_gpu = True if torch.cuda.device_count() > 1 else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Cuda device not available make sure CUDA has been installed")
        return
    else:
        print(f"Running with {torch.cuda.device_count()} GPUs")

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()
    evaluation_config_file = args.config_file

    with open(evaluation_config_file) as file:
        evaluation_config = yaml.load(file, Loader=yaml.FullLoader)

    no_memory_file_path = f"results_teapot/{evaluation_config['experiment_name']}_{evaluation_config['memory_dataset']}_{evaluation_config['input_dataset']}_before.npz"
    extended_memory_file_path = f"results_teapot/{evaluation_config['experiment_name']}_{evaluation_config['memory_dataset']}_{evaluation_config['input_dataset']}_after.npz"

    no_memory_data = np.load(no_memory_file_path)
    extended_memory_data = np.load(extended_memory_file_path)

    memory_labels = no_memory_data['memory_labels']
    y_score = no_memory_data['y_score']
    true_labels = no_memory_data['true_labels']
    complete_cls_set = no_memory_data['complete_cls_set']
    new_class = no_memory_data['new_class']
    probability_threshold = no_memory_data['probability_threshold']
    unknown_label = complete_cls_set.max() + 1


    e_u_new, e_u_other = getUnknown_error(memory_labels,
                                          y_score,
                                          true_labels,
                                          complete_cls_set,
                                          probability_threshold,
                                          new_class)

    memory_labels_after = extended_memory_data['memory_labels']
    y_score_after = extended_memory_data['y_score']
    true_labels_after = extended_memory_data['true_labels']
    complete_cls_set_after = extended_memory_data['complete_cls_set']
    new_class_after = extended_memory_data['new_class']


    F1, F1_all, e_k_new_cls = get_f1_scores(memory_labels_after,
                                            y_score_after,
                                            true_labels_after,
                                            complete_cls_set_after,
                                            probability_threshold,
                                            new_class_after)

    plot_new_cls_score(true_labels, y_score, unknown_label, new_class)


    return


def getUnknown_error(memory_labels, y_score, true_labels, complete_cls_set, probability_threshold, new_class):
    # Unknown label is set to max idx + 1
    unknown_label = complete_cls_set.max() + 1

    # retrieve final label from top n X1 by getting the max prediction score
    final_label = memory_labels[np.arange(len(memory_labels)), y_score.argmax(axis=1)]

    # Set all final labels lower than threshold to unknown
    final_label[np.where(y_score.max(axis=1) < probability_threshold)] = unknown_label

    # Set all true input labels not in memory to unknown
    new_cls_idx = np.where(true_labels == new_class)[0]
    other_cls_idx = np.where(true_labels != new_class)[0]
    e_u_new = (np.where(final_label[new_cls_idx] != unknown_label)[0]).shape[0] / new_cls_idx.shape[0]
    try:
        e_u_other = (np.where(final_label[other_cls_idx] != unknown_label)[0]).shape[0] / other_cls_idx.shape[0]
    except ZeroDivisionError:
        e_u_other = 0

    return e_u_new, e_u_other


def get_f1_scores(memory_labels, y_score, true_labels, complete_cls_set, probability_threshold, new_class):
    # Unknown label is set to max idx + 1
    unknown_label = complete_cls_set.max() + 1

    # retrieve final label from top n X1 by getting the max prediction score
    final_label = memory_labels[np.arange(len(memory_labels)), y_score.argmax(axis=1)]

    # Set all final labels lower than threshold to unknown
    final_label[np.where(y_score.max(axis=1) < probability_threshold)] = unknown_label

    # Set all true input labels not in memory to unknown
    new_cls_idx = np.where(true_labels == new_class)[0]
    other_cls_idx = np.where(true_labels != new_class)[0]
    # true_labels[np.isin(true_labels, unknown_class_labels)] = unknown_label
    input_class_set = np.unique(true_labels)
    unknown_class_set = input_class_set[~np.isin(input_class_set, np.unique(memory_labels))]

    true_labels[np.isin(true_labels, unknown_class_set)] = unknown_label
    true_unknowns = np.where(true_labels == unknown_label)[0]

    incorrect_unknowns = np.where(final_label[true_unknowns] != unknown_label)[0]
    try:
        e_u = incorrect_unknowns.shape[0] / true_unknowns.shape[0]
    except ZeroDivisionError:
        e_u = 0
    e_k_new_cls = (np.where(final_label[new_cls_idx] != new_class)[0]).shape[0] / new_cls_idx.shape[0]
    # e_u_other_cls = (np.where(final_label[other_cls_idx] != unknown_label)[0]).shape[0]/other_cls_idx.shape[0]

    new_cls_true = np.ones(new_cls_idx.shape)
    new_cls_pred = np.ones(new_cls_idx.shape)
    new_cls_pred[np.where(final_label[new_cls_idx] != new_class)] = 0
    # true_labels[np.isin(true_labels, unknown_class_labels)] = unknown_label
    F1 = sklearn.metrics.f1_score(y_true=new_cls_true, y_pred=new_cls_pred, zero_division=0)

    ## F1 seconde verison, over all data
    new_cls_true_all = np.ones(final_label.shape)
    new_cls_true_all[np.where(true_labels != new_class)] = 0

    new_cls_pred_all = np.ones(final_label.shape)
    new_cls_pred_all[np.where(final_label != new_class)] = 0

    F1_all = sklearn.metrics.f1_score(y_true=new_cls_true_all, y_pred=new_cls_pred_all, zero_division=0)
    return F1, F1_all, e_k_new_cls


def plot_new_cls_score(true_labels_initial, score_initial, unknown_label, new_label):
    fig, axs = plt.subplots(1, 1, figsize=(15, 10))

    unknown_scores_initial = score_initial[np.where(true_labels_initial == unknown_label)]

    # known_scores = score[np.where(true_labels != unknown_label)]
    #
    # score_dict = {x_label: np.concatenate([known_scores, unknown_scores], axis=0),
    #                    legend_label: np.concatenate([np.full((len(known_scores)), 'Known classes'), np.full((len(unknown_scores)), 'Unknown classes')], axis=0)}
    # # score_dict = {x_label: np.concatenate([unknown_scores], axis=0),
    # #               legend_label: np.concatenate([np.full((len(unknown_scores)), 'Unknown classes')], axis=0)}
    #
    # scores = pd.DataFrame.from_dict(score_dict)
    # axs[ii].set_title(f"{results[exp][results[exp]['known_unknown']][ii]} {title}", fontweight="bold", size=18)
    # sns.histplot(ax=axs[ii], data=scores, x=x_label, hue=legend_label, stat='probability', kde=False,
    #      common_norm=True,
    #      element='bars', binrange=(0, 1), binwidth=0.005)
    #
    # if figure_path is not None:
    #     fig.savefig(f"{figure_path}_{exp}_score_distribution")

if __name__ == "__main__":
    main()
