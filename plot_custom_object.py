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

    experiment_names = ['l_t_teapot_no_lstm',
                        'l_t_teapot_no_lstm',
                        # 'l_t_teapot_no_lstm_same',
                        # 'l_t_teapot_no_lstm_same',
                        ]

    memory_names = [
                    'tinyimagenet_train',
                    'reconstruction_site',
                    'webots_dataset_224',
                    'webots_dataset_224_close',
                    # 'webots_dataset_224',
                    # 'tinyimagenet_train'
                    ]

    input_names = ['teapot_photos',
                   'teapot_photos',
                   # 'teapot_photos',
                   # 'teapot_photos',
                   ]

    for experiment_name, input_name in zip(experiment_names, input_names):
        make_teapot_table(experiment_name, memory_names, input_name)



    return


def make_teapot_table(experiment_name, memory_names,input_name):
    e_k_new_list_same = []
    e_k_new_list_diff = []
    e_k_list_same = []
    e_k_list_diff = []
    e_u_list_same = []
    e_u_list_diff = []
    e_u_new_list_same = []
    e_u_new_list_diff = []
    F1_list_diff = []
    F1_list_same = []
    F1_new_list_same = []
    F1_new_list_diff = []



    for memory_name in memory_names:
        file_path_same = f"results_teapot/{experiment_name}_same_{memory_name}_{input_name}_after.npz"
        file_path_diff = f"results_teapot/{experiment_name}_same_{memory_name}_{input_name}_after.npz"

        F1_new_cls_same, F1_same, e_k_new_cls_same, e_k_same, e_u_same = get_metrics(file_path_same)
        F1_new_cls_diff, F1_diff, e_k_new_cls_diff, e_k_diff, e_u_diff = get_metrics(file_path_diff)

        e_k_new_list_same.append(e_k_new_cls_same)
        e_k_new_list_diff.append(e_k_new_cls_diff)
        e_k_list_same.append(e_k_same)
        e_k_list_diff.append(e_k_diff)
        e_u_list_same.append(e_u_same)
        e_u_list_diff.append(e_u_diff)
        e_u_new_list_same.append('-')
        e_u_new_list_diff.append('-')
        F1_new_list_same.append(F1_new_cls_same)
        F1_new_list_diff.append(F1_new_cls_diff)
        F1_list_same.append(F1_same)
        F1_list_diff.append(F1_diff)

    results_new = np.stack([F1_new_list_same,F1_new_list_diff ,e_u_new_list_same, e_u_new_list_diff, e_k_new_list_same, e_k_new_list_diff, ], axis=1)
    results_all = np.stack([F1_list_same, F1_list_diff, e_u_list_same, e_u_list_diff, e_k_list_same, e_k_list_diff ], axis=1)

    print(f'\n Model: {experiment_name}, Input dataset: {input_name}\n')
    for row_new, row_all in zip(results_new, results_all):

        print(f"& \\textit{{New}} &{row_new[0]} & {row_new[1]} & {row_new[2]} & {row_new[3]} & {row_new[4]} & {row_new[5]} \\\\ ")
        print(f"& \\textit{{All}} &{row_all[0]} & {row_all[1]} & {row_all[2]} & {row_all[3]} & {row_all[4]} & {row_all[5]} \\\\ ")
        print('\n')




    return

def get_metrics(file_path):
    data = np.load(file_path)

    memory_labels = data['memory_labels']
    y_score = data['y_score']
    true_labels = data['true_labels']
    complete_cls_set = data['complete_cls_set']
    new_class = data['new_class']
    probability_threshold = data['probability_threshold']
    unknown_label = complete_cls_set.max() + 1

    e_k_new_cls, e_k, e_u = calculate_classification_errors(y_score, memory_labels, true_labels, new_class,
                                                            unknown_label, probability_threshold)
    F1_new_cls, F1 = calc_f1(y_score, memory_labels, true_labels, new_class, unknown_label, probability_threshold)

    return round(F1_new_cls,3), round(F1,3), round(e_k_new_cls,3), round(e_k,3), round(e_u,3)

def calculate_classification_errors(y_score, memory_labels, true_labels, new_class, unknown_label, probability_threshold):

    final_labels = memory_labels[np.arange(len(memory_labels)), y_score.argmax(axis=1)]
    unique_memory_set = np.unique(memory_labels)
    # Set class labels not in memory to unknown
    final_labels[np.where(y_score.max(axis=1) < probability_threshold)] = unknown_label
    true_labels[~np.isin(true_labels, unique_memory_set)] = unknown_label
    unknown_cls_idx = np.where(true_labels == unknown_label)[0]
    known_cls_idx = np.where(true_labels != unknown_label)[0]
    new_cls_idx = np.where(true_labels == new_class)[0]
    e_u = np.where(final_labels[unknown_cls_idx] != unknown_label)[0].shape[0]/unknown_cls_idx.shape[0]
    e_k_new_cls = np.where(final_labels[new_cls_idx] != new_class)[0].shape[0]/new_cls_idx.shape[0]


    e_k = np.where(final_labels[known_cls_idx] != true_labels[known_cls_idx])[0].shape[0]/known_cls_idx.shape[0]


    return e_k_new_cls, e_k, e_u

def calc_f1(y_score, memory_labels, true_labels, new_class, unknown_label, probability_threshold):

    final_labels = memory_labels[np.arange(len(memory_labels)), y_score.argmax(axis=1)]
    unique_memory_set = np.unique(memory_labels)
    # Set class labels not in memory to unknown
    final_labels[np.where(y_score.max(axis=1) < probability_threshold)] = unknown_label
    true_labels[~np.isin(true_labels, unique_memory_set)] = unknown_label
    unknown_cls_idx = np.where(true_labels == unknown_label)[0]
    known_cls_idx = np.where(true_labels != unknown_label)[0]
    new_cls_idx = np.where(true_labels == new_class)[0]

    new_cls_true = np.ones(new_cls_idx.shape)
    new_cls_pred = np.ones(new_cls_idx.shape)
    new_cls_pred[np.where(final_labels[new_cls_idx] != new_class)] = 0
    # true_labels[np.isin(true_labels, unknown_class_labels)] = unknown_label
    F1_new_cls = sklearn.metrics.f1_score(y_true=new_cls_true, y_pred=new_cls_pred, zero_division=0)
    F1 = sklearn.metrics.f1_score(y_true=true_labels, y_pred=final_labels, zero_division=0,average='weighted')

    return F1_new_cls, F1


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
