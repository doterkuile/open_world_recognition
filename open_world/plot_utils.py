import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from open_world import OpenWorldUtils
import torch
from torch.utils.data import DataLoader
import imageio
import os
import torch.nn as nn
from torchvision.utils import make_grid


import open_world.loss_functions as loss_func


def plot_losses(train_losses, test_losses, figure_path):
    fig = plt.figure()
    plt.plot(train_losses, label='training loss')
    plt.plot(test_losses, label='validation loss')
    plt.ylabel('Loss function')
    plt.xlabel('Epochs')
    plt.title('Losses')
    plt.legend()
    fig.savefig(figure_path + '_losses')
    plt.close(fig)
    return


def plot_accuracy(train_acc, test_acc, figure_path):
    fig = plt.figure()
    plt.plot(train_acc, label='training accuracy')
    plt.plot(test_acc, label='validation accuracy')
    plt.ylabel('accuracy [%]')
    plt.xlabel('Epochs')
    plt.title('Accuracy')
    plt.legend()
    fig.savefig(figure_path + '_accuracy')
    plt.close(fig)
    return


def plot_precision(train_precision, test_precision, figure_path):
    fig = plt.figure()
    plt.plot(train_precision, label='training precision')
    plt.plot(test_precision, label='validation precision')
    plt.ylabel('precision')
    plt.xlabel('Epochs')
    plt.title('Precision')
    plt.legend()
    fig.savefig(figure_path + '_precision')
    plt.close(fig)
    return


def plot_recall(train_recall, test_recall, figure_path):
    fig = plt.figure()
    plt.plot(train_recall, label='training recall')
    plt.plot(test_recall, label='validation recall')
    plt.ylabel('recall')
    plt.xlabel('Epochs')
    plt.title('Recall')
    plt.legend()
    fig.savefig(figure_path + '_recall')
    plt.close(fig)
    return


def plot_F1(train_F1, test_F1, figure_path):
    fig = plt.figure()
    plt.plot(train_F1, label='training F1')
    plt.plot(test_F1, label='validation F1')
    plt.ylabel('F1')
    plt.xlabel('Epochs')
    plt.title('F1 ')
    plt.legend()
    fig.savefig(figure_path + '_F1')
    plt.close(fig)
    return


def plot_mean_prediction(trn_pred, trn_true, tst_pred, tst_true, figure_path):
    fig = plt.figure()
    plt.plot(trn_pred, label='Training predicted mean', alpha=1, color='blue')
    plt.plot(trn_true, label='Training true mean', alpha=0.4, color='blue')
    plt.plot(tst_pred, label='Validation predicted mean', alpha=1, color='orange')
    plt.plot(tst_true, label='Validation true mean', alpha=0.4, color='orange')
    plt.ylabel('Mean value')
    plt.xlabel('Epochs')
    plt.title('Mean prediction score')
    plt.legend()
    fig.savefig(figure_path + '_mean_pred')
    plt.close(fig)

    return


def plot_intermediate_similarity(trn_same_cls, trn_diff_cls, tst_same_cls, tst_diff_cls, figure_path):
    fig = plt.figure()
    plt.plot(trn_same_cls, label='Training same class', alpha=1, color='blue')
    plt.plot(trn_diff_cls, label='Training diff class', alpha=0.4, color='blue')
    plt.plot(tst_same_cls, label='Validation same class', alpha=1, color='orange')
    plt.plot(tst_diff_cls, label='Validation diff class', alpha=0.4, color='orange')
    plt.ylabel('Similarity score')
    plt.xlabel('Epochs')
    plt.title('Intermediate similarity_score')
    plt.legend()
    fig.savefig(figure_path + '_intermediate_similarity')
    plt.close(fig)
    return


def plot_final_similarity(trn_same_cls, trn_diff_cls, tst_same_cls, tst_diff_cls, figure_path):
    fig = plt.figure()
    plt.plot(trn_same_cls, label='Training same class', alpha=1, color='blue')
    plt.plot(trn_diff_cls, label='Training diff class', alpha=0.4, color='blue')
    plt.plot(tst_same_cls, label='Validation same class', alpha=1, color='orange')
    plt.plot(tst_diff_cls, label='Validation diff class', alpha=0.4, color='orange')
    plt.ylabel('Similarity score')
    plt.xlabel('Epochs')
    plt.title('Final similarity_score')
    plt.legend()
    fig.savefig(figure_path + '_final_similarity')
    plt.close(fig)
    return


def create_gif_image(trn_ml_out, trn_y_true, tst_ml_out, tst_y_true, trn_y_raw, tst_y_raw, epoch, gif_path):
    fig_sim, axs_sim = plt.subplots(2, 1, figsize=(15, 10))
    fig_final, axs_final = plt.subplots(2, 1, figsize=(15, 10))
    title = f'Intermediate similarity score\n Epoch = {epoch + 1}'
    fig_ml_name = f'{gif_path}/sim_{epoch}.png'
    # Make gif of similarity function score

    ml_trn_same_idx = (trn_y_true == 1).nonzero()[0].squeeze()
    ml_trn_diff_idx = (trn_y_true == 0).nonzero()[0].squeeze()
    ml_tst_same_idx = (tst_y_true == 1).nonzero()[0].squeeze()
    ml_tst_diff_idx = (tst_y_true == 0).nonzero()[0].squeeze()


    plot_prob_density(fig_sim, axs_sim, trn_ml_out, ml_trn_same_idx, ml_trn_diff_idx, tst_ml_out, ml_tst_same_idx,
                      ml_tst_diff_idx, title, fig_ml_name)
    plt.close(fig_sim)

    title = f'Final similarity score\n Epoch = {epoch + 1}'
    fig_al_name = f'{gif_path}/final_{epoch}.png'

    al_trn_same_idx = (trn_y_true == 1).nonzero()[0].squeeze()
    al_trn_diff_idx = (trn_y_true == 0).nonzero()[0].squeeze()
    al_tst_same_idx = (tst_y_true == 1).nonzero()[0].squeeze()
    al_tst_diff_idx = (tst_y_true == 0).nonzero()[0].squeeze()


    # Make gif of similarity function score
    plot_prob_density(fig_final, axs_final, trn_y_raw, al_trn_same_idx, al_trn_diff_idx, tst_y_raw, al_tst_same_idx,
                      al_tst_diff_idx, title, fig_al_name)
    plt.close(fig_final)

    return fig_ml_name, fig_al_name


def save_gif_file(fig_list, gif_path):
    images_sim = []
    for filename in fig_list:
        images_sim.append(imageio.imread(filename))
        os.remove(filename)

    imageio.mimsave(gif_path, images_sim, fps=2, loop=1)


def plot_prob_density(fig, axs, trn_score, trn_same_idx, trn_diff_idx, tst_score, tst_same_idx, tst_diff_idx, title,
                      figure_path=None):
    if len(axs) != 2:
        print('Axes not correct, no prob density plotted')
        return

    x_label = 'Output score'
    legend_label = 'Dataset'

    trn_score_same = trn_score[trn_same_idx].reshape(-1)
    trn_score_diff = trn_score[trn_diff_idx].reshape(-1)
    trn_label = np.full((len(trn_score_same)), 'train')
    trn_label2 = np.full((len(trn_score_diff)), 'train')

    tst_score_same = tst_score[tst_same_idx].reshape(-1)
    tst_score_diff = tst_score[tst_diff_idx].reshape(-1)

    tst_label = np.full((len(tst_score_same)), 'test')
    tst_label2 = np.full((len(tst_score_diff)), 'test')

    same_score_dict = {x_label: np.concatenate([trn_score_same, tst_score_same], axis=0),
                       legend_label: np.concatenate([trn_label, tst_label], axis=0)}

    same_scores = pd.DataFrame.from_dict(same_score_dict)
    same_scores[x_label] = same_scores[x_label].astype(float)
    same_scores[legend_label] = same_scores[legend_label].astype(str)

    diff_score_dict = {x_label: np.concatenate([trn_score_diff, tst_score_diff], axis=0),
                       legend_label: np.concatenate([trn_label2, tst_label2], axis=0)}

    diff_scores = pd.DataFrame.from_dict(diff_score_dict)
    diff_scores[x_label] = diff_scores[x_label].astype(float)
    diff_scores[legend_label] = diff_scores[legend_label].astype(str)

    # giving title to the plot
    fig.suptitle(title, fontsize=24)

    axs[0].set_title('Same class', fontweight="bold", size=18)
    axs[1].set_title('Different class', fontweight="bold", size=18)
    axs[0].set_ylabel('Density')
    axs[1].set_ylabel('Density')

    sns.histplot(ax=axs[0], data=same_scores, x=x_label, hue=legend_label, stat='probability', kde=False,
                 common_norm=False,
                 element='bars', binrange=(0, 1), binwidth=0.005)

    sns.histplot(ax=axs[1], data=diff_scores, x=x_label, hue=legend_label, stat='probability', kde=False,
                 common_norm=False,
                 element='bars', binrange=(0, 1), binwidth=0.005)

    if figure_path is not None:
        fig.savefig(figure_path)

    plt.close(fig)


def plot_final_weighted_F1(results, figure_labels, title, figure_path):
    fig = plt.figure()

   
    ii = 0
    for key in results.keys():
        y = results[key]['weighted_f1']
        x = results[key][results[key]['known_unknown']]
        plt.plot(x, y, f'-o', label=f'{figure_labels[key]}')
        ii = ii + 1

    plt.ylabel('F1-score')
    plt.xlabel(results[key]['known_unknown'])
    plt.title('Weighted F1-score')
    plt.legend()
    # plt.show()

    fig.savefig(f'{figure_path}_w_f1')

def plot_final_macro_F1(results, figure_labels, title, figure_path):
    fig = plt.figure()

    ii = 0
    for key in results.keys():
        y = results[key]['macro_f1']
        x = results[key][results[key]['known_unknown']]
        plt.plot(x, y, f'-o', label=f'{figure_labels[key]}')
        ii = ii + 1

    plt.ylabel('F1-score')
    plt.xlabel(results[key]['known_unknown'])
    plt.title('Macro F1-score')
    plt.legend()
    # plt.show()

    fig.savefig(f'{figure_path}_m_f1')

def plot_final_accuracy(results, figure_labels, title, figure_path):
    fig = plt.figure()

    ii = 0
    for key in results.keys():
        y = results[key]['accuracy']
        x = results[key][results[key]['known_unknown']]
        plt.plot(x, y, f'-o', label=f'{figure_labels[key]}')
        ii = ii + 1

    plt.ylabel('Accuracy')
    plt.xlabel(results[key]['known_unknown'])
    plt.title('Accuracy')
    plt.legend()
    # plt.show()

    fig.savefig(f'{figure_path}_acc')


    return

def plot_final_open_world_error(results, figure_labels, title, figure_path):
    fig = plt.figure()

    ii = 0
    for key in results.keys():
        y = results[key]['open_world_error']
        x = results[key][results[key]['known_unknown']]
        plt.plot(x, y, f'-o', label=f'{figure_labels[key]}')
        ii = ii + 1

    plt.ylabel('Open World Error')
    plt.xlabel(results[key]['known_unknown'])
    plt.title('Open world error')
    plt.legend()
    # plt.show()

    fig.savefig(f'{figure_path}_e_ow')
    return

def plot_final_known_error(results, figure_labels, title, figure_path):
    fig = plt.figure()

    ii = 0
    for key in results.keys():
        y = results[key]['error_knowns']
        x = results[key][results[key]['known_unknown']]
        plt.plot(x, y, f'-o', label=f'{figure_labels[key]}')
        ii = ii + 1

    plt.ylabel('E_k')
    plt.xlabel(results[key]['known_unknown'])
    plt.title('Prediction error known classes E_K')
    plt.legend()
    # plt.show()

    fig.savefig(f'{figure_path}_p_known')
    return

def plot_final_unknown_error(results, figure_labels, title, figure_path):
    fig = plt.figure()

    ii = 0
    for key in results.keys():
        y = results[key]['error_unknowns']
        x = results[key][results[key]['known_unknown']]
        plt.plot(x, y, f'-o', label=f'{figure_labels[key]}')
        ii = ii + 1

    plt.ylabel('E_U')
    plt.xlabel(results[key]['known_unknown'])
    plt.title('Prediction error unknown classes E_U')
    plt.legend()
    # plt.show()

    fig.savefig(f'{figure_path}_p_unknown')
    return


def plot_confusion_matrix(results, figure_labels, title, figure_path):

    for exp in results.keys():

        if not os.path.exists(f"{figure_path}_{exp}"):
            os.mkdir(f"{figure_path}_{exp}")

        for ii in range(0,len(results[exp]['unknown_classes'])):
            fig = plt.figure()



            cf_matrix = results[exp]['confusion_matrix'][ii]
            unknowns = results[exp][results[exp]['known_unknown']][ii]
            sns.heatmap(np.multiply(cf_matrix, 1.0/(cf_matrix.sum(axis=1).reshape(-1, 1))),annot=False, cmap='Blues')
            plt.title(f"{unknowns} {results[exp]['known_unknown']} objects")
            fig.savefig(f"{figure_path}_{exp}/{unknowns}")
            # plt.show()
            plt.close(fig)

    return

def plot_final_wilderness_impact(results, figure_labels, title, figure_path):
    fig = plt.figure()

    ii = 0
    for key in results.keys():
        y = results[key]['wilderness_impact']
        x = results[key]['wilderness_ratio']
        plt.plot(x, y, f'-o', label=f'{figure_labels[key]}')
        ii = ii + 1

    plt.ylabel('Wilderness Impact')
    plt.xlabel('Wilderness Ratio')
    plt.title('Wilderness Impact')
    plt.legend()
    # plt.show()

    fig.savefig(f'{figure_path}_WI')
    return


def plot_final_score_distribution(results, title,
                      figure_path=None):
    x_label = 'Output score'
    legend_label = 'Dataset'

    for exp in results.keys():


        fig, axs = plt.subplots(len(results[exp]['true_labels']), 1, figsize=(15, 10))

        if results[exp]['known_unknown'] == "unknown_classes":
            label = "Known classes"
            title = "unknown classes"
        else:
            label = "Unknown classes"
            title = "known classes"


        for ii in range(0,len(results[exp]['true_labels'])):
            true_labels = results[exp]['true_labels'][ii]
            final_labels = results[exp]['final_labels'][ii]
            final_score = results[exp]['final_score'][ii]
            unknown_label = results[exp]['unknown_label'][ii]
            unknown_scores = final_score[np.where(true_labels == unknown_label)]

            known_scores = final_score[np.where(true_labels != unknown_label)]

            score_dict = {x_label: np.concatenate([known_scores, unknown_scores], axis=0),
                               legend_label: np.concatenate([np.full((len(known_scores)), 'Known classes'), np.full((len(unknown_scores)), 'Unknown classes')], axis=0)}
            # score_dict = {x_label: np.concatenate([unknown_scores], axis=0),
            #               legend_label: np.concatenate([np.full((len(unknown_scores)), 'Unknown classes')], axis=0)}

            scores = pd.DataFrame.from_dict(score_dict)
            axs[ii].set_title(f"{results[exp][results[exp]['known_unknown']][ii]} {title}", fontweight="bold", size=18)
            sns.histplot(ax=axs[ii], data=scores, x=x_label, hue=legend_label, stat='probability', kde=False,
                 common_norm=True,
                 element='bars', binrange=(0, 1), binwidth=0.005)

        if figure_path is not None:
            fig.savefig(f"{figure_path}_{exp}_score_distribution")

    plt.close(fig)



def plot_best_loss(loss, loop_variable, figure_path):
    fig = plt.figure()
    var_name = list(loop_variable.keys())[0]
    plt.plot(loop_variable[var_name], loss, '-o', label='loss')
    plt.ylabel('loss')
    plt.xlabel(f'{var_name}')
    plt.title('loss score')
    plt.xticks(rotation=45)

    plt.legend()
    fig.savefig(figure_path + f'_{var_name}', bbox_inches='tight')
    plt.close(fig)

    return


def plot_best_accuracy(accuracy, loop_variable, figure_path):
    fig = plt.figure()
    var_name = list(loop_variable.keys())[0]
    plt.plot(loop_variable[var_name], accuracy, '-o', label='accuracy')
    plt.ylabel('accuracy')
    plt.xlabel(f'{var_name}')
    plt.title('Accuracy')
    plt.xticks(rotation=45)

    plt.legend()
    fig.savefig(figure_path + f'_{var_name}', bbox_inches='tight')
    plt.close(fig)

    return


def plot_feature_vector(vector, title, figure_path, y_max):
    fig = plt.figure()
    y = vector.view(-1).numpy()
    x = np.arange(0, vector.shape[0])

    data = pd.DataFrame({'indices': x, 'values': y})

    ax = sns.barplot(x='indices', y='values', data=data)
    ax.set_title(title)
    ax.set_ylim(0, y_max)

    # plt.show()
    fig.savefig(figure_path, bbox_inches='tight')
    plt.close(fig)

    pass


def plot_final_classification(label_list, images, final_label):



    im_grid = make_grid(images[0:-1], nrow=images.shape[0]-1)
    im_grid_images = images.shape[0]-1
    im_grid2 = make_grid(images[-1], nrow=1)
    if final_label < 0 :

        final_label_class: str = "Unknown"
    else:
    #     im_grid = make_grid(images[0:-1], nrow=images.shape[0]-2)
    #     im_grid2 = make_grid(images[-2:], nrow=1)
    #     im_grid_images = images.shape[0]-2
        final_label_class = label_list[-1]


    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    axs[1].imshow(np.transpose(im_grid.numpy(), (1, 2, 0)))
    axs[0].imshow(np.transpose(im_grid2.numpy(), (1, 2, 0)))

    axs[0].set_title(f'input image = {final_label_class} ')
    half_image_size = int(im_grid.shape[2]/im_grid_images/2)
    axs[1].set_xticks(np.linspace(half_image_size, im_grid.shape[2] - half_image_size, im_grid_images))
    label_list_2 = [" "]
    label_list_2.extend(label_list[0:im_grid_images])
    label_list_2.append(" ")
    axs[1].set_xticklabels(label_list[0:im_grid_images])
    plt.show()


def plot_classes_surface(results,metric, figure_labels, figure_path):


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    for exp in results.keys():
        n = np.unique(np.array(results[exp]['memory_classes'])).shape[0]
        known_classes = np.array(results[exp]['memory_classes']).reshape(-1, n)
        unknown_classes = np.array(results[exp]['unknown_classes']).reshape(-1, n)
        F1 = np.array(results[exp][f'{metric}']).reshape(-1, n)

        light = matplotlib.colors.LightSource(90,45)
        ill_surf = light.shade(F1,cmap=matplotlib.cm.coolwarm)
        surf = ax.plot_surface(known_classes, unknown_classes, F1, shade=True, antialiased=True, alpha=0.6,
                               label=f'{figure_labels[exp]}')
        surf._facecolors2d=surf._facecolor3d
        surf._edgecolors2d=surf._edgecolor3d

        wf = ax.plot_wireframe(known_classes, unknown_classes, F1, color='white', linewidth=0.4)
    ax.legend()
    ax.set(
        xlabel='# of known classes in memory',
        ylabel='# of unknown classes',
        zlabel=f'{metric}',
        xlim=[known_classes.min(), known_classes.max()],
        xticks=np.unique(known_classes),
        ylim = [unknown_classes.max(), unknown_classes.min()],
        yticks = np.unique(unknown_classes),
    )

    fig.savefig(f"{figure_path}_surface_plot_{metric}", bbox_inches='tight')
    plt.close(fig)

    return



def main():


    knowns = np.array([20,  40, 60 ,80])
    unknowns = np.array([0, 20,  40, 60 ,80])
    X, Y = np.meshgrid(knowns, unknowns)

    F1 = np.random.rand(X.shape[0], X.shape[1])

    plot_classes_surface(X, Y, F1)

    return


if __name__ == '__main__':
    main()
