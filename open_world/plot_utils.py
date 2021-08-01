import numpy as np
import matplotlib.pyplot as plt
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
    # Make gif of similarity function score

    ml_trn_same_idx = (y_trn == 0).nonzero()[0].squeeze()
    ml_trn_diff_idx = (y_trn == 1).nonzero()[0].squeeze()


    plot_prob_density(fig_sim, axs_sim, trn_ml_out, trn_y_true, tst_ml_out, tst_y_true,
                                 title)
    fig_sim.savefig(f'{gif_path}/sim_{epoch}.png')
    plt.close(fig_sim)
    fig_ml_name = f'{gif_path}/sim_{epoch}.png'

    title = f'Final similarity score\n Epoch = {epoch + 1}'
    # Make gif of similarity function score
    plot_prob_density(fig_final, axs_final, trn_y_raw, trn_y_true, tst_y_raw, tst_y_true,
                                 title)
    fig_final.savefig(f'{gif_path}/final_{epoch}.png')
    fig_al_name = f'{gif_path}/final_{epoch}.png'
    plt.close(fig_final)

    return fig_ml_name, fig_al_name


def save_gif_file(fig_list, gif_path):
    images_sim = []
    for filename in fig_list:
        images_sim.append(imageio.imread(filename))
        os.remove(filename)

    imageio.mimsave(gif_path, images_sim, fps=2, loop=1)





def plot_prob_density(fig, axs, trn_score, trn_same_idx, trn_diff_idx, tst_score, tst_same_idx, tst_diff_idx, title, figure_path=None):
    if len(axs) != 2:
        print('Axes not correct, no prob density plotted')
        return

    x_label = 'Output score'
    legend_label = 'Dataset'

    # make sure to select only x1 samples of a different class
    trn_idx_diff_class = (y_trn == 0).nonzero()[0].squeeze()
    trn_idx_same_class = (y_trn == 1).nonzero()[0].squeeze()

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

    diff_score_dict = {x_label:  np.concatenate([trn_score_diff, tst_score_diff], axis=0),
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


    sns.histplot(ax=axs[0], data=same_scores, x=x_label, hue=legend_label, stat='probability', kde=False, common_norm=False,
                 element='bars', binrange=(0, 1), binwidth=0.005)

    sns.histplot(ax=axs[1], data=diff_scores, x=x_label, hue=legend_label, stat='probability', kde=False, common_norm=False,
                 element='bars', binrange=(0, 1), binwidth=0.005)

    if figure_path is not None:
        fig.savefig(figure_path)

    plt.close(fig)



def plot_best_F1(F1, loop_variable, figure_path):
    fig = plt.figure()
    var_name = list(loop_variable.keys())[0]
    plt.plot(loop_variable[var_name], F1, '-o', label='F1 score')
    plt.ylabel('F1')
    plt.xlabel(f'{var_name}')
    plt.title('F1 score')
    plt.xticks(rotation=45)
    plt.legend()
    fig.savefig(figure_path + f'_{var_name}', bbox_inches='tight')
    plt.close(fig)

    return


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

def plot_feature_vector(vector, title,figure_path, y_max):
    fig = plt.figure()
    y = vector.view(-1).numpy()
    x = np.arange(0,vector.shape[0])

    data = pd.DataFrame({'indices': x,'values': y})

    ax = sns.barplot(x='indices', y='values', data=data)
    ax.set_title(title)
    ax.set_ylim(0, y_max)

    # plt.show()
    fig.savefig(figure_path, bbox_inches='tight')
    plt.close(fig)

    pass


def main():

    # multiple_gpu = True if torch.cuda.device_count() > 1 else False
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # Parse config file
    # (train_dataset, model, criterion, optimizer, epochs, batch_size, learning_rate, config) = OpenWorldUtils.parseConfigFile(
    #     device, multiple_gpu)
    #
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)#, num_workers=4)
    #
    # dataset_name = 'cifar'
    # figure_path = f'figures/{dataset_name}'
    # for b, ((X0_train, X1_train), y_train, [X0_labels, X1_labels]) in enumerate(train_loader):
    #
    #     optimizer.zero_grad()
    #
    #     # X0_train = X0_train.to(device)
    #     # X1_train = X1_train.to(device)
    #     # y_train = y_train.view(-1, 1).to(device)
    #
    #     # Limit the number of batches
    #     if b == 0:
    #         break
    #
    #
    #
    # versions = ['same', 'diff']
    #
    # for version in versions:
    #
    #     if version == 'same':
    #         idx = (y_train == 1).nonzero()[0]
    #     if version == 'diff':
    #         idx = (y_train == 0).nonzero()[0]
    #
    #     x0 = X0_train[idx].view(-1)
    #     x1 = X1_train[idx][0][0].view(-1)
    #
    #     x_abssub = x0.sub((x1))
    #     x_abssub.abs_()
    #     x_add = x0.add(x1)
    #     similarity_vector = torch.cat((x_abssub, x_add))
    #     plot_feature_vector(x0, f'x_0_{version}', f'{figure_path}_x0_{version}', 4)
    #     plot_feature_vector(x1, f'x_1_{version}', f'{figure_path}_x1_{version}', 4)
    #     plot_feature_vector(x_abssub, f'x_abssub_{version}', f'{figure_path}_x_abssub_{version}', 4)
    #     plot_feature_vector(x_add, f'x_add_{version}', f'{figure_path}_x_add_{version}', 10)
    #     plot_feature_vector(similarity_vector, f'similarity_vector_{version}', f'{figure_path}_similarity_vector_{version}',10)

    x = torch.tensor(np.arange(0, 1, 0.01)).view(-1,1)
    # y = torch.nn.functional.leaky_relu(x-0.5)*-torch.log(1-x)
    targets = [0, 1]
    fig = plt.figure()
    f1 = lambda x: - 1 * x.log()
    # f2 = lambda x:(target * nn.functional.relu(0.55 - x) + (1 - target) * nn.functional.relu(x - 0.45))
    f2 = lambda x: 20 * nn.functional.relu(0.5 - x)
    losses= []
    for target in targets:

        logloss = target * f1(x) + (1-target) * f1(1-x)
        y = target * (f2(x) + f1(x)) + (1-target) * (f2(1-x) + f1(1 - x))
        losses.append(y)
        plt.plot(x.numpy(), y.numpy(), 'r', label=f"log*relu_{target}")
        plt.plot(x.numpy(), logloss.numpy(), 'b', label=f"log_{target}")
        # plt.plot(x.numpy(), f2(x), 'g', label="relu")

    plt.legend()

    plt.show()


    return


if __name__ == '__main__':
    main()
