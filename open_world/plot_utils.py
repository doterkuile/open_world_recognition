import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from open_world import OpenWorldUtils
import torch
from torch.utils.data import DataLoader


def plot_losses(train_losses, test_losses, figure_path):
    fig = plt.figure()
    plt.plot(train_losses, label='training loss')
    plt.plot(test_losses, label='validation loss')
    plt.ylabel('Loss function')
    plt.xlabel('Epochs')
    plt.title('Losses')
    plt.legend()
    fig.savefig(figure_path + '_losses')
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
    return


def plot_prob_density(fig, axs, trn_score, y_trn, tst_score, y_tst, title, figure_path=None):
    if len(axs) != 2:
        print('Axes not correct, no prob density plotted')
        return

    x_label = 'Output score'
    legend_label = 'Dataset'

    # make sure to select only x1 samples of a different class
    trn_idx_diff_class = (y_trn == 0).nonzero()[0].squeeze()
    trn_idx_same_class = (y_trn == 1).nonzero()[0].squeeze()

    trn_score_same = trn_score[trn_idx_same_class].reshape(-1)
    trn_score_diff = trn_score[trn_idx_diff_class].reshape(-1)
    trn_label = np.full((len(trn_score_same)), 'train')
    trn_label2 = np.full((len(trn_score_diff)), 'train')

    tst_idx_diff_class = (y_tst == 0).nonzero()[0].squeeze()
    tst_idx_same_class = (y_tst == 1).nonzero()[0].squeeze()

    tst_score_same = tst_score[tst_idx_same_class].reshape(-1)
    tst_score_diff = tst_score[tst_idx_diff_class].reshape(-1)

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
    return

def plot_feature_vector(vector, title,figure_path):
    fig = plt.figure()
    y = vector.view(-1).numpy()
    x = np.arange(0,vector.shape[0])

    data = pd.DataFrame({'indices': x,'values': y})

    ax = sns.barplot(x='indices', y='values', data=data)
    ax.set_title(title)

    # plt.show()
    fig.savefig(figure_path, bbox_inches='tight')

    pass


def main():

    multiple_gpu = True if torch.cuda.device_count() > 1 else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Parse config file
    (train_dataset, model, criterion, optimizer, epochs, batch_size, learning_rate, config) = OpenWorldUtils.parseConfigFile(
        device, multiple_gpu)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)#, num_workers=4)

    dataset_name = 'amazon'
    figure_path = f'figures/{dataset_name}'
    for b, ((X0_train, X1_train), y_train, [X0_labels, X1_labels]) in enumerate(train_loader):

        optimizer.zero_grad()

        # X0_train = X0_train.to(device)
        # X1_train = X1_train.to(device)
        # y_train = y_train.view(-1, 1).to(device)

        # Limit the number of batches
        if b == 0:
            break



    versions = ['same', 'diff']

    for version in versions:

        if version == 'same':
            idx = (y_train == 1).nonzero()[0]
        if version == 'diff':
            idx = (y_train == 0).nonzero()[0]

        x0 = X0_train[idx].view(-1)
        x1 = X1_train[idx][0][0].view(-1)

        x_abssub = x0.sub((x1))
        x_abssub.abs_()
        x_add = x0.add(x1)
        similarity_vector = torch.cat((x_abssub, x_add))
        plot_feature_vector(x0, f'x_0_{version}', f'{figure_path}_x0_{version}')
        plot_feature_vector(x1, f'x_1_{version}', f'{figure_path}_x1_{version}')
        plot_feature_vector(x_abssub, f'x_abssub_{version}', f'{figure_path}_x_abssub_{version}')
        plot_feature_vector(x_add, f'x_add_{version}', f'{figure_path}_x_add_{version}')
        plot_feature_vector(similarity_vector, f'similarity_vector_{version}', f'{figure_path}_similarity_vector_{version}')




    return


if __name__ == '__main__':
    main()
