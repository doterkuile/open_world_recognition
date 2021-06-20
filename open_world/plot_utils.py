import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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


def plot_prob_density(trn_sim_score, y_trn, tst_sim_score, y_tst, title, figure_path):


    # make sure to select only x1 samples of a different class
    trn_idx_diff_class = (y_trn == 0).nonzero()[0].squeeze()
    trn_idx_same_class = (y_trn == 1).nonzero()[0].squeeze()

    if len(trn_sim_score.shape) == 2:
        trn_sim_score_same = trn_sim_score[trn_idx_same_class,:].reshape(-1)
        trn_sim_score_diff = trn_sim_score[trn_idx_diff_class,:].reshape(-1)
    else:
        trn_sim_score_same = trn_sim_score[trn_idx_same_class]
        trn_sim_score_diff = trn_sim_score[trn_idx_diff_class]
    trn_label = np.full((len(trn_sim_score_same)), 'trn_same')
    trn_label2 = np.full((len(trn_sim_score_diff)), 'trn_diff')

    tst_idx_diff_class = (y_tst == 0).nonzero()[0].squeeze()
    tst_idx_same_class = (y_tst == 1).nonzero()[0].squeeze()

    if len(trn_sim_score.shape) == 2:
        tst_sim_score_same = tst_sim_score[tst_idx_same_class,:].reshape(-1)
        tst_sim_score_diff = tst_sim_score[tst_idx_diff_class,:].reshape(-1)
    else:
        tst_sim_score_same = tst_sim_score[tst_idx_same_class]
        tst_sim_score_diff = tst_sim_score[tst_idx_diff_class]

    tst_label = np.full((len(tst_sim_score_same)), 'tst_same')
    tst_label2 = np.full((len(tst_sim_score_diff)), 'tst_diff')

    scores_combined = np.concatenate([trn_sim_score_same, trn_sim_score_diff, tst_sim_score_same, tst_sim_score_diff], axis=0)
    labels_combined = np.concatenate([trn_label, trn_label2, tst_label, tst_label2], axis=0)
    # scores = pd.DataFrame(scores_combined, columns=['score'])
    score_dict = {'score': scores_combined,
                   'class': labels_combined}

    scores = pd.DataFrame.from_dict(score_dict)
    scores['score'] = scores['score'].astype(float)
    scores['class'] = scores['class'].astype(str)


    fig = plt.figure()
    sns.kdeplot(
        data=scores, x="score", hue="class",
        fill=True, common_norm=False, palette="muted",
        alpha=0.8, linewidth=0, multiple='layer'
    )

    # giving title to the plot
    plt.title(title)

    fig.savefig(figure_path)
