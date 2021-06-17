import numpy as np
import matplotlib.pyplot as plt

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
