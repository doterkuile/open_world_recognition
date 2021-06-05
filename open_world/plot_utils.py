import numpy as np
import matplotlib.pyplot as plt

def plot_losses(train_losses, test_losses, figure_path):


    fig = plt.figure()
    plt.plot(train_losses, label='training loss')
    plt.plot(test_losses, label='validation loss')
    plt.ylabel('Loss function')
    plt.xlabel('Epochs')
    plt.title('Losses ')
    plt.legend()
    fig.savefig(figure_path + '_losses')
    return


def plot_accuracy(train_acc, test_acc, figure_path):


    fig = plt.figure()
    plt.plot(train_acc, label='training accuracy')
    plt.plot(test_acc, label='validation accuracy')
    plt.ylabel('accuracy [%]')
    plt.xlabel('Epochs')
    plt.title('Accuracy ')
    plt.legend()
    fig.savefig(figure_path + '_accuracy')
    return