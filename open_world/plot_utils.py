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
    fig.savefig(figure_path + 'losses')
    return