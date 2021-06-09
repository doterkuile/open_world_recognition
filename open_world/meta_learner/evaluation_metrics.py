from sklearn import metrics
import numpy as np
import torch
import time
import open_world.plot_utils as plot_utils



def precision(y_pred, y_true):

    ## precision = tp/(tp + fp)
    precision = metrics.precision_score(y_pred, y_true)
    return precision



def recall(y_pred, y_true):

    ## recall = tp/(tp + fn)
    recall= metrics.recall_score(y_pred, y_true)
    return recall

def accuracy():
    ## Accuracy = (tp + tn)/ (tp + fp + tn + fn)
    pass

def F1_score():

    ## F1 = 2 * (precision * recall)/ (precision + recall)
    pass

def wilderness_impact():
    pass

def wilderness_ratio():
    pass


def main():
    y_pred = [1, 1 , 1, 1, 1 , 1]
    y_true = [1, 0, 1, 1, 1, 1]

    precision(y_pred, y_true)
    print(tp)

    pass



if __name__ == "__main__":
    main()