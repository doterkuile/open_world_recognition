import torch.nn as nn
import torch
import numpy as np





class bce_loss_default(nn.Module):

    def __init__(self, weight=None):
        super(bce_loss_default, self).__init__()
        self.positive_weight = weight
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight, reduction='mean')


    def forward(self, input, target):
        
        epsilon = 10 ** -7
        x = input.sigmoid().clamp(epsilon, 1 - epsilon)
        f1 = lambda x: - x.log()

        loss = self.positive_weight * target * f1(x) + (1-target) * f1(1 - x)
        # loss = self.criterion(input, target)

        idx0 = len((target == 0).nonzero())
        idx1 = len((target == 1).nonzero())

        lossmean = loss.sum() / (idx1 * int(self.positive_weight) + idx0)

        return lossmean


class bce_loss_matching_layer(nn.Module):

    def __init__(self, weight=None):
        super(bce_loss_matching_layer, self).__init__()
        self.positive_weight = weight
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight, reduction='mean')

    def forward(self, input, target):
        epsilon = 10 ** -7
        x = input.sigmoid().clamp(epsilon, 1 - epsilon)
        f1 = lambda x: - x.log()

        loss = target * f1(x) + self.positive_weight * (1 - target) * f1(1 - x)
        # loss = self.criterion(input, target)

        idx0 = len((target == 0).nonzero())
        idx1 = len((target == 1).nonzero())

        lossmean = loss.sum() / (idx1 + idx0 * int(self.positive_weight))

        return lossmean


class bce_loss_custom(nn.Module):

    def __init__(self, weight=None):
        super(bce_loss_custom, self).__init__()
        self.positive_weight = weight


    def forward(self, input, target):
        epsilon = 10 ** -7
        x = input.sigmoid().clamp(epsilon, 1 - epsilon)

        f1 = lambda x: - 1 * x.log()
        # f2 = lambda x:(target * nn.functional.relu(0.55 - x) + (1 - target) * nn.functional.relu(x - 0.45))
        f2 = lambda x: 10 * nn.functional.relu(0.55 - x)

        loss = self.positive_weight * target * (f2(x) + f1(x)) +  (1-target) * (f2(1-x) + f1(1 - x))

        idx0 = len((target == 0).nonzero())
        idx1 = len((target == 1).nonzero())

        lossmean = loss.sum()/(idx1 * int(self.positive_weight) + idx0)

        return lossmean

class matching_layer_loss(nn.Module):

    def __init__(self, weight=None):
        super(matching_layer_loss, self).__init__()
        self.weight = weight


    def forward(self, input, target):
        epsilon = 10 ** -7
        x = input.sigmoid().clamp(epsilon, 1 - epsilon)

        f1 = lambda x: - 1 * x.log()
        # f2 = lambda x:(target * nn.functional.relu(0.55 - x) + (1 - target) * nn.functional.relu(x - 0.45))
        f2 = lambda x: 10 * nn.functional.relu(0.55 - x)

        loss = target * (f2(x) + f1(x)) + self.weight * (1-target) * (f2(1-x) + f1(1 - x))

        idx0 = len((target == 0).nonzero())
        idx1 = len((target == 1).nonzero())

        lossmean = loss.sum()/(idx1 + idx0* int(self.weight) )

        return lossmean



def main():

    w=torch.tensor(9,dtype=torch.float)
    crit1 = bce_loss_custom(w)
    crit2 = bce_loss_default(w)
    crit3 = torch.nn.BCEWithLogitsLoss(pos_weight=w)

    sample_size = 400000
    trn_y_true = torch.zeros(int(0.9 * sample_size))
    trn_y_true = torch.cat([trn_y_true, torch.ones(int(0.1 * sample_size))])
    trn_y_pred = 0.3 * torch.ones(sample_size)
    shuffle_idx = np.random.permutation(sample_size)
    trn_y_true = trn_y_true[shuffle_idx]
    trn_y_pred = trn_y_pred[shuffle_idx]

    tst_y_true = torch.zeros(int(0.5 * sample_size))
    tst_y_true = torch.cat([tst_y_true, torch.ones(int(0.5 * sample_size))])
    tst_y_pred = 0.3 * torch.ones(sample_size)
    shuffle_idx = np.random.permutation(sample_size)
    tst_y_true = tst_y_true[shuffle_idx]
    tst_y_pred = tst_y_pred[shuffle_idx]

    tst_w=torch.tensor(1,dtype=torch.float)
    tst_crit = bce_loss_custom(tst_w)
    # tst_crit = torch.nn.BCEWithLogitsLoss(pos_weight=tst_w)
    trn_loss = crit1(trn_y_pred, trn_y_true)
    tst_loss = tst_crit(tst_y_pred, tst_y_true)

    f1 = lambda x: - 1 * x.log()

    x = torch.ones(5) * 0.3
    x1 = torch.ones(1) * 0.3
    x0 = torch.ones(9) * 0.3

    loss1 = torch.cat([f1(x),f1(1-x)])
    loss2 = torch.cat([9 * f1(x1), f1(1-x0)])





    y_true = torch.randint(2, (1,sample_size),dtype=torch.float)
    y_pred = 10000*torch.rand((1, sample_size))
    mn = []

    mn = torch.cat(mn)


    print(mn.mean())
    print(y_pred.mean())


    loss1 = crit1(y_pred, y_true)
    loss2 = crit2(y_pred.view(-1), y_true.view(-1))
    loss3 = crit3(y_pred, y_true)



    return

if __name__ == '__main__':
    main()