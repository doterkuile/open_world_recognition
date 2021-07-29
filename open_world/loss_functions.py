import torch.nn as nn
import torch





class bce_loss_default(nn.Module):

    def __init__(self, weight=None):
        super(bce_loss_default).__init__()


    def forward(self, input, target):

        loss = nn.BCEWithLogitsLoss(target, input)

        return loss


class bce_loss_custom(nn.Module):

    def __init__(self, weight=None):
        super(bce_loss_default).__init__()


    def forward(self, input, target):

        x = input.sigmoid()

        f1 = lambda x: - 2 * nn.functional.log(x)
        f2 = lambda x: 1 + 2 * nn.functional.leaky_relu(0.5 - x)

        loss = y * f2(x) * f1(x) + (1-y) * f2(1-x) * f1(1 - x)

        return loss


def main():

    crit1 = bce_loss_custom()
    crit2 = bce_loss_default()
    crit3 = torch.nn.BCEWithLogitsLoss()

    sample_size = 10

    y_true = torch.randint(2, (1,sample_size))
    y_pred = torch.rand((1, sample_size))

    loss1 = crit1(y_pred, y_true)
    loss2 = crit2(y_pred, y_true)
    loss3 = crit3(y_pred, y_true)



    return

if __name__ == '__main__':
    main()