import torch.nn as nn
import torch





class custom_bce_loss(nn.Module):

    def __init__(self, weight=None):
        super(custom_bce_loss).__init__()

        self.base_loss = nn.BCEWithLogitsLoss()


        def forward(self, input, target):

            input = nn.Sigmoid(input)

            input = input.view(-1)

            target = target.view(-1)

            loss = -[target * torch.log(input) + (1 - target) * torch.log(1 - input)]

            test_loss = self.base_loss(target, input)

            return loss


