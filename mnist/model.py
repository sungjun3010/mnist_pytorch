from argparse import Namespace

import torch.nn as nn


class MNISTClassifier(nn.Module):
    def __init__(self, config: Namespace):
        super(MNISTClassifier, self).__init__()
        self.config = config

        self.linear = nn.Linear(784, 10, bias=True)

        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, img, label=None):
        img = img.view(-1, 28 * 28)

        logits = self.linear(img)

        loss = None
        if label is not None:
            loss = self.criterion(logits, label)

        return logits, loss
