import random

import torch

from mnist.config import parse_args
from mnist.dataset import MNISTDataset
from mnist.model import MNISTClassifier


class MNISTTrainer:
    def __init__(self):
        self.config = parse_args()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.set_seed(self.config.seed)

        self.dataset = MNISTDataset(self.config)

        self.model = MNISTClassifier(self.config)
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def fit(self):
        train_data_loader = self.dataset.get_data_loader(mode='train')

        for epoch in range(self.config.epochs):
            total_loss = 0
            num_data = 0

            for img, label in train_data_loader:
                self.optimizer.zero_grad()
                logits, loss = self.model(img, label)
                loss.backward()
                self.optimizer.step()

                total_loss += loss
                num_data += len(img)

            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.9f}'.format(total_loss/num_data))
        print('Learning finished')

    def test(self):
        test_data_loader = self.dataset.get_data_loader(mode='test')

        total_loss = 0
        num_data = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                logits, loss = self.model(img, label)

                total_loss += loss
                num_data += len(img)
        print('loss =', '{:.9f}'.format(total_loss/num_data))

    def set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed_all(seed)
