from argparse import Namespace

import torchvision.datasets as ds
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class MNISTDataset:
    def __init__(self, config: Namespace):
        self.batch_size = config.batch_size
        self.train = ds.MNIST(
            root='MNIST_data/',
            train=True,
            transform=transforms.ToTensor(),
            download=True)

        self.test = ds.MNIST(
            root='MNIST_data/',
            train=False,
            transform=transforms.ToTensor(),
            download=True)

    def get_data_loader(self, mode='train'):
        if mode == 'train':
            return DataLoader(
                dataset=self.train,
                batch_size=self.batch_size,  # 배치 크기는 100
                shuffle=True,
                drop_last=True)

        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,  # 배치 크기는 100
            shuffle=False,
            drop_last=False)
