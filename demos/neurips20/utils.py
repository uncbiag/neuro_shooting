"""Utility functions for loading the rotated MNIST data."""

import torch
from torch.utils import data
from scipy.io import loadmat


class Dataset(data.Dataset):
    def __init__(self, Xtr):
        self.Xtr = Xtr # N,16,784
    def __len__(self):
        return len(self.Xtr)
    def __getitem__(self, idx):
        return self.Xtr[idx]


def load_data():
    X = loadmat('../../data/rot_mnist/rot-mnist-3s.mat')['X'].squeeze() # (N, 16, 784)
    N, T = 500, 16
    Xtr   = torch.tensor(X[:N],dtype=torch.float32).view([N,T,1,28,28])
    Xtest = torch.tensor(X[N:],dtype=torch.float32).view([-1,T,1,28,28])
    return Xtr, Xtest, N, T