"""Utility functions for loading the NeurIPS 20 experiments"""

import torch
import numpy as np
from torch.utils import data
from torch.utils.data import Subset
from torch.utils.data import Dataset
from scipy.io import loadmat
from random import random

"""Rotated MNIST related classes/functions"""


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


"""Concentric circles related classes/functions"""


def random_point_in_sphere(dim, min_radius, max_radius):
    unif = random()
    distance = (max_radius - min_radius) * (unif ** (1. / dim)) + min_radius
    direction = torch.randn(dim)
    unit_direction = direction / torch.norm(direction, 2)
    return distance * unit_direction


class ConcentricSphere(Dataset):
    def __init__(self, dim, inner_range, outer_range, num_points_inner,
                 num_points_outer):
        self.dim = dim
        self.inner_range = inner_range
        self.outer_range = outer_range
        self.num_points_inner = num_points_inner
        self.num_points_outer = num_points_outer

        self.data = []
        self.targets = []

        # Generate data for inner sphere
        for _ in range(self.num_points_inner):
            self.data.append(
                random_point_in_sphere(dim, inner_range[0], inner_range[1])
            )
            self.targets.append(torch.Tensor([0]))

        # Generate data for outer sphere
        for _ in range(self.num_points_outer):
            self.data.append(
                random_point_in_sphere(dim, outer_range[0], outer_range[1])
            )
            self.targets.append(torch.Tensor([1]))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


def dataset_to_numpy(loader):
    X,y = [], []
    for X_tensor, y_tensor in loader:
        X.append(X_tensor.cpu().numpy())
        y.append(y_tensor.cpu().numpy())
    return np.vstack(X), np.vstack(y)


def sample(ds, N):
    assert N <= len(ds) 
    assert len(np.unique(ds.targets)) == 2
    tmp = np.array([int(l.item()) for l in ds.targets])

    idx0 = np.where(tmp==1)[0]
    idx1 = np.where(tmp==0)[0]

    idx = np.concatenate((
        idx0[np.random.randint(0,len(idx0),N)],
        idx1[np.random.randint(0,len(idx1),N)]))

    return Subset(ds, idx)
    
