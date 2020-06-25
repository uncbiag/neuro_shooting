




import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
import neuro_shooting.resnet_fx as resnet
#import neuro_shooting.res_net as resnet
import simple_discrete_neural_networks as sdnn
import neuro_shooting.utils as utils
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet', 'shooting'], default='shooting')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--method', type=str, choices=['dopri5', 'adams','rk4'], default='euler', help='Selects the desired integrator')
parser.add_argument('--step_size', type=float, default=None, help='Step size for the integrator (if not adaptive).')
parser.add_argument('--max_num_steps', type=int, default=None, help='Maximum number of steps (for dopri5).')

parser.add_argument('--particle_number', type=int, default=10, help='Number of particles for shooting.')
parser.add_argument('--particle_size', type=int, default=6, help='Particle size for shooting.')

parser.add_argument('--downsampling-method', type=str, default='res', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=80)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_batch_size', type=int, default=1000)

parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=9)

parser.add_argument('--seed', required=False, type=int, default=1234,
                    help='Sets the random seed which affects data shuffling')

args = parser.parse_args()

# random seeds
utils.setup_random_seed(seed=args.seed)
# takes care of the GPU setup
device = utils.setup_device(desired_gpu=args.gpu)
def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0, num_workers=0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=num_workers, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=num_workers, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/nmnist', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=num_workers, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()



def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        y = one_hot(np.array(y.cpu().detach().numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x.to(device)).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


if __name__ == '__main__':

    ## define the model
    model = sdnn.MyCNNResNet(nr_layers = 10,inflation_factor = 3,in_channels = 64)
    train_loader, test_loader, train_eval_loader = get_mnist_loaders(args.data_aug, args.batch_size, args.test_batch_size)

    #define the data generator
    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)
    dataiter = iter(train_loader)

    # prepare the model (for first pass, and register parameters)
    images, labels = dataiter.next()
    images = images.to(device)
    model(images)
    model.parameters()
    print("number of parameters ",sum(p.numel() for p in model.parameters() if p.requires_grad))

    # define criterion and optimizer
    criterion = nn.CrossEntropyLoss()


    def setup_optimizer_and_scheduler(params, lr=0.1, weight_decay=0):

        #optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True, factor=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.8)
        return optimizer, scheduler

    optimizer, scheduler = setup_optimizer_and_scheduler(params=model.parameters(), lr=args.lr)

    best_acc = 0
    start = time.time()
    for itr in range(args.nepochs):
        scheduler.step()
        for itr_batch, sampled_batch in enumerate(train_loader):
            optimizer.zero_grad()
            x,y = data_gen.__next__()
    #for itr in range(args.nepochs * batches_per_epoch):
    #    scheduler.step()
    #    optimizer.zero_grad()
    #    x, y = data_gen.__next__()
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            #print(loss)
        #if itr%500==0:
        #    print(itr," loss ",loss.item())

        #if itr % batches_per_epoch == 0:
        print(loss.data)
        with torch.no_grad():
            stop = time.time()
            train_acc = accuracy(model, train_eval_loader)
            val_acc = accuracy(model, test_loader)
            if val_acc > best_acc:
                best_acc = val_acc
                print(best_acc)
        print("Epoch {:04d} | Time {:.3f} | Train Acc {:.4f} ".format(itr , stop - start, val_acc))
        start = time.time()