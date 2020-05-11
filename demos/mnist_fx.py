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
import neuro_shooting.shooting_models as shooting_models
import neuro_shooting.utils as utils



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
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)

parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=2)

parser.add_argument('--seed', required=False, type=int, default=1234,
                    help='Sets the random seed which affects data shuffling')

args = parser.parse_args()


# takes care of the GPU setup
#utils.setup_device(desired_gpu=args.gpu)


print('Setting the random seed to {:}'.format(args.seed))
random.seed(args.seed)
torch.manual_seed(args.seed)
utils.setup_device(desired_gpu=args.gpu)
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
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
        datasets.MNIST(root='.data/fashionmnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/fashionmnist', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/fashionmnist', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
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
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)




if __name__ == '__main__':

    ## define the model
    model = resnet.BasicResNet(nr_of_image_channels=1,nr_of_blocks_per_layer=[1],layer_channels=[32],
                 downsampling_stride=2,
                 nonlinearity='relu',
                 particle_sizes=[[4,4]],
                 nr_of_particles=30,
                 nr_of_classes=10,parameter_weight=1.0,inflation_factor=2)

    train_loader, test_loader, train_eval_loader = get_mnist_loaders(args.data_aug, args.batch_size, args.test_batch_size)

    #define the data generator
    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)
    dataiter = iter(train_loader)

    # prepare the model (for first pass, and register parameters)
    images, labels = dataiter.next()
    model.to(device)
    images = images.to(device)
    model(images)
    model.parameters()
    print("number of parameters ",sum(p.numel() for p in model.parameters() if p.requires_grad))

    # define criterion and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    best_acc = 0
    start = time.time()
    for itr in range(args.nepochs * batches_per_epoch):

        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()
        if itr%30==0:
            print(itr," loss ",loss)

        if itr % batches_per_epoch == 0:
            #print(loss.data)
            # with torch.no_grad():
            stop = time.time()
            #train_acc = accuracy(model, train_eval_loader)
            #val_acc = accuracy(model, test_loader)
            #if val_acc > best_acc:
            #    best_acc = val_acc
            #    print(best_acc)
            print("Epoch {:04d} | Time {:.3f} | Train Acc {:.4f} ".format(itr // batches_per_epoch, stop - start, loss.cpu().detach().numpy()))
            start = time.time()