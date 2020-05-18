# This is the code for a very basic rasnet full implemented using shooting

import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from tqdm import trange
    has_TQDM = True
    range_command = trange
except:
    has_TQDM = False
    print('If you want to display progress bars install TQDM; conda install tqdm')
    range_command = range


import neuro_shooting.shooting_models as shooting_models
import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.generic_integrator as generic_integrator
import neuro_shooting.parameter_initialization as parameter_initialization
import neuro_shooting.res_net as res_net

import neuro_shooting.utils as utils
import neuro_shooting.data_loaders as data_loaders

# Setup

def setup_cmdline_parsing():
    # Command line arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('--tol', type=float, default=1e-3)
    parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
    parser.add_argument('--method', type=str, choices=['dopri5', 'adams','rk4'], default='rk4', help='Selects the desired integrator')
    parser.add_argument('--stepsize', type=float, default=0.1, help='Step size for the integrator (if not adaptive).')
    parser.add_argument('--max_num_steps', type=int, default=None, help='Maximum number of steps (for dopri5).')
    parser.add_argument('--shooting_model', type=str, default='conv_updown', choices=['conv_updown', 'conv_simple'])

    parser.add_argument('--particle_size', type=int, default=3, help='Particle size for shooting.')
    parser.add_argument('--nr_of_particles', type=int, default=25,
                        help='Number of particles to parameterize the initial condition')
    parser.add_argument('--sim_weight', type=float, default=100.0, help='Weight for the similarity measure')
    parser.add_argument('--norm_weight', type=float, default=0.01, help='Weight for the similarity measure')

    parser.add_argument('--inflation_factor', type=int, default=5,
                        help='Multiplier for state dimension for updown shooting model types')
    parser.add_argument('--optimize_over_data_initial_conditions', action='store_true', default=False)
    parser.add_argument('--optimize_over_data_initial_conditions_type', type=str,
                        choices=['direct', 'linear', 'mini_nn'], default='linear',
                        help='Different ways to predict the initial conditions for higher order models. Currently only supported for updown model.')

    parser.add_argument('--dataset', type=str, choices=['MNIST','CIFAR10', 'CIFAR100', 'FashionMNIST'], default='CIFAR10', help='Dataset to run. Pretty much all pyTorch datasets are supported.')

    parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
    parser.add_argument('--nepochs', type=int, default=160)
    parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=128)

    parser.add_argument('--save', type=str, default='./experiment1')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--checkpointing_time_interval', type=float, default=0.0,
                        help='If specified puts a checkpoint after every interval (hence dynamically changes with the integration time). If a fixed number is deisred use --nr_of_checkpoints instead.')
    parser.add_argument('--nr_of_checkpoints', type=int, default=0,
                        help='If specified will add that many checkpoints for integration. If integration times differ it is more convenient to set --checkpointing_time_interval instead.')

    parser.add_argument('--seed', required=False, type=int, default=1234, help='Sets the random seed which affects data shuffling')

    args = parser.parse_args()

    return args

def setup_integrator(method, use_adjoint, step_size, rtol=1e-8, atol=1e-12, nr_of_checkpoints=None, checkpointing_time_interval=None):

    integrator_options = dict()

    if method not in ['dopri5', 'adams']:
        integrator_options  = {'step_size': step_size}

    integrator = generic_integrator.GenericIntegrator(integrator_library = 'odeint', integrator_name = method,
                                                      use_adjoint_integration=use_adjoint, integrator_options=integrator_options, rtol=rtol, atol=atol,
                                                      nr_of_checkpoints=nr_of_checkpoints,
                                                      checkpointing_time_interval=checkpointing_time_interval)
    return integrator

def setup_shooting_model(shooting_model, inflation_factor=2, optimize_over_data_initial_conditions=True, optimize_over_data_initial_conditions_type='linear'):

    available_shooting_models = ['conv_updown', 'conv_simple']

    if shooting_model=='conv_updown':
        shooting_model = shooting_models.AutoShootingIntegrandModelUpDownConv2D
        shooting_model_kwargs = {'inflation_factor': inflation_factor,
                                 'optimize_over_data_initial_conditions': optimize_over_data_initial_conditions,
                                 'optimize_over_data_initial_conditions_type': optimize_over_data_initial_conditions_type}

    elif shooting_model=='conv_simple':
        shooting_model = shooting_models.AutoShootingIntegrandModelSimpleConv2D
        shooting_model_kwargs = {}
    else:
        raise ValueError('Unkonw shooting model: choices are {}'.format(available_shooting_models))

    return shooting_model, shooting_model_kwargs


def setup_optimizer_and_scheduler(params):

    optimizer = optim.Adam(params, lr=0.1)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,verbose=True)

    return optimizer, scheduler

# Defining the differential equation and data


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)



class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


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


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader, nr_of_labels):
    total_correct = 0
    for x, y in dataset_loader:
        y = one_hot(np.array(y.cpu().detach().numpy()), nr_of_labels)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x.to(device)).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath=None, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    if filepath is not None:
        logger.info(filepath)
        with open(filepath, "r") as f:
            logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

class ExtractFirstOutput(nn.Module):

    def __init__(self, shooting_block):
        super(ExtractFirstOutput, self).__init__()
        self.shooting_block = shooting_block

    def forward(self,x):
        ret_all = self.shooting_block(x)
        return ret_all[0]


if __name__ == '__main__':

    args = setup_cmdline_parsing()

    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'))
    logger.info(args)

    utils.setup_random_seed(seed=args.seed)
    device = utils.setup_device(desired_gpu=args.gpu)

    # optional checkpointing support for integration
    if args.checkpointing_time_interval > 0:
        checkpointing_time_interval = args.checkpointing_time_interval
    else:
        checkpointing_time_interval = None

    if args.nr_of_checkpoints > 0:
        nr_of_checkpoints = args.nr_of_checkpoints
    else:
        nr_of_checkpoints = None

    integrator = setup_integrator(method=args.method, step_size=args.stepsize, use_adjoint=args.adjoint, nr_of_checkpoints=nr_of_checkpoints, checkpointing_time_interval=checkpointing_time_interval)


    train_loader, test_loader, train_eval_loader, nr_of_labels = data_loaders.get_data_loaders(
        dataset=args.dataset, data_aug=args.data_aug, batch_size=args.batch_size, test_batch_size=args.test_batch_size
    )

    if args.downsampling_method == 'conv':
        downsampling_layers = [
            nn.Conv2d(1, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        ]
    elif args.downsampling_method == 'res':
        downsampling_layers = [
            nn.Conv2d(1, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]

    # setup the initializers
    state_initializer = parameter_initialization.ConvolutionEvolutionParameterInitializer(
        only_random_initialization=True, random_initialization_magnitude=0.5)
    costate_initializer = parameter_initialization.ConvolutionEvolutionParameterInitializer(
        only_random_initialization=True, random_initialization_magnitude=0.5)

    # shooting_model = shooting_models.AutoShootingIntegrandModelSimpleConv2D(in_features=64,
    #                                                                         nonlinearity='relu',
    #                                                                         nr_of_particles=args.nr_of_particles,
    #                                                                         state_initializer=state_initializer,
    #                                                                         costate_initializer=costate_initializer,
    #                                                                         particle_size=[args.particle_size,args.particle_size],
    #                                                                         particle_dimension=64)
    #
    # shooting_block = ExtractFirstOutput(shooting_blocks.ShootingBlockBase(name='test_block', shooting_integrand=shooting_model, integrator=integrator))
    #feature_layers = [shooting_block]
    # feature_layers = [res_net_shooting_block]
    # fc_layers = [norm(layer_channels[-1]), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(),
    #              nn.Linear(layer_channels[-1], nr_of_labels)]

    shooting_model, shooting_model_kwargs = setup_shooting_model(shooting_model=args.shooting_model,
                                                                 inflation_factor=args.inflation_factor,
                                                                 optimize_over_data_initial_conditions=args.optimize_over_data_initial_conditions,
                                                                 optimize_over_data_initial_conditions_type=args.optimize_over_data_initial_conditions_type)

    layer_channels = [32,64,128]
    particle_sizes = [[11,11],[7,7],[5,5]]
    res_net_shooting_model = res_net.BasicResNet(nr_of_image_channels=3,
                 layer_channels=layer_channels,
                 nr_of_blocks_per_layer=[1]*len(layer_channels),
                 downsampling_stride=2,
                 nonlinearity='relu',
                 particle_sizes=particle_sizes,
                 nr_of_particles=args.nr_of_particles,
                 nr_of_classes=nr_of_labels,
                 state_initializer=state_initializer,
                 costate_initializer=costate_initializer,
                 integrator=integrator,
                 shooting_model=shooting_model,
                 shooting_model_kwargs=shooting_model_kwargs)


    model = res_net_shooting_model

    criterion = nn.CrossEntropyLoss()

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)
    dataiter = iter(train_loader)

    # prepare the model (for first pass, and register parameters)
    images, labels = dataiter.next()
    images = images.to(device)
    model(images)

    # setup optimizer
    optimizer, scheduler = setup_optimizer_and_scheduler(params=model.parameters())
    nr_of_pars = utils.compute_number_of_parameters(model=model)

    logger.info(model)
    logger.info('Number of parameters: {}'.format(nr_of_pars))

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    for itr in range_command(args.nepochs * batches_per_epoch):

        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        if itr%50==0:
            print(itr," loss ",loss.item())

        # todo: maybe only do this every couple of iterations
        if itr % 100 == 0:
            try:
                scheduler.step()
            except:
                scheduler.step(loss)

        batch_time_meter.update(time.time() - end)

        end = time.time()

        if itr % batches_per_epoch == 0:
            # with torch.no_grad():
            train_acc = accuracy(model=model, dataset_loader=train_eval_loader, nr_of_labels=nr_of_labels)
            val_acc = accuracy(model=model, dataset_loader=test_loader, nr_of_labels=nr_of_labels)
            if val_acc > best_acc:
                torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                best_acc = val_acc
            logger.info(
                "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                "Train Acc {:.4f} | Test Acc {:.4f}".format(
                    itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                    b_nfe_meter.avg, train_acc, val_acc
                )
            )
