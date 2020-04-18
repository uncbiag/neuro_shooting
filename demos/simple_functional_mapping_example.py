# the goal of this model is to map the input [-2,2] to a desired functional output
# for simplicity we will try to do this with a very simple 1D shooting model

import os
import sys
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torchdiffeq import odeint

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.insert(0, '../neuro_shooting')

import neuro_shooting
import neuro_shooting.shooting_blocks as sblocks
import neuro_shooting.shooting_models as smodels
import neuro_shooting.generic_integrator as gi
import neuro_shooting.parameter_initialization as pi

def setup_cmdline_parsing():
    parser = argparse.ArgumentParser('Shooting spiral')
    parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--niters', type=int, default=5000)

    # shooting model parameters
    parser.add_argument('--shooting_model', type=str, default='resnet_updown', choices=['resnet_updown','simple', '2nd_order', 'updown'])
    parser.add_argument('--nonlinearity', type=str, default='relu', choices=['identity', 'relu', 'tanh', 'sigmoid'], help='Nonlinearity for shooting.')
    parser.add_argument('--pw', type=float, default=0.01, help='parameter weight')
    parser.add_argument('--nr_of_particles', type=int, default=20, help='Number of particles to parameterize the initial condition')

    # non-shooting networks implemented
    parser.add_argument('--use_updown',action='store_true')
    parser.add_argument('--use_double_resnet',action='store_true')
    parser.add_argument('--use_rnn',action='store_true')
    parser.add_argument('--use_simple_resnet',action='store_true')
    parser.add_argument('--use_neural_ode',action='store_true')

    parser.add_argument('--test_freq', type=int, default=100)
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    return args

def get_sample_batch(nr_of_samples=10):

    sample_batch_in = 4*torch.rand([nr_of_samples,1,1])-2 # creates uniform samples in [-2,2]
    sample_batch_out = sample_batch_in**3 # and takes them to the power of three for the output

    return sample_batch_in, sample_batch_out

class UpDownNet(nn.Module):

    def __init__(self):
        super(UpDownNet, self).__init__()

        self.l1 = nn.Linear(1,5,bias=True)
        self.l2 = nn.Linear(5, 1,bias=True)

    def forward(self, x):

        x = self.l1(F.relu(x))
        x = self.l2(F.relu(x))

        return x

class UpDownDoubleResNet(nn.Module):

    def __init__(self):
        super(UpDownDoubleResNet, self).__init__()

        self.l1 = nn.Linear(5,1,bias=True)
        self.l2 = nn.Linear(1,5,bias=True)

    def forward(self, x1, x2):

        x1 = x1 + self.l1(F.relu(x2))
        #x2 = x2 + self.l2(F.relu(x1))
        x2 = self.l2(F.relu(x1))

        return x1, x2

class SuperSimpleDoubleResNetUpDown(nn.Module):

    def __init__(self):
        super(SuperSimpleDoubleResNetUpDown, self).__init__()

        self.l1 = UpDownDoubleResNet()
        self.l2 = UpDownDoubleResNet()
        self.l3 = UpDownDoubleResNet()
        self.l4 = UpDownDoubleResNet()
        self.l5 = UpDownDoubleResNet()


    def forward(self, x1, x2):

        x1,x2 = self.l1(x1,x2)
        x1,x2 = self.l2(x1,x2)
        x1,x2 = self.l3(x1,x2)
        x1,x2 = self.l4(x1,x2)
        x1,x2 = self.l5(x1,x2)

        return x1,x2

class SuperSimpleResNetUpDown(nn.Module):

    def __init__(self):
        super(SuperSimpleResNetUpDown, self).__init__()

        self.l1 = UpDownNet()
        self.l2 = UpDownNet()
        self.l3 = UpDownNet()
        self.l4 = UpDownNet()
        self.l5 = UpDownNet()

    def forward(self, x):

        x = x + self.l1(F.relu(x))
        x = x + self.l2(F.relu(x))
        x = x + self.l3(F.relu(x))
        x = x + self.l4(F.relu(x))
        x = x + self.l5(F.relu(x))

        return x

class SuperSimpleResNet(nn.Module): # corresponds to our simple shooting model

    def __init__(self):
        super(SuperSimpleResNet, self).__init__()

        self.l1 = nn.Linear(1,1,bias=True)
        self.l2 = nn.Linear(1,1,bias=True)
        self.l3 = nn.Linear(1,1,bias=True)
        self.l4 = nn.Linear(1,1,bias=True)
        self.l5 = nn.Linear(1,1,bias=True)

    def forward(self, x):

        x = x + self.l1(F.relu(x))
        x = x + self.l2(F.relu(x))
        x = x + self.l3(F.relu(x))
        x = x + self.l4(F.relu(x))
        x = x + self.l5(F.relu(x))

        return x

class SuperSimpleRNNResNet(nn.Module):

    def __init__(self):
        super(SuperSimpleRNNResNet, self).__init__()
        self.l1 = nn.Linear(1,1,bias=True)

    def forward(self, x):

        x = x + self.l1(F.relu(x))
        x = x + self.l1(F.relu(x))
        x = x + self.l1(F.relu(x))
        x = x + self.l1(F.relu(x))
        x = x + self.l1(F.relu(x))

        return x

class ODESimpleFunc(nn.Module):

    def __init__(self):
        super(ODESimpleFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 1),
            nn.Tanh(),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)

if __name__ == '__main__':

    args = setup_cmdline_parsing()
    if args.verbose:
        print(args)

    par_init = pi.VectorEvolutionParameterInitializer(
        only_random_initialization=True,
        random_initialization_magnitude=0.01)


    shootingintegrand_kwargs = {'in_features': 1,
                                'nonlinearity': args.nonlinearity,
                                'nr_of_particles': args.nr_of_particles,
                                'parameter_weight': args.pw,
                                'particle_dimension': 1,
                                'particle_size': 1}

    if args.shooting_model == 'simple':
        smodel = smodels.AutoShootingIntegrandModelSimple(**shootingintegrand_kwargs,use_analytic_solution=True)
    elif args.shooting_model == '2nd_order':
        smodel = smodels.AutoShootingIntegrandModelSecondOrder(**shootingintegrand_kwargs)
    elif args.shooting_model == 'updown':
        smodel = smodels.AutoShootingIntegrandModelUpDown(**shootingintegrand_kwargs)
    elif args.shooting_model == 'resnet_updown':
        smodel = smodels.AutoShootingIntegrandModelResNetUpDown(**shootingintegrand_kwargs)

    sblock = sblocks.ShootingBlockBase(
        name='simple',
        shooting_integrand=smodel,
        integrator_name='rk4',
        use_adjoint_integration=False,
        intgrator_options = {'stepsize':0.5}
    )

    use_shooting = False
    if args.use_rnn:
        weight_decay = 0.0000001
        print('Using SuperSimpleRNNResNet: weight = {}'.format(weight_decay))
        simple_resnet = SuperSimpleRNNResNet()
    elif args.use_updown:
        weight_decay = 0.025
        print('Using SuperSimpleResNetUpDown: weight = {}'.format(weight_decay))
        simple_resnet = SuperSimpleResNetUpDown()
    elif args.use_simple_resnet:
        weight_decay = 0.0000001
        print('Using SuperSimpleResNet: weight = {}'.format(weight_decay))
        simple_resnet = SuperSimpleResNet()
    elif args.use_double_resnet:
        weight_decay = 0.025
        print('Using SuperSimpleDoubleResNetUpDown: weight = {}'.format(weight_decay))
        simple_resnet = SuperSimpleDoubleResNetUpDown()
    elif args.use_neural_ode:
        print('Using neural ode')
        func = ODESimpleFunc()
        optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
        integrator = gi.GenericIntegrator(integrator_library='odeint',
                                                          integrator_name=args.method,
                                                          use_adjoint_integration=False,
                                                          rtol=1e-8, atol=1e-12)

    else:
        use_shooting = True
        print('Using shooting')


    if not use_shooting:
        if args.use_neural_ode:
            optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
        else:
            optimizer = optim.Adam(simple_resnet.parameters(), lr=1e-2, weight_decay=weight_decay)
    else:

        sample_batch_in, sample_batch_out = get_sample_batch(nr_of_samples=args.batch_size)
        sblock(x=sample_batch_in)

        # do some parameter freezing
        for pn,pp in sblock.named_parameters():
            if pn=='q1':
                # freeze the locations
                pp.requires_grad = False

        optimizer = optim.Adam(sblock.parameters(), lr=1e-5)

    track_loss = []
    for itr in range(1, args.niters + 1):

        # get current batch data
        batch_in, batch_out = get_sample_batch(nr_of_samples=args.batch_size)

        optimizer.zero_grad()

        if not use_shooting:
            if args.use_neural_ode:

                pred_y = integrator.integrate(func=func, x0=batch_in, t=torch.tensor([0, 1]).float())
                pred_y = pred_y[1, :, :, :] # need prediction at time 1

            elif args.use_double_resnet:
                x20 = torch.zeros_like(batch_in)
                sz = [1] * len(x20.shape)
                sz[-1] = 5
                x20 = x20.repeat(sz)

                pred_y, pred_y2 = simple_resnet(x1=batch_in, x2=x20)
            else:
                pred_y = simple_resnet(x=batch_in)
        else:
            # set integration time
            # sblock.set_integration_time(time_to=1.0) # try to do this mapping in unit time
            pred_y, _, _, _ = sblock(x=batch_in)


        loss = torch.mean((pred_y - batch_out)**2)  # + 1e-2 * sblock.get_norm_penalty()
        loss.backward()

        track_loss.append(loss.item())
        optimizer.step()

        if itr % args.test_freq == 0:

            fig = plt.figure()
            plt.plot(batch_in.detach().numpy().squeeze(),batch_out.detach().numpy().squeeze(),'g+')
            plt.plot(batch_in.detach().numpy().squeeze(),pred_y.detach().numpy().squeeze(),'r*')
            plt.show()

            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
