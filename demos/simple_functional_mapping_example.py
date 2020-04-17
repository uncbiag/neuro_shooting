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
    parser.add_argument('--data_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--niters', type=int, default=2000)
    parser.add_argument('--test_freq', type=int, default=100)
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    return args

def get_sample_batch(nr_of_samples=10):

    sample_batch_in = 4*torch.rand([nr_of_samples,1,1])-2 # creates uniform samples in [-2,2]
    sample_batch_out = sample_batch_in**3 # and takes them to the power of three for the output

    return sample_batch_in, sample_batch_out


if __name__ == '__main__':

    args = setup_cmdline_parsing()
    if args.verbose:
        print(args)

    par_init = pi.VectorEvolutionParameterInitializer(
        only_random_initialization=True,
        random_initialization_magnitude=1.0)

    smodel = smodels.AutoShootingIntegrandModelUpDown(
        in_features=1,
        nonlinearity='tanh',
        parameter_weight=0.05,
        nr_of_particles=50,
        particle_dimension=1,
        particle_size=1)

    # smodel = smodels.AutoShootingIntegrandModelSimple(
    #     in_features=1,
    #     nonlinearity='tanh',
    #     parameter_weight=0.05,
    #     nr_of_particles=50,
    #     particle_dimension=1,
    #     particle_size=1)

    sblock = sblocks.ShootingBlockBase(
        name='simple',
        shooting_integrand=smodel,
        integrator_name='rk4',
        use_adjoint_integration=False,
        intgrator_options = {'stepsize':0.01}
    )

    sample_batch_in, sample_batch_out = get_sample_batch(nr_of_samples=args.batch_size)

    sblock(x=sample_batch_in)
    optimizer = optim.Adam(sblock.parameters(), lr=1e-4)

    track_loss = []
    for itr in range(1, args.niters + 1):

        # get current batch data
        batch_in, batch_out = get_sample_batch(nr_of_samples=args.batch_size)

        # set integration time
        #sblock.set_integration_time(time_to=1.0) # try to do this mapping in unit time

        optimizer.zero_grad()

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
