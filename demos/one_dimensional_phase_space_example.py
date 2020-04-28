# the goal of this model is to map the input [-2,2] to a desired functional output
# for simplicity we will try to do this with a very simple 1D shooting model

import os
import sys
import time
import random
import argparse
import numpy as np

from collections import OrderedDict
from collections import defaultdict

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
    parser.add_argument('--niters', type=int, default=4000)

    # shooting model parameters
    parser.add_argument('--nonlinearity', type=str, default='relu', choices=['identity', 'relu', 'tanh', 'sigmoid',"softmax"], help='Nonlinearity for shooting.')
    parser.add_argument('--pw', type=float, default=0.1, help='parameter weight')
    parser.add_argument('--nr_of_particles', type=int, default=10, help='Number of particles to parameterize the initial condition')

    parser.add_argument('--test_freq', type=int, default=100)
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    return args


def get_sample_batch(nr_of_samples=10):

    # create some random input
    sample_batch_in = 4*torch.rand([nr_of_samples,1,1])-2 # creates uniform samples in [-2,2]

    # and now create the random output, will also be random, but shifted up

    sample_batch_out = 0.5*4*torch.rand([nr_of_samples,1,1])+2 # creates uniform samples in 0.5*[-2,2]+4

    return sample_batch_in, sample_batch_out


def print_all_parameters(model):

    print('\n Model parameters:\n')
    for pn,pv in model.named_parameters():
        print('{} = {}\n'.format(pn, pv))

def compute_number_of_parameters_and_print_all_parameters(model):

    nr_of_fixed_parameters = 0
    nr_of_optimized_parameters = 0
    print('\n Model parameters:\n')
    for pn, pv in model.named_parameters():
        print('{} = {}'.format(pn, pv))
        current_number_of_parameters = np.prod(list(pv.size()))
        print('# of parameters = {}\n'.format(current_number_of_parameters))
        if pv.requires_grad:
            nr_of_optimized_parameters += current_number_of_parameters
        else:
            nr_of_fixed_parameters += current_number_of_parameters

    print('\n')
    print('Number of fixed parameters = {}'.format(nr_of_fixed_parameters))
    print('Number of optimized parameters = {}'.format(nr_of_optimized_parameters))
    print('Overall number of parameters = {}'.format(nr_of_fixed_parameters + nr_of_optimized_parameters))


def plot_temporal_data(data, block_name):

    # time
    t = np.asarray(data['t'])
    # energy
    energy = np.asarray(data['energy'])

    # first plot the energy over time
    plt.figure()
    plt.plot(t,energy)
    plt.xlabel('time')
    plt.ylabel('energy')
    plt.show()

    # do the phase-space plot (i.e., q1 vs p_q1)
    q1 = np.asarray(data['{}.state.q1'.format(block_name)]).squeeze()
    p_q1 = np.asarray(data['{}.costate.p_q1'.format(block_name)]).squeeze()
    dot_q1 = np.asarray(data['{}.dot_state.dot_q1'.format(block_name)]).squeeze()
    dot_p_q1 = np.asarray(data['{}.dot_costate.dot_p_q1'.format(block_name)]).squeeze()

    # plot the phase space
    plt.figure()

    nr_of_particles = q1.shape[1]
    for n in range(nr_of_particles):
        plt.plot(q1[:,n],p_q1[:,n])

    plt.xlabel('q1')
    plt.ylabel('p_q1')
    plt.show()

    # exclude list (what not to plot, partial initial match is fine)
    do_not_plot = ['t', 'energy', 'dot_state','dot_costate','dot_data']

    for k in data:

        # first check if we should plot this
        do_plotting = True
        for dnp in do_not_plot:
            if k.startswith(dnp) or k.startswith('{}.{}'.format(block_name,dnp)):
                do_plotting = False

        if do_plotting:
            plt.figure()

            cur_vals = np.asarray(data[k]).squeeze()
            cur_shape = cur_vals.shape
            if len(cur_shape)==3: # multi-dimensional state
                for cur_dim in range(cur_shape[2]):
                    plt.plot(t,cur_vals[:,:,cur_dim])
            else:
                plt.plot(t,cur_vals)

            plt.xlabel('time')
            plt.ylabel(k)
            plt.show()


if __name__ == '__main__':

    args = setup_cmdline_parsing()
    if args.verbose:
        print(args)


    def record_generic_dict_of_dicts(custom_hook_data, d,d_name):
        for block_name in d:
            cur_block = d[block_name]
            for cur_state_name in cur_block:
                cur_key = '{}.{}.{}'.format(block_name, d_name, cur_state_name)
                custom_hook_data[cur_key].append(cur_block[cur_state_name].detach().numpy())

    def parameters_hook(module, t, state_dicts, costate_dicts, data_dict_of_dicts,
                              dot_state_dicts, dot_costate_dicts, dot_data_dict_of_dicts, parameter_objects,
                              custom_hook_data):

        with torch.no_grad():

            # record time
            custom_hook_data['t'].append(t.item())

            current_energy = torch.zeros(1)
            # record all parameters
            for k in parameter_objects:
                cur_par_dict = parameter_objects[k]._parameter_dict
                for p in cur_par_dict:
                    cur_key = '{}.{}'.format(k,p)
                    custom_hook_data[cur_key].append(cur_par_dict[p].detach().numpy())
                    # add to current energy
                    current_energy += 0.5*torch.sum(cur_par_dict[p]**2)

            # record the current energy
            custom_hook_data['energy'].append(current_energy.item())

            # now record all the states
            record_generic_dict_of_dicts(custom_hook_data=custom_hook_data, d=state_dicts, d_name='state')
            # now record all the costates
            record_generic_dict_of_dicts(custom_hook_data=custom_hook_data, d=costate_dicts, d_name='costate')
            # now record all the data states
            record_generic_dict_of_dicts(custom_hook_data=custom_hook_data, d=data_dict_of_dicts, d_name='data')

            # now record all the current derivatives
            record_generic_dict_of_dicts(custom_hook_data=custom_hook_data, d=dot_state_dicts, d_name='dot_state')
            record_generic_dict_of_dicts(custom_hook_data=custom_hook_data, d=dot_costate_dicts, d_name='dot_costate')
            record_generic_dict_of_dicts(custom_hook_data=custom_hook_data, d=dot_data_dict_of_dicts, d_name='dot_data')

        return None

    par_init = pi.VectorEvolutionSampleBatchParameterInitializer(
        only_random_initialization=True,
        random_initialization_magnitude=0.1,
        sample_batch = torch.linspace(0,1,args.nr_of_particles))


    shootingintegrand_kwargs = {'in_features': 1,
                                'nonlinearity': args.nonlinearity,
                                'nr_of_particles': args.nr_of_particles,
                                'parameter_weight': args.pw,
                                'particle_dimension': 1,
                                'particle_size': 1,
                                "costate_initializer":pi.VectorEvolutionParameterInitializer(random_initialization_magnitude=0.1)}

    smodel = smodels.AutoShootingIntegrandModelSimple(**shootingintegrand_kwargs,use_analytic_solution=True)
    block_name = 'sblock'

    sblock = sblocks.ShootingBlockBase(
        name=block_name,
        shooting_integrand=smodel,
        integrator_name='rk4',
        use_adjoint_integration=False,
        integrator_options = {'step_size':0.1}
    )

    sample_batch_in, sample_batch_out = get_sample_batch(nr_of_samples=args.batch_size)
    sblock(x=sample_batch_in)

    # do some parameter freezing
    for pn,pp in sblock.named_parameters():
        if pn=='q1':
            # freeze the locations
            #pp.requires_grad = False
            pass
    optimizer = optim.Adam(sblock.parameters(), lr=1e-2)

    track_loss = []
    for itr in range(1, args.niters + 1):

        # get current batch data
        batch_in, batch_out = get_sample_batch(nr_of_samples=args.batch_size)

        optimizer.zero_grad()

        # set integration time
        # sblock.set_integration_time(time_to=1.0) # try to do this mapping in unit time
        pred_y, _, _, _ = sblock(x=batch_in)


        loss = torch.mean((pred_y - batch_out)**2)  # + 1e-2 * sblock.get_norm_penalty()
        loss.backward()

        track_loss.append(loss.item())
        optimizer.step()

        if itr % args.test_freq == 0:
            fig = plt.figure(figsize=(8, 12), facecolor='white')
            ax1 = fig.add_subplot(121, frameon=False)
            ax2 = fig.add_subplot(122, frameon=False)

            #ax.cla()

            z = torch.zeros_like(batch_in).detach().numpy().squeeze()
            o = torch.ones_like(batch_out).detach().numpy().squeeze()

            oz = (np.stack((z,o),axis=-1)).transpose()

            true_correspondences = (np.stack((batch_in.detach().numpy().squeeze(),
                                              batch_out.detach().numpy().squeeze()),axis=-1)).transpose()

            pred_correspondences = (np.stack((batch_in.detach().numpy().squeeze(),
                                              pred_y.detach().numpy().squeeze()), axis=-1)).transpose()

            ax1.plot(oz,true_correspondences,'g-',linewidth=0.5)
            ax2.plot(oz,pred_correspondences,'r-',linewidth=0.5)

            plt.show()

            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

    # now print the paramerers
    compute_number_of_parameters_and_print_all_parameters(sblock)

    custom_hook_data = defaultdict(list)
    hook = sblock.shooting_integrand.register_lagrangian_gradient_hook(parameters_hook)
    sblock.shooting_integrand.set_custom_hook_data(data=custom_hook_data)

    batch_in, batch_out = get_sample_batch(nr_of_samples=args.batch_size)
    pred_y, _, _, _ = sblock(x=batch_in)
    hook.remove()

    plot_temporal_data(data=custom_hook_data, block_name=block_name)
