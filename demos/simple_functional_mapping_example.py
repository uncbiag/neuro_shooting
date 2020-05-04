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

import torch.optim as optim
from torchdiffeq import odeint

import neuro_shooting
import neuro_shooting.shooting_blocks as sblocks
import neuro_shooting.shooting_models as smodels
import neuro_shooting.generic_integrator as gi
import neuro_shooting.parameter_initialization as pi

import simple_discrete_neural_networks as sdnn

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0, '../neuro_shooting')


def setup_cmdline_parsing():
    parser = argparse.ArgumentParser('Shooting spiral')
    parser.add_argument('--method', type=str, choices=['dopri5', 'adams', 'rk4'], default='rk4', help='Selects the desired integrator')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--niters', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--adjoint', action='store_true', help='Use adjoint integrator to avoid storing values during forward pass.')
    parser.add_argument('--gpu', type=int, default=0, help='Enable GPU computation on specified GPU.')
    parser.add_argument('--stepsize', type=float, default=0.1, help='Step size for the integrator (if not adaptive).')

    # shooting model parameters
    parser.add_argument('--shooting_model', type=str, default='universal', choices=["universal",'dampened_updown','simple', '2nd_order', 'updown'])
    parser.add_argument('--nonlinearity', type=str, default='relu', choices=['identity', 'relu', 'tanh', 'sigmoid',"softmax"], help='Nonlinearity for shooting.')
    parser.add_argument('--pw', type=float, default=1.0, help='parameter weight')
    parser.add_argument('--nr_of_particles', type=int, default=3, help='Number of particles to parameterize the initial condition')
    parser.add_argument('--inflation_factor', type=int, default=5, help='Multiplier for state dimension for updown shooting model types')
    parser.add_argument('--use_particle_rnn_mode', action='store_true', help='When set then parameters are only computed at the initial time and used for the entire evolution; mimicks a particle-based RNN model.')
    parser.add_argument('--use_particle_free_rnn_mode', action='store_true', help='This is directly optimizing over the parameters -- no particles here; a la Neural ODE')
    parser.add_argument('--use_parameter_penalty_energy', action='store_true', default=False)

    # non-shooting networks implemented
    parser.add_argument('--nr_of_layers', type=int, default=30, help='Number of layers for the non-shooting networks')
    parser.add_argument('--use_updown',action='store_true')
    parser.add_argument('--use_double_resnet',action='store_true')
    parser.add_argument('--use_rnn',action='store_true')
    parser.add_argument('--use_double_resnet_rnn',action="store_true")
    parser.add_argument('--use_simple_resnet',action='store_true')
    parser.add_argument('--use_neural_ode',action='store_true')

    parser.add_argument('--test_freq', type=int, default=100)
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    return args

# --shooting_model updown --nr_of_particles 5 --pw 0.1 --viz --niters 100 --nonlinearity relu

def get_sample_batch(nr_of_samples=10):

    sample_batch_in = 4*torch.rand([nr_of_samples,1,1])-2 # creates uniform samples in [-2,2]
    sample_batch_out = sample_batch_in**3 # and takes them to the power of three for the output

    return sample_batch_in, sample_batch_out

def get_uniform_sample_batch(nr_of_samples=10):

    sample_batch_in = (torch.linspace(-2,2,nr_of_samples)).view([nr_of_samples,1,1]) # creates uniform samples in [-2,2]
    sample_batch_out = sample_batch_in**3 # and takes them to the power of three for the output

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

def collect_and_sort_parameter_values_across_layers(model):

    # TODO: in principle I would like to extend this so that we sort the states so it is possible
    # to follow how they change over the layers; this should happen automatically for the continuous
    # models, but states can swap arbitrarily for the resnet models

    print('\n Collecting and sorting parameter values across layers')
    named_pars = list(model.named_parameters())

    nr_of_pars= len(named_pars)

    par_names_per_block = []
    par_size_per_block = []

    # first get the
    for i in range(nr_of_pars):
        cur_par_name = named_pars[i][0]
        cur_par_val = named_pars[i][1]
        names = cur_par_name.split('.')

        if i==0:
            first_layer_name = names[1]

        if first_layer_name==names[1]:
            par_names_per_block.append('.'.join(names[2:]))
            par_size_per_block.append(cur_par_val.size())
        else:
            break

    # nr of pars per block
    nr_of_pars_per_block = len(par_names_per_block)

    # nr of layers
    nr_of_layers = nr_of_pars//nr_of_pars_per_block

    # create the torch arrays where we will store the variables as they change over the layers
    layer_pars = dict()
    for pn,ps in zip(par_names_per_block,par_size_per_block):
        layer_pars[pn] = torch.zeros([nr_of_layers] + list(ps))

    # now we can put the values in these arrays
    layer_nr = 0
    current_layer_name = first_layer_name

    for i in range(nr_of_pars):
        cur_par_name = named_pars[i][0]
        cur_par_val = named_pars[i][1]
        names = cur_par_name.split('.')

        if current_layer_name != names[1]:
            current_layer_name = names[1]
            layer_nr+=1

        layer_par_name = '.'.join(names[2:])
        layer_pars[layer_par_name][layer_nr,...] = cur_par_val

    print('\nCollected (not yet sorted: TODO) block parameters across layers:\n')
    for k in layer_pars:
        print('{} = {}'.format(k,layer_pars[k]))

    import matplotlib.pyplot as plt
    import matplotlib
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from matplotlib import gridspec

    spec = gridspec.GridSpec(ncols=1, nrows=4,
                             width_ratios=[4], height_ratios=[5,1,5,5])

    fig = plt.figure() #(figsize=(12, 4), facecolor='white')
    l1w = fig.add_subplot(spec[0])
    l1b = fig.add_subplot(spec[1])
    l2w = fig.add_subplot(spec[2])
    l2b = fig.add_subplot(spec[3])

    if 'l1.weight' in layer_pars:
        l1w.cla()
        l1w.set_title('l1-weight')
        #l1w.set_xlabel('layers')
        A = layer_pars['l1.weight'].detach().numpy().squeeze()
        A.sort()
        im1 = l1w.imshow(A.transpose(),
                         norm=matplotlib.colors.SymLogNorm(linthresh=0.01, linscale=0.03,vmin=-1.0, vmax=1.0),
                         aspect='auto')

        divider = make_axes_locatable(l1w)
        cax = divider.append_axes('bottom', size='20%', pad=0.25)
        fig.colorbar(im1, cax=cax, orientation='horizontal')

    if 'l1.bias' in layer_pars:
        l1b.cla()
        l1b.set_title('l1-bias')
        l1b.set_xlabel('layers')
        im2 = l1b.imshow(layer_pars['l1.bias'].detach().numpy().transpose(),
                        norm=matplotlib.colors.SymLogNorm(linthresh=0.01, linscale=0.03,vmin=-1.0, vmax=1.0),
                        aspect='auto')

        divider = make_axes_locatable(l1b)
        cax = divider.append_axes('bottom', size='50%', pad=0.25)
        fig.colorbar(im2, cax=cax, orientation='horizontal')

    if 'l2.weight' in layer_pars:
        l2w.cla()
        l2w.set_title('l2-weight')
        #l2w.set_xlabel('layers')
        B = layer_pars['l2.weight'].detach().numpy().squeeze()
        B.sort()
        im3 = l2w.imshow(B.transpose(),
                         #norm=matplotlib.colors.SymLogNorm(linthresh=0.01, linscale=0.03, vmin=-1.0, vmax=1.0),
                         aspect='auto')

        divider = make_axes_locatable(l2w)
        cax = divider.append_axes('bottom', size='20%', pad=0.25)
        fig.colorbar(im3, cax=cax, orientation='horizontal')

    if 'l2.bias' in layer_pars:
        l2b.cla()
        l2b.set_title('l2-bias')
        #l2b.set_xlabel('layers')
        im4 = l2b.imshow(layer_pars['l2.bias'].detach().numpy().squeeze().transpose(),
                         norm=matplotlib.colors.SymLogNorm(linthresh=0.01, linscale=0.03, vmin=-1.0, vmax=1.0),
                         aspect='auto')

        divider = make_axes_locatable(l2b)
        cax = divider.append_axes('bottom', size='20%', pad=0.25)
        fig.colorbar(im4, cax=cax, orientation='horizontal')

    fig.show()

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

    seed = args.seed
    print('Setting the random seed to {:}'.format(seed))

    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

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

    inflation_factor = args.inflation_factor  # for the up-down models (i.e., how much larger is the internal state; default is 5)
    use_particle_rnn_mode = args.use_particle_rnn_mode
    use_particle_free_rnn_mode = args.use_particle_free_rnn_mode

    use_analytic_solution = True # True is the proper setting here for models that have analytic solutions implemented
    write_out_first_five_gradients = False # for debugging purposes; use jointly with check_gradient_over_iterations.py
    use_fixed_sample_batch = write_out_first_five_gradients # has to be set to True if we want to compare autodiff and analytic gradients (as otherwise there will be different random initializations

    if write_out_first_five_gradients and not use_fixed_sample_batch:
        print('WARNING: if you want to compare autodiff/analytic gradient then use_fixed_sample_batch should be set to True')

    if args.shooting_model == 'simple':
        smodel = smodels.AutoShootingIntegrandModelSimple(**shootingintegrand_kwargs,use_analytic_solution=use_analytic_solution, use_rnn_mode=use_particle_rnn_mode)
    elif args.shooting_model == '2nd_order':
        smodel = smodels.AutoShootingIntegrandModelSecondOrder(**shootingintegrand_kwargs, use_rnn_mode=use_particle_rnn_mode)
    elif args.shooting_model == 'updown':
        smodel = smodels.AutoShootingIntegrandModelUpDown(**shootingintegrand_kwargs,use_analytic_solution=use_analytic_solution, inflation_factor=inflation_factor, use_rnn_mode=use_particle_rnn_mode)
    elif args.shooting_model == 'dampened_updown':
        smodel = smodels.AutoShootingIntegrandModelDampenedUpDown(**shootingintegrand_kwargs, inflation_factor=inflation_factor, use_rnn_mode=use_particle_rnn_mode)
    elif args.shooting_model == 'universal':
        smodel = smodels.AutoShootingIntegrandModelUniversal(**shootingintegrand_kwargs,use_analytic_solution=use_analytic_solution, inflation_factor=inflation_factor, use_rnn_mode=use_particle_rnn_mode)

    block_name = 'sblock'

    sblock = sblocks.ShootingBlockBase(
        name=block_name,
        shooting_integrand=smodel,
        integrator_name=args.method,
        use_particle_free_rnn_mode=use_particle_free_rnn_mode,
        integrator_options = {'step_size': args.stepsize}
    )

    #sblock.to(device)

    use_shooting = False
    if args.use_rnn:
        weight_decay = 0.0001
        lr = 1e-3
        print('Using ResNetRNN: weight = {}'.format(weight_decay))
        simple_resnet = sdnn.ResNetRNN(nr_of_layers=args.nr_of_layers, inflation_factor=inflation_factor)
    elif args.use_double_resnet_rnn:
        weight_decay = 0.025
        lr = 1e-3
        print('Using DoubleResNetRNN: weight = {}'.format(weight_decay))
        simple_resnet = sdnn.DoubleResNetUpDownRNN(nr_of_layers=args.nr_of_layers, inflation_factor=inflation_factor)
    elif args.use_updown:
        weight_decay = 0.01
        lr = 1e-3
        print('Using ResNetUpDown: weight = {}'.format(weight_decay))
        simple_resnet = sdnn.ResNetUpDown(nr_of_layers=args.nr_of_layers, inflation_factor=inflation_factor)
    elif args.use_simple_resnet:
        weight_decay = 0.0001
        lr = 1e-2
        print('Using ResNet: weight = {}'.format(weight_decay))
        simple_resnet = sdnn.ResNet(nr_of_layers=args.nr_of_layers)
    elif args.use_double_resnet:
        #weight_decay = 0.025
        weight_decay = 0.01
        lr = 1e-2
        print('Using DoubleResNetUpDown: weight = {}'.format(weight_decay))
        simple_resnet = sdnn.DoubleResNetUpDown(nr_of_layers=args.nr_of_layers, inflation_factor=inflation_factor)
    elif args.use_neural_ode:
        print('Using neural ode')
        func = sdnn.ODESimpleFunc()
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
            optimizer = optim.RMSprop(func.parameters(), lr=1e-2)
        else:
            optimizer = optim.Adam(simple_resnet.parameters(), lr=lr, weight_decay=weight_decay)
    else:

        sample_batch_in, sample_batch_out = get_sample_batch(nr_of_samples=args.batch_size)
        if use_fixed_sample_batch:
            fixed_batch_in, fixed_batch_out = get_sample_batch(nr_of_samples=args.batch_size)

        sblock(x=sample_batch_in)

        # do some parameter freezing
        for pn,pp in sblock.named_parameters():
            if pn=='q1':
                # freeze the locations
                #pp.requires_grad = False
                pass
        optimizer = optim.Adam(sblock.parameters(), lr=1e-2)

    # fig = plt.figure(figsize=(12, 4), facecolor='white')
    # ax = fig.add_subplot(111, frameon=False)
    # plt.show(block=False)

    track_loss = []

    if write_out_first_five_gradients:
        max_iter = 5
    else:
        max_iter = args.niters

    for itr in range(1, max_iter + 1):

        # get current batch data
        if use_fixed_sample_batch:
            batch_in = fixed_batch_in
            batch_out = fixed_batch_out
        else:
            batch_in, batch_out = get_sample_batch(nr_of_samples=args.batch_size)

        optimizer.zero_grad()

        if not use_shooting:
            if args.use_neural_ode:

                pred_y = integrator.integrate(func=func, x0=batch_in, t=torch.tensor([0, 1]).float())
                pred_y = pred_y[1, :, :, :] # need prediction at time 1

            elif args.use_double_resnet or args.use_double_resnet_rnn:
                x20 = torch.zeros_like(batch_in)
                sz = [1] * len(x20.shape)
                sz[-1] = inflation_factor
                x20 = x20.repeat(sz)
                x1x2 = (batch_in, x20)
                pred_y, pred_y2 = simple_resnet(x1x2)
            else:
                pred_y = simple_resnet(x=batch_in)
        else:
            # set integration time
            # sblock.set_integration_time(time_to=1.0) # try to do this mapping in unit time
            pred_y, _, _, _ = sblock(x=batch_in)

        if use_shooting:
            loss = torch.mean((pred_y - batch_out)**2)
            if args.use_parameter_penalty_energy:
                loss = loss + sblock.get_norm_penalty()
        else:
            loss = torch.mean((pred_y - batch_out)**2)

        loss.backward()

        if write_out_first_five_gradients:
            # save gradient
            grad_dict = dict()
            for n,v in sblock.named_parameters():
                grad_dict[n] = v
                grad_dict['{}_grad'.format(n)] = v.grad

            torch.save(grad_dict,'grad_iter_{}_analytic_{}.pt'.format(itr,use_analytic_solution))

        track_loss.append(loss.item())
        optimizer.step()

        if itr % args.test_freq == 0:
            fig = plt.figure(figsize=(8, 12), facecolor='white')
            ax = fig.add_subplot(111, frameon=False)
            #ax.cla()
            ax.plot(batch_in.detach().numpy().squeeze(),batch_out.detach().numpy().squeeze(),'g+')
            ax.plot(batch_in.detach().numpy().squeeze(),pred_y.detach().numpy().squeeze(),'r*')
            ax.set_xlim(-2, 2)
            ax.set_ylim(-8, 8)
            plt.draw()
            #plt.pause(0.001)
            plt.show()
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

    # now print the paramerers
    if not use_shooting:
        if args.use_neural_ode:
            compute_number_of_parameters_and_print_all_parameters(func)
        else:
            compute_number_of_parameters_and_print_all_parameters(simple_resnet)

            if args.use_double_resnet:
                collect_and_sort_parameter_values_across_layers(simple_resnet)
    else:
        #print_all_parameters(sblock)
        compute_number_of_parameters_and_print_all_parameters(sblock)


    if use_shooting:
        custom_hook_data = defaultdict(list)
        hook = sblock.shooting_integrand.register_lagrangian_gradient_hook(parameters_hook)
        sblock.shooting_integrand.set_custom_hook_data(data=custom_hook_data)

        uniform_batch_in, uniform_batch_out = get_uniform_sample_batch(nr_of_samples=args.batch_size)
        pred_y, _, _, _ = sblock(x=uniform_batch_in)
        hook.remove()

        plot_temporal_data(data=custom_hook_data, block_name=block_name)
