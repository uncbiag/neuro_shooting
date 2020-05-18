# the goal of this model is to map the input [-2,2] to a desired functional output
# for simplicity we will try to do this with a very simple 1D shooting model

import os
import sys
import time
import random
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

try:
    from tqdm import trange
    has_TQDM = True
    range_command = trange
except:
    has_TQDM = False
    print('If you want to display progress bars install TQDM; conda install tqdm')
    range_command = range

from collections import OrderedDict
from collections import defaultdict

import torch.optim as optim
from torchdiffeq import odeint

import neuro_shooting
import neuro_shooting.shooting_blocks as sblocks
import neuro_shooting.shooting_models as smodels
import neuro_shooting.generic_integrator as gi
import neuro_shooting.parameter_initialization as pi
import neuro_shooting.shooting_hooks as sh
import neuro_shooting.vector_visualization as vector_visualization
import neuro_shooting.validation_measures as validation_measures
import neuro_shooting.utils as utils
import neuro_shooting.figure_settings as figure_settings

import simple_discrete_neural_networks as sdnn

import matplotlib.cm as cm
import matplotlib.pyplot as plt
figure_settings.setup_plotting()


import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0, '../neuro_shooting')


def setup_cmdline_parsing():
    parser = argparse.ArgumentParser('Simple functional mapping')
    parser.add_argument('--method', type=str, choices=['dopri5', 'adams', 'rk4'], default='rk4', help='Selects the desired integrator')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--niters', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--adjoint', action='store_true', help='Use adjoint integrator to avoid storing values during forward pass.')
    parser.add_argument('--gpu', type=int, default=0, help='Enable GPU computation on specified GPU.')
    parser.add_argument('--stepsize', type=float, default=0.1, help='Step size for the integrator (if not adaptive).')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate for optimizer')

    # shooting model parameters
    parser.add_argument('--shooting_model', type=str, default='updown', choices=['univeral','dampened_updown','simple', '2nd_order', 'updown'])
    parser.add_argument('--nonlinearity', type=str, default='relu', choices=['identity', 'relu', 'tanh', 'sigmoid',"softmax"], help='Nonlinearity for shooting.')
    parser.add_argument('--pw', type=float, default=1.0, help='parameter weight')
    parser.add_argument('--sim_weight', type=float, default=100.0, help='Weight for the similarity measure')
    parser.add_argument('--norm_weight', type=float, default=0.01, help='Weight for the similarity measure')
    parser.add_argument('--nr_of_particles', type=int, default=10, help='Number of particles to parameterize the initial condition')
    parser.add_argument('--inflation_factor', type=int, default=10, help='Multiplier for state dimension for updown shooting model types')
    parser.add_argument('--use_particle_rnn_mode', action='store_true', help='When set then parameters are only computed at the initial time and used for the entire evolution; mimicks a particle-based RNN model.')
    parser.add_argument('--use_particle_free_rnn_mode', action='store_true', help='This is directly optimizing over the parameters -- no particles here; a la Neural ODE')
    parser.add_argument('--do_not_use_parameter_penalty_energy', action='store_true', default=False)

    parser.add_argument('--xrange', type=float, default=1.5, help='Desired range in x direction')
    parser.add_argument('--clamp_range', action='store_true', help='Clamps the range of the q1 particles to [-xrange,xrange]')

    parser.add_argument('--optimize_over_data_initial_conditions', action='store_true', default=False)
    parser.add_argument('--optimize_over_data_initial_conditions_type', type=str,
                        choices=['direct', 'linear', 'mini_nn'], default='linear',
                        help='Different ways to predict the initial conditions for higher order models. Currently only supported for updown model.')

    parser.add_argument('--custom_parameter_freezing', action='store_true', default=False,
                        help='Enable custom code for parameter freezing -- development mode')
    parser.add_argument('--unfreeze_parameters_at_iter', type=int, default=-1, help='Allows unfreezing parameters later during the iterations')
    parser.add_argument('--custom_parameter_initialization', action='store_true', default=False,
                        help='Enable custom code for parameter initialization -- development mode')

    # non-shooting networks implemented
    parser.add_argument('--nr_of_layers', type=int, default=30, help='Number of layers for the non-shooting networks')
    parser.add_argument('--use_updown',action='store_true')
    parser.add_argument('--use_double_resnet',action='store_true')
    parser.add_argument('--use_rnn',action='store_true')
    parser.add_argument('--use_double_resnet_rnn',action="store_true")
    parser.add_argument('--use_simple_resnet',action='store_true')
    parser.add_argument('--use_neural_ode',action='store_true')

    parser.add_argument('--test_freq', type=int, default=10)
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--verbose', action='store_true', default=False)

    parser.add_argument('--save_figures', action='store_true', help='If specified figures are saved (in current output directory) instead of displayed')
    parser.add_argument('--output_directory', type=str, default='results_simple_functional_mapping', help='Directory in which the results are saved')

    args = parser.parse_args()

    return args

# --shooting_model updown --nr_of_particles 5 --pw 0.1 --viz --niters 100 --nonlinearity relu

def setup_optimizer_and_scheduler(params,lr=0.1, weight_decay=0):

    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,verbose=True)

    return optimizer, scheduler

def get_sample_batch(nr_of_samples=10,max_x=1):

    sample_batch_in = (2*max_x)*torch.rand([nr_of_samples,1,1])-max_x # creates uniform samples in [-2,2]
    sample_batch_out = sample_batch_in**3 # and takes them to the power of three for the output

    return sample_batch_in, sample_batch_out

def get_uniform_sample_batch(nr_of_samples=10,max_x=1):

    sample_batch_in = (torch.linspace(-max_x,max_x,nr_of_samples)).view([nr_of_samples,1,1]) # creates uniform samples in [-2,2]
    sample_batch_out = sample_batch_in**3 # and takes them to the power of three for the output

    return sample_batch_in, sample_batch_out

class FunctionDataset(Dataset):
    def __init__(self, nr_of_samples, uniform_sample=False, max_x=1):
        """
        Args:

        :param nr_of_samples: number of samples to create for this dataset
        :param uniform_sample: if set to False input samples will be random, otherwise they will be uniform in [-2,2]
        """
        self.nr_of_samples = nr_of_samples
        # create these samples, we create them once and for all
        if uniform_sample:
            self.input, self.output = get_uniform_sample_batch(nr_of_samples=nr_of_samples, max_x=max_x)
        else:
            self.input, self.output = get_sample_batch(nr_of_samples=nr_of_samples, max_x=max_x)

    def __len__(self):
        return self.nr_of_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'input': self.input[idx,...], 'output': self.output[idx, ...]}

        return sample

def get_function_data_loaders(training_samples=2000,training_evaluation_samples=1000, testing_samples=1000, visualization_samples=50, batch_size=100, num_workers=0, max_x=1.0):

    train_loader = DataLoader(
        FunctionDataset(nr_of_samples=training_samples,uniform_sample=False, max_x=max_x), batch_size=batch_size,
        shuffle=True, num_workers=num_workers, drop_last=True
    )

    train_eval_loader = DataLoader(
        FunctionDataset(nr_of_samples=training_evaluation_samples,uniform_sample=False, max_x=max_x),
        batch_size=training_evaluation_samples, shuffle=False, num_workers=num_workers, drop_last=True
    )

    test_loader = DataLoader(
        FunctionDataset(nr_of_samples=testing_samples,uniform_sample=True, max_x=max_x),
        batch_size=testing_samples, shuffle=False, num_workers=num_workers, drop_last=True
    )

    visualization_loader = DataLoader(
        FunctionDataset(nr_of_samples=visualization_samples, uniform_sample=True, max_x=max_x),
        batch_size=visualization_samples, shuffle=False, num_workers=num_workers, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader, visualization_loader


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

def compute_loss(args, use_shooting, integrator, sblock, func, batch_in, batch_out):
    if not use_shooting:
        if args.use_neural_ode:

            pred_y = integrator.integrate(func=func, x0=batch_in, t=torch.tensor([0, 1]).float())
            pred_y = pred_y[1, :, :, :]  # need prediction at time 1

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
        sim_loss = args.sim_weight * torch.mean((pred_y - batch_out) ** 2)

        norm_penalty = sblock.get_norm_penalty()

        if args.do_not_use_parameter_penalty_energy:
            norm_loss = torch.tensor([0])
        else:
            norm_loss = args.norm_weight * norm_penalty

        loss = sim_loss + norm_loss

    else:
        sim_loss = args.sim_weight*torch.mean((pred_y - batch_out) ** 2)
        norm_penalty = torch.tensor([0])
        norm_loss = args.norm_weight * norm_penalty
        loss = sim_loss

    return loss, sim_loss, norm_loss, norm_penalty, pred_y,


def show_figure(args, batch_in, batch_out, pred_y, use_shooting, sblock):

    fig = plt.figure(figsize=(8, int(np.ceil(8 * (args.xrange ** 3) / args.xrange))), facecolor='white')
    ax = fig.add_subplot(111, frameon=True)
    ax.set_aspect('equal')
    # ax.cla()
    ax.plot(batch_in.detach().cpu().numpy().squeeze(), batch_out.detach().cpu().numpy().squeeze(), 'g+')
    ax.plot(batch_in.detach().cpu().numpy().squeeze(), pred_y.detach().cpu().numpy().squeeze(), 'r*')
    ax.set_xlim(-args.xrange, args.xrange)
    ax.set_ylim(-args.xrange ** 3, args.xrange ** 3)

    if use_shooting:
        sd = sblock.state_dict()
        q1 = sd['q1'].detach().cpu().numpy().squeeze()

        for v in q1:
            ax.plot([v, v], [-args.xrange ** 3, args.xrange ** 3], 'k-')

    figure_settings.set_font_size_for_axis(ax, fontsize=20)
    plt.show()

if __name__ == '__main__':

    args = setup_cmdline_parsing()
    if args.verbose:
        print(args)

    # takes care of the GPU setup
    utils.setup_random_seed(seed=args.seed)
    utils.setup_device(desired_gpu=args.gpu)

    # create the data immediately after setting the random seed to make sure it is always consistent across the experiments
    train_loader, test_loader, train_eval_loader, visualization_loader = get_function_data_loaders(batch_size=args.batch_size, max_x=args.xrange)

    par_init = pi.VectorEvolutionSampleBatchParameterInitializer(
        only_random_initialization=True,
        random_initialization_magnitude=0.1, 
        sample_batch = torch.linspace(-args.xrange,args.xrange,args.nr_of_particles))


    shootingintegrand_kwargs = {'in_features': 1,
                                'nonlinearity': args.nonlinearity,
                                'nr_of_particles': args.nr_of_particles,
                                'parameter_weight': args.pw,
                                'particle_dimension': 1,
                                'particle_size': 1,
                                'costate_initializer':pi.VectorEvolutionParameterInitializer(random_initialization_magnitude=0.1),
                                'optimize_over_data_initial_conditions': args.optimize_over_data_initial_conditions,
                                'optimize_over_data_initial_conditions_type': args.optimize_over_data_initial_conditions_type}

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

    use_shooting = False
    integrator = None
    func = None

    if args.use_rnn:
        weight_decay = 0.0001
        print('Using ResNetRNN: weight = {}'.format(weight_decay))
        simple_resnet = sdnn.ResNetRNN(nr_of_layers=args.nr_of_layers, inflation_factor=inflation_factor)
    elif args.use_double_resnet_rnn:
        weight_decay = 0.025
        print('Using DoubleResNetRNN: weight = {}'.format(weight_decay))
        simple_resnet = sdnn.DoubleResNetUpDownRNN(nr_of_layers=args.nr_of_layers, inflation_factor=inflation_factor)
    elif args.use_updown:
        weight_decay = 0.01
        print('Using ResNetUpDown: weight = {}'.format(weight_decay))
        simple_resnet = sdnn.ResNetUpDown(nr_of_layers=args.nr_of_layers, inflation_factor=inflation_factor)
    elif args.use_simple_resnet:
        weight_decay = 0.0001
        print('Using ResNet: weight = {}'.format(weight_decay))
        simple_resnet = sdnn.ResNet(nr_of_layers=args.nr_of_layers)
    elif args.use_double_resnet:
        #weight_decay = 0.025
        weight_decay = 0.01
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
            params = func.parameters()
        else:
            params = simple_resnet.parameters()
    else:

        sample_batch_in, sample_batch_out = get_sample_batch(nr_of_samples=args.batch_size, max_x=args.xrange)
        if use_fixed_sample_batch:
            fixed_batch_in, fixed_batch_out = get_sample_batch(nr_of_samples=args.batch_size, max_x=args.xrange)

        sblock(x=sample_batch_in)

        # custom initialization
        if args.custom_parameter_initialization:
            # first get the state dictionary of the shooting block which contains all parameters
            ss_sd = sblock.state_dict()

            # get initial positions on the trajectory (to place the particles there)
            init_q1, _ = get_uniform_sample_batch(nr_of_samples=args.nr_of_particles, max_x=args.xrange)
            with torch.no_grad():
                ss_sd['q1'].copy_(init_q1)

            # init.uniform_(shooting_block.state_dict()['q1'],-2,2) # just for initialization experiments, not needed

        uses_particles = not args.use_particle_free_rnn_mode
        if uses_particles:
            if args.custom_parameter_freezing:
                utils.freeze_parameters(sblock, ['q1'])

        params = sblock.parameters()
        weight_decay = 0.0

    # create optimizer
    optimizer, scheduler = setup_optimizer_and_scheduler(params=params, lr=args.lr, weight_decay=weight_decay)

    # now iterate over the data

    track_loss = []

    if write_out_first_five_gradients:
        max_iter = 5
        current_itr = 0

    for itr in range_command(0, args.niters + 1): # number of epochs

        if uses_particles and args.custom_parameter_freezing:
            if itr==args.unfreeze_parameters_at_iter:
                utils.unfreeze_parameters(sblock, ['q1'])

        # now iterate over the entire training dataset
        for itr_batch, sampled_batch in enumerate(train_loader):

            # get current batch data
            if use_fixed_sample_batch:
                # force it back to the fixed value; this is just for debugging!
                batch_in = fixed_batch_in
                batch_out = fixed_batch_out
            else:
                batch_in, batch_out = sampled_batch['input'], sampled_batch['output']

            optimizer.zero_grad()

            loss, sim_loss, norm_loss, norm_penalty, pred_y = compute_loss(args=args, use_shooting=use_shooting, integrator=integrator,
                                                                   sblock=sblock, func=func, batch_in=batch_in, batch_out=batch_out)

            loss.backward()

            if write_out_first_five_gradients:
                # save gradient
                grad_dict = dict()
                for n,v in sblock.named_parameters():
                    grad_dict[n] = v
                    grad_dict['{}_grad'.format(n)] = v.grad

                torch.save(grad_dict,'grad_iter_{}_analytic_{}.pt'.format(current_itr,use_analytic_solution))
                current_itr += 1
                if current_itr>=max_iter:
                    exit() # end, this is just for debugging


            track_loss.append(loss.item())
            optimizer.step()

            if args.clamp_range:
                if use_shooting:
                    # clip q1 if needed
                    sd = sblock.state_dict()
                    q1 = sd['q1']
                    q1.data.clamp_(min=-args.xrange, max=args.xrange)
                    print('q1 = {}'.format(q1.detach().cpu().numpy()))

        # now compute testing evaluation loss
        if itr % args.test_freq == 0:

            with torch.no_grad():
                dataiter = iter(train_eval_loader)
                train_eval_batch = dataiter.next()
                batch_in, batch_out = train_eval_batch['input'], train_eval_batch['output']

                loss, sim_loss, norm_loss, norm_penalty, pred_y = compute_loss(args=args, use_shooting=use_shooting,
                                                                               integrator=integrator,
                                                                               sblock=sblock, func=func,
                                                                               batch_in=batch_in,
                                                                               batch_out=batch_out)

                try:
                    scheduler.step()
                except:
                    scheduler.step(loss)

                if args.viz:
                    show_figure(args=args, batch_in=batch_in, batch_out=batch_out, pred_y=pred_y, use_shooting=use_shooting, sblock=sblock)

                if use_shooting:
                    print('\nIter {:04d} | Loss {:.4f}; sim_loss = {:.4f}; norm_loss = {:.4f}; par_norm = {:.4f}'.format(itr, loss.item(), sim_loss.item(), norm_loss.item(), norm_penalty.item()))
                else:
                    print('Iter {:04d} | Loss {:.6f}'.format(itr, loss.item()))

    # now print the parameters
    if not use_shooting:
        if args.use_neural_ode:
            utils.compute_number_of_parameters(func,print_parameters=True)
        else:
            utils.compute_number_of_parameters(simple_resnet,print_parameters=True)

            if args.use_double_resnet:
                collect_and_sort_parameter_values_across_layers(simple_resnet)
    else:
        utils.compute_number_of_parameters(sblock,print_parameters=True)

    # get the data for final testing, there will only be one batch
    dataiter = iter(test_loader)
    test_batch = dataiter.next()
    uniform_batch_in, uniform_batch_out = test_batch['input'], test_batch['output']

    # for visualization
    dataiter = iter(visualization_loader)
    test_batch = dataiter.next()
    visualization_batch_in, visualization_batch_out = test_batch['input'], test_batch['output']

    if use_shooting:

        # get measures
        custom_hook_data = defaultdict(list)
        hook = sblock.shooting_integrand.register_lagrangian_gradient_hook(sh.parameter_evolution_hook)
        sblock.shooting_integrand.set_custom_hook_data(data=custom_hook_data)

        loss, sim_loss, norm_loss, norm_penalty, pred_y = compute_loss(args=args, use_shooting=use_shooting,
                                                                       integrator=integrator,
                                                                       sblock=sblock, func=func,
                                                                       batch_in=uniform_batch_in,
                                                                       batch_out=uniform_batch_out)
        hook.remove()

        log_complexity_measures = validation_measures.compute_complexity_measures(data=custom_hook_data)
        print('Validation measures = {}'.format(log_complexity_measures))

        # now repeat this for visualization

        # get measures
        custom_hook_data = defaultdict(list)
        hook = sblock.shooting_integrand.register_lagrangian_gradient_hook(sh.parameter_evolution_hook)
        sblock.shooting_integrand.set_custom_hook_data(data=custom_hook_data)

        loss, sim_loss, norm_loss, norm_penalty, pred_y = compute_loss(args=args, use_shooting=use_shooting,
                                                                       integrator=integrator,
                                                                       sblock=sblock, func=func,
                                                                       batch_in=visualization_batch_in,
                                                                       batch_out=visualization_batch_out)
        hook.remove()

        vector_visualization.plot_temporal_data(data=custom_hook_data, block_name=block_name)
