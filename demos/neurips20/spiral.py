import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.init as init
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

from collections import defaultdict

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.shooting_models as shooting_models
import neuro_shooting.generic_integrator as generic_integrator
import neuro_shooting.shooting_hooks as sh
import neuro_shooting.vector_visualization as vector_visualization
import neuro_shooting.validation_measures as validation_measures
import neuro_shooting.utils as utils

# Setup

def setup_cmdline_parsing():
    # Command line arguments

    parser = argparse.ArgumentParser('ODE demo')
    parser.add_argument('--method', type=str, choices=['dopri5', 'adams','rk4'], default='rk4', help='Selects the desired integrator')
    parser.add_argument('--stepsize', type=float, default=0.05, help='Step size for the integrator (if not adaptive).')
    parser.add_argument('--data_size', type=int, default=200, help='Length of the simulated data that should be matched.')
    parser.add_argument('--batch_time', type=int, default=5, help='Length of the training samples.')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of training samples.')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate for optimizer')
    parser.add_argument('--niters', type=int, default=2000, help='Maximum nunber of iterations.')
    parser.add_argument('--batch_validation_size', type=int, default=25, help='Length of the samples for validation.')
    parser.add_argument('--seed', required=False, type=int, default=-1,
                        help='Sets the random seed which affects data shuffling')

    # experimental
    parser.add_argument('--optional_weight', type=float, default=10.0,
                        help='Optional weight (meaning is model dependent) that we can use to sweep across additional model-specific settings.')

    parser.add_argument('--linear', action='store_true', help='If specified the ground truth system will be linear, otherwise nonlinear.')

    parser.add_argument('--test_freq', type=int, default=100, help='Frequency with which the validation measures are to be computed.')
    parser.add_argument('--viz_freq', type=int, default=100, help='Frequency with which the results should be visualized; if --viz is set.')

    parser.add_argument('--validate_with_long_range', action='store_true', help='If selected, a long-range trajectory will be used; otherwise uses batches as for training')
    parser.add_argument('--chunk_time', type=int, default=5, help='For a long range valdation solution chunks the solution together in these pieces.')

    parser.add_argument('--shooting_model', type=str, default='updown_universal', choices=['updown_universal', 'simple', 'updown', 'periodic'])
    parser.add_argument('--nr_of_particles', type=int, default=25, help='Number of particles to parameterize the initial condition')
    parser.add_argument('--pw', type=float, default=1.0, help='Parameter weight; controls the weight internally for the shooting equations; probably best left at zero and to control the weight with --sim_weight and --norm_weight.')
    parser.add_argument('--sim_weight', type=float, default=100.0, help='Weight for the similarity measure')
    parser.add_argument('--norm_weight', type=float, default=0.01, help='Weight for the similarity measure')
    parser.add_argument('--sim_norm', type=str, choices=['l1','l2'], default='l2', help='Norm for the similarity measure.')
    parser.add_argument('--nonlinearity', type=str, choices=['identity', 'relu', 'tanh', 'sigmoid'], default='relu', help='Nonlinearity for shooting.')

    parser.add_argument('--inflation_factor', type=int, default=5,
                        help='Multiplier for state dimension for updown shooting model types')
    parser.add_argument('--use_particle_rnn_mode', action='store_true',
                        help='When set then parameters are only computed at the initial time and used for the entire evolution; mimicks a particle-based RNN model.')
    parser.add_argument('--use_particle_free_rnn_mode', action='store_true',
                        help='This is directly optimizing over the parameters -- no particles here; a la Neural ODE')
    parser.add_argument('--use_particle_free_time_dependent_mode', action='store_true',
                        help='This is directly optimizing over time-dependent parameters -- no particles here; a la Neural ODE')
    parser.add_argument('--nr_of_particle_free_time_dependent_steps', type=int, default=10, help='Number of parameter sets(!) for the time-dependent particle-free mode')

    parser.add_argument('--do_not_use_parameter_penalty_energy', action='store_true', default=False)
    parser.add_argument('--optimize_over_data_initial_conditions', action='store_true', default=False)
    parser.add_argument('--optimize_over_data_initial_conditions_type', type=str, choices=['direct','linear','mini_nn'], default='linear', help='Different ways to predict the initial conditions for higher order models. Currently only supported for updown model.')


    parser.add_argument('--disable_distance_based_sampling', action='store_true', default=False, help='If specified uses the original trajectory sampling, otherwise samples based on trajectory length.')

    parser.add_argument('--custom_parameter_freezing', action='store_true', default=False, help='Enable custom code for parameter freezing -- development mode')
    parser.add_argument('--unfreeze_parameters_at_iter', type=int, default=-1, help='Allows unfreezing parameters later during the iterations')
    parser.add_argument('--custom_parameter_initialization', action='store_true', default=False, help='Enable custom code for parameter initialization -- development mode')

    parser.add_argument('--viz', action='store_true', help='Enable visualization.')
    parser.add_argument('--gpu', type=int, default=0, help='Enable GPU computation on specified GPU.')
    parser.add_argument('--adjoint', action='store_true', help='Use adjoint integrator to avoid storing values during forward pass.')

    parser.add_argument('--checkpointing_time_interval', type=float, default=0.0, help='If specified puts a checkpoint after every interval (hence dynamically changes with the integration time). If a fixed number is deisred use --nr_of_checkpoints instead.')
    parser.add_argument('--nr_of_checkpoints', type=int, default=0, help='If specified will add that many checkpoints for integration. If integration times differ it is more convenient to set --checkpointing_time_interval instead.')

    parser.add_argument('--do_not_use_analytic_solution', action='store_true', help='Use adjoint integrator to avoid storing values during forward pass.')

    parser.add_argument('--create_animation', action='store_true', help='Creates animated gif for the evolution of the particles.')

    parser.add_argument('--save_figures', action='store_true',
                        help='If specified figures are saved (in current output directory) instead of displayed')
    parser.add_argument('--output_directory', type=str, default='results_spiral',
                        help='Directory in which the results are saved')
    parser.add_argument('--output_basename', type=str, default='spiral', help='Base name for the resulting figures.')

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

def setup_shooting_block(integrator=None, shooting_model='updown', parameter_weight=1.0, nr_of_particles=10,
                         inflation_factor=2, nonlinearity='relu',
                         use_particle_rnn_mode=False, use_particle_free_rnn_mode=False, use_particle_free_time_dependent_mode=False,
                         nr_of_particle_free_time_dependent_steps=5,
                         optimize_over_data_initial_conditions=False,
                         optimize_over_data_initial_conditions_type='linear',
                         use_analytic_solution=True,
                         optional_weight=10,
                         max_integraton_time=1.0):

    if shooting_model=='updown':
        smodel = shooting_models.AutoShootingIntegrandModelUpDown(in_features=2, nonlinearity=nonlinearity,
                                                                  parameter_weight=parameter_weight,
                                                                  inflation_factor=inflation_factor,
                                                                  nr_of_particles=nr_of_particles, particle_dimension=1,
                                                                  particle_size=2,
                                                                  use_analytic_solution=use_analytic_solution,
                                                                  use_rnn_mode=use_particle_rnn_mode,
                                                                  optimize_over_data_initial_conditions=optimize_over_data_initial_conditions,
                                                                  optimize_over_data_initial_conditions_type=optimize_over_data_initial_conditions_type)
    elif shooting_model == 'updown_universal':
        smodel = shooting_models.AutoShootingIntegrandModelUpDownUniversal(in_features=2, nonlinearity=nonlinearity,
                                                                           parameter_weight=parameter_weight,
                                                                           inflation_factor=inflation_factor,
                                                                           nr_of_particles=nr_of_particles, particle_dimension=1,
                                                                           particle_size=2,
                                                                           use_analytic_solution=use_analytic_solution,
                                                                           use_rnn_mode=use_particle_rnn_mode,
                                                                           optimize_over_data_initial_conditions=optimize_over_data_initial_conditions,
                                                                           optimize_over_data_initial_conditions_type=optimize_over_data_initial_conditions_type,
                                                                           optional_weight=optional_weight)
    elif shooting_model=='periodic':
        smodel = shooting_models.AutoShootingIntegrandModelUpdownPeriodic(in_features=2, nonlinearity=nonlinearity,
                                                                  parameter_weight=parameter_weight,
                                                                  inflation_factor=inflation_factor,
                                                                  nr_of_particles=nr_of_particles, particle_dimension=1,
                                                                  particle_size=2,
                                                                  use_analytic_solution=use_analytic_solution,
                                                                  use_rnn_mode=use_particle_rnn_mode,
                                                                  optimize_over_data_initial_conditions=optimize_over_data_initial_conditions,
                                                                  optimize_over_data_initial_conditions_type=optimize_over_data_initial_conditions_type)
    elif shooting_model=='simple':
        smodel = shooting_models.AutoShootingIntegrandModelSimple(in_features=2, nonlinearity=nonlinearity,
                                                                  parameter_weight=parameter_weight,
                                                                  nr_of_particles=nr_of_particles, particle_dimension=1,
                                                                  particle_size=2,
                                                                  use_analytic_solution=True,
                                                                  use_rnn_mode=use_particle_rnn_mode)

    print('Using shooting model {}'.format(shooting_model))

    import neuro_shooting.parameter_initialization as pi
    # par_initializer = pi.VectorEvolutionSampleBatchParameterInitializer(only_random_initialization=False,
    #     random_initialization_magnitude=0.1,
    #     sample_batch=batch_y0)

    par_initializer = pi.VectorEvolutionParameterInitializer(only_random_initialization=True, random_initialization_magnitude=1.0)

    smodel.set_state_initializer(state_initializer=par_initializer)
    shooting_block = shooting_blocks.ShootingBlockBase(name='simple', shooting_integrand=smodel,
                                                       use_particle_free_rnn_mode=use_particle_free_rnn_mode,
                                                       use_particle_free_time_dependent_mode=use_particle_free_time_dependent_mode,
                                                       nr_of_particle_free_time_dependent_steps=nr_of_particle_free_time_dependent_steps,
                                                       integrator=integrator,
                                                       max_integration_time=max_integration_time)

    return shooting_block

def setup_optimizer_and_scheduler(params,lr=0.1):

    #optimizer = optim.Adam(params, lr=0.025)
    optimizer = optim.Adam(params, lr=lr)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,verbose=True)

    return optimizer, scheduler

# Defining the differential equation and data

class DiffEqRHS(nn.Module):
    """
    RHS of the differential equation:
    linear: \dot{y}^T = y^T A
    nonlinear: \dot{y}^T = (y**3)^T A
    """

    def __init__(self, A, linear=False):
        super(DiffEqRHS, self).__init__()
        self.A = A
        self.linear = linear

    def forward(self, t, y):
        if self.linear:
            return torch.mm(y, self.A)
        else:
            return torch.mm(y**3, self.A)


def create_uniform_distance_selection_array(data_dict, batch_time):
    # distances between successive data points
    dists = torch.sqrt(torch.sum((data_dict['y'][1:, 0, :] - data_dict['y'][0:-1, 0, :]) ** 2, dim=1))

    all_dists = torch.cat((dists,dists[-1::1])) # repeats the last value. We can use these as weights

    # remove the last batch_time minus 1 ones (as we need to be able to create trajectories at least of this size)
    dists = dists[0:-(batch_time - 1)]
    scaled_dists = (dists / dists.min()).detach().cpu().numpy().round().astype(np.int32)
    # create sampling array where each index has as many repetitions
    indices = []
    for i, sd in enumerate(scaled_dists):
        indices += [i]*sd

    indices = np.array(indices).astype(np.int64)
    return indices,all_dists

def generate_data(integrator, data_size, batch_time, linear=False):

    d = dict()

    d['y0'] = torch.tensor([[2., 0.]])
    d['t'] = torch.linspace(0., 10., data_size)
    d['A'] = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])

    # pure slow oscillation
    #d['A'] = torch.tensor([[0, 0.025], [-0.025, 0]])
    # small section
    #d['A'] = torch.tensor([[0, 0.1], [-0.1, 0]])

    with torch.no_grad():
        # integrate it
        d['y'] = integrator.integrate(func=DiffEqRHS(A=d['A'], linear=linear), x0=d['y0'], t=d['t'])

    d['uniform_sample_indices'],d['dists'] = create_uniform_distance_selection_array(d, batch_time=batch_time)

    return d


def get_batch(data_dict, batch_size, batch_time, distance_based_sampling=True):

    data_size = len(data_dict['t'])

    if distance_based_sampling:
        s = torch.from_numpy(np.random.choice(data_dict['uniform_sample_indices'], size=batch_size, replace=True))
    else:
        s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), size=batch_size, replace=True))

    batch_y0 = data_dict['y'][s]  # (M, D)
    batch_t = data_dict['t'][:batch_time]  # (T)
    batch_y = torch.stack([data_dict['y'][s + i] for i in range(batch_time)], dim=0)  # (T, M, D)

    return batch_y0, batch_t, batch_y

class FunctionDataset(Dataset):
    def __init__(self, args, data_dict, nr_of_samples, batch_time, distance_based_sampling=True, validation_mode=False,online_mode=False):
        """
        Args:

        :param nr_of_samples: number of samples to create for this dataset
        :param uniform_sample: if set to False input samples will be random, otherwise they will be uniform in [-2,2]
        """
        self.data_dict = data_dict
        self.validation_mode = validation_mode
        self.validate_with_long_range = args.validate_with_long_range
        self.online_mode = online_mode
        self.distance_based_sampling = distance_based_sampling
        self.batch_time = batch_time
        self.nr_of_samples = nr_of_samples

        if not online_mode:
            self.y0, self.t, self.y = self.create_data(nr_of_samples=nr_of_samples)
        else:
            self.y0 = None
            self.t = None
            self.y = None

    def create_data(self,nr_of_samples):
        if (self.validation_mode) and (self.validate_with_long_range):
            # just return the entire data trajectory
            y0 = self.data_dict['y0'].unsqueeze(dim=0)
            t = self.data_dict['t']
            y = self.data_dict['y'].unsqueeze(dim=1)
        else:
            self.nr_of_samples = nr_of_samples
            # create these samples, we create them once and for all

            y0, t, y = get_batch(data_dict=self.data_dict, batch_size=nr_of_samples,
                                 batch_time=self.batch_time, distance_based_sampling=self.distance_based_sampling)

        return y0,t,y

    def __len__(self):
        return self.nr_of_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not self.online_mode:
            sample = {'y0': self.y0[idx,...], 'y': self.y[:,idx,...]}
        else:
            if type(idx)==int:
                batch_size = 1
                eff_idx = 0
            elif type(idx)==list:
                batch_size = len(idx)
                eff_idx = list(range(0,batch_size))
            else:
                raise ValueError('Unknown index type {}'.format(type(idx)))

            y0, self.t, y = get_batch(data_dict=self.data_dict, batch_size=batch_size,
                                 batch_time=self.batch_time, distance_based_sampling=self.distance_based_sampling)
            sample = {'y0': y0[eff_idx, ...], 'y': y[:, eff_idx, ...]}

        return sample

def get_function_data_loaders(args, data_dict,
                              training_samples=100,training_evaluation_samples=100, short_range_testing_samples=1000, testing_samples=100,
                              num_workers=0):

    batch_time = args.batch_time
    batch_size = args.batch_size
    distance_based_sampling = not args.disable_distance_based_sampling

    if args.validate_with_long_range:
        testing_samples = 1

    train_loader = DataLoader(
        FunctionDataset(args=args,data_dict=data_dict, nr_of_samples=training_samples, batch_time=batch_time,
                        distance_based_sampling=distance_based_sampling,online_mode=True),
        batch_size=batch_size,shuffle=True, num_workers=num_workers, drop_last=True
    )

    train_eval_loader = DataLoader(
        FunctionDataset(args=args,data_dict=data_dict, nr_of_samples=training_evaluation_samples, batch_time=batch_time,
                        distance_based_sampling=distance_based_sampling,online_mode=False),
        batch_size=training_evaluation_samples, shuffle=False, num_workers=num_workers, drop_last=True
    )

    short_range_test_loader = DataLoader(
        FunctionDataset(args=args, data_dict=data_dict, nr_of_samples=short_range_testing_samples,
                        batch_time=batch_time,
                        distance_based_sampling=distance_based_sampling, online_mode=False),
        batch_size=short_range_testing_samples, shuffle=False, num_workers=num_workers, drop_last=True
    )

    test_loader = DataLoader(
        FunctionDataset(args=args,data_dict=data_dict, nr_of_samples=testing_samples, batch_time=batch_time,
                        distance_based_sampling=distance_based_sampling,validation_mode=True,online_mode=False),
        batch_size=testing_samples, shuffle=False, num_workers=num_workers, drop_last=True
    )

    return train_loader, train_eval_loader, short_range_test_loader, test_loader


def get_time_chunks(t, chunk_time, start_time_chunk_from_zero=True):
    time_chunks = []

    if chunk_time<=0:
        time_chunks.append(t)
    else:

        last_t = len(t)
        cur_idx = 0

        continue_chunking = True
        while continue_chunking:
            desired_idx = cur_idx+chunk_time
            if desired_idx>=last_t:
                desired_idx = last_t
                continue_chunking = False
            if start_time_chunk_from_zero:
                # if the equation is not time-dependent (as is the case for the spiral), we can just start the time
                # at the beginning. This will allow us to also run the direct time-dependent model
                current_chunk = t[0:(desired_idx-cur_idx)]
            else:
                current_chunk = t[cur_idx:desired_idx]
            time_chunks.append(current_chunk)
            cur_idx = desired_idx-1

    return time_chunks

def compute_validation_data(shooting_block, t, y0, validate_with_long_range, chunk_time, start_time_chunk_from_zero=True):

    if validate_with_long_range:

        sz = [len(t)] + list(y0.shape)
        val_pred_y = torch.zeros(sz,device=y0.device,dtype=y0.dtype)

        # now we chunk it
        cur_idx = 0
        time_chunks = get_time_chunks(t=t, chunk_time=chunk_time, start_time_chunk_from_zero=start_time_chunk_from_zero)
        for i,time_chunk in enumerate(time_chunks):
            shooting_block.set_integration_time_vector(integration_time_vector=time_chunk, suppress_warning=True)
            if i==0:
                cur_pred_y, _, _, _ = shooting_block(x=y0)
                current_norm_penalty = shooting_block.get_norm_penalty()
            else:
                cur_pred_y, _, _, _ = shooting_block(x=val_pred_y[cur_idx,...])

            val_pred_y[cur_idx:cur_idx + len(time_chunk), ...] = cur_pred_y
            cur_idx += len(time_chunk) - 1

    else:
        shooting_block.set_integration_time_vector(integration_time_vector=t, suppress_warning=True)
        val_pred_y, _, _, _ = shooting_block(x=y0)
        current_norm_penalty = shooting_block.get_norm_penalty()

    return val_pred_y, current_norm_penalty


def compute_loss(args, batch_y0, batch_t, batch_y, shooting_block, validation_mode=False):

    if validation_mode and args.validate_with_long_range:
        pred_y, norm_penalty = compute_validation_data(shooting_block=shooting_block, t=batch_t, y0=batch_y0,
                                                       validate_with_long_range=args.validate_with_long_range,
                                                       chunk_time=args.chunk_time)
    else:
        shooting_block.set_integration_time_vector(integration_time_vector=batch_t, suppress_warning=True)
        pred_y, _, _, _ = shooting_block(x=batch_y0)
        norm_penalty = shooting_block.get_norm_penalty()

    if args.sim_norm == 'l1':
        sim_loss = args.sim_weight * torch.mean(torch.abs(pred_y - batch_y))
    elif args.sim_norm == 'l2':
        sim_loss = args.sim_weight * torch.mean(torch.norm(pred_y - batch_y, dim=3))
    else:
        raise ValueError('Unknown norm {}.'.format(args.sim_norm))

    if args.do_not_use_parameter_penalty_energy:
        norm_loss = torch.tensor([0])
    else:
        norm_loss = args.norm_weight * norm_penalty

    loss = sim_loss + norm_loss

    return loss, sim_loss, norm_loss, norm_penalty, pred_y


if __name__ == '__main__':

    # do some initial setup
    args = setup_cmdline_parsing()

    # optional checkpointing support for integration
    if args.checkpointing_time_interval>0:
        checkpointing_time_interval = args.checkpointing_time_interval
    else:
        checkpointing_time_interval = None

    if args.nr_of_checkpoints>0:
        nr_of_checkpoints = args.nr_of_checkpoints
    else:
        nr_of_checkpoints = None

    # takes care of the GPU setup
    utils.setup_random_seed(seed=args.seed)
    utils.setup_device(desired_gpu=args.gpu)

    # setup the integrator
    integrator = setup_integrator(method=args.method, step_size=args.stepsize, use_adjoint=args.adjoint, nr_of_checkpoints=nr_of_checkpoints, checkpointing_time_interval=checkpointing_time_interval)

    # generate the true data that we want to match
    data = generate_data(integrator=integrator, data_size=args.data_size, batch_time=args.batch_time, linear=args.linear)

    # create the data immediately after setting the random seed to make sure it is always consistent across the experiments
    train_loader, train_eval_loader, short_range_test_loader, test_loader = get_function_data_loaders(args=args, data_dict=data)

    # draw an initial batch from it
    batch_y0, batch_t, batch_y = get_batch(data_dict=data, batch_time=args.batch_time, batch_size=args.batch_size,
                                           distance_based_sampling=not args.disable_distance_based_sampling)

    # determine the maximum integration time (important for proper setup of particle_free_time_dependent_mode
    max_integration_time = (torch.max(batch_t)).item()

    # create the shooting block
    shooting_block = setup_shooting_block(integrator=integrator,
                                          shooting_model=args.shooting_model,
                                          parameter_weight=args.pw,
                                          nr_of_particles=args.nr_of_particles,
                                          inflation_factor=args.inflation_factor,
                                          nonlinearity=args.nonlinearity,
                                          use_particle_rnn_mode=args.use_particle_rnn_mode,
                                          use_particle_free_rnn_mode=args.use_particle_free_rnn_mode,
                                          use_particle_free_time_dependent_mode=args.use_particle_free_time_dependent_mode,
                                          nr_of_particle_free_time_dependent_steps=args.nr_of_particle_free_time_dependent_steps,
                                          optimize_over_data_initial_conditions=args.optimize_over_data_initial_conditions,
                                          optimize_over_data_initial_conditions_type=args.optimize_over_data_initial_conditions_type,
                                          use_analytic_solution=not args.do_not_use_analytic_solution,
                                          optional_weight=args.optional_weight,
                                          max_integraton_time=max_integration_time)


    # run through the shooting block once (to get parameters as needed)
    shooting_block(x=batch_y)

    uses_particles = not (args.use_particle_free_rnn_mode or args.use_particle_free_time_dependent_mode)

    # custom initialization
    if args.custom_parameter_initialization:

        # first get the state dictionary of the shooting block which contains all parameters
        ss_sd = shooting_block.state_dict()

        # get initial positions on the trajectory (to place the particles there)
        init_q1, _, _ = get_batch(data_dict=data, batch_time=1, batch_size=args.nr_of_particles, distance_based_sampling=True)

        if uses_particles:
            with torch.no_grad():
                ss_sd['q1'].copy_(init_q1)

        #init.uniform_(shooting_block.state_dict()['q1'],-2,2) # just for initialization experiments, not needed

    if uses_particles:
        if args.custom_parameter_freezing:
            utils.freeze_parameters(shooting_block,['q1'])

    optimizer, scheduler = setup_optimizer_and_scheduler(params=shooting_block.parameters(), lr=args.lr)
    nr_of_pars = utils.compute_number_of_parameters(model=shooting_block)

    for itr in range_command(0, args.niters+1):

        if args.custom_parameter_freezing:
            if itr == args.unfreeze_parameters_at_iter:
                if uses_particles:
                    utils.unfreeze_parameters(shooting_block, ['q1'])

        # now iterate over the entire training dataset
        for itr_batch, sampled_batch in enumerate(train_loader):

            batch_t = train_loader.dataset.t
            batch_y0, batch_y = sampled_batch['y0'], sampled_batch['y'].transpose(0,1)

            optimizer.zero_grad()

            loss, sim_loss, norm_loss, norm_penalty, pred_y = compute_loss(args=args,
                                                                           batch_y0 = batch_y0,
                                                                           batch_t=batch_t,
                                                                           batch_y=batch_y,
                                                                           shooting_block=shooting_block)

            loss.backward()

            optimizer.step()

        # now compute testing evaluation loss
        if (itr % args.test_freq == 0) or (itr % args.viz_freq == 0) or (itr == args.niters):
            try:
                print('\nIter {:04d} | Training Loss {:.4f}; sim_loss = {:.4f}; norm_loss = {:.4f}; par_norm = {:.4f}; lr = {:.6f}'.format(itr, loss.item(), sim_loss.item(), norm_loss.item(), norm_penalty.item(), scheduler.get_last_lr()[0]))
            except:
                print('\nIter {:04d} | Training Loss {:.4f}; sim_loss = {:.4f}; norm_loss = {:.4f}; par_norm = {:.4f}'.format(itr, loss.item(), sim_loss.item(), norm_loss.item(), norm_penalty.item()))

            with torch.no_grad():
                # TODO: currently evaluates only based on one batch
                dataiter = iter(train_eval_loader)
                train_eval_batch = dataiter.next()
                batch_t = train_eval_loader.dataset.t
                batch_y0, batch_y = train_eval_batch['y0'], train_eval_batch['y'].transpose(0,1)

                loss, sim_loss, norm_loss, norm_penalty, pred_y = compute_loss(args=args,
                                                                               batch_y0=batch_y0,
                                                                               batch_t=batch_t,
                                                                               batch_y=batch_y,
                                                                               shooting_block=shooting_block)
                try:
                    print(
                        '\nIter {:04d} | Training evaluation loss {:.4f}; sim_loss = {:.4f}; norm_loss = {:.4f}; par_norm = {:.4f}'.format(
                            itr, loss.item(), sim_loss.item(), norm_loss.item(), norm_penalty.item()))
                    scheduler.step(loss)
                except:
                    print(
                        '\nIter {:04d} | Training evaluation loss {:.4f}; sim_loss = {:.4f}; norm_loss = {:.4f}; par_norm = {:.4f}; lr = {:.6f}'.format(
                            itr, loss.item(), sim_loss.item(), norm_loss.item(), norm_penalty.item(),
                            scheduler.get_last_lr()[0]))
                    scheduler.step()


        if itr % args.viz_freq == 0:
            # visualize what is happening based on the validation data

            with torch.no_grad():
                # TODO: currently evaluates only based on one batch
                dataiter = iter(test_loader)
                test_batch = dataiter.next()
                val_batch_t = test_loader.dataset.t
                val_batch_y0, val_batch_y = test_batch['y0'], test_batch['y'].transpose(0,1)

                loss, sim_loss, norm_loss, norm_penalty, val_pred_y = compute_loss(args=args,
                                                                                   batch_y0=val_batch_y0,
                                                                                   batch_t=val_batch_t,
                                                                                   batch_y=val_batch_y,
                                                                                   shooting_block=shooting_block,
                                                                                   validation_mode=True
                                                                                   )

            losses_to_print = {'model_name': args.shooting_model, 'loss': loss.item(), 'sim_loss': sim_loss.item(), 'norm_loss': norm_loss.item(), 'par_norm': norm_penalty.item()}

            vector_visualization.basic_visualize(shooting_block, val_batch_y, val_pred_y, val_batch_t, batch_y, pred_y, batch_t, itr,
                                                 uses_particles=uses_particles, losses_to_print=losses_to_print, nr_of_pars=nr_of_pars,args=args)

    # now print the parameters
    nr_of_parameters = utils.compute_number_of_parameters(shooting_block, print_parameters=True)

    values_to_save = dict()
    values_to_save['args'] = args._get_kwargs()
    values_to_save['nr_of_parameters'] = nr_of_parameters

    if args.validate_with_long_range:
        with torch.no_grad():
            # TODO: currently evaluates only based on one batch
            dataiter = iter(test_loader)
            test_batch = dataiter.next()
            val_batch_t = test_loader.dataset.t
            val_batch_y0, val_batch_y = test_batch['y0'], test_batch['y'].transpose(0,1)
            loss, sim_loss, norm_loss, norm_penalty, val_pred_y = compute_loss(args=args,
                                                                               batch_y0=val_batch_y0,
                                                                               batch_t=val_batch_t,
                                                                               batch_y=val_batch_y,
                                                                               shooting_block=shooting_block,
                                                                               validation_mode=True)



            values_to_save['test_loss'] = loss.item()
            values_to_save['sim_loss'] = sim_loss.item()
            values_to_save['norm_loss'] = norm_loss.item()
            values_to_save['norm_penalty'] = norm_loss.item()

            vector_visualization.basic_visualize(shooting_block, val_batch_y, val_pred_y, val_batch_t, batch_y, pred_y,
                                                 batch_t, itr,
                                                 uses_particles=uses_particles, losses_to_print=losses_to_print,
                                                 nr_of_pars=nr_of_pars,args=args)

    # now with the evaluation data (short range)
    with torch.no_grad():
        # TODO: currently evaluates only based on one batch
        dataiter = iter(short_range_test_loader)
        test_batch = dataiter.next()
        val_batch_t = short_range_test_loader.dataset.t
        val_batch_y0, val_batch_y = test_batch['y0'], test_batch['y'].transpose(0, 1)
        loss, sim_loss, norm_loss, norm_penalty, val_pred_y = compute_loss(args=args,
                                                                           batch_y0=val_batch_y0,
                                                                           batch_t=val_batch_t,
                                                                           batch_y=val_batch_y,
                                                                           shooting_block=shooting_block,
                                                                           validation_mode=True)

        values_to_save['short_range_test_loss'] = loss.item()
        values_to_save['short_range_sim_loss'] = sim_loss.item()
        values_to_save['short_range_norm_loss'] = norm_loss.item()
        values_to_save['short_range_norm_penalty'] = norm_loss.item()

    if args.validate_with_long_range:
        with torch.no_grad():

            # now evaluate the evolution over time
            custom_hook_data = defaultdict(list)
            hook = shooting_block.shooting_integrand.register_lagrangian_gradient_hook(sh.parameter_evolution_hook)
            shooting_block.shooting_integrand.set_custom_hook_data(data=custom_hook_data)

            # run the evaluation for the validation data and record it via the hook
            dataiter = iter(test_loader)
            test_batch = dataiter.next()
            val_batch_t = test_loader.dataset.t
            val_batch_y0, val_batch_y = test_batch['y0'], test_batch['y'].transpose(0,1)

            loss, sim_loss, norm_loss, norm_penalty, val_pred_y = compute_loss(args=args,
                                                                               batch_y0=val_batch_y0,
                                                                               batch_t=val_batch_t,
                                                                               batch_y=val_batch_y,
                                                                               shooting_block=shooting_block,
                                                                               validation_mode=True)
            hook.remove()

            log_complexity_measures = validation_measures.compute_complexity_measures(data=custom_hook_data)
            print('Validation measures = {}'.format(log_complexity_measures))

            values_to_save['log_complexity_measures'] = log_complexity_measures

            if args.create_animation:
                # now plot the evolution over time
                vector_visualization.visualize_time_evolution(val_batch_y, data=custom_hook_data, block_name='simple', save_to_directory='result-{}'.format(args.shooting_model))

    # now with the evaluation data (short range)
    with torch.no_grad():

        # now evaluate the evolution over time
        custom_hook_data = defaultdict(list)
        hook = shooting_block.shooting_integrand.register_lagrangian_gradient_hook(sh.parameter_evolution_hook)
        shooting_block.shooting_integrand.set_custom_hook_data(data=custom_hook_data)

        # run the evaluation for the validation data and record it via the hook
        dataiter = iter(short_range_test_loader)
        test_batch = dataiter.next()
        val_batch_t = short_range_test_loader.dataset.t
        val_batch_y0, val_batch_y = test_batch['y0'], test_batch['y'].transpose(0, 1)

        loss, sim_loss, norm_loss, norm_penalty, val_pred_y = compute_loss(args=args,
                                                                           batch_y0=val_batch_y0,
                                                                           batch_t=val_batch_t,
                                                                           batch_y=val_batch_y,
                                                                           shooting_block=shooting_block,
                                                                           validation_mode=True)
        hook.remove()

        log_complexity_measures = validation_measures.compute_complexity_measures(data=custom_hook_data)
        print('Validation measures = {}'.format(log_complexity_measures))

        values_to_save['short_range_log_complexity_measures'] = log_complexity_measures

    if args.save_figures:
        # in this case we also save the results
        if not os.path.exists(args.output_directory):
            os.mkdir(args.output_directory)

        result_filename = os.path.join(args.output_directory,'{}-results.pt'.format(args.output_basename))
        print('Saving results in {}'.format(result_filename))
        torch.save(values_to_save,result_filename)