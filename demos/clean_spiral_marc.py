import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.init as init

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
import neuro_shooting.tensorboard_shooting_hooks as thooks
import neuro_shooting.shooting_hooks as sh
import neuro_shooting.vector_visualization as vector_visualization
import neuro_shooting.validation_measures as validation_measures

# Setup

def setup_cmdline_parsing():
    # Command line arguments

    parser = argparse.ArgumentParser('ODE demo')
    parser.add_argument('--method', type=str, choices=['dopri5', 'adams','rk4'], default='rk4', help='Selects the desired integrator')
    parser.add_argument('--stepsize', type=float, default=0.05, help='Step size for the integrator (if not adaptive).')
    parser.add_argument('--data_size', type=int, default=200, help='Length of the simulated data that should be matched.')
    parser.add_argument('--batch_time', type=int, default=25, help='Length of the training samples.')
    parser.add_argument('--batch_size', type=int, default=50, help='Number of training samples.')
    parser.add_argument('--niters', type=int, default=10000, help='Maximum nunber of iterations.')
    parser.add_argument('--batch_validation_size', type=int, default=25, help='Length of the samples for validation.')
    parser.add_argument('--seed', required=False, type=int, default=-1,
                        help='Sets the random seed which affects data shuffling')

    parser.add_argument('--linear', action='store_true', help='If specified the ground truth system will be linear, otherwise nonlinear.')

    parser.add_argument('--test_freq', type=int, default=100, help='Frequency with which the validation measures are to be computed.')
    parser.add_argument('--viz_freq', type=int, default=100, help='Frequency with which the results should be visualized; if --viz is set.')

    parser.add_argument('--validate_with_long_range', action='store_true', help='If selected, a long-range trajectory will be used; otherwise uses batches as for training')
    parser.add_argument('--chunk_time', type=int, default=15, help='For a long range valdation solution chunks the solution together in these pieces.')

    parser.add_argument('--shooting_model', type=str, default='updown', choices=['simple', 'updown', 'periodic'])
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
    parser.add_argument('--do_not_use_parameter_penalty_energy', action='store_true', default=False)
    parser.add_argument('--optimize_over_data_initial_conditions', action='store_true', default=False)
    parser.add_argument('--optimize_over_data_initial_conditions_type', type=str, choices=['direct','linear','mini_nn'], default='linear', help='Different ways to predict the initial conditions for higher order models. Currently only supported for updown model.')

    parser.add_argument('--disable_distance_based_sampling', action='store_true', default=False, help='If specified uses the original trajectory sampling, otherwise samples based on trajectory length.')

    parser.add_argument('--custom_parameter_freezing', action='store_true', default=False, help='Enable custom code for parameter freezing -- development mode')
    parser.add_argument('--custom_parameter_initialization', action='store_true', default=False, help='Enable custom code for parameter initialization -- development mode')

    parser.add_argument('--viz', action='store_true', help='Enable visualization.')
    parser.add_argument('--gpu', type=int, default=0, help='Enable GPU computation on specified GPU.')
    parser.add_argument('--adjoint', action='store_true', help='Use adjoint integrator to avoid storing values during forward pass.')

    parser.add_argument('--create_animation', action='store_true', help='Creates animated gif for the evolution of the particles.')

    args = parser.parse_args()

    return args

def setup_random_seed(seed):
    if seed==-1:
        print('No seed was specified, leaving everthing at random. Use --seed to specify a seed if you want repeatable results.')
    else:
        print('Setting the random seed to {:}'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def compute_number_of_parameters(model):

    nr_of_fixed_parameters = 0
    nr_of_optimized_parameters = 0
    print('\nModel parameters:\n')
    print('-----------------')
    for pn, pv in model.named_parameters():
        #print('{} = {}'.format(pn, pv))
        current_number_of_parameters = np.prod(list(pv.size()))
        print('{}: # of parameters = {}\n'.format(pn,current_number_of_parameters))
        if pv.requires_grad:
            nr_of_optimized_parameters += current_number_of_parameters
        else:
            nr_of_fixed_parameters += current_number_of_parameters

    print('Number of fixed parameters = {}'.format(nr_of_fixed_parameters))
    print('Number of optimized parameters = {}'.format(nr_of_optimized_parameters))
    overall_nr_of_parameters = nr_of_fixed_parameters + nr_of_optimized_parameters
    print('Overall number of parameters = {}\n'.format(overall_nr_of_parameters))

    nr_of_pars = dict()
    nr_of_pars['fixed'] = nr_of_fixed_parameters
    nr_of_pars['optimized'] = nr_of_optimized_parameters
    nr_of_pars['overall'] = overall_nr_of_parameters

    return nr_of_pars


def setup_integrator(method, use_adjoint, step_size, rtol=1e-8, atol=1e-12):

    integrator_options = dict()

    if method not in ['dopri5', 'adams']:
        integrator_options  = {'step_size': step_size}

    integrator = generic_integrator.GenericIntegrator(integrator_library = 'odeint', integrator_name = method,
                                                     use_adjoint_integration=use_adjoint, integrator_options=integrator_options, rtol=rtol, atol=atol)


    return integrator

def setup_shooting_block(integrator=None, shooting_model='updown', parameter_weight=1.0, nr_of_particles=10,
                         inflation_factor=2, nonlinearity='relu',
                         use_particle_rnn_mode=False, use_particle_free_rnn_mode=False,
                         optimize_over_data_initial_conditions=False,
                         optimize_over_data_initial_conditions_type='linear',
                         device='cpu'):

    if shooting_model=='updown':
        smodel = shooting_models.AutoShootingIntegrandModelUpDown(in_features=2, nonlinearity=nonlinearity,
                                                                  parameter_weight=parameter_weight,
                                                                  inflation_factor=inflation_factor,
                                                                  nr_of_particles=nr_of_particles, particle_dimension=1,
                                                                  particle_size=2,
                                                                  use_analytic_solution=True,
                                                                  use_rnn_mode=use_particle_rnn_mode,
                                                                  optimize_over_data_initial_conditions=optimize_over_data_initial_conditions,
                                                                  optimize_over_data_initial_conditions_type=optimize_over_data_initial_conditions_type)
    elif shooting_model=='periodic':
        smodel = shooting_models.AutoShootingIntegrandModelUpdownPeriodic(in_features=2, nonlinearity=nonlinearity,
                                                                  parameter_weight=parameter_weight,
                                                                  inflation_factor=inflation_factor,
                                                                  nr_of_particles=nr_of_particles, particle_dimension=1,
                                                                  particle_size=2,
                                                                  use_analytic_solution=True,
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
                                                       use_particle_free_rnn_mode=use_particle_free_rnn_mode, integrator=integrator)
    shooting_block = shooting_block.to(device)

    return shooting_block

def setup_optimizer_and_scheduler(params):

    #optimizer = optim.Adam(params, lr=0.025)
    optimizer = optim.Adam(params, lr=0.1)

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
    scaled_dists = (dists / dists.min()).numpy().round().astype(np.int32)
    # create sampling array where each index has as many repetitions
    indices = []
    for i, sd in enumerate(scaled_dists):
        indices += [i]*sd

    indices = np.array(indices).astype(np.int64)
    return indices,all_dists

def generate_data(integrator, data_size, batch_time, linear=False, device='cpu'):

    d = dict()

    d['y0'] = torch.tensor([[2., 0.]]).to(device)
    #d['t'] = torch.linspace(0., 25., data_size).to(device)

    d['t'] = torch.linspace(0., 10., data_size).to(device)

    d['A'] = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)
    #d['A'] = torch.tensor([[-0.05, 0.025], [-0.025, -0.05]]).to(device)

    # pure slow oscillation
    #d['A'] = torch.tensor([[0, 0.025], [-0.025, 0]]).to(device)
    # small section
    #d['A'] = torch.tensor([[0, 0.1], [-0.1, 0]]).to(device)

    with torch.no_grad():
        # integrate it
        d['y'] = integrator.integrate(func=DiffEqRHS(A=d['A'], linear=linear), x0=d['y0'], t=d['t'])

    d['uniform_sample_indices'],d['dists'] = create_uniform_distance_selection_array(d, batch_time=batch_time)

    return d


def get_batch(data_dict, batch_size, batch_time, distance_based_sampling=True):

    data_size = len(data_dict['t'])

    if distance_based_sampling:
        s = torch.from_numpy(np.random.choice(data_dict['uniform_sample_indices'], size=batch_size, replace=True)).to(device)
    else:
        s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), size=batch_size, replace=True)).to(device)

    batch_y0 = data_dict['y'][s]  # (M, D)
    batch_t = data_dict['t'][:batch_time]  # (T)
    batch_y = torch.stack([data_dict['y'][s + i] for i in range(batch_time)], dim=0)  # (T, M, D)

    return batch_y0, batch_t, batch_y


def freeze_parameters(shooting_block,parameters_to_freeze):

    # get all the parameters that we are optimizing over
    pars = shooting_block.state_dict()
    for pn in parameters_to_freeze:
        print('Freezing {}'.format(pn))
        pars[pn].requires_grad = False

def get_time_chunks(t, chunk_time):
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
            current_chunk = t[cur_idx:desired_idx]
            time_chunks.append(current_chunk)
            cur_idx = desired_idx-1

    return time_chunks

def compute_validation_data(shooting_block, t, y0, validate_with_long_range, chunk_time):

    if validate_with_long_range:

        sz = [len(t)] + list(y0.shape)
        val_pred_y = torch.zeros(sz,device=y0.device,dtype=y0.dtype)

        # now we chunk it
        cur_idx = 0
        time_chunks = get_time_chunks(t=t, chunk_time=chunk_time)
        for i,time_chunk in enumerate(time_chunks):
            shooting_block.set_integration_time_vector(integration_time_vector=time_chunk, suppress_warning=True)
            if i==0:
                cur_pred_y, _, _, _ = shooting_block(x=y0)
                current_norm_penalty = shooting_block.get_norm_penalty()
            else:
                cur_pred_y, _, _, _ = shooting_block(x=val_pred_y[cur_idx-1,...])

            val_pred_y[cur_idx:cur_idx + len(time_chunk), ...] = cur_pred_y
            cur_idx += len(time_chunk) - 1

    else:
        shooting_block.set_integration_time_vector(integration_time_vector=t, suppress_warning=True)
        val_pred_y, _, _, _ = shooting_block(x=y0)
        current_norm_penalty = shooting_block.get_norm_penalty()

    return val_pred_y, current_norm_penalty


if __name__ == '__main__':

    # do some initial setup
    args = setup_cmdline_parsing()
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    use_distance_based_sampling = not args.disable_distance_based_sampling

    setup_random_seed(seed=args.seed)

    integrator = setup_integrator(method=args.method, step_size=args.stepsize, use_adjoint=args.adjoint)

    shooting_block = setup_shooting_block(integrator=integrator,
                                          shooting_model=args.shooting_model,
                                          parameter_weight=args.pw,
                                          nr_of_particles=args.nr_of_particles,
                                          inflation_factor=args.inflation_factor,
                                          nonlinearity=args.nonlinearity,
                                          use_particle_rnn_mode=args.use_particle_rnn_mode,
                                          use_particle_free_rnn_mode=args.use_particle_free_rnn_mode,
                                          optimize_over_data_initial_conditions=args.optimize_over_data_initial_conditions,
                                          optimize_over_data_initial_conditions_type=args.optimize_over_data_initial_conditions_type,
                                          device=device)

    # generate the true data tha we want to match
    data = generate_data(integrator=integrator, data_size=args.data_size, batch_time=args.batch_time, linear=args.linear, device=device)

    # draw an initial batch from it
    batch_y0, batch_t, batch_y = get_batch(data_dict=data, batch_time=args.batch_time, batch_size=args.batch_size, distance_based_sampling=use_distance_based_sampling)

    # create validation data
    if args.validate_with_long_range:
        print('Validating with long range data')
        val_y0 = data['y0'].unsqueeze(dim=0)
        val_t = data['t']
        val_y = data['y'].unsqueeze(dim=1)
    else:
        # draw a FIXED validation batch
        print('Validating with fixed validation batch of size {} and with {} time-points'.format(args.batch_size,args.batch_time))
        # TODO: maybe want to support a new validation batch every time
        val_y0, val_t, val_y = get_batch(data_dict=data, batch_time=args.batch_time, batch_size=args.batch_size, distance_based_sampling=use_distance_based_sampling)

    # run through the shooting block once (to get parameters as needed)
    shooting_block(x=batch_y)

    # custom initialization
    if args.custom_parameter_initialization:

        # first get the state dictionary of the shooting block which contains all parameters
        ss_sd = shooting_block.state_dict()

        # get initial positions on the trajectory (to place the particles there)
        init_q1, _, _ = get_batch(data_dict=data, batch_time=1, batch_size=args.nr_of_particles, distance_based_sampling=True)
        with torch.no_grad():
            ss_sd['q1'].copy_(init_q1)

        #init.uniform_(shooting_block.state_dict()['q1'],-2,2) # just for initialization experiments, not needed

    uses_particles = not args.use_particle_free_rnn_mode
    if uses_particles:
        if args.custom_parameter_freezing:
            freeze_parameters(shooting_block,['q1'])

    optimizer, scheduler = setup_optimizer_and_scheduler(params=shooting_block.parameters())
    nr_of_pars = compute_number_of_parameters(model=shooting_block)

    for itr in range_command(0, args.niters):

        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(data_dict=data, batch_time=args.batch_time, batch_size=args.batch_size, distance_based_sampling=use_distance_based_sampling)

        shooting_block.set_integration_time_vector(integration_time_vector=batch_t, suppress_warning=True)
        pred_y,_,_,_ = shooting_block(x=batch_y0)

        if args.sim_norm == 'l1':
            sim_loss = args.sim_weight*torch.mean(torch.abs(pred_y - batch_y))
        elif args.sim_norm == 'l2':
            sim_loss = args.sim_weight*torch.mean(torch.norm(pred_y-batch_y,dim=3))
        else:
            raise ValueError('Unknown norm {}.'.format(args.sim_norm))

        norm_penalty = shooting_block.get_norm_penalty()

        if args.do_not_use_parameter_penalty_energy:
            norm_loss = torch.tensor([0])
        else:
            norm_loss = args.norm_weight*norm_penalty

        loss = sim_loss + norm_loss

        loss.backward()

        optimizer.step()

        if itr % args.test_freq == 0:
            try:
                print('\nIter {:04d} | Training Loss {:.4f}; sim_loss = {:.4f}; norm_loss = {:.4f}; par_norm = {:.4f}; lr = {:.6f}'.format(itr, loss.item(), sim_loss.item(), norm_loss.item(), norm_penalty.item(), scheduler.get_last_lr()[0]))
                scheduler.step()
            except:
                print('\nIter {:04d} | Training Loss {:.4f}; sim_loss = {:.4f}; norm_loss = {:.4f}; par_norm = {:.4f}'.format(itr, loss.item(), sim_loss.item(), norm_loss.item(), norm_penalty.item()))
                scheduler.step(loss)

        if itr % args.test_freq == 0:

            with torch.no_grad():

                val_pred_y, current_norm = compute_validation_data(shooting_block=shooting_block,
                                                     t=val_t, y0=val_y0,
                                                     validate_with_long_range=args.validate_with_long_range,
                                                     chunk_time=args.chunk_time)

                if args.sim_norm=='l1':
                    sim_loss = args.sim_weight*torch.mean(torch.abs(val_pred_y - val_y))
                elif args.sim_norm=='l2':
                    sim_loss = args.sim_weight*torch.mean(torch.norm(val_pred_y - val_y, dim=3))
                else:
                    raise ValueError('Unknown norm {}.'.format(args.sim_norm))

                norm_penalty = current_norm
                if args.do_not_use_parameter_penalty_energy:
                    norm_loss = torch.tensor([0])
                else:
                    norm_loss = args.norm_weight*norm_penalty

                loss = sim_loss + norm_loss

                print('Iter {:04d} | Validation Loss {:.4f}; sim_loss = {:.4f}; norm_loss = {:.4f}; par_norm = {:.4f}'.format(itr, loss.item(), sim_loss.item(), norm_loss.item(), norm_penalty.item()))

            if itr % args.viz_freq == 0:

                losses_to_print = {'model_name': args.shooting_model, 'loss': loss.item(), 'sim_loss': sim_loss.item(), 'norm_loss': norm_loss.item(), 'par_norm': norm_penalty.item()}

                vector_visualization.basic_visualize(shooting_block, val_y, val_pred_y, val_t, batch_y, pred_y, batch_t, itr, uses_particles=uses_particles, losses_to_print=losses_to_print, nr_of_pars=nr_of_pars)

    # now evaluate the evolution over time
    custom_hook_data = defaultdict(list)
    hook = shooting_block.shooting_integrand.register_lagrangian_gradient_hook(sh.parameter_evolution_hook)
    shooting_block.shooting_integrand.set_custom_hook_data(data=custom_hook_data)

    # run the evaluation for the validation data and record it via the hook
    val_pred_y, current_norm = compute_validation_data(shooting_block=shooting_block,
                                                       t=val_t, y0=val_y0,
                                                       validate_with_long_range=args.validate_with_long_range,
                                                       chunk_time=args.chunk_time)

    hook.remove()

    complexity_measures = validation_measures.compute_complexity_measures(data=custom_hook_data)
    print('\nLog complexity measures:')
    for m in complexity_measures:
        print('  {} = {:.3f}'.format(m,complexity_measures[m]))

    if args.create_animation:
        # now plot the evolution over time
        vector_visualization.visualize_time_evolution(val_y, data=custom_hook_data, block_name='simple', save_to_directory='result-{}'.format(args.shooting_model))

