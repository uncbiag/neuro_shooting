import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.init as init

import matplotlib.pyplot as plt

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.shooting_models as shooting_models
import neuro_shooting.generic_integrator as generic_integrator
import neuro_shooting.tensorboard_shooting_hooks as thooks


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
    parser.add_argument('--seed', required=False, type=int, default=1234,
                        help='Sets the random seed which affects data shuffling')

    parser.add_argument('--linear', action='store_true', help='If specified the ground truth system will be linear, otherwise nonlinear.')

    parser.add_argument('--test_freq', type=int, default=100, help='Frequency with which the validation measures are to be computed.')
    parser.add_argument('--viz_freq', type=int, default=100, help='Frequency with which the results should be visualized; if --viz is set.')

    parser.add_argument('--validate_with_long_range', action='store_true', help='If selected, a long-range trajectory will be used; otherwise uses batches as for training')
    parser.add_argument('--chunk_time', type=int, default=15, help='For a long range valdation solution chunks the solution together in these pieces.')

    parser.add_argument('--shooting_model', type=str, default='updown', choices=['simple', 'updown'])
    parser.add_argument('--nr_of_particles', type=int, default=25, help='Number of particles to parameterize the initial condition')
    parser.add_argument('--pw', type=float, default=1.0, help='parameter weight')
    parser.add_argument('--sim_norm', type=str, choices=['l1','l2'], default='l2', help='Norm for the similarity measure.')
    parser.add_argument('--nonlinearity', type=str, choices=['identity', 'relu', 'tanh', 'sigmoid'], default='relu', help='Nonlinearity for shooting.')

    parser.add_argument('--inflation_factor', type=int, default=5,
                        help='Multiplier for state dimension for updown shooting model types')
    parser.add_argument('--use_particle_rnn_mode', action='store_true',
                        help='When set then parameters are only computed at the initial time and used for the entire evolution; mimicks a particle-based RNN model.')
    parser.add_argument('--use_particle_free_rnn_mode', action='store_true',
                        help='This is directly optimizing over the parameters -- no particles here; a la Neural ODE')
    parser.add_argument('--use_parameter_penalty_energy', action='store_true', default=False)
    parser.add_argument('--optimize_over_data_initial_conditions', action='store_true', default=False)
    parser.add_argument('--optimize_over_data_initial_conditions_type', type=str, choices=['direct','linear','mini_nn'], default='mini_nn', help='Different ways to predict the initial conditions for higher order models. Currently only supported for updown model.')

    parser.add_argument('--disable_distance_based_sampling', action='store_true', default=False, help='If specified uses the original trajectory sampling, otherwise samples based on trajectory length.')

    parser.add_argument('--viz', action='store_true', help='Enable visualization.')
    parser.add_argument('--gpu', type=int, default=0, help='Enable GPU computation on specified GPU.')
    parser.add_argument('--adjoint', action='store_true', help='Use adjoint integrator to avoid storing values during forward pass.')

    args = parser.parse_args()

    return args

def setup_random_seed(seed):
    print('Setting the random seed to {:}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def setup_integrator(method='rk4', use_adjoint=False, step_size=0.05, rtol=1e-8, atol=1e-12):

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
                                                                  use_particle_rnn_mode=use_particle_rnn_mode,
                                                                  optimize_over_data_initial_conditions=optimize_over_data_initial_conditions,
                                                                  optimize_over_data_initial_conditions_type=optimize_over_data_initial_conditions_type)
    elif shooting_model=='simple':
        smodel = shooting_models.AutoShootingIntegrandModelSimple(in_features=2, nonlinearity=nonlinearity,
                                                                  parameter_weight=parameter_weight,
                                                                  nr_of_particles=nr_of_particles, particle_dimension=1,
                                                                  particle_size=2,
                                                                  use_analytic_solution=True,
                                                                  use_particle_rnn_mode=use_particle_rnn_mode)

    import neuro_shooting.parameter_initialization as pi
    # par_initializer = pi.VectorEvolutionSampleBatchParameterInitializer(only_random_initialization=False,
    #     random_initialization_magnitude=0.1,
    #     sample_batch=batch_y0)

    par_initializer = pi.VectorEvolutionParameterInitializer(only_random_initialization=True, random_initialization_magnitude=1.0)

    smodel.set_state_initializer(state_initializer=par_initializer)
    shooting_block = shooting_blocks.ShootingBlockBase(name='simple', shooting_integrand=smodel, use_particle_free_rnn_mode=use_particle_free_rnn_mode, integrator=integrator)
    shooting_block = shooting_block.to(device)

    return shooting_block

def setup_optimizer_and_scheduler(params):

    optimizer = optim.Adam(params, lr=0.025)

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

# Visualization
# TODO: revamp

def plot_particles(shooting_block,ax):

    quiver_scale = 1.0 # to scale the magnitude of the quiver vectors for visualization

    # get all the parameters that we are optimizing over
    pars = shooting_block.state_dict()

    # let's first just plot the positions
    ax.plot(pars['q1'][:,0,0], pars['q1'][:,0,1],'k+',markersize=12)
    ax.quiver(pars['q1'][:,0,0], pars['q1'][:,0,1], pars['p_q1'][:,0,0], pars['p_q1'][:,0,1], color='r', scale=quiver_scale)

    ax.set_title('q1 and p_q1')

def plot_higher_order_state(shooting_block,ax):


    quiver_scale = 1.0 # to scale the magnitude of the quiver vectors for visualization

    # get all the parameters that we are optimizing over
    pars = shooting_block.state_dict()

    # let's first just plot the positions
    ax.plot(pars['q2'][:,0,0], pars['q2'][:,0,1],'k+',markersize=12)
    ax.quiver(pars['q2'][:,0,0], pars['q2'][:,0,1], pars['p_q2'][:,0,0], pars['p_q2'][:,0,1], color='r', scale=quiver_scale)

    ax.set_title('q2 and p_q2')


def plot_trajectories(val_y, pred_y, sim_time, batch_y, batch_pred_y, batch_t, itr, ax):
    for n in range(val_y.size()[1]):
        ax.plot(val_y.detach().numpy()[:, n, 0, 0], val_y.detach().numpy()[:, n, 0, 1], 'g-')
        ax.plot(pred_y.detach().numpy()[:, n, 0, 0], pred_y.detach().numpy()[:, n, 0, 1], 'b--+')

    for n in range(batch_y.size()[1]):
        ax.plot(batch_y.detach().numpy()[:, n, 0, 0], batch_y.detach().numpy()[:, n, 0, 1], 'k-')
        ax.plot(batch_pred_y.detach().numpy()[:, n, 0, 0], batch_pred_y.detach().numpy()[:, n, 0, 1], 'r--')

    ax.set_title('trajectories: iter = {}'.format(itr))

def basic_visualize(shooting_block, val_y, pred_y, sim_time, batch_y, batch_pred_y, batch_t, itr):

    fig = plt.figure(figsize=(12, 4), facecolor='white')

    ax = fig.add_subplot(131, frameon=False)
    ax_lo = fig.add_subplot(132, frameon=False)
    ax_ho = fig.add_subplot(133, frameon=False)

    # plot it without any additional information
    plot_trajectories(val_y, pred_y, sim_time, batch_y, batch_pred_y, batch_t, itr, ax=ax)

    # now plot the information from the state variables
    plot_particles(shooting_block=shooting_block,ax=ax_lo)
    plot_higher_order_state(shooting_block=shooting_block,ax=ax_ho)

    plt.show()

def visualize(true_y, pred_y, sim_time, odefunc, itr, is_higher_order_model=True):

    quiver_scale = 2.5 # to scale the magnitude of the quiver vectors for visualization

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)

    ax_traj.cla()
    ax_traj.set_title('Trajectories')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('x,y')

    for n in range(true_y.size()[1]):
        ax_traj.plot(sim_time.numpy(), true_y.detach().numpy()[:, n, 0, 0], sim_time.numpy(), true_y.numpy()[:, n, 0, 1],
                 'g-')
        ax_traj.plot(sim_time.numpy(), pred_y.detach().numpy()[:, n, 0, 0], '--', sim_time.numpy(),
                 pred_y.detach().numpy()[:, n, 0, 1],
                 'b--')

    ax_traj.set_xlim(sim_time.min(), sim_time.max())
    ax_traj.set_ylim(-2, 2)
    ax_traj.legend()

    ax_phase.cla()
    ax_phase.set_title('Phase Portrait')
    ax_phase.set_xlabel('x')
    ax_phase.set_ylabel('y')

    for n in range(true_y.size()[1]):
        ax_phase.plot(true_y.detach().numpy()[:, n, 0, 0], true_y.detach().numpy()[:, n, 0, 1], 'g-')
        ax_phase.plot(pred_y.detach().numpy()[:, n, 0, 0], pred_y.detach().numpy()[:, n, 0, 1], 'b--')

    try:
        q = (odefunc.q_params)
        p = (odefunc.p_params)

        q_np = q.cpu().detach().squeeze(dim=1).numpy()
        p_np = p.cpu().detach().squeeze(dim=1).numpy()

        ax_phase.scatter(q_np[:,0],q_np[:,1],marker='+')
        ax_phase.quiver(q_np[:,0],q_np[:,1], p_np[:,0],p_np[:,1],color='r', scale=quiver_scale)
    except:
        pass

    ax_phase.set_xlim(-2, 2)
    ax_phase.set_ylim(-2, 2)


    ax_vecfield.cla()
    ax_vecfield.set_title('Learned Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')

    y, x = np.mgrid[-2:2:21j, -2:2:21j]

    current_y = torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))

    # print("q_params",q_params.size())

    x_0 = current_y.unsqueeze(dim=1)

    #viz_time = t[:5] # just 5 timesteps ahead
    viz_time = sim_time[:5] # just 5 timesteps ahead

    odefunc.set_integration_time_vector(integration_time_vector=viz_time,suppress_warning=True)
    dydt_pred_y,_,_,_ = odefunc(x=x_0)

    if is_higher_order_model:
        dydt = (dydt_pred_y[-1,...]-dydt_pred_y[0,...]).detach().numpy()
        dydt = dydt[:,0,...]
    else:
        dydt = dydt_pred_y[-1,0,...]

    mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
    dydt = (dydt / mag)
    dydt = dydt.reshape(21, 21, 2)

    ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")

    try:
        ax_vecfield.scatter(q_np[:, 0], q_np[:, 1], marker='+')
        ax_vecfield.quiver(q_np[:,0],q_np[:,1], p_np[:,0],p_np[:,1],color='r', scale=quiver_scale)
    except:
        pass

    ax_vecfield.set_xlim(-2, 2)
    ax_vecfield.set_ylim(-2, 2)

    fig.tight_layout()

    plt.show()

def freeze_parameters(shooting_block,parameters_to_freeze):

    # get all the parameters that we are optimizing over
    pars = shooting_block.state_dict()
    for pn in parameters_to_freeze:
        print('Freezing {}'.format(pn))
        pars[pn].requires_grad = False

def get_time_chunks(t, chunk_time):
    time_chunks = []

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
            else:
                cur_pred_y, _, _, _ = shooting_block(x=val_pred_y[cur_idx-1,...])

            val_pred_y[cur_idx:cur_idx + len(time_chunk), ...] = cur_pred_y
            cur_idx += len(time_chunk) - 1

    else:
        shooting_block.set_integration_time_vector(integration_time_vector=t, suppress_warning=True)
        val_pred_y, _, _, _ = shooting_block(x=y0)

    return val_pred_y


if __name__ == '__main__':

    # do some initial setup
    args = setup_cmdline_parsing()
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    use_distance_based_sampling = not args.disable_distance_based_sampling

    setup_random_seed(seed=args.seed)

    integrator = setup_integrator(method=args.method, use_adjoint=args.adjoint)

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

    #init.uniform_(shooting_block.state_dict()['q1'],-2,2) # just for initialization experiments, not needed
    freeze_parameters(shooting_block,['q1'])

    optimizer, scheduler = setup_optimizer_and_scheduler(params=shooting_block.parameters())

    for itr in range(0, args.niters):

        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(data_dict=data, batch_time=args.batch_time, batch_size=args.batch_size, distance_based_sampling=use_distance_based_sampling)

        shooting_block.set_integration_time_vector(integration_time_vector=batch_t, suppress_warning=True)
        pred_y,_,_,_ = shooting_block(x=batch_y0)

        if args.sim_norm == 'l1':
            loss = torch.mean(torch.abs(pred_y - batch_y))
        elif args.sim_norm == 'l2':
            loss = torch.mean(torch.norm(pred_y-batch_y,dim=3))
        else:
            raise ValueError('Unknown norm {}.'.format(args.sim_norm))

        if args.use_parameter_penalty_energy:
            loss = loss + shooting_block.get_norm_penalty()

        loss.backward()

        optimizer.step()

        if itr % args.test_freq == 0:
            try:
                print('Iter {:04d} | Training Loss {:.6f}; lr = {:.6f}'.format(itr, loss.item(), scheduler.get_last_lr()[0]))
                scheduler.step()
            except:
                print('Iter {:04d} | Training Loss {:.6f}'.format(itr, loss.item()))
                scheduler.step(loss)

        if itr % args.test_freq == 0:

            with torch.no_grad():

                val_pred_y = compute_validation_data(shooting_block=shooting_block,
                                                     t=val_t, y0=val_y0,
                                                     validate_with_long_range=args.validate_with_long_range,
                                                     chunk_time=args.chunk_time)

                if args.sim_norm=='l1':
                    loss = torch.mean(torch.abs(val_pred_y - val_y))
                elif args.sim_norm=='l2':
                    loss = torch.mean(torch.norm(val_pred_y - val_y, dim=3))
                else:
                    raise ValueError('Unknown norm {}.'.format(args.sim_norm))

                if args.use_parameter_penalty_energy:
                    loss = loss + shooting_block.get_norm_penalty()

                print('Iter {:04d} | Validation Loss {:.6f}'.format(itr, loss.item()))

            if itr % args.viz_freq == 0:
                basic_visualize(shooting_block, val_y, val_pred_y, val_t, batch_y, pred_y, batch_t, itr)

                #visualize(val_y, val_pred_y, val_t, shooting_block, itr)

