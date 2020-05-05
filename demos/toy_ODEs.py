from scipy import integrate
import argparse
import numpy as np
from itertools import cycle
import torch
import torch.optim as optim
import random

import matplotlib.pyplot as plt

import os

import neuro_shooting.shooting_blocks as sblocks
import neuro_shooting.shooting_models as smodels
import neuro_shooting.parameter_initialization as pi
import neuro_shooting.generic_integrator as generic_integrator


def setup_cmdline_parsing():
    parser = argparse.ArgumentParser('toy ODEs')
    parser.add_argument('--method', type=str, choices=['dopri5', 'adams', 'rk4'], default='rk4', help='Selects the desired integrator')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--batch_timesteps', type=int, default=25,help='batch trajectories evaluated on 0:batch_tmax:batch_timesteps')
    parser.add_argument('--batch_tmax', type=int, default=1, help='batch trajectories integration time [0,batch_tmax]')
    parser.add_argument('--niters', type=int, default=2000)
    parser.add_argument('--seed', required=False, type=int, default=1234,
                        help='Sets the random seed which affects data shuffling')
    parser.add_argument('--adjoint', action='store_true', help='Use adjoint integrator to avoid storing values during forward pass.')
    parser.add_argument('--stepsize', type=float, default=0.05, help='Step size for the integrator (if not adaptive).')
    parser.add_argument('--ode_name', type=str, default='3d_lorenz',choices=['3d_lorenz','3d_time_spiral']) #TODO: code up more ODEs, only 3d lorenz is currently in here

    # shooting model parameters
    parser.add_argument('--shooting_model', type=str, default='updown', choices=['dampened_updown','simple', '2nd_order', 'updown'])
    parser.add_argument('--nonlinearity', type=str, default='relu', choices=['identity', 'relu', 'tanh', 'sigmoid',"softmax"], help='Nonlinearity for shooting.')
    parser.add_argument('--pw', type=float, default=1.0, help='parameter weight')
    parser.add_argument('--nr_of_particles', type=int, default=40, help='Number of particles to parameterize the initial condition')
    parser.add_argument('--inflation_factor', type=int, default=5, help='Multiplier for state dimension for updown shooting model types')
    parser.add_argument('--use_particle_rnn_mode', action='store_true', help='When set then parameters are only computed at the initial time and used for the entire evolution; mimicks a particle-based RNN model.')
    parser.add_argument('--use_particle_free_rnn_mode', action='store_true', help='This is directly optimizing over the parameters -- no particles here; a la Neural ODE')
    parser.add_argument('--use_analytic_solution',action='store_true')
    parser.add_argument('--sim_norm', type=str, choices=['l1', 'l2'], default='l2',help='Norm for the similarity measure.')
    parser.add_argument('--shooting_norm_penalty', type=float, default=1e-4,help='Factor to penalize the norm with; default 0, but 0.1 or so might be a good value')
    parser.add_argument('--optimize_over_data_initial_conditions', action='store_true', default=False)

    # non-shooting networks implemented
    parser.add_argument('--nr_of_layers', type=int, default=30, help='Number of layers for the non-shooting networks')
    parser.add_argument('--use_updown',action='store_true')
    parser.add_argument('--use_double_resnet',action='store_true')
    parser.add_argument('--use_rnn',action='store_true')
    parser.add_argument('--use_double_resnet_rnn',action="store_true")
    parser.add_argument('--use_simple_resnet',action='store_true')
    parser.add_argument('--use_neural_ode',action='store_true')

    parser.add_argument('--taskid',type=int,default=None)
    parser.add_argument('--test_freq', type=int, default=100)
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    print(args)
    return args


def lorentz_deriv(initial, t0, sigma=10., beta=8./3, rho=28.0):
    x = initial[0]
    y = initial[1]
    z = initial[2]
    """Compute the time-derivative of a Lorenz system."""
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def plot_trajectory(trajectory_dict, nr_trajectories):
    '''

    :param dictionary of trajectories where each value has size [timesteps, nr_trajectories, 1, dim]
    :return: plots each trajectory and labels according to dictionary key (typically predicted versus actual)
    '''

    colors = cycle('rgcmkb')

    for trajectory_num in range(0, nr_trajectories):

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        for key, value in trajectory_dict.items():

            # make trajectory shape [nr_trajectories, timesteps, dim]
            trajectory = value.permute(1,0,2,3).squeeze(dim=2).detach().numpy()

            # pick a random trajectory to visualize
            ax.plot(trajectory[trajectory_num,:,0],
                    trajectory[trajectory_num,:,1],
                    trajectory[trajectory_num,:,2],
                    lw=0.5,color=next(colors),label=key)

        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Lorenz Attractor")
        plt.legend()

        saveresultspath = 'toy_ODEs/taskid{}'.format(args.taskid)
        makedirs(saveresultspath)
        plt.savefig('{}/validation_trajectory{}.png'.format(saveresultspath,trajectory_num))
        plt.show()


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def setup_random_seed(seed):
    print('Setting the random seed to {:}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_integrator(method='rk4', use_adjoint=False, step_size=0.05, rtol=1e-8, atol=1e-12):

    integrator_options = dict()

    if method not in ['dopri5', 'adams']:
        integrator_options  = {'step_size': step_size}

    integrator = generic_integrator.GenericIntegrator(integrator_library = 'odeint', integrator_name = 'rk4',
                                                     use_adjoint_integration=use_adjoint, integrator_options=integrator_options, rtol=rtol, atol=atol)

    return integrator


def setup_shootingblock(integrator):

    batch_initial, batch_t, batch_trajectory = get_batch()
    in_features = batch_initial.shape[2]

    shootingintegrand_kwargs = {'in_features': in_features,
                                'nonlinearity': args.nonlinearity,
                                'nr_of_particles': args.nr_of_particles,
                                'parameter_weight': args.pw,
                                'particle_dimension': 1,
                                'particle_size': in_features}

    inflation_factor = args.inflation_factor  # for the up-down models (i.e., how much larger is the internal state; default is 5)
    use_particle_rnn_mode = args.use_particle_rnn_mode
    use_particle_free_rnn_mode = args.use_particle_free_rnn_mode

    if args.shooting_model == 'simple':
        smodel = smodels.AutoShootingIntegrandModelSimple(**shootingintegrand_kwargs,
                                                          use_analytic_solution=args.use_analytic_solution,
                                                          use_particle_rnn_mode=use_particle_rnn_mode)
    elif args.shooting_model == '2nd_order':
        smodel = smodels.AutoShootingIntegrandModelSecondOrder(**shootingintegrand_kwargs,
                                                               use_rnn_mode=use_particle_rnn_mode)
    elif args.shooting_model == 'updown':
        smodel = smodels.AutoShootingIntegrandModelUpDown(**shootingintegrand_kwargs,
                                                          use_analytic_solution=args.use_analytic_solution,
                                                          use_particle_rnn_mode=args.use_particle_rnn_mode,
                                                          inflation_factor=args.inflation_factor,
                                                          optimize_over_data_initial_conditions=args.optimize_over_data_initial_conditions)
    elif args.shooting_model == 'dampened_updown':
        smodel = smodels.AutoShootingIntegrandModelDampenedUpDown(**shootingintegrand_kwargs,
                                                                  inflation_factor=inflation_factor,
                                                                  use_rnn_mode=use_particle_rnn_mode)

    par_init = pi.VectorEvolutionParameterInitializer(
        only_random_initialization=True,
        random_initialization_magnitude=1.0)
    smodel.set_state_initializer(state_initializer=par_init)

    block_name = 'sblock'
    sblock = sblocks.ShootingBlockBase(
        name=block_name,
        shooting_integrand=smodel,
        integrator=integrator,
        use_particle_free_rnn_mode=use_particle_free_rnn_mode
    )
    sblock = sblock.to(device)

    # run through the shooting block once (to get parameters as needed)
    sblock(x=batch_initial)

    return sblock


def setup_optimizer_and_scheduler(params):

    optimizer = optim.Adam(params, lr=0.01)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,verbose=True)

    return optimizer, scheduler


def get_batch(batch_size=None,batch_tmax=None,batch_timesteps=None):

    if batch_size is None:
        batch_size = args.batch_size
    if batch_tmax is None:
        batch_tmax = args.batch_tmax
    if batch_timesteps is None:
        batch_timesteps = args.batch_timesteps

    # Choose random starting points, uniformly distributed from -15 to 15
    # np.random.seed(1)
    batch_initial = -15 + 30 * np.random.random((batch_size, 3))
    batch_t = np.linspace(0, batch_tmax, batch_timesteps)

    if args.ode_name == '3d_lorenz':
        batch_trajectory = np.asarray([integrate.odeint(lorentz_deriv, x0i, batch_t)
                          for x0i in batch_initial])
    elif args.ode_name == '3d_time_spiral': #TODO: add NeuPDE 3d time-dependent spiral
        batch_trajectory = np.asarray([integrate.odeint(lorentz_deriv, x0i, batch_t)
                          for x0i in batch_initial])

    # some formatting to get the right tensor size
    # batch_initial [nr_trajectories,1,dim]
    # batch_t [timesteps]
    # batch_trajectory [timesteps, nr_trajectories, 1, dim]

    batch_initial = torch.from_numpy(batch_initial).float().unsqueeze(dim=1)
    batch_t = torch.from_numpy(batch_t).float()
    batch_trajectory = torch.from_numpy(batch_trajectory).float()
    batch_trajectory = batch_trajectory.permute(1, 0, 2).unsqueeze(dim=2)

    return batch_initial, batch_t, batch_trajectory


def freeze_parameters(shooting_block,parameters_to_freeze):

    # get all the parameters that we are optimizing over
    pars = shooting_block.state_dict()
    for pn in parameters_to_freeze:
        print('Freezing {}'.format(pn))
        pars[pn].requires_grad = False

if __name__ == '__main__':

    args = setup_cmdline_parsing()
    if args.verbose:
        print(args)
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    setup_random_seed(seed=args.seed)

    integrator = setup_integrator(method=args.method, use_adjoint=args.adjoint)

    sblock = setup_shootingblock(integrator)

    freeze_parameters(sblock,['q1'])
    optimizer, scheduler = setup_optimizer_and_scheduler(params=sblock.parameters())

    for itr in range(0, args.niters):

        optimizer.zero_grad()

        batch_initial, batch_t, batch_trajectory = get_batch()
        sblock.set_integration_time_vector(integration_time_vector=batch_t, suppress_warning=True)
        pred_trajectory,_,_,_ = sblock(x=batch_initial)

        # todo: figure out wht the norm penality does not work
        if args.sim_norm == 'l1':
            loss = torch.mean(torch.abs(pred_trajectory - batch_trajectory))
        elif args.sim_norm == 'l2':
            loss = torch.mean(torch.norm(pred_trajectory-batch_trajectory,dim=3))
        else:
            raise ValueError('Unknown norm {}.'.format(args.sim_norm))

        loss = loss + args.shooting_norm_penalty * sblock.get_norm_penalty()

        if itr % args.test_freq == 0:
            print('Iter {:04d} | Training Loss {:.6f}'.format(itr, loss.item()))

        loss.backward()
        optimizer.step()

    print('Finished training')
    # evaluate on some unseen trajectories, visualize one random trajectory
    batch_initial, batch_t, batch_trajectory = get_batch()
    pred_trajectory, _, _, _ = sblock(x=batch_initial)

    if args.sim_norm == 'l1':
        loss = torch.mean(torch.abs(pred_trajectory - batch_trajectory))
    elif args.sim_norm == 'l2':
        loss = torch.mean(torch.norm(pred_trajectory - batch_trajectory, dim=3))
    else:
        raise ValueError('Unknown norm {}.'.format(args.sim_norm))
    loss = loss + args.shooting_norm_penalty * sblock.get_norm_penalty()

    print('Testing Loss {:.6f}'.format(loss.item()))
    nr_trajectories = batch_trajectory.shape[1]
    plot_trajectory({'predicted':pred_trajectory,'actual':batch_trajectory},nr_trajectories)
