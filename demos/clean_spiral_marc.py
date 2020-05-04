import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

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
    parser.add_argument('--batch_time', type=int, default=10, help='Length of the training samples.')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of training samples.')
    parser.add_argument('--niters', type=int, default=10000, help='Maximum nunber of iterations.')
    parser.add_argument('--batch_validation_size', type=int, default=5, help='Length of the samples for validation.')
    parser.add_argument('--seed', required=False, type=int, default=1234,
                        help='Sets the random seed which affects data shuffling')

    parser.add_argument('--linear', action='store_true', help='If specified the ground truth system will be linear, otherwise nonlinear.')

    parser.add_argument('--test_freq', type=int, default=200, help='Frequency with which the validation measures are to be computed.')
    parser.add_argument('--viz_freq', type=int, default=100, help='Frequency with which the results should be visualized; if --viz is set.')

    parser.add_argument('--validate_with_long_range', action='store_true', help='If selected, a long-range trajectory will be used; otherwise uses batches as for training')

    parser.add_argument('--nr_of_particles', type=int, default=40, help='Number of particles to parameterize the initial condition')
    parser.add_argument('--sim_norm', type=str, choices=['l1','l2'], default='l2', help='Norm for the similarity measure.')
    parser.add_argument('--shooting_norm_penalty', type=float, default=0, help='Factor to penalize the norm with; default 0, but 0.1 or so might be a good value')
    parser.add_argument('--nonlinearity', type=str, choices=['identity', 'relu', 'tanh', 'sigmoid'], default='relu', help='Nonlinearity for shooting.')


    parser.add_argument('--viz', action='store_true', help='Enable visualization.')
    parser.add_argument('--gpu', type=int, default=0, help='Enable GPU computation on specified GPU.')
    parser.add_argument('--adjoint', action='store_true', help='Use adjoint integrator to avoid storing values during forward pass.')

    args = parser.parse_args()

    return args

def setup_random_seed(seed):
    print('Setting the random seed to {:}'.format(seed))
    random.seed(seed)
    torch.manual_seed(seed)

def setup_integrator(method='rk4', use_adjoint=False, step_size=0.05, rtol=1e-8, atol=1e-12):

    integrator_options = dict()

    if method not in ['dopri5', 'adams']:
        integrator_options  = {'step_size': step_size}

    # TODO: remove, purely for debug
    integrator_options = {'step_size': 0.05}

    integrator = generic_integrator.GenericIntegrator(integrator_library = 'odeint', integrator_name = 'rk4',
                                                     use_adjoint_integration=use_adjoint, integrator_options=integrator_options, rtol=rtol, atol=atol)

    return integrator

def setup_shooting_block(nonlinearity='relu', device='cpu'):
    # TODO: make the selection of the model more flexible
    # shooting_model = shooting_models.AutoShootingIntegrandModelUpDown(in_features=2, nonlinearity=nonlinearity,
    #                                                                   parameter_weight=0.5,
    #                                                                   inflation_factor=1,
    #                                                                   nr_of_particles=50, particle_dimension=1,
    #                                                                   particle_size=2,
    #                                                                   use_analytic_solution=True,
    #                                                                   optimize_over_data_initial_conditions=True)

    shooting_model = shooting_models.AutoShootingIntegrandModelSimple(in_features=2, nonlinearity=nonlinearity,
                                                                      parameter_weight=0.1,
                                                                      nr_of_particles=50, particle_dimension=1,
                                                                      particle_size=2, use_analytic_solution=True)


    import neuro_shooting.parameter_initialization as pi
    # par_initializer = pi.VectorEvolutionSampleBatchParameterInitializer(only_random_initialization=False,
    #     random_initialization_magnitude=0.1,
    #     sample_batch=batch_y0)

    par_initializer = pi.VectorEvolutionParameterInitializer(only_random_initialization=True,
                                                             random_initialization_magnitude=1.0)

    shooting_model.set_state_initializer(state_initializer=par_initializer)
    shooting_block = shooting_blocks.ShootingBlockBase(name='simple', shooting_integrand=shooting_model, use_particle_free_rnn_mode=True)
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

def generate_data(integrator, data_size, linear=False, device='cpu'):

    d = dict()

    d['y0'] = torch.tensor([[2., 0.]]).to(device)
    #d['t'] = torch.linspace(0., 25., data_size).to(device)

    d['t'] = torch.linspace(0., 10., data_size).to(device)

    d['A'] = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)

    # pure slow oscillation
    #d['A'] = torch.tensor([[0, 0.025], [-0.025, 0]]).to(device)

    # small section
    #d['A'] = torch.tensor([[0, 0.1], [-0.1, 0]]).to(device)

    with torch.no_grad():
        # integrate it
        d['y'] = integrator.integrate(func=DiffEqRHS(A=d['A'], linear=linear), x0=d['y0'], t=d['t'])

    return d

def get_batch(data_dict, batch_size, batch_time):

    data_size = len(data_dict['t'])

    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), size=batch_size, replace=True)).to(device)
    batch_y0 = data_dict['y'][s]  # (M, D)
    batch_t = data_dict['t'][:batch_time]  # (T)
    batch_y = torch.stack([data_dict['y'][s + i] for i in range(batch_time)], dim=0)  # (T, M, D)

    return batch_y0, batch_t, batch_y

def __get_batch(data_dict, batch_size, batch_time):

    # not random, just directly return the simulated data. This should be the easiest test case

    batch_y0 = data_dict['y0'].unsqueeze(dim=0)
    batch_t = data_dict['t']
    batch_y = data_dict['y'].unsqueeze(dim=1)

    return batch_y0, batch_t, batch_y

# Visualization
# TODO: revamp

def basic_visualize(val_y, pred_y, sim_time, batch_y, batch_pred_y, batch_t):

    plt.figure()

    for n in range(val_y.size()[1]):
        plt.plot(val_y.detach().numpy()[:, n, 0, 0], val_y.detach().numpy()[:, n, 0, 1], 'g-')
        plt.plot(pred_y.detach().numpy()[:, n, 0, 0], pred_y.detach().numpy()[:, n, 0, 1], 'b--+')

    for n in range(batch_y.size()[1]):
        plt.plot(batch_y.detach().numpy()[:, n, 0, 0], batch_y.detach().numpy()[:, n, 0, 1], 'k-')
        plt.plot(batch_pred_y.detach().numpy()[:, n, 0, 0], batch_pred_y.detach().numpy()[:, n, 0, 1], 'r--')

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

if __name__ == '__main__':

    # do some initial setup
    args = setup_cmdline_parsing()
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    setup_random_seed(seed=args.seed)

    integrator = setup_integrator(method=args.method, use_adjoint=args.adjoint)
    shooting_block = setup_shooting_block(nonlinearity=args.nonlinearity, device=device)

    # generate the true data tha we want to match
    data = generate_data(integrator=integrator, data_size=args.data_size, linear=args.linear, device=device)

    # draw an initial batch from it
    batch_y0, batch_t, batch_y = get_batch(data_dict=data, batch_time=args.batch_time, batch_size=args.batch_size)

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
        val_y0, val_t, val_y = get_batch(data_dict=data, batch_time=args.batch_time, batch_size=args.batch_size)

    # run through the shooting block once (to get parameters as needed)
    shooting_block(x=batch_y)

    optimizer, scheduler = setup_optimizer_and_scheduler(params=shooting_block.parameters())

    # t_0 = time.time()
    # ### time clock
    # t_1 = time.time()
    # print("time", t_1 - t_0)
    # t_0 = t_1

    for itr in range(0, args.niters):

        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(data_dict=data, batch_time=args.batch_time, batch_size=args.batch_size)

        shooting_block.set_integration_time_vector(integration_time_vector=batch_t, suppress_warning=True)
        pred_y,_,_,_ = shooting_block(x=batch_y0)

        # if args.sim_norm == 'l1':
        #     loss = torch.mean(torch.abs(pred_y - batch_y))
        # elif args.sim_norm == 'l2':
        #     loss = torch.mean(torch.norm(pred_y-batch_y,dim=3))
        # else:
        #     raise ValueError('Unknown norm {}.'.format(args.sim_norm))

        # TODO: maybe put this norm loss back in
        #loss = loss + args.shooting_norm_penalty * shooting_block.get_norm_penalty()

        loss = torch.mean(torch.abs(pred_y - batch_y))
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

                shooting_block.set_integration_time_vector(integration_time_vector=val_t, suppress_warning=True)
                val_pred_y,_,_,_ = shooting_block(x=val_y0)

                # if args.sim_norm=='l1':
                #     loss = torch.mean(torch.abs(val_pred_y - val_y))
                # elif args.sim_norm=='l2':
                #     loss = torch.mean(torch.norm(val_pred_y - val_y, dim=3))
                # else:
                #     raise ValueError('Unknown norm {}.'.format(args.sim_norm))

                #loss = loss #+ shooting_block.get_norm_penalty()

                loss = torch.mean(torch.abs(val_pred_y - val_y))

                print('Iter {:04d} | Validation Loss {:.6f}'.format(itr, loss.item()))

            if itr % args.viz_freq == 0:
                #basic_visualize(val_y, val_pred_y, val_t, batch_y, pred_y, batch_t)

                visualize(val_y, val_pred_y, val_t, shooting_block, itr)

                # # test two different time intervals
                # val_t0 = data['t'][0:50]
                # val_t1 = data['t'][0:100]
                #
                # shooting_block.set_integration_time_vector(integration_time_vector=val_t0, suppress_warning=True)
                # val_pred_y0, _, _, _ = shooting_block(x=val_y0)
                #
                # shooting_block.set_integration_time_vector(integration_time_vector=val_t1, suppress_warning=True)
                # val_pred_y1, _, _, _ = shooting_block(x=val_y0)
                #
                # tst = val_pred_y1[0:50, 0, 0, :] - val_pred_y0[0:50, 0, 0, :]
                #
                # print('Hello world')

        #end = time.time()