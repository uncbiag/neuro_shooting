# Perform sanity check for continuous affine layer: f(t,y(t)) = A_0 \sigma(y(t)) + b_0
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys

import neuro_shooting.generic_integrator as generic_integrator
from neuro_shooting.shooting_models import AutoShootingIntegrandModelSimple as Simple
from neuro_shooting.shooting_blocks import ShootingBlockBase as Base
import neuro_shooting.shooting_blocks as sblocks
import neuro_shooting.shooting_models as smodels
import neuro_shooting.parameter_initialization as pi

# the simulation involves random b_0 and random initial condition y_0
random_seed = 7
torch.manual_seed(random_seed)
np.random.seed(random_seed)
gpu = 0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')

def setup_cmdline_parsing():
    parser = argparse.ArgumentParser('Sanity check: shooting affine layer')
    parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
    parser.add_argument('--in_features', type=int, default=2)
    parser.add_argument('--data_size', type=int, default=250)
    parser.add_argument('--t_max', type=int, default=10)
    parser.add_argument('--loss_terminal_only',action='store_true', help='If selected, loss only penalizes terminal prediction')
    parser.add_argument('--batch_time', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--true_nonlinearity', type=str, default='tanh', choices=['identity', 'relu', 'tanh', 'sigmoid'], help='Nonlinearity that generates the data.')
    parser.add_argument('--assumed_nonlinearity', type=str, default='tanh', choices=['identity', 'relu', 'tanh', 'sigmoid'], help='Nonlinearity for shooting.')
    parser.add_argument('--pw', type=float, default=0.5, help='parameter weight')
    parser.add_argument('--nr_of_particles', type=int, default=10,
                        help='Number of particles to parameterize the initial condition')
    parser.add_argument('--niters', type=int, default=500)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--validate_with_long_range',action='store_true', help='If selected, a long-range trajectory will be used; otherwise uses batches as for training')
    parser.add_argument('--batch_validation_size', type=int, default=100,
                        help='number of validation trajectories (each of time length batch_time).')
    parser.add_argument('--log_to_file',action='store_true',help='If selected, all things printed on screen are saved in file and not displayed on screen')

    args = parser.parse_args()
    return args


def setup_problem(args):
    """Sets up the true trajectory"""

    # random initial condition
    true_y0 = torch.rand((1, args.in_features)).to(device)

    true_t = torch.linspace(0., args.t_max, args.data_size).to(device)

    true_A = torch.tensor([
        [-0.1, -1.0],
        [1.0, -0.1]]).to(device)
    true_b = torch.rand((1, args.in_features)).to(device)

    ## Generate (noiseless) trajectory from dy/dt = f(t,y(t))
    ## where f(t,y(t))  = true_A \sigma(y(t)) + true_b
    class Lambda(nn.Module):
        def forward(self, t, y):
            if args.true_nonlinearity == 'tanh':
                return torch.mm(torch.tanh(y), true_A) + true_b
            elif args.true_nonlinearity == 'relu':
                return torch.mm(torch.relu(y), true_A) + true_b
            if args.true_nonlinearity == 'sigmoid':
                return torch.mm(torch.sigmoid(y), true_A) + true_b
            if args.true_nonlinearity == 'identity':
                return torch.mm(y, true_A) + true_b

    stepsize = 0.5
    integrator_options = {'step_size': stepsize}
    rtol = 1e-8
    atol = 1e-10
    adjoint = False
    integrator = generic_integrator.GenericIntegrator(integrator_library='odeint',
                                                      integrator_name=args.method,
                                                      use_adjoint_integration=adjoint,
                                                      integrator_options=integrator_options,
                                                      rtol=rtol, atol=atol)

    with torch.no_grad():
        true_y = integrator.integrate(func=Lambda(), x0=true_y0, t=true_t)

    return true_y0, true_t, true_A, true_b, true_y


def get_batch(batch_size,batch_time):
    s = torch.from_numpy(
        np.random.choice(np.arange(args.data_size - batch_time, dtype=np.int64), batch_size, replace=False)).to(device)
    batch_y0 = true_y[s]  # (M, D)
    batch_t = true_t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


def to_np(x):
    return x.detach().cpu().numpy()


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def plot_trajectories(true_y, pred_y=None, sim_time=None, save=None, figsize=(16, 8)):
    # plt.figure(figsize=figsize)
    plt.subplot(122)

    if true_y is not None:
        if sim_time is None:
            sim_time = [None] * len(true_y)
        for o, t in zip(true_y, sim_time):
            o, t = to_np(o), to_np(t)
            plt.scatter(o[:, :, 0], o[:, :, 1], c=t, cmap=cm.plasma, label='observations (colored by time)')

    if pred_y is not None:
        for z in pred_y:
            z = to_np(z)
            plt.plot(z[:, :, 0], z[:, :, 1], lw=1.5, label="prediction")
        if save is not None:
            plt.savefig(save)

    plt.legend()
    plt.title('Trajectory: observed versus predicted')
    plt.xlabel('y_1')
    plt.ylabel('y_2')
    plt.show()


def visualize(true_y, pred_y, sim_time, odefunc, is_odenet=False, is_higher_order_model=False, savepath=None):


    quiver_scale = 2.5 # to scale the magnitude of the quiver vectors for visualization

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)

    ax_traj.cla()
    ax_traj.set_title('Trajectories')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('y1,y2')

    # true_y and pred_y: [time points, sample size, 1, dimension of y]
    # this graphic is unintelligible, too much overcrowding
    for n in range(true_y.size()[1]):
        ax_traj.plot(sim_time.numpy(), true_y.detach().numpy()[:, n, 0, 0], 'g-',
                     sim_time.numpy(), true_y.numpy()[:, n, 0, 1], 'b-')
        ax_traj.plot(sim_time.numpy(), pred_y.detach().numpy()[:, n, 0, 0], 'g--',
                     sim_time.numpy(), pred_y.detach().numpy()[:, n, 0, 1],'b--')


    ax_traj.set_xlim(sim_time.min(), sim_time.max())
    ax_traj.set_ylim(-2, 2)
    # ax_traj.legend()

    ax_phase.cla()
    ax_phase.set_title('Phase Portrait')
    ax_phase.set_xlabel('y1')
    ax_phase.set_ylabel('y2')

    xlim_min, xlim_max, ylim_min, ylim_max = [], [], [], []
    for n in range(true_y.size()[1]):
        ax_phase.plot(true_y.detach().numpy()[:, n, 0, 0], true_y.detach().numpy()[:, n, 0, 1], 'g-')
        ax_phase.plot(pred_y.detach().numpy()[:, n, 0, 0], pred_y.detach().numpy()[:, n, 0, 1], 'b--')
        xlim_min += [np.min([np.min(true_y.detach().numpy()[:, n, 0, 0]),np.min(pred_y.detach().numpy()[:, n, 0, 0])])]
        xlim_max += [np.max([np.max(true_y.detach().numpy()[:, n, 0, 0]),np.max(pred_y.detach().numpy()[:, n, 0, 0])])]
        ylim_min += [np.min([np.min(true_y.detach().numpy()[:, n, 0, 1]),np.min(pred_y.detach().numpy()[:, n, 0, 1])])]
        ylim_max += [np.max([np.max(true_y.detach().numpy()[:, n, 0, 1]),np.max(pred_y.detach().numpy()[:, n, 0, 1])])]

    if not is_odenet:

        try:
            q = (odefunc.q_params)
            p = (odefunc.p_params)

            q_np = q.cpu().detach().squeeze(dim=1).numpy()
            p_np = p.cpu().detach().squeeze(dim=1).numpy()

            ax_phase.scatter(q_np[:,0],q_np[:,1],marker='+')
            ax_phase.quiver(q_np[:,0],q_np[:,1], p_np[:,0],p_np[:,1],color='r', scale=quiver_scale)
        except:
            pass


    ax_phase.set_xlim(min(xlim_min),max(xlim_max))
    ax_phase.set_ylim(min(ylim_min),max(ylim_max))


    ax_vecfield.cla()
    ax_vecfield.set_title('Learned Vector Field')
    ax_vecfield.set_xlabel('y1')
    ax_vecfield.set_ylabel('y2')

    y, x = np.mgrid[-2:2:21j, -2:2:21j]

    current_y = torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))

    # print("q_params",q_params.size())

    if not is_odenet:
        x_0 = current_y.unsqueeze(dim=1)

        viz_time = sim_time[:5] # just 5 timesteps ahead

        odefunc.set_integration_time_vector(integration_time_vector=viz_time,suppress_warning=True)
        dydt_pred_y,_,_,_ = odefunc(x=x_0)

        if is_higher_order_model:
            dydt = (dydt_pred_y[-1,...]-dydt_pred_y[0,...]).detach().numpy()
            dydt = dydt[:,0,...]
        else:
            dydt = dydt_pred_y[-1,0,...]

    else:
        dydt = odefunc(0, current_y).cpu().detach().numpy()

    mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
    dydt = (dydt / mag)
    dydt = dydt.reshape(21, 21, 2)

    ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")

    if not is_odenet:
        try:
            ax_vecfield.scatter(q_np[:, 0], q_np[:, 1], marker='+')
            ax_vecfield.quiver(q_np[:,0],q_np[:,1], p_np[:,0],p_np[:,1],color='r', scale=quiver_scale)
        except:
            pass

    ax_vecfield.set_xlim(-2, 2)
    ax_vecfield.set_ylim(-2, 2)

    fig.tight_layout()

    print('Plotting')
    plt.savefig('{}/traj_phase_vf.png'.format(savepath))
    plt.draw()
    plt.pause(0.001)
    plt.show()


if __name__ == '__main__':

    args = setup_cmdline_parsing()

    saveresultspath = \
        'sanity_check_results/t_max{}/numparticles{}/pw{}/true_nonlinearity{}/assumed_nonlinearity{}'.\
            format(args.t_max,
                   args.nr_of_particles,
                   args.pw,
                   args.true_nonlinearity,
                   args.assumed_nonlinearity)


    def makedirs(dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)


    makedirs(saveresultspath)

    if args.log_to_file:
        stdoutOrigin = sys.stdout
        sys.stdout = open("{}/log.txt".format(saveresultspath), "w")


    print(args)


    true_y0, true_t, true_A, true_b, true_y = setup_problem(args)
    print('true_y0: {}'.format(true_y0))
    print('true_A: {}'.format(true_A))
    print('true_b: {}'.format(true_b))

    assert true_y.size() == torch.Size([args.data_size, 1, 2])

    sample_batch_y0, sample_batch_t, sample_batch_y = get_batch(args.batch_size,args.batch_time)
    assert sample_batch_t.size() == torch.Size([args.batch_time])
    assert sample_batch_y0.size() == torch.Size([args.batch_size, 1, 2])
    assert sample_batch_y.size() == torch.Size([args.batch_time, args.batch_size, 1, 2])

    # model = Model(in_features=args.in_features, pw=args.pw, nr_of_particles=args.nr_of_particles)

    par_init = pi.VectorEvolutionParameterInitializer(
        only_random_initialization=True,
        random_initialization_magnitude=50)

    smodel = smodels.AutoShootingIntegrandModelSimple(
        in_features=args.in_features,
        nonlinearity=args.assumed_nonlinearity,
        parameter_weight=args.pw,
        nr_of_particles=args.nr_of_particles,
        particle_dimension=1,
        particle_size=2)
    smodel.set_state_initializer(state_initializer=par_init)

    sblock = sblocks.ShootingBlockBase(
        name='simple',
        shooting_integrand=smodel,
        integrator_name='dopri5'
        # intgrator_options = {'stepsize':0.1}
    )

    sblock(x=sample_batch_y)
    opt = torch.optim.Adam(sblock.parameters(), lr=0.1)

    print('Begin training on trajectories of length {}'.format(args.batch_time))
    for epoch in range(1, args.niters + 1):

        # each minibatch trajectory is a chopped up part of the original trajectory
        batch_y0, batch_t, batch_y = get_batch(args.batch_size,args.batch_time)
        # batch_y0 (batch_size, 1, dimension)
        # batch_y (args.batch_time, batch_size, 1, dimension)
        # batch_t (args.batch_time)

        # zero-out gradients
        sblock.zero_grad()

        # get output from shooting block + model output
        # set integration time
        sblock.set_integration_time_vector(
            integration_time_vector=batch_t,
            suppress_warning=True)

        pred_batch_terminal, _, _, _ = sblock(batch_y0)  # (batch_size, 1, dimension)
        
        if args.loss_terminal_only:
            loss = torch.mean(torch.norm(pred_batch_terminal[args.batch_time - 1, :, :] - batch_y[args.batch_time - 1, :, :], dim=2))
            if epoch % args.print_freq == 0:
                print('Epoch {} | Total Loss (terminal only) {:.2f}'.format(epoch, loss.item()))
        else:

            loss = torch.mean(torch.norm(pred_batch_terminal - batch_y, dim=2))
            if epoch % args.print_freq == 0:
                print('Epoch {} | Total Loss (trajectory) {:.4f}'.format(epoch, loss.item()))

        # backprop
        loss.backward()
        # take gradient step
        opt.step()


    print('Finished training on trajectories of length {}'.format(args.batch_time))

    # evaluate
    with torch.no_grad():

        if args.validate_with_long_range:
            print('original trajectory of length {}: visualizing actual versus predicted'.format(args.data_size))

            sblock.set_integration_time_vector(
                integration_time_vector=true_t,
                suppress_warning=True)

            pred_y_trajectory, _, _, _ = sblock(true_y0)

            visualize(true_y, pred_y_trajectory, true_t, sblock, is_odenet=False,
                      is_higher_order_model=True, savepath=saveresultspath)

        else:
            print('Pick {} random trajectories of length {}: visualizing actual versus predicted'
                  .format(args.batch_validation_size,args.batch_time))
            batch_y0, batch_t, batch_y = get_batch(batch_size=args.batch_validation_size, batch_time=args.batch_time)

            sblock.set_integration_time_vector(
                integration_time_vector=batch_t,
                suppress_warning=True)

            pred_batch_trajectory,_,_,_ = sblock(batch_y0)

            visualize(batch_y, pred_batch_trajectory, batch_t, sblock, is_odenet=False,
                      is_higher_order_model=True, savepath=saveresultspath)

    if args.log_to_file:
        sys.stdout.close()
        sys.stdout=stdoutOrigin
