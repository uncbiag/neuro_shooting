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
    parser.add_argument('--batch_time', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--niters', type=int, default=2000)
    parser.add_argument('--test_freq', type=int, default=10)
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    return args
    

def setup_problem(args):
    """Sets up the true trajectory"""

    true_y0 = torch.tensor([[2., 0.]])
    true_t = torch.linspace(0., 25., args.data_size)
    true_A = torch.tensor([
        [-0.1, 2.0],
        [-2.0, -0.1]])

    class Lambda(nn.Module):
        def forward(self, t, y):
            return torch.mm(y, true_A)

    with torch.no_grad():
        true_y = odeint(Lambda(), true_y0, true_t, method='dopri5')

    return true_y0, true_t, true_A, true_y


def get_batch(args, true_y, true_t):
    # select random indices i1,...,iB
    s = torch.from_numpy(
        np.random.choice(
            np.arange(args.data_size - args.batch_time, dtype=np.int64), 
            args.batch_size, 
            replace=False))
    # get the starting points corresponding to i1,...,iB 
    # (i.e., the points on the true trajectory)
    batch_y0 = true_y[s]  
    # each batch trajectory will start at t=0,...
    batch_t = true_t[:args.batch_time] 
    # create a batch such that each batch element represents couple of 
    # sequential points along the trajectory, when starting from i1,i2, ...
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)
    return batch_y0, batch_t, batch_y


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def visualize(true_y, pred_y, t, ax_traj, ax_phase, ax_vecfield, odefunc):

    ax_traj.cla()
    ax_traj.set_title('Trajectories')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('x,y')

    # plot checked
    ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-')
    ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], '--', t.numpy(), pred_y.numpy()[:, 0, 1], 'b--')
    ax_traj.set_xlim(t.min(), t.max())
    ax_traj.set_ylim(-2, 2)
    ax_traj.legend()

    # plot checked
    ax_phase.cla()
    ax_phase.set_title('Phase Portrait')
    ax_phase.set_xlabel('x')
    ax_phase.set_ylabel('y')
    ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
    ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')
    ax_phase.set_xlim(-3, 3)
    ax_phase.set_ylim(-3, 3)

    # vector field
    ax_vecfield.cla()
    ax_vecfield.set_title('Learned Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')

    y, x = np.mgrid[-2:2:21j, -2:2:21j]
    viz_time = t[:10]
    odefunc.set_integration_time_vector(
        integration_time_vector=viz_time,
        suppress_warning=True
    )
    x_0 = torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).unsqueeze(dim=1)
    dydt_pred_y,_,_,_ = odefunc(x=x_0)
    dydt = dydt_pred_y[-1,:,0,:].cpu().detach().numpy()

    mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
    dydt = (dydt / mag)
    dydt = dydt.reshape(21, 21, 2)

    ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    ax_vecfield.set_xlim(-2, 2)
    ax_vecfield.set_ylim(-2, 2)

    fig.tight_layout()
    plt.draw()
    plt.pause(0.001)


if __name__ == '__main__':

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)

    args = setup_cmdline_parsing()
    if args.verbose:
        print(args)

    true_y0, true_t, true_A, true_y = setup_problem(args)
    assert true_y.size() == torch.Size([args.data_size, 1, 2])

    sample_batch_y0, sample_batch_t, sample_batch_y = get_batch(args, true_y, true_t)
    assert sample_batch_t.size() == torch.Size([args.batch_time])
    assert sample_batch_y0.size() == torch.Size([args.batch_size, 1, 2])
    assert sample_batch_y.size() == torch.Size([args.batch_time, args.batch_size, 1, 2])

    par_init = pi.VectorEvolutionParameterInitializer(
        only_random_initialization=True,
        random_initialization_magnitude=50)
    
    smodel = smodels.AutoShootingIntegrandModelSimple(
        in_features=2,
        nonlinearity='tanh',
        parameter_weight=30,
        nr_of_particles=300,
        particle_dimension=1,
        particle_size=2)
    smodel.set_state_initializer(state_initializer=par_init)

    sblock = sblocks.ShootingBlockBase(
        name='simple',
        shooting_integrand=smodel,
        integrator_name='dopri5'
        #intgrator_options = {'stepsize':0.1}
    )

    sblock(x=sample_batch_y)
    optimizer = optim.Adam(sblock.parameters(), lr=1e-1)

    track_loss = []
    for itr in range(1, args.niters + 1):
    
        # get batch data
        batch_y0, batch_t, batch_y = get_batch(args, true_y, true_t)

        # set integration time
        sblock.set_integration_time_vector(
            integration_time_vector=batch_t, 
            suppress_warning=True)
        
        pred_y,_,_,_ = sblock(x=batch_y0)
        
        loss = torch.mean(torch.abs(pred_y - batch_y)) #+ 1e-2 * sblock.get_norm_penalty()
        loss.backward()
        
        track_loss.append(loss.item())
        optimizer.step()

        sblock.set_integration_time_vector(
                    integration_time_vector=true_t, 
                    suppress_warning=True)
        
        if itr % args.test_freq == 0:
            with torch.no_grad():
                sblock.set_integration_time_vector(
                        integration_time_vector=true_t, 
                        suppress_warning=True)
                val_pred_y,_,_,_ = sblock(x=true_y0.unsqueeze(0))
                val_loss = torch.mean(torch.abs(val_pred_y[:,0,:,:] - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, val_loss.item()))
                visualize(true_y, 
                    val_pred_y[:,0,:,:], 
                    true_t, 
                    ax_traj, 
                    ax_phase, 
                    ax_vecfield,
                    sblock)




