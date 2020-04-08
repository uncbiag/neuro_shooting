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

import neuro_shooting.generic_integrator as generic_integrator
from neuro_shooting.shooting_models import AutoShootingIntegrandModelUpDown as UpDown
from neuro_shooting.shooting_models import AutoShootingIntegrandModelSimple as Simple
from neuro_shooting.shooting_models import AutoShootingIntegrandModelSecondOrder as SecOrder
from neuro_shooting.shooting_blocks import ShootingBlockBase as Base

# the simulation involves random b_0 and random initial condition y_0
random_seed = 7
torch.manual_seed(random_seed)
np.random.seed(random_seed)

gpu = 0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')


## Generate (noiseless) trajectory from dy/dt = f(t,y(t))
## where f(t,y(t))  = A_0 \sigma(y(t)) + b_0

dimension = 2
# true_A = torch.rand((dimension,dimension)).to(device)
true_A = torch.tensor([[-0.1, -1.0], [1.0, -0.1]]).to(device)
# random b
true_b = torch.rand((1, dimension)).to(device)


class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(F.relu(y), true_A) + true_b


stepsize = 0.5
integrator_options = dict()
integrator_options = {'step_size': stepsize}
rtol = 1e-8
atol = 1e-10
adjoint = False
integrator = generic_integrator.GenericIntegrator(integrator_library='odeint',
                                                  integrator_name='rk4',
                                                  use_adjoint_integration=adjoint,
                                                  integrator_options=integrator_options,
                                                  rtol=rtol, atol=atol)
# random initial condition
true_y0 = torch.rand((1,dimension)).to(device)
# true_y0 = torch.tensor([[0.6, 0.3]]).to(device)
t_max = 10
data_size = 250
t = torch.linspace(0., t_max, data_size).to(device)
with torch.no_grad():
    true_y = integrator.integrate(func=Lambda(), x0=true_y0, t=t)

fig = plt.figure(figsize=(16, 8), facecolor='white')
ax_truetraj = fig.add_subplot(121, frameon=False)
ax_trueversuspred_batch = fig.add_subplot(122, frameon=False)

ax_truetraj.scatter(true_y[:,:,0],true_y[:,:,1],c=t, cmap=cm.plasma, label='observations (colored by time)')
ax_truetraj.scatter(true_y0[:,0],true_y0[:,1],marker="x", s= 160, label='initial condition')
ax_truetraj.set_xlabel('y_1')
ax_truetraj.set_xlabel('y_2')
ax_truetraj.legend()
ax_truetraj.set_title('true dynamics')



def get_batch(batch_size, batch_time):
    s = torch.from_numpy(
        np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False)).to(device)
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


def to_np(x):
    return x.detach().cpu().numpy()


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

saveimgname = 'affinelayer_sanitycheck'
makedirs('png/{}'.format(saveimgname))


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


class Model(nn.Module):
    def __init__(self, in_features, nr_of_particles=5, pw=1.0):
        super(Model, self).__init__()

        self.int = Simple(in_features, 'relu', nr_of_particles=nr_of_particles, parameter_weight=pw)
        self.blk = Base('shooting_block', shooting_integrand=self.int)

    def trajectory(self, batch_y0, batch_t):
        # batch_t defines time steps for trajectory
        # set time steps
        self.blk.set_integration_time_vector(batch_t, suppress_warning=True)
        # run through shooting block
        out = self.blk(batch_y0)
        # reset integration time
        self.blk.set_integration_time(t_max)
        return out

    def forward(self, batch_y0):
        pred_y, _, _, _ = self.blk(batch_y0)
        return pred_y


N_epochs = 100
N_particles = 50
pw = 0.5
batch_size = 10
batch_time = 25

model = Model(in_features=dimension, pw=pw, nr_of_particles=N_particles)
opt = torch.optim.Adam(model.parameters(), lr=0.1)

print_freq = 10
print('Begin training on trajectories of length {}'.format(batch_time))
for epoch in range(1, N_epochs + 1):

    # each minibatch trajectory is a chopped up part of the original trajectory
    batch_y0, batch_t, batch_y = get_batch(batch_size, batch_time)

    # zero-out gradients
    model.zero_grad()
    # get output from shooting block + model output
    pred_batch_terminal = model(batch_y0)
    # compute loss

    loss = torch.mean(torch.norm(pred_batch_terminal - batch_y[batch_time - 1, :, :], dim=2))
    if epoch % print_freq == 0:
        print('Epoch {} | Total Loss (terminal only) {:.2f}'.format(epoch, loss.item()))

    # backprop
    loss.backward()
    # take gradient step
    opt.step()
print('Finished training on trajectories of length {}'.format(batch_time))

print('Pick a random trajectory of length {}: visualize actual versus predicted'.format(batch_time))
batch_y0, batch_t, batch_y = get_batch(batch_size=1, batch_time=batch_time)
pred_batch_trajectory, _, _, _ = model.trajectory(batch_y0, batch_t)
plot_trajectories([batch_y[:, 0, :, :]],
                  [pred_batch_trajectory[:, 0, :, :]],
                  [batch_t],
                  save='png/affinelayer_sanitycheck/seed{}'.format(random_seed),
                  figsize=(8, 8))