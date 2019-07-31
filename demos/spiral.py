import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import custom_lr_scheduler

from functools import partial

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--network', type=str, choices=['odenet', 'shooting'], default='shooting')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[2., 0.]])
t = torch.linspace(0., 25., args.data_size)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])


class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(y**3, true_A)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')


def get_batch(batch_size=None):
    if batch_size is None:
        batch_size = args.batch_size
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], '--', t.numpy(), pred_y.numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.min(), t.max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y**3)


class ShootingBlock(nn.Module):
    def __init__(self, batch_y0, Mbar=None, Mbar_b=None):
        super(ShootingBlock, self).__init__()

        self.K = batch_y0.shape[0]
        self.d = batch_y0.shape[2]

        if Mbar is None:
            self.Mbar = torch.eye(self.d)
        else:
            self.Mbar = Mbar
        if Mbar_b is None:
            self.Mbar_b = torch.eye(self.d)
        else:
            self.Mbar_b = Mbar_b

        self.x_params = nn.Parameter(batch_y0)
        self.p_params = nn.Parameter(torch.zeros(self.K,1,self.d))


    def forward(self, input, batch_t):
        """

        :param input: 3D tensor of minibatch x 1 x feature dimension holding initial conditions
        :param batch_t: 1D tensor holding time points for evaluation
        :return: |batch_t| x minibatch x 1 x feature dimension
        """

        # solve Equation 3 for theta
        theta = torch.matmul(-self.p_params.squeeze().transpose(0, 1), nn.functional.relu(self.x_params.squeeze()))
        theta = torch.matmul(torch.inverse(self.Mbar), theta)

        # solve Equation 4 for bias
        bias = torch.matmul(-self.p_params.squeeze().transpose(0, 1), torch.ones([K, 1]))
        bias = torch.matmul(torch.inverse(self.Mbar_b), bias)

        # right hand side of Equation 1
        def odefunc_x(t, x, theta, bias):
            relux = nn.functional.relu(x.squeeze(dim=1).transpose(0, 1))
            prod = torch.matmul(theta, relux)
            rhs = prod + bias
            return rhs.unsqueeze(dim=1).transpose(0, 2)

        # right hand side of Equation 2
        def odefunc_p(t, p, x, theta):
            a = (x>=0).type(torch.FloatTensor).squeeze(dim=1)
            b = torch.eye(a.size(1))
            c = a.unsqueeze(2).expand(*a.size(), a.size(1))
            Drelu = c * b
            repeat_theta = torch.cat(self.K*[theta.transpose(0, 1).unsqueeze(0)])
            prod = torch.einsum('ijk,ikl->ijl', [Drelu, repeat_theta])
            return torch.einsum('ijk,ikl->ijl', [-prod, p.transpose(1,2)]).transpose(1,2)

        def RHS(t, concat_input, theta, bias, K):
            x = torch.index_select(concat_input,0,torch.arange(0,K))
            p = torch.index_select(concat_input,0,torch.arange(K,2*K))
            input = torch.index_select(concat_input,0,torch.arange(2*self.K,concat_input.shape[0]))
            return torch.cat((odefunc_x(t, x, theta, bias), odefunc_p(t, p, x, theta), odefunc_x(t, input, theta, bias)),0)

        func = partial(RHS, theta=theta,bias=bias, K=self.K)
        concat_input = torch.cat((self.x_params,self.p_params,input),0)
        output = odeint(func,concat_input,batch_t)
        return torch.index_select(output,1,torch.arange(2*self.K,concat_input.shape[0]))

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0

    is_odenet = args.network == 'odenet'

    if is_odenet:
        func = ODEFunc()
        optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    else:

        # parameters to play with for shooting
        K = 25
        Mbar = 2*torch.eye(2)
        Mbar_b = 2*torch.eye(2)
        #

        x_params, _, _ = get_batch(K)
        shooting = ShootingBlock(x_params, Mbar, Mbar_b)
        optimizer = optim.RMSprop(shooting.parameters(), lr=1e-3)
        # there are also parameters that can be adjusted in the custom lr scheduler
        scheduler = custom_lr_scheduler.CustomReduceLROnPlateau(optimizer, 'min', verbose=True)

    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)


    for itr in range(1, args.niters + 1):

        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()

        if is_odenet:
            pred_y = odeint(func, batch_y0, batch_t)
        else:
            pred_y = shooting(batch_y0, batch_t)

        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        # between epochs
        if not is_odenet:
            scheduler.step(loss.item())
            if scheduler.has_convergence_been_reached():
                print('INFO: Converence has been reached. Stopping iterations.')
                break

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():

                if is_odenet:
                    pred_y = odeint(func, true_y0, t)
                    loss = torch.mean(torch.abs(pred_y.squeeze(dim=1) - true_y))
                    print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                    # TODO: default visualize does not work for odenet
                    # visualize(true_y, pred_y, func, ii)
                else:
                    pred_y = shooting(true_y0.unsqueeze(dim=0), t)
                    loss = torch.mean(torch.abs(pred_y.squeeze(dim=1) - true_y))
                    print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                    # TODO: implement visualize for shooting
                    # visualize(true_y, pred_y, func, ii)

                ii += 1

        end = time.time()
