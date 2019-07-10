import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Function

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

class ShootingFunc(Function):

    @staticmethod
    def forward(ctx, input, batch_t, x_params, p_params, Mbar, Mbar_b):

        ctx.save_for_backward(input, batch_t, x_params, p_params, Mbar, Mbar_b)

        K = x_params.shape[0]
        d = x_params.shape[2]

        # solve equation 3 for theta
        theta = torch.matmul(-p_params.squeeze().transpose(0,1), nn.functional.relu(x_params.squeeze()))
        theta = torch.matmul(torch.inverse(Mbar), theta)

        # solve equation 4 for bias
        bias = torch.matmul(-p_params.squeeze().transpose(0,1), torch.ones([K,1]))
        bias = torch.matmul(torch.inverse(Mbar_b), bias)

        # right hand side of equation 1
        def odefunc_x(t, x, theta, bias):
            shapex = nn.functional.relu(x.squeeze(dim=1).transpose(0,1))
            prod = torch.matmul(theta,shapex)
            rhs = prod + bias
            return rhs.unsqueeze(dim=1).transpose(0,2)

        # right hand side of equation 2
        def odefunc_p(t, p, x, theta):

            # Drelu = torch.eye(d)
            # prod = torch.matmul(Drelu,theta.transpose(0,1))
            # rhs = torch.matmul(prod, p.squeeze().transpose(0,1))

            rhs=[]
            for i in range(0,K):
                Drelu = torch.diag((x[i,0,:]>=0).type(torch.FloatTensor))
                prod = torch.matmul(Drelu, theta.transpose(0, 1))
                rhs.append(torch.matmul(prod, p[i,:,:].transpose(0,1)))
            rhs = torch.cat(rhs,1)

            return rhs.unsqueeze(dim=1).transpose(0,2)

        # Equation 1 at time 1 for INITIAL conditions x_params
        func = partial(odefunc_x,theta=theta,bias=bias)
        x_params = odeint(func, x_params, torch.tensor([1]).float())
        x_params = torch.squeeze(x_params, dim=0)

        # Equation 2 at time 1 for FINAL conditions p_params
        func = partial(odefunc_p, x=x_params, theta=theta)
        p_params = odeint(func, p_params, torch.tensor([-1]).float())
        p_params = torch.squeeze(p_params, dim=0)

        # Once again solve equations 3 and 4
        theta = torch.matmul(-p_params.squeeze().transpose(0,1), nn.functional.relu(x_params.squeeze()))
        bias = torch.matmul(-p_params.squeeze().transpose(0,1), torch.ones([K,1]))

        # Equation 1 at time 1 for initial condition xsample
        func = partial(odefunc_x,theta=theta,bias=bias)
        output = odeint(func, input, batch_t)

        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx,grad_output):

        return None, None, None, None, None, None

class ShootingBlock(nn.Module):
    def __init__(self, batch_y0, Mbar=None, Mbar_b=None):
        super(ShootingBlock, self).__init__()

        K = batch_y0.shape[0]
        d = batch_y0.shape[2]

        if Mbar is None:
            self.Mbar = torch.eye(d)
        if Mbar_b is None:
            self.Mbar_b = torch.eye(d)

        self.x_params = nn.Parameter(batch_y0)
        self.p_params = nn.Parameter(torch.zeros(K,1,d))


    def forward(self, input, batch_t):
        """

        :param input: 3D tensor of minibatch x 1 x feature dimension holding initial conditions
        :param batch_t: 1D tensor holding time points for evaluation
        :return: |batch_t| x minibatch x 1 x feature dimension
        """

        return ShootingFunc.apply(input, batch_t, self.x_params, self.p_params, self.Mbar, self.Mbar_b)

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
        K = 5
        batch_y0, batch_t, batch_y = get_batch(K)
        shooting = ShootingBlock(batch_y0)
        optimizer = optim.RMSprop(shooting.parameters(), lr=1e-3)

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

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():

                if is_odenet:
                    pred_y = odeint(func, true_y0, t)
                else:
                    pred_y = shooting(true_y0.unsqueeze(dim=0), t)

                loss = torch.mean(torch.abs(pred_y.squeeze(dim=1) - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                # visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()