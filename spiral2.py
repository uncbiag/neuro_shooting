import os
import argparse
import time
import numpy as np
import custom_optimizers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function,Variable

from functools import partial

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--network', type=str, choices=['odenet', 'shooting'], default='shooting')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=1)
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



class ShootingBlock2(nn.Module):
    def __init__(self, batch_y0, Kbar=None, Kbar_b=None):
        super(ShootingBlock2, self).__init__()

        self.k = batch_y0.shape[0]
        self.d = batch_y0.shape[2]

        if Kbar is None:
            self.Kbar = 1. / float(self.k) * torch.eye(self.d)
        else:
            self.Kbar = 1. / float(self.k) * Mbar
        if Mbar_b is None:
            self.Kbar_b = 1. / float(self.k) * torch.eye(self.d)
        else:
            self.Kbar_b = 1. / float(self.k) * Mbar_b

        self.q_params = nn.Parameter(batch_y0)
        self.p_params = nn.Parameter(torch.zeros(self.k, 1, self.d))

    def forward(self, t,input):
        """
        :param input: containing q_params, p_params, x
        :param batch_t: 1D tensor holding time points for evaluation
        :return: |batch_t| x minibatch x 1 x feature dimension
        """
        # q_params and p_params are K x 1 x feature dim tensors
        # x is a |batch| x 1 x feature dim tensor
        q_params, p_params,x = input[:self.k, ...], input[self.k:2 * self.k, ...], input[2 * self.k:, ...]

        # Update theta according to the (p,q) equations
        # With Kbar = \bar M_\theta}^{-1}
        #\theta = Kbar(-\sum_i p_i \sigma(x_i)^T
        temp = torch.matmul(-p_params.squeeze().transpose(0, 1), nn.functional.relu(q_params.squeeze()))
        theta = torch.matmul(self.Kbar, temp)


        # Update bias according to the (p,q)
        # With Kbar_b = \bar M_b^{-1}
        # b = Kbar_b(-\sum_i p_i)
        temp = torch.matmul(-p_params.squeeze().transpose(0, 1), torch.ones([self.k, 1]))
        bias = torch.matmul(self.Kbar_b, temp)

        # Compute the advection equation for q_params and x
        #\dot x_i = \theta \sigma(x_i) + b
        # \dot q_i = \theta \sigma(q_i) + b
        temp_q = nn.functional.relu(q_params.squeeze(dim=1).transpose(0, 1))
        temp_x = nn.functional.relu(x.squeeze(dim=1).transpose(0, 1))
        dot_x = torch.matmul(theta, temp_x) + bias
        dot_q = torch.matmul(theta, temp_q) + bias
        dot_x = dot_x.unsqueeze(dim=1).transpose(0, 2)
        dot_q = dot_q.unsqueeze(dim=1).transpose(0, 2)

        # compute the advection equation for p_params
        # \dot p_i =  - [d\sigma(x_i)^T]\theta^T p_i
        # but here, d\sigma(x_i)^T = d\sigma(x_i) is a diagonal matrix composed with the derivative of the relu.

        rhs = []
        for i in range(0, self.k):
            Drelu = torch.diag((q_params[i, 0, :] >= 0).type(torch.FloatTensor))
            prod = torch.matmul(Drelu, theta.transpose(0, 1))
            rhs.append(torch.matmul(-prod, p_params[i, :, :].transpose(0, 1)))
        rhs = torch.cat(rhs, 1)
        rhs = rhs.unsqueeze(dim=1).transpose(0, 2)
        return torch.cat((dot_q,rhs,dot_x))



class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.9):
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
        K = 4
        #Mbar = torch.inverse(torch.tensor([[1.0,0.3],[0.3,1.0]]))
        Mbar = 1 * torch.tensor([[1.0, 0.], [0., 1.0]])
        #Mbar_b = torch.inverse(torch.tensor([[1.0,0.3],[0.3,1.0]]))
        Mbar_b = 1 * torch.tensor([[1.0, 0.], [0., 1.0]])

        batch_y0, batch_t, batch_y = get_batch(K)
        print(batch_t)
        shooting = ShootingBlock2(batch_y0, Mbar, Mbar_b)
        #optimizer = optim.RMSprop(shooting.parameters(), lr=1.e-3)
        optimizer = optim.Adam(shooting.parameters(), lr=3e-3)
        #optimizer = custom_optimizers.LBFGS_LS(shooting.parameters())
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):

        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()

        if is_odenet:
            pred_y = odeint(func, batch_y0, batch_t)
            print(batch_t.size())
            print("t",t.size())
        else:
            #print("size batch y0",batch_y0.size())

            #print(shooting.q_params.size())
            q_params = (shooting.q_params).clone()
            p_params = (shooting.p_params).clone()
            z_0 = torch.cat((batch_y0,q_params,p_params))

            #print("size input in shooting",z_0.size())
            temp_pred_y = odeint(shooting,z_0 , batch_t)
            pred_y = temp_pred_y[:, 2 * K:, ...]
            #print("size output shooting",pred_y.size())

        #print("batch",batch_y.size())
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        #print("size of tensor",loss.size())


        y = q_params.grad

        optimizer.step()
        #print(shooting.p_params)
        #time_meter.update(time.time() - end)
        #loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():

                if is_odenet:
                    pred_y = odeint(func, true_y0, t)
                    print("true y", true_y.size())
                    print("pred y", pred_y.size())
                    loss = torch.mean(torch.abs(pred_y.squeeze(dim=1) - true_y))
                    print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                    # TODO: visualize does not work for odenet
                    visualize(true_y, pred_y, func, ii)
                else:
                    q_params = (shooting.q_params).clone()
                    p_params = (shooting.p_params).clone()
                    #print("q_params",q_params.size())
                    z_0 = torch.cat((true_y0.unsqueeze(dim=0), q_params, p_params))
                    temp_pred_y = odeint(shooting, z_0, t)
                    pred_y = temp_pred_y[:, 2 * K:, ...]
                    #print("actually",pred_y.size())
                    #print("true y",true_y.size())
                    loss = torch.mean(torch.abs(pred_y.squeeze(dim=1) - true_y))
                    print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                    # TODO: implement visualize for shooting
                    # visualize(true_y, pred_y, func, ii)

                ii += 1

        end = time.time()
