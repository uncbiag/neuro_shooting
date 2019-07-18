import os
import argparse
import time
import numpy as np
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
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--niters', type=int, default=10000)

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

true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)

odeint_method = 'dopri5'
options = dict()
rtol = 1e-6
atol = 1e-7

#odeint_method = 'rk4'
#options  = {'step_size': 0.01}

class Lambda(nn.Module):

    def forward(self, t, y):
        #return torch.mm(y**3, true_A)
        return torch.mm(y, true_A)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method=odeint_method, atol=atol, rtol=rtol, options=options)


def get_batch(batch_size=None):
    if batch_size is None:
        batch_size = args.batch_size
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), batch_size, replace=False)).to(device)
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
    # fig = plt.figure(figsize=(12, 4), facecolor='white')
    # ax_traj = fig.add_subplot(131, frameon=False)
    # ax_phase = fig.add_subplot(132, frameon=False)
    # ax_vecfield = fig.add_subplot(133, frameon=False)
    #plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        quiver_scale = 2.5 # to scale the magnitude of the quiver vectors for visualization

        fig = plt.figure(figsize=(12, 4), facecolor='white')
        ax_traj = fig.add_subplot(131, frameon=False)
        ax_phase = fig.add_subplot(132, frameon=False)
        ax_vecfield = fig.add_subplot(133, frameon=False)

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], '--', t.numpy(), pred_y.numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.min(), t.max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        q = (odefunc.q_params)
        p = (odefunc.p_params)

        q_np = q.cpu().detach().squeeze(dim=1).numpy()
        p_np = p.cpu().detach().squeeze(dim=1).numpy()


        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')

        ax_phase.scatter(q_np[:,0],q_np[:,1],marker='+')
        ax_phase.quiver(q_np[:,0],q_np[:,1], p_np[:,0],p_np[:,1],color='r', scale=quiver_scale)

        #ax_phase.scatter(p_np[:,0],p_np[:,1],marker='*')

        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)


        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]

        current_y = torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))

        # print("q_params",q_params.size())
        z_0 = torch.cat((q, p, current_y.unsqueeze(dim=1)))
        dydt_tmp = odefunc(0, z_0).cpu().detach().numpy()
        dydt = dydt_tmp[2 * K:, 0,...]

        #dydt = odefunc(0, ).cpu().detach().numpy()

        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")

        ax_vecfield.scatter(q_np[:, 0], q_np[:, 1], marker='+')
        ax_vecfield.quiver(q_np[:,0],q_np[:,1], p_np[:,0],p_np[:,1],color='r', scale=quiver_scale)

        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()

        print('Plotting')
        # plt.savefig('png/{:03d}'.format(itr))
        # plt.draw()
        # plt.pause(0.001)
        plt.show()


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
        #return self.net(y**3)
        return self.net(y)

def drelu(x):
    # derivative of relu
    res = (x>=0)
    res = res.type(x.type())
    return res

def dtanh(x):
    # derivative of tanh
    return 1.0-torch.tanh(x)**2

def identity(x):
    return x

def didentity(x):
    return torch.ones_like(x)

class ShootingBlock(nn.Module):
    def __init__(self, batch_y0=None, Kbar=None, Kbar_b=None, nonlinearity=None):
        super(ShootingBlock, self).__init__()

        self.k = batch_y0.size()[0]
        self.d = batch_y0.size()[2]

        mult_theta = 1.0
        mult_b = 1.0

        if Kbar is None:
            self.Kbar = 1./mult_theta*torch.eye(self.d**2)
        else:
            self.Kbar = 1./mult_theta*Kbar
        if Kbar_b is None:
            self.Kbar_b = 1./mult_b*torch.eye(self.d)
        else:
            self.Kbar_b = 1./mult_b*Kbar_b

        self.inv_Kbar_b = self.Kbar_b.inverse()
        self.inv_Kbar = self.Kbar.inverse()

        self.rand_mag = 0.01

        if batch_y0 is None:
            # do a fully random initialization
            self.q_params = nn.Parameter(self.rand_mag * torch.rand_like(batch_y0))
            self.p_params = nn.Parameter(self.rand_mag * torch.rand([self.k, 1, self.d]))
        else:
            self.q_params = nn.Parameter(batch_y0 + self.rand_mag*torch.rand_like(batch_y0))
            self.p_params = nn.Parameter(torch.zeros(self.k, 1, self.d) + self.rand_mag*torch.rand([self.k,1,self.d]))


        supported_nonlinearities = ['identity', 'relu', 'tanh']

        if nonlinearity is None:
            use_nonlinearity = 'identity'
        else:
            use_nonlinearity = nonlinearity.lower()

        if use_nonlinearity not in supported_nonlinearities:
            raise ValueError('Unsupported nonlinearity {}'.format(use_nonlinearity))

        if use_nonlinearity=='relu':
            self.nl = nn.functional.relu
            self.dnl = drelu
        elif use_nonlinearity=='tanh':
            self.nl = torch.tanh
            self.dnl = dtanh
        elif use_nonlinearity=='identity':
            self.nl = identity
            self.dnl = didentity
        else:
            raise ValueError('Unknown nonlinearity {}'.format(use_nonlinearity))

    def get_norm_penalty(self):

        p = self.p_params.transpose(1,2)
        q = self.q_params.transpose(1,2)

        theta = self.compute_theta(q=q,p=p)
        bias = self.compute_bias(p=p)

        theta_penalty = torch.mm(theta.view(1,-1),torch.mm(self.inv_Kbar,theta.view(-1,1)))
        bias_penalty = torch.mm(bias.t(),torch.mm(self.inv_Kbar_b,bias))

        penalty = theta_penalty + bias_penalty
        return penalty

    def compute_theta(self,q,p):
        # Update theta according to the (p,q) equations
        # With Kbar = \bar M_\theta}^{-1}
        # \theta = Kbar(-\sum_i p_i \sigma(x_i)^T
        # computing the negative sum of the outer product
        temp = -torch.bmm(p, self.nl(q.transpose(1, 2))).sum(dim=0)

        # now multiply it with the inverse of the regularizer (needs to be vectorized first and then back)
        theta = (torch.mm(self.Kbar, temp.view(-1,1))).view(temp.size())

        return theta

    def compute_bias(self,p):
        # Update bias according to the (p,q)
        # With Kbar_b = \bar M_b^{-1}
        # b = Kbar_b(-\sum_i p_i)
        # temp = torch.matmul(-p.squeeze().transpose(0, 1), torch.ones([self.k, 1],device=device))
        # keep in mind that by convention the vectors are stored as row vectors here, hence the transpose
        temp = -p.sum(dim=0)
        bias = torch.mm(self.Kbar_b, temp)

        return bias

    def forward(self, t,input):
        """
        :param input: containing q, p, x
        :param batch_t: 1D tensor holding time points for evaluation
        :return: |batch_t| x minibatch x 1 x feature dimension
        """
        # q and p are K x 1 x feature dim tensors
        # x is a |batch| x 1 x feature dim tensor
        qt,pt,xt = input[:self.k, ...], input[self.k:2 * self.k, ...], input[2 * self.k:, ...]

        # let's first convert everything to column vectors (as this is closer to our notation)
        q = qt.transpose(1,2)
        p = pt.transpose(1,2)
        x = xt.transpose(1,2)


        # compute theta
        theta = self.compute_theta(q=q,p=p)

        # compute b
        bias = self.compute_bias(p=p)

        # let't first compute the right hand side of the evolution equation for q and the same for x
        # \dot x_i = \theta \sigma(x_i) + b
        # \dot q_i = \theta \sigma(q_i) + b

        temp_q = self.nl(q)
        temp_x = self.nl(x)

        dot_x = torch.matmul(theta, temp_x) + bias
        dot_q = torch.matmul(theta, temp_q) + bias

        # compute the advection equation for p
        # \dot p_i =  - [d\sigma(x_i)^T]\theta^T p_i
        # but here, d\sigma(x_i)^T = d\sigma(x_i) is a diagonal matrix composed with the derivative of the relu.

        # first compute \theta^T p_i
        tTp = torch.matmul(theta.t(),p)
        # now compute element-wise sigma-prime xi
        sigma_p = self.dnl(q)
        # and multiply the two
        dot_p = -sigma_p*tTp

        # as we transposed the vectors before we need to transpose on the way back
        dot_qt = dot_q.transpose(1, 2)
        dot_pt = dot_p.transpose(1, 2)
        dot_xt = dot_x.transpose(1, 2)

        return torch.cat((dot_qt,dot_pt,dot_xt))



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
        K = 10

        batch_y0, batch_t, batch_y = get_batch(K)
        print(batch_t)
        shooting = ShootingBlock(batch_y0)
        shooting = shooting.to(device)
        optimizer = optim.RMSprop(shooting.parameters(), lr=2.5e-3)
        #optimizer = optim.Adam(shooting.parameters(), lr=1e-3)
        #optimizer = custom_optimizers.LBFGS_LS(shooting.parameters())
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(0, args.niters):

        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()

        if is_odenet:
            pred_y = odeint(func, batch_y0, batch_t, method=odeint_method, atol=atol, rtol=rtol, options=options)
            print(batch_t.size())
            print("t",t.size())
        else:
            q = (shooting.q_params)
            p = (shooting.p_params)
            z_0 = torch.cat((q,p,batch_y0))

            temp_pred_y = odeint(shooting,z_0 , batch_t, method=odeint_method, atol=atol, rtol=rtol, options=options)

            # we are actually only interested in the prediction of the batch itself (not the parameterization)
            pred_y = temp_pred_y[:, 2 * K:, ...]

        # todo: figure out wht the norm penality does not work
        loss = torch.mean(torch.abs(pred_y - batch_y)) # + shooting.get_norm_penalty()
        loss.backward()
        #print(torch.sum(shooting.p_params.grad**2))
        #print("size of tensor",loss.size())

        optimizer.step()
        #print(shooting.p_params)
        #time_meter.update(time.time() - end)
        #loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():

                if is_odenet:
                    pred_y = odeint(func, true_y0, t, method=odeint_method, atol=atol, rtol=rtol, options=options)
                    print("true y", true_y.size())
                    print("pred y", pred_y.size())
                    loss = torch.mean(torch.abs(pred_y.squeeze(dim=1) - true_y))
                    print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                    # TODO: visualize does not work for odenet
                    visualize(true_y, pred_y, func, ii)
                else:
                    q = (shooting.q_params)
                    p = (shooting.p_params)
                    #print("q_params",q_params.size())
                    z_0 = torch.cat(( q, p,true_y0.unsqueeze(dim=0)))
                    temp_pred_y = odeint(shooting, z_0, t, method=odeint_method, atol=atol, rtol=rtol, options=options)
                    pred_y = temp_pred_y[:, 2 * K:, ...]
                    #print("actually",pred_y.size())
                    #print("true y",true_y.size())
                    loss = torch.mean(torch.abs(pred_y.squeeze(dim=1) - true_y))
                    print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

                    if itr % 100 == 0:
                        visualize(true_y, pred_y.squeeze(dim=1), shooting, ii)
                        ii += 1

        end = time.time()
