import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import random


# Command line arguments

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--network', type=str, choices=['odenet', 'shooting'], default='shooting', help='Sets the network training appproach.')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams','rk4'], default='rk4', help='Selects the desired integrator')
parser.add_argument('--stepsize', type=float, default=0.1, help='Step size for the integrator (if not adaptive).')
parser.add_argument('--data_size', type=int, default=250, help='Length of the simulated data that should be matched.')
parser.add_argument('--batch_time', type=int, default=25, help='Length of the training samples.')
parser.add_argument('--batch_size', type=int, default=10, help='Number of training samples.')
parser.add_argument('--niters', type=int, default=10000, help='Maximum nunber of iterations.')
parser.add_argument('--batch_validation_size', type=int, default=100, help='Length of the samples for validation.')
parser.add_argument('--seed', required=False, type=int, default=1234,
                    help='Sets the random seed which affects data shuffling')
parser.add_argument('--test_freq', type=int, default=20, help='Frequency with which the validation measures are to be computed.')
parser.add_argument('--viz_freq', type=int, default=100, help='Frequency with which the results should be visualized; if --viz is set.')

parser.add_argument('--validate_with_long_range', action='store_true', help='If selected, a long-range trajectory will be used; otherwise uses batches as for training')

parser.add_argument('--nr_of_particles', type=int, default=5, help='Number of particles to parameterize the initial condition')
parser.add_argument('--sim_norm', type=str, choices=['l1','l2'], default='l2', help='Norm for the similarity measure.')
parser.add_argument('--shooting_norm_penalty', type=float, default=0, help='Factor to penalize the norm with; default 0, but 0.1 or so might be a good value')

parser.add_argument('--viz', action='store_true', help='Enable visualization.')
parser.add_argument('--gpu', type=int, default=0, help='Enable GPU computation on specified GPU.')
parser.add_argument('--adjoint', action='store_true', help='Use adjoint integrator to avoid storing values during forward pass.')

args = parser.parse_args()

print('Setting the random seed to {:}'.format(args.seed))
random.seed(args.seed)
torch.manual_seed(args.seed)

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
#true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)
#true_A = torch.tensor([[-0.025, 2.0], [-2.0, -0.025]]).to(device)
#true_A = torch.tensor([[-0.05, 2.0], [-2.0, -0.05]]).to(device)
true_A = torch.tensor([[-0.01, 0.25], [-0.25, -0.01]]).to(device)


options = dict()

# default tolerance settings
#rtol=1e-6
#atol=1e-12

rtol = 1e-8
atol = 1e-10

options  = {'step_size': args.stepsize}

class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(y**3, true_A)
        #return torch.mm(y, true_A)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method=args.method, atol=atol, rtol=rtol, options=options)


def get_batch(batch_size=None):
    if batch_size is None:
        batch_size = args.batch_size
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), batch_size, replace=False)).to(device)
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y

def visualize_batch(batch_t,batch_y,thetas=None,real_thetas=None,bias=None):

    # convention for batch_t: t x B x (row-vector)

    if args.viz:

        batch_size = batch_y.size()[1]

        if (thetas is None) or (bias is None) or (real_thetas is None):
            fig = plt.figure(figsize=(8, 4), facecolor='white')
            ax_traj = fig.add_subplot(121, frameon=False)
            ax_phase = fig.add_subplot(122, frameon=False)
        else:
            fig = plt.figure(figsize=(8, 8), facecolor='white')
            ax_traj = fig.add_subplot(221, frameon=False)
            ax_phase = fig.add_subplot(222, frameon=False)
            ax_thetas = fig.add_subplot(223, frameon=False)
            ax_bias = fig.add_subplot(224, frameon=False)

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')

        for b in range(batch_size):
            c_values = batch_y[:,b,0,:]

            ax_traj.plot(batch_t.numpy(), c_values.numpy()[:, 0], batch_t.numpy(), c_values.numpy()[:, 1], 'g-')

        ax_traj.set_xlim(batch_t.min(), batch_t.max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')

        for b in range(batch_size):
            c_values = batch_y[:,b,0,:]

            ax_phase.plot(c_values.numpy()[:, 0], c_values.numpy()[:, 1], 'g-')

        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        if (thetas is not None) and (bias is not None) and (real_thetas is not None):
            ax_thetas.cla()
            ax_thetas.set_title('theta elements over time')
            nr_t_el = thetas.shape[1]
            colors = ['r','b','c','k']
            for n in range(nr_t_el):
                ax_thetas.plot(thetas[:,n],color=colors[n])
                ax_thetas.plot(real_thetas[:,n],'--', color=colors[n])

            ax_bias.cla()
            ax_bias.set_title('bias elements over time')
            nr_b_el = bias.shape[1]
            for n in range(nr_b_el):
                ax_bias.plot(bias[:,n])

        fig.tight_layout()

        print('Plotting')
        plt.show()


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt

def visualize(true_y, pred_y, sim_time, odefunc, itr, is_odenet=False):

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

        if not is_odenet:
            q = (odefunc.q_params)
            p = (odefunc.p_params)

            q_np = q.cpu().detach().squeeze(dim=1).numpy()
            p_np = p.cpu().detach().squeeze(dim=1).numpy()

            ax_phase.scatter(q_np[:,0],q_np[:,1],marker='+')
            ax_phase.quiver(q_np[:,0],q_np[:,1], p_np[:,0],p_np[:,1],color='r', scale=quiver_scale)

        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)


        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]

        current_y = torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))

        # print("q_params",q_params.size())

        if not is_odenet:
            z_0 = torch.cat((q, p, current_y.unsqueeze(dim=1)))
            dydt_tmp = odefunc(0, z_0).cpu().detach().numpy()
            dydt = dydt_tmp[2 * K:, 0,...]
        else:
            dydt = odefunc(0, current_y).cpu().detach().numpy()

        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")

        if not is_odenet:
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
        return self.net(y)

class ODESimpleFunc(nn.Module):

    def __init__(self):
        super(ODESimpleFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 2),
            nn.Tanh(),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)

class ODESimpleFuncWithIssue(nn.Module):
# order matters. If linear transform comes after the tanh it cannot move the nonlinearity to a point where it does not matter
# (and hence will produce the 45 degree tanh angle phenonmenon)

    def __init__(self):
        super(ODESimpleFuncWithIssue, self).__init__()

        self.net = nn.Sequential(
            nn.Tanh(),
            nn.Linear(2, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
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
    def __init__(self, batch_y0=None, Kbar=None, Kbar_b=None, nonlinearity=None, only_random_initialization=False):
        super(ShootingBlock, self).__init__()

        nonlinearity = 'tanh'

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

        self.Kbar = self.Kbar.to(device)
        self.Kbar_b = self.Kbar_b.to(device)

        self.inv_Kbar_b = self.Kbar_b.inverse()
        self.inv_Kbar = self.Kbar.inverse()

        self.rand_mag_q = 0.1
        self.rand_mag_p = 0.1

        if only_random_initialization:
            # do a fully random initialization
            self.q_params = nn.Parameter(self.rand_mag_q * torch.randn_like(batch_y0))
            self.p_params = nn.Parameter(self.rand_mag_p * torch.randn([self.k, 1, self.d]))
        else:
            self.q_params = nn.Parameter(batch_y0 + self.rand_mag_q * torch.randn_like(batch_y0))
            self.p_params = nn.Parameter(torch.zeros(self.k, 1, self.d) + self.rand_mag_p * torch.randn([self.k, 1, self.d]))

        supported_nonlinearities = ['identity', 'relu', 'tanh', 'sigmoid']

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
        elif use_nonlinearity=='sigmoid':
            self.nl = torch.sigmoid
            self.dnl = torch.sigmoid
        else:
            raise ValueError('Unknown nonlinearity {}'.format(use_nonlinearity))


        # keeping track of variables
        self._number_of_calls = 0

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

        temp = -torch.bmm(p, self.nl(q.transpose(1, 2))).mean(dim=0)

        # now multiply it with the inverse of the regularizer (needs to be vectorized first and then back)
        theta = (torch.mm(self.Kbar, temp.view(-1,1))).view(temp.size())

        return theta

    def compute_bias(self,p):
        # Update bias according to the (p,q)
        # With Kbar_b = \bar M_b^{-1}
        # b = Kbar_b(-\sum_i p_i)
        # temp = torch.matmul(-p.squeeze().transpose(0, 1), torch.ones([self.k, 1],device=device))
        # keep in mind that by convention the vectors are stored as row vectors here, hence the transpose

        #temp = -p.sum(dim=0)
        temp = -p.mean(dim=0)

        bias = torch.mm(self.Kbar_b, temp)

        return bias

    def forward(self, t,input):
        """
        :param input: containing q, p, x
        :param batch_t: 1D tensor holding time points for evaluation
        :return: |batch_t| x minibatch x 1 x feature dimension
        """

        self._number_of_calls += 1
        if (self._number_of_calls%10000==0):
            # just to test; this is a way we can keep track of state variables, for example to initialize iterative solvers
            print('Number of calls: {}'.format(self._number_of_calls))


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

class ShootingBlock2(nn.Module):
    def __init__(self, batch_y0=None, Kbar=None, Kbar_b=None, nonlinearity=None, only_random_initialization=False):
        super(ShootingBlock2, self).__init__()

        nonlinearity = 'tanh'

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

        self.Kbar = self.Kbar.to(device)
        self.Kbar_b = self.Kbar_b.to(device)

        self.inv_Kbar_b = self.Kbar_b.inverse()
        self.inv_Kbar = self.Kbar.inverse()

        self.rand_mag_q = 0.1
        self.rand_mag_p = 0.1

        if only_random_initialization:
            # do a fully random initialization
            self.q_params = nn.Parameter(self.rand_mag_q * torch.randn_like(batch_y0))
            self.p_params = nn.Parameter(self.rand_mag_p * torch.randn([self.k, 1, self.d]))
        else:
            self.q_params = nn.Parameter(batch_y0 + self.rand_mag_q * torch.randn_like(batch_y0))
            self.p_params = nn.Parameter(torch.zeros(self.k, 1, self.d) + self.rand_mag_p * torch.randn([self.k, 1, self.d]))

        supported_nonlinearities = ['identity', 'relu', 'tanh', 'sigmoid']

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
        elif use_nonlinearity=='sigmoid':
            self.nl = torch.sigmoid
            self.dnl = torch.sigmoid
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

        #temp = -torch.bmm(p, self.nl(q.transpose(1, 2))).sum(dim=0)
        temp = -torch.bmm(p, self.nl(q.transpose(1, 2))).mean(dim=0)

        # now multiply it with the inverse of the regularizer (needs to be vectorized first and then back)
        theta = (torch.mm(self.Kbar, temp.view(-1,1))).view(temp.size())

        return theta

    def compute_bias(self,p):
        # Update bias according to the (p,q)
        # With Kbar_b = \bar M_b^{-1}
        # b = Kbar_b(-\sum_i p_i)
        # temp = torch.matmul(-p.squeeze().transpose(0, 1), torch.ones([self.k, 1],device=device))
        # keep in mind that by convention the vectors are stored as row vectors here, hence the transpose

        #temp = -p.sum(dim=0)
        temp = -p.mean(dim=0)

        bias = torch.mm(self.Kbar_b, temp)

        return bias

    def advect_x(self,x,theta,bias):
        """
        Forward equation which  is applied on the data. In principle similar to advect_q
        :param x:
        :param theta:
        :param bias:
        :return: \dot x_i = \theta \sigma(x_i) + b
        """
        temp_x = self.nl(x)
        return torch.matmul(theta, temp_x) + bias

    def advect_q(self,q,theta,bias):
        """
        Forward equation which  is applied on the data. In principle similar to advect_q
        :param x:
        :param theta:
        :param bias:
        :return: \dot q_i = \theta \sigma(q_i) + b
        """
        temp_q = self.nl(q)
        return torch.matmul(theta, temp_q) + bias

    def advect_p(self,p,q,theta,bias):
        theta = theta.detach()
        bias = bias.detach()
        compute = torch.sum(p*self.advect_q(q,theta,bias))

        xgrad, = autograd.grad(compute, q,
                               grad_outputs=compute.data.new(compute.shape).fill_(1),
                               create_graph=True,
                               retain_graph=True,
                               allow_unused=True)
        return -xgrad

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
        dot_x = self.advect_x(x,theta,bias)
        dot_q = self.advect_q(q,theta,bias)

        # compute the advection equation for p
        # \dot p_i =  - [d\sigma(x_i)^T]\theta^T p_i
        # but here, d\sigma(x_i)^T = d\sigma(x_i) is a diagonal matrix composed with the derivative of the relu.

        # first compute \theta^T p_i

        dot_p =  self.advect_p(p,q,theta,bias)

        #theta_bis = torch.empty_like(theta).copy_(theta)
        #p_bis = torch.empty_like(p).copy_(p)
        #q_bis = torch.empty_like(q).copy_(q)
        #tTp = torch.matmul(theta_bis.t(), p_bis)
        # now compute element-wise sigma-prime xi
        #sigma_p = self.dnl(q_bis)
        # and multiply the two
        #dot_p_2 = -sigma_p * tTp
        #print("comparison",torch.sum((dot_p_2 - dot_p)**2))
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

    t_0 = time.time()
    ii = 0

    is_odenet = args.network == 'odenet'

    if is_odenet:
        #func = ODEFunc()
        func = ODESimpleFuncWithIssue()
        optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
        #optimizer = optim.SGD(func.parameters(), lr=2.5e-3, momentum=0.5, dampening=0.0, nesterov=True)

    else:

        # parameters to play with for shooting
        K = args.nr_of_particles

        batch_y0, batch_t, batch_y = get_batch(K)
        shooting = ShootingBlock2(batch_y0,only_random_initialization=True)
        shooting = shooting.to(device)

        optimizer = optim.RMSprop(shooting.parameters(), lr=5e-3)
        #optimizer = optim.Adam(shooting.parameters(), lr=1e-1)
        #optimizer = optim.SGD(shooting.parameters(), lr=2.5e-3, momentum=0.5, dampening=0.0, nesterov=True)
        #optimizer = custom_optimizers.LBFGS_LS(shooting.parameters())

    all_thetas = None
    all_real_thetas = None
    all_bs = None

    validate_with_batch_data = not args.validate_with_long_range
    validate_with_random_batch_each_time = False

    if validate_with_batch_data:
        if not validate_with_random_batch_each_time:
            val_batch_y0, val_batch_t, val_batch_y = get_batch(batch_size=args.batch_validation_size)

    for itr in range(0, args.niters):

        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()

        if itr % args.test_freq == 0:
            if itr % 100 == 0:

                if not is_odenet:
                    theta_np = (shooting.compute_theta(q=shooting.q_params.transpose(1,2),p=shooting.p_params.transpose(1,2))).view(1,-1).detach().cpu().numpy()
                    bias_np = (shooting.compute_bias(p=shooting.p_params.transpose(1,2))).view(1,-1).detach().cpu().numpy()

                    if all_thetas is None:
                        all_thetas = theta_np
                    else:
                        all_thetas = np.append(all_thetas,theta_np,axis=0)

                    c_true_A = true_A.view(1,-1).detach().cpu().numpy()
                    if all_real_thetas is None:
                        all_real_thetas = c_true_A
                    else:
                        all_real_thetas = np.append(all_real_thetas,c_true_A,axis=0)

                    if all_bs is None:
                        all_bs = bias_np
                    else:
                        all_bs = np.append(all_bs,bias_np,axis=0)

                visualize_batch(batch_t,batch_y,thetas=all_thetas,real_thetas=all_real_thetas,bias=all_bs)

        if is_odenet:
            pred_y = odeint(func, batch_y0, batch_t, method=args.method, atol=atol, rtol=rtol, options=options)
        else:
            q = (shooting.q_params)
            p = (shooting.p_params)
            z_0 = torch.cat((q,p,batch_y0))

            temp_pred_y = odeint(shooting,z_0 , batch_t, method=args.method, atol=atol, rtol=rtol, options=options)

            # we are actually only interested in the prediction of the batch itself (not the parameterization)
            pred_y = temp_pred_y[:, 2 * K:, ...]

        # todo: figure out wht the norm penality does not work
        if args.sim_norm == 'l1':
            loss = torch.mean(torch.abs(pred_y - batch_y))
        elif args.sim_norm == 'l2':
            loss = torch.mean(torch.norm(pred_y-batch_y,dim=3))
        else:
            raise ValueError('Unknown norm {}.'.format(args.sim_norm))

        if not is_odenet:
            loss = loss + args.shooting_norm_penalty * shooting.get_norm_penalty()

        loss.backward()

        optimizer.step()

        if itr % args.test_freq == 0:
            # we need to keep computing the gradient here as the forward model may require gradient computations

            if validate_with_batch_data:
                if validate_with_random_batch_each_time:
                    # draw new batch. This will be like a moving target for the evaluation
                    val_batch_y0, val_batch_t, val_batch_y = get_batch()
                val_y0 = val_batch_y0
                val_t = val_batch_t
                val_y = val_batch_y
            else:
                val_y0 = true_y0.unsqueeze(dim=0)
                val_t = t
                val_y = true_y.unsqueeze(dim=1)

            if is_odenet:
                val_pred_y = odeint(func, val_y0, val_t, method=args.method, atol=atol, rtol=rtol, options=options)

                if args.sim_norm=='l1':
                    loss = torch.mean(torch.abs(val_pred_y - val_y))
                elif args.sim_norm=='l2':
                    loss = torch.mean(torch.norm(val_pred_y - val_y, dim=3))
                else:
                    raise ValueError('Unknown norm {}.'.format(args.sim_norm))

                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

                if itr % args.viz_freq == 0:
                    visualize(val_y, val_pred_y, val_t, func, ii, is_odenet=is_odenet)
                    ii += 1

            else:
                ### time clock
                t_1 = time.time()

                print("time",t_1 - t_0)
                t_0 = t_1

                q = (shooting.q_params)
                p = (shooting.p_params)
                val_z_0 = torch.cat((q, p, val_y0))

                temp_pred_y = odeint(shooting, val_z_0, val_t, method=args.method, atol=atol, rtol=rtol,
                                     options=options)

                # we are actually only interested in the prediction of the batch itself (not the parameterization)
                val_pred_y = temp_pred_y[:, 2 * K:, ...]

                if args.sim_norm=='l1':
                    loss = torch.mean(torch.abs(val_pred_y - val_y))
                elif args.sim_norm=='l2':
                    loss = torch.mean(torch.norm(val_pred_y - val_y, dim=3))
                else:
                    raise ValueError('Unknown norm {}.'.format(args.sim_norm))

                loss = loss + args.shooting_norm_penalty * shooting.get_norm_penalty()

                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

                if itr % 100 == 0:
                    visualize(val_y, val_pred_y, val_t, shooting, ii, is_odenet=is_odenet)
                    ii += 1


        end = time.time()
