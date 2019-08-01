import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

import neuro_shooting.shooting_models as shooting_models

# Command line arguments

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--network', type=str, choices=['odenet', 'shooting'], default='shooting', help='Sets the network training appproach.')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams','rk4'], default='rk4', help='Selects the desired integrator')
parser.add_argument('--stepsize', type=float, default=0.5, help='Step size for the integrator (if not adaptive).')
parser.add_argument('--data_size', type=int, default=250, help='Length of the simulated data that should be matched.')
parser.add_argument('--batch_time', type=int, default=25, help='Length of the training samples.')
parser.add_argument('--batch_size', type=int, default=10, help='Number of training samples.')
parser.add_argument('--niters', type=int, default=10000, help='Maximum nunber of iterations.')
parser.add_argument('--batch_validation_size', type=int, default=100, help='Length of the samples for validation.')
parser.add_argument('--seed', required=False, type=int, default=1234,
                    help='Sets the random seed which affects data shuffling')

parser.add_argument('--linear', action='store_true', help='If specified the ground truth system will be linear, otherwise nonlinear.')

parser.add_argument('--test_freq', type=int, default=20, help='Frequency with which the validation measures are to be computed.')
parser.add_argument('--viz_freq', type=int, default=100, help='Frequency with which the results should be visualized; if --viz is set.')

parser.add_argument('--validate_with_long_range', action='store_true', help='If selected, a long-range trajectory will be used; otherwise uses batches as for training')

parser.add_argument('--nr_of_particles', type=int, default=10, help='Number of particles to parameterize the initial condition')
parser.add_argument('--sim_norm', type=str, choices=['l1','l2'], default='l2', help='Norm for the similarity measure.')
parser.add_argument('--shooting_norm_penalty', type=float, default=0, help='Factor to penalize the norm with; default 0, but 0.1 or so might be a good value')
parser.add_argument('--nonlinearity', type=str, choices=['identity', 'relu', 'tanh', 'sigmoid'], default='tanh', help='Nonlinearity for shooting.')


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
        if args.linear:
            return torch.mm(y, true_A)
        else:
            return torch.mm(y**3, true_A)

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

def visualize(true_y, pred_y, sim_time, odefunc, itr, is_odenet=False, is_higher_order_model=False):

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

        if not is_odenet:
            z_0 = odefunc.get_initial_condition(x=current_y.unsqueeze(dim=1))

            if is_higher_order_model:

                viz_time = t[:5] # just 5 timesteps ahead
                temp_pred_y = odeint(shooting, z_0, viz_time, method=args.method, atol=atol, rtol=rtol, options=options)
                dydt_pred_y = shooting.disassemble(temp_pred_y, dim=1)

                dydt = (dydt_pred_y[-1,...]-dydt_pred_y[0,...]).detach().numpy()

                dydt = dydt[:,0,...]
            else:
                temp_pred_y = shooting(0,z_0)
                dydt_tmp = shooting.disassemble(temp_pred_y, dim=0)
                dydt = dydt_tmp[:,0,...].detach().numpy()

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




if __name__ == '__main__':

    t_0 = time.time()
    ii = 0

    is_odenet = args.network == 'odenet'

    is_higher_order_model = True

    if is_odenet:
        #func = ODEFunc()
        func = ODESimpleFuncWithIssue()
        optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
        #optimizer = optim.SGD(func.parameters(), lr=2.5e-3, momentum=0.5, dampening=0.0, nesterov=True)

    else:

        # parameters to play with for shooting
        K = args.nr_of_particles

        batch_y0, batch_t, batch_y = get_batch(K)

        shooting = shooting_models.AutoShootingIntegrandModelSimple(name='simple', batch_y0=batch_y0, only_random_initialization=True, nonlinearity=args.nonlinearity)
        #shooting = shooting_models.AutoShootingBlockModelSecondOrder(name='second_order', batch_y0=batch_y0, only_random_initialization=True, nonlinearity=args.nonlinearity)
        #shooting = shooting_models.AutoShootingBlockModelUpDown(name='up_down', batch_y0=batch_y0, only_random_initialization=True, nonlinearity=args.nonlinearity)

        shooting = shooting.to(device)

        #optimizer = optim.RMSprop(shooting.parameters(), lr=5e-3)
        optimizer = optim.Adam(shooting.parameters(), lr=2e-2)
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
            if itr % args.viz_freq == 0:

                try:
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
                except:
                    pass

        if is_odenet:
            pred_y = odeint(func, batch_y0, batch_t, method=args.method, atol=atol, rtol=rtol, options=options)
        else:

            z_0 = shooting.get_initial_condition(x=batch_y0)
            temp_pred_y = odeint(shooting,z_0 , batch_t, method=args.method, atol=atol, rtol=rtol, options=options)

            # we are actually only interested in the prediction of the batch itself (not the parameterization)
            pred_y = shooting.disassemble(temp_pred_y,dim=1)

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
                    visualize(val_y, val_pred_y, val_t, func, ii, is_odenet=is_odenet, is_higher_order_model=is_higher_order_model)
                    ii += 1

            else:
                ### time clock
                t_1 = time.time()

                print("time",t_1 - t_0)
                t_0 = t_1

                val_z_0 = shooting.get_initial_condition(x=val_y0)

                temp_pred_y = odeint(shooting, val_z_0, val_t, method=args.method, atol=atol, rtol=rtol,
                                     options=options)

                # we are actually only interested in the prediction of the batch itself (not the parameterization)
                val_pred_y = shooting.disassemble(temp_pred_y,dim=1)

                if args.sim_norm=='l1':
                    loss = torch.mean(torch.abs(val_pred_y - val_y))
                elif args.sim_norm=='l2':
                    loss = torch.mean(torch.norm(val_pred_y - val_y, dim=3))
                else:
                    raise ValueError('Unknown norm {}.'.format(args.sim_norm))

                loss = loss + args.shooting_norm_penalty * shooting.get_norm_penalty()

                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

                if itr % args.viz_freq == 0:
                    visualize(val_y, val_pred_y, val_t, shooting, ii, is_odenet=is_odenet, is_higher_order_model=is_higher_order_model)
                    ii += 1


        end = time.time()
