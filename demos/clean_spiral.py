# modeled after torchdiffeq's ode_demo.py

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import sys

import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.shooting_models as shooting_models
import neuro_shooting.generic_integrator as generic_integrator
import neuro_shooting.tensorboard_shooting_hooks as thooks

# Command line arguments

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--network', type=str, choices=['odenet', 'shooting'], default='shooting', help='Sets the network training appproach.')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams','rk4'], default='rk4', help='Selects the desired integrator')
parser.add_argument('--stepsize', type=float, default=0.5, help='Step size for the integrator (if not adaptive).')
parser.add_argument('--data_size', type=int, default=250, help='number of time points on the simulated ODE.')
parser.add_argument('--batch_time', type=int, default=25, help='Length of the training trajectories.')
parser.add_argument('--batch_size', type=int, default=10, help='Number of training trajectories.')
parser.add_argument('--niters', type=int, default=10000, help='Maximum number of iterations.')
parser.add_argument('--batch_validation_size', type=int, default=100, help='number of validation trajectories (each of time length batch_time).')
parser.add_argument('--seed', required=False, type=int, default=1234,
                    help='Sets the random seed which affects data shuffling')

parser.add_argument('--linear', action='store_true', help='If specified the ground truth system will be linear, otherwise nonlinear.')

parser.add_argument('--test_freq', type=int, default=100, help='Frequency with which the validation measures are to be computed.')
parser.add_argument('--viz_freq', type=int, default=100, help='Frequency with which the results should be visualized; if --viz is set.')
parser.add_argument('--saveimgname',type=str, default='spiral', help='any images will be saved with this name')
parser.add_argument('--validate_with_long_range', action='store_true', help='If selected, a long-range trajectory will be used; otherwise uses batches as for training')

parser.add_argument('--shooting_model',type=str,default='simple',choices=['simple','2nd_order','updown'])
parser.add_argument('--pw',type=float,default=0.5,help='parameter weight')
parser.add_argument('--nr_of_particles', type=int, default=10, help='Number of particles to parameterize the initial condition')
parser.add_argument('--sim_norm', type=str, choices=['l1','l2'], default='l2', help='Norm for the similarity measure.')
parser.add_argument('--shooting_norm_penalty', type=float, default=0, help='Factor to penalize the norm with; default 0, but 0.1 or so might be a good value')
parser.add_argument('--nonlinearity', type=str, choices=['identity', 'relu', 'tanh', 'sigmoid'], default='tanh', help='Nonlinearity for shooting.')

parser.add_argument('--use_analytic_solution',action='store_true',help='Enable analytic solution in shooting model')
parser.add_argument('--viz', action='store_true', help='Enable visualization.')
parser.add_argument('--gpu', type=int, default=0, help='Enable GPU computation on specified GPU.')
parser.add_argument('--adjoint', action='store_true', help='Use adjoint integrator to avoid storing values during forward pass.')

args = parser.parse_args()

saveresultspath = '{}/shootingmodel{}/numparticles{}/pw{}/nonlinearity{}'.format(args.saveimgname,
                                                                                 args.shooting_model,
                                                                                 args.nr_of_particles,
                                                                                 args.pw,
                                                                                 args.nonlinearity)
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

makedirs(saveresultspath)
stdoutOrigin=sys.stdout
sys.stdout = open("{}/log.txt".format(saveresultspath), "w")

print(args)

print('Setting the random seed to {:}'.format(args.seed))
random.seed(args.seed)
torch.manual_seed(args.seed)

# default tolerance settings
rtol=1e-6
atol=1e-12

integrator_options = dict()
integrator_options = {'step_size': args.stepsize}

integrator = generic_integrator.GenericIntegrator(integrator_library = 'odeint', integrator_name = args.method,
                                                  use_adjoint_integration=args.adjoint, integrator_options=integrator_options, rtol=rtol, atol=atol)


device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

t_max = 25
print('tmax: {}'.format(t_max))
t = torch.linspace(0., t_max, args.data_size).to(device)
true_y0 = torch.tensor([[2., 0.]]).to(device)
print('true_y0: {}'.format(true_y0))
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)
print('true_A: {}'.format(true_A))

# true_y0 = torch.tensor([[0.6, 0.3]]).to(device)
# true_A = torch.tensor([[-0.1, -1.0], [1.0, -0.1]]).to(device)
# true_A = torch.tensor([[-0.025, 2.0], [-2.0, -0.025]]).to(device)
# true_A = torch.tensor([[-0.05, 2.0], [-2.0, -0.05]]).to(device)
# true_A = torch.tensor([[-0.01, 0.25], [-0.25, -0.01]]).to(device)

class Lambda(nn.Module):

    def forward(self, t, y):
        if args.linear:
            return torch.mm(y, true_A)
        else:
            return torch.mm(y**3, true_A)

# TODO: if integrated with rk4, true_y will contain nan's
with torch.no_grad():
    true_y = integrator.integrate(func=Lambda(), x0=true_y0, t=t)

def get_batch(batch_size=None):
    if batch_size is None:
        batch_size = args.batch_size
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), batch_size, replace=False)).to(device)
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y

def to_np(x):
    return x.detach().cpu().numpy()

def visualize(true_y, pred_y, sim_time, odefunc, itr, is_odenet=False, is_higher_order_model=False, savepath=None):

    if args.viz:

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
        ax_vecfield.set_xlabel('y1')
        ax_vecfield.set_ylabel('y2')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]

        current_y = torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))

        # print("q_params",q_params.size())

        if not is_odenet:
            x_0 = current_y.unsqueeze(dim=1)

            viz_time = t[:5] # just 5 timesteps ahead

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
        plt.savefig('{}/itr{}.png'.format(savepath,itr))
        plt.draw()
        plt.pause(0.001)
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
        shootingintegrand_kwargs = {'in_features': 2,
                                    'nonlinearity': args.nonlinearity,
                                    'nr_of_particles': args.nr_of_particles,
                                    'parameter_weight': args.pw}
        if args.use_analytic_solution:
            shootingintegrand_kwargs.update({'use_analytic_solution': args.use_analytic_solution})

        if args.shooting_model =='simple':
            shooting_model = shooting_models.AutoShootingIntegrandModelSimple(**shootingintegrand_kwargs)
        elif args.shooting_model == '2nd_order':
            shooting_model = shooting_models.AutoShootingIntegrandModelSecondOrder(**shootingintegrand_kwargs)
        elif args.shooting_model == 'updown':
            shooting_model = shooting_models.AutoShootingIntegrandModelUpDown(**shootingintegrand_kwargs)

        shooting_block = shooting_blocks.ShootingBlockBase(name='simple', shooting_integrand=shooting_model)
        shooting_block = shooting_block.to(device)

        # run through the shooting block once (to get parameters as needed)
        _,_,sample_batch = get_batch()
        shooting_block(x=sample_batch)

        optimizer = optim.Adam(shooting_block.parameters(), lr=2e-2)
        #optimizer = optim.RMSprop(shooting_block.parameters(), lr=5e-3)
        #optimizer = optim.SGD(shooting_block.parameters(), lr=2.5e-3, momentum=0.5, dampening=0.0, nesterov=True)
        # optimizer = custom_optimizers.LBFGS_LS(shooting.parameters())

    all_thetas = None
    all_real_thetas = None
    all_bs = None

    validate_with_batch_data = not args.validate_with_long_range
    validate_with_random_batch_each_time = False

    if validate_with_batch_data:
        if not validate_with_random_batch_each_time:
            val_batch_y0, val_batch_t, val_batch_y = get_batch(batch_size=args.batch_validation_size)

    custom_hook_data = dict()

    for itr in range(0, args.niters):

        custom_hook_data['epoch'] = itr
        custom_hook_data['batch'] = 0 # we do not really have batches here

        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()

        if is_odenet:
            pred_y = integrator.integrate(func=func, x0=batch_y0, t=batch_t)
        else:

            # register a hook so we can log via tensorboard
            # shooting_hook = shooting_block.shooting_integrand.register_lagrangian_gradient_hook(thooks.linear_transform_hook)
            # shooting_block.shooting_integrand.set_custom_hook_data(data=custom_hook_data)

            shooting_block.set_integration_time_vector(integration_time_vector=batch_t, suppress_warning=True)
            pred_y,_,_,_ = shooting_block(x=batch_y0)
            # TODO: do we need to reset integration time?
            # shooting_block.set_integration_time(t_max)

            # get rid of the hook again, so we don't get any issues with the testing later on (we do not want to log there)
            #shooting_hook.remove()

        # TODO: figure out wht the norm penality does not work
        if args.sim_norm == 'l1':
            loss = torch.mean(torch.abs(pred_y - batch_y))
        elif args.sim_norm == 'l2':
            loss = torch.mean(torch.norm(pred_y - batch_y, dim=3))
        else:
            raise ValueError('Unknown norm {}.'.format(args.sim_norm))

        if not is_odenet:
            loss = loss + args.shooting_norm_penalty * shooting_block.get_norm_penalty()

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
                val_pred_y = integrator.integrate(func=func, x0=val_y0, t=val_t)

                if args.sim_norm=='l1':
                    loss = torch.mean(torch.abs(val_pred_y - val_y))
                elif args.sim_norm=='l2':
                    loss = torch.mean(torch.norm(val_pred_y - val_y, dim=3))
                else:
                    raise ValueError('Unknown norm {}.'.format(args.sim_norm))

                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

                if args.viz & (itr % args.viz_freq == 0):
                    visualize(val_y, val_pred_y, val_t, func, ii, is_odenet=is_odenet, is_higher_order_model=is_higher_order_model)
                    ii += 1

            else:
                ### time clock
                t_1 = time.time()

                print("time",t_1 - t_0)
                t_0 = t_1

                # TODO: if val_y0 is the entire trajectory then val_pred_y will eventually have nan's
                shooting_block.set_integration_time_vector(integration_time_vector=val_t, suppress_warning=True)
                val_pred_y,_,_,_ = shooting_block(x=val_y0)
                # TODO: do we need to reset integration time?
                # shooting_block.set_integration_time(t_max)

                if args.sim_norm=='l1':
                    loss = torch.mean(torch.abs(val_pred_y - val_y))
                elif args.sim_norm=='l2':
                    loss = torch.mean(torch.norm(val_pred_y - val_y, dim=3))
                else:
                    raise ValueError('Unknown norm {}.'.format(args.sim_norm))

                loss = loss + args.shooting_norm_penalty * shooting_block.get_norm_penalty()

                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

                if args.viz & (itr % args.viz_freq == 0):
                    visualize(val_y, val_pred_y, val_t, shooting_block, itr, is_odenet=is_odenet, is_higher_order_model=is_higher_order_model, savepath = saveresultspath)

        end = time.time()


# TODO: this is no longer used, deprecate?
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
        ax_traj.set_ylabel('y1,y2')

        for b in range(batch_size):
            c_values = batch_y[:,b,0,:]

            ax_traj.plot(batch_t.numpy(), c_values.numpy()[:, 0], batch_t.numpy(), c_values.numpy()[:, 1], 'g-')

        ax_traj.set_xlim(batch_t.min(), batch_t.max())
        ax_traj.set_ylim(-2, 2)
        # ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('y1')
        ax_phase.set_ylabel('y2')

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

sys.stdout.close()
sys.stdout=stdoutOrigin