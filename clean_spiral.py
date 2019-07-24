import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import random

from abc import ABCMeta, abstractmethod
# may require conda install sortedcontainers
from sortedcontainers import SortedDict

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
            if is_higher_order_model:
                z_0 = odefunc.get_initial_condition(x=current_y.unsqueeze(dim=1))
            else:
                z_0 = torch.cat((q, p, current_y.unsqueeze(dim=1)))

            if is_higher_order_model:

                viz_time = t[:5] # just 5 timesteps ahead
                temp_pred_y = odeint(shooting, z_0, viz_time, method=args.method, atol=atol, rtol=rtol, options=options)
                dydt_pred_y = shooting.disassemble(temp_pred_y, dim=1)

                dydt = (dydt_pred_y[-1,...]-dydt_pred_y[0,...]).detach().numpy()

                dydt = dydt[:,0,...]
            else:
                dydt_tmp = odefunc(0, z_0).cpu().detach().numpy()
                dydt = dydt_tmp[2 * K:, 0,...]
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

def softmax(x,epsilon = 1.0):
  return x*(torch.ones_like(x))/(torch.exp(-x*epsilon) + torch.ones_like(x))


def dsoftmax(x,epsilon = 1.0):
  return epsilon*softmax(x,epsilon)*(torch.ones_like(x))/(torch.exp(epsilon*x) + torch.ones_like(x)) + (torch.ones_like(x))/(torch.exp(-x*epsilon) + torch.ones_like(x))

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



class ShootingBlockBase(nn.Module):
    def __init__(self, batch_y0=None, nonlinearity=None, transpose_state_when_forward=True):
        super(ShootingBlockBase, self).__init__()

        self.nl, self.dnl = self._get_nonlinearity(nonlinearity=nonlinearity)
        self.use_iterative_parameter_solution = True

        self._state_parameter_dict = None
        self._costate_parameter_dict = None

        self.transpose_state_when_forward = transpose_state_when_forward

        # this allows to define one at some point and it will then be used going forward until it is reset
        self.auto_assembly_plans = None

        # norm penalty
        self.current_norm_penalty = None

    def get_current_norm_penalty(self):
        return self.current_norm_penalty

    def get_norm_penalty(self):
        return self.get_current_norm_penalty()

    def _get_nonlinearity(self, nonlinearity):
        supported_nonlinearities = ['identity', 'relu', 'tanh', 'sigmoid', 'softmax']

        if nonlinearity is None:
            use_nonlinearity = 'identity'
        else:
            use_nonlinearity = nonlinearity.lower()

        if use_nonlinearity not in supported_nonlinearities:
            raise ValueError('Unsupported nonlinearity {}'.format(use_nonlinearity))

        if use_nonlinearity == 'relu':
            nl = nn.functional.relu
            dnl = drelu
        elif use_nonlinearity == 'tanh':
            nl = torch.tanh
            dnl = dtanh
        elif use_nonlinearity == 'identity':
            nl = identity
            dnl = didentity
        elif use_nonlinearity == 'sigmoid':
            nl = torch.sigmoid
            dnl = torch.sigmoid
        elif use_nonlinearity == 'softmax':
            nl = softmax
            dnl = dsoftmax
        else:
            raise ValueError('Unknown nonlinearity {}'.format(use_nonlinearity))

        return nl,dnl

    @abstractmethod
    def create_initial_state_and_costate_parameters(self,batch_y0,only_random_initialization):
        # creates these as a sorted dictionary and returns them as a tupe (state_dict,costate_dict) (engtries need to be in the same order!!)
        pass

    def create_auto_assembly_plans(self,input):
        # does this based on the given vectorized input and the parameters that exist
        # assumes that the data_dict has the same structure as the state_dict
        auto_assembly_plans = dict()

        _, state_assembly_plan = self._assemble_generic_dict(self._state_parameter_dict)
        _, costate_assembly_plan = self._assemble_generic_dict(self._costate_parameter_dict)

        auto_assembly_plans['state'] = state_assembly_plan
        auto_assembly_plans['costate'] = costate_assembly_plan

        # the data vector is at the end and has the same structure as the state vector
        # so let's try to constitute the corresponding assembly plan

        # let's first go through the state and the costate and see how big they are

        incr = 0
        begin_state = None
        end_state = None

        # measure the size
        for k in state_assembly_plan:
            current_shape = state_assembly_plan[k]
            if begin_state is None:
                begin_state = incr

            incr += current_shape[0]
            end_state = incr

        nr = input.shape[0]
        nr_state = end_state-begin_state
        nr_costate = nr_state
        nr_data = nr-(nr_state+nr_costate)

        factor = int(nr_data/nr_state)

        data_assembly_plan = SortedDict()

        for k in state_assembly_plan:
            cp = list(state_assembly_plan[k])
            cp[0] *= factor
            data_assembly_plan[k] = torch.Size(cp)

        auto_assembly_plans['data'] = data_assembly_plan

        return auto_assembly_plans

    def _assemble_generic_dict(self,d):
        # d is a sorted dictionary
        # first test that this assumption is true
        if type(d)!=SortedDict:
            raise ValueError('Expected a SortedDict, but got {}'.format(type(d)))

        d_list = []
        assembly_plan = SortedDict()
        for k in d:
            d_list.append(d[k])
            assembly_plan[k] = d[k].shape
        return torch.cat(tuple(d_list)), assembly_plan

    def assemble_tensor(self,state_dict,costate_dict,data_dict):
        # these are all ordered dictionaries, will assemble all into a big torch vector
        state_vector,state_assembly_plan = self._assemble_generic_dict(state_dict)
        costate_vector,costate_assembly_plan = self._assemble_generic_dict(costate_dict)
        data_vector,data_assembly_plan = self._assemble_generic_dict(data_dict)

        assembly_plans = dict()
        assembly_plans['state'] = state_assembly_plan
        assembly_plans['costate'] = costate_assembly_plan
        assembly_plans['data'] = data_assembly_plan

        return torch.cat((state_vector,costate_vector,data_vector)),assembly_plans

    @abstractmethod
    def disassemble(self, input):
        #Is supposed to return the desired data state (possibly only one) from an input vector
        pass

    def disassemble_tensor(self, input, assembly_plans=None, dim=0):
        # will create sorted dictionaries for state, costate and data based on the assembly plans

        supported_dims = [0,1]
        if dim not in supported_dims:
            raise ValueError('Only supports dimensions 0 and 1; if 1, then the 0-th dimension is time')


        if assembly_plans is None:
            if self.auto_assembly_plans is None:
                print('WARNING: created data assembly plan automatically, this could go wrong, if data dimension changed, call create_auto_assembly_plan again.')
                self.auto_assembly_plans = self.create_auto_assembly_plans(input=input)

            assembly_plans = self.auto_assembly_plans

        state_dict = SortedDict()
        costate_dict = SortedDict()
        data_dict = SortedDict()

        incr = 0
        for ap in ['state','costate','data']:

            assembly_plan = assembly_plans[ap]

            for k in assembly_plan:
                current_shape = assembly_plan[k]

                if dim==0:
                    if ap=='state':
                        state_dict[k] = input[incr:incr+current_shape[0],...]
                    elif ap=='costate':
                        costate_dict[k] = input[incr:incr+current_shape[0],...]
                    elif ap=='data':
                        data_dict[k] = input[incr:incr+current_shape[0],...]
                    else:
                        raise ValueError('Unknown key {}'.format(ap))
                else:
                    if ap=='state':
                        state_dict[k] = input[:,incr:incr+current_shape[0],...]
                    elif ap=='costate':
                        costate_dict[k] = input[:,incr:incr+current_shape[0],...]
                    elif ap=='data':
                        data_dict[k] = input[:,incr:incr+current_shape[0],...]
                    else:
                        raise ValueError('Unknown key {}'.format(ap))

                incr += current_shape[0]

        return state_dict,costate_dict,data_dict

    def compute_potential_energy(self,state_dict,costate_dict,parameter_dict):

        # this is really only how one propagates through the system given the parameterization

        rhs_state_dict = self.rhs_advect_state(state_dict=state_dict,parameter_dict=parameter_dict)

        potential_energy = 0

        for ks,kcs in zip(rhs_state_dict,costate_dict):
            potential_energy = potential_energy + torch.mean(costate_dict[kcs]*rhs_state_dict[ks])

        return potential_energy

    def compute_kinetic_energy(self,parameter_dict,parameter_weight_dict=None):
        # as a default it just computes the square norms of all of them (overwrite this if it is not the desired behavior)
        # a weight dictionary can be specified for the individual parameters (default is all uniform weight)

        kinetic_energy = 0

        if parameter_weight_dict is not None:
            for k,w in zip(parameter_dict,parameter_weight_dict):
                cpar = parameter_dict[k]
                weight = parameter_weight_dict[w]
                cpar_penalty = (cpar ** 2).sum()*weight
                kinetic_energy = kinetic_energy + cpar_penalty
        else:
            for k in parameter_dict:
                cpar = parameter_dict[k]
                cpar_penalty = (cpar ** 2).sum()
                kinetic_energy = kinetic_energy + cpar_penalty

        kinetic_energy = 0.5*kinetic_energy

        return kinetic_energy

    def compute_lagrangian(self, state_dict, costate_dict, parameter_dict):

        kinetic_energy = self.compute_kinetic_energy(parameter_dict=parameter_dict)
        potential_energy = self.compute_potential_energy(state_dict=state_dict,costate_dict=costate_dict,parameter_dict=parameter_dict)

        lagrangian = kinetic_energy-potential_energy

        return lagrangian, kinetic_energy, potential_energy

    @abstractmethod
    def get_initial_condition(self,x):
        # Initial condition from given data vector
        # easiest to first build a data dictionary and then call get_initial_conditions_from_data_dict(self,data_dict):
        pass

    def get_initial_conditions_from_data_dict(self,data_dict):
        # initialize the second state of x with zero so far
        initial_conditions,assembly_plans = self.assemble_tensor(state_dict=self._state_parameter_dict, costate_dict=self._costate_parameter_dict, data_dict=data_dict)
        if self.auto_assembly_plans is not None:
            #print('INFO: updated the assembly plans')
            self.auto_assembly_plans = assembly_plans
        return initial_conditions

    @abstractmethod
    def create_default_parameter_dict(self):
        raise ValueError('Not implemented. Needs to return a SortedDict of parameters')

    def extract_dict_from_tuple_based_on_generic_dict(self,data_tuple,generic_dict,prefix=''):
        extracted_dict = SortedDict()
        indx = 0
        for k in generic_dict:
            extracted_dict[prefix+k] = data_tuple[indx]
            indx += 1

        return extracted_dict

    def compute_tuple_from_generic_dict(self,generic_dict):
        # form a tuple of all the state variables (because this is what we take the derivative of)
        sv_list = []
        for k in generic_dict:
            sv_list.append(generic_dict[k])

        return tuple(sv_list)

    @abstractmethod
    def rhs_advect_state(self, state_dict, parameter_dict):
        pass

    def rhs_advect_data(self,data_dict,parameter_dict):
        return self.rhs_advect_state(state_dict=data_dict,parameter_dict=parameter_dict)

    @abstractmethod
    def rhs_advect_costate(self, state_dict, costate_dict, parameter_dict):
        # now that we have the parameters we can get the rhs for the costate using autodiff
        # returns a dictionary of the RHS of the costate
        pass

    def add_multiple_to_parameter_dict(self,pd_1,pd_2,multiplier=1.0):

        res = SortedDict()
        for k1,k2 in zip(pd_1,pd_2):
            res[k1] = pd_1[k1]+multiplier*pd_2[k2]

        return res

    def negate_divide_and_store_as_parameter_dict(self,generic_dict,parameter_dict_for_keys,parameter_weight_dict=None):

        parameter_dict = SortedDict()

        if parameter_weight_dict is None:

            for kg,kp in zip(generic_dict,parameter_dict_for_keys):
                parameter_dict[kp] = -generic_dict[kg]

        else:

            for kgp,kw in zip(zip(generic_dict,parameter_dict_for_keys),parameter_weight_dict):
                parameter_dict[kgp[1]] = -parameter_weight_dict[kw]*generic_dict[kgp[0]]

        return parameter_dict

    @abstractmethod
    def compute_parameters(self,state_dict,costate_dict):
        """
        Computes parameters and returns a tuple of a SortedDict parameter dictionary and the current kinectic energy (i.e., penalizer on parameters)
        :param state_dict:
        :param costate_dict:
        :return:
        """
        pass

    def compute_gradients(self,state_dict,costate_dict,data_dict):

        parameter_dict, current_kinetic_energy = self.compute_parameters(state_dict=state_dict,costate_dict=costate_dict)

        self.current_norm_penalty = current_kinetic_energy

        dot_state_dict = self.rhs_advect_state(state_dict=state_dict,parameter_dict=parameter_dict)
        dot_data_dict = self.rhs_advect_data(data_dict=data_dict,parameter_dict=parameter_dict)
        dot_costate_dict = self.rhs_advect_costate(state_dict=state_dict,costate_dict=costate_dict,parameter_dict=parameter_dict)

        return dot_state_dict,dot_costate_dict,dot_data_dict


    def transpose_state(self,generic_dict):
        ret = SortedDict()
        for k in generic_dict:
            ret[k] = (generic_dict[k]).transpose(1,2)
        return ret

    def register_state_and_costate_parameters(self,state_dict,costate_dict):

        self._state_parameter_dict = state_dict
        self._costate_parameter_dict = costate_dict

        if type(self._state_parameter_dict) != SortedDict:
            raise ValueError('state parameter dictionrary needs to be an SortedDict and not {}'.format(
                type(self._state_parameter_dict)))

        if type(self._costate_parameter_dict) != SortedDict:
            raise ValueError('costate parameter dictionrary needs to be an SortedDict and not {}'.format(
                type(self._costate_parameter_dict)))

        for k in self._state_parameter_dict:
            self.register_parameter(k,self._state_parameter_dict[k])

        for k in self._costate_parameter_dict:
            self.register_parameter(k,self._costate_parameter_dict[k])

    def forward(self, t,input):

        state_dict,costate_dict,data_dict = self.disassemble_tensor(input)

        if self.transpose_state_when_forward:
            state_dict = self.transpose_state(state_dict)
            costate_dict = self.transpose_state(costate_dict)
            data_dict = self.transpose_state(data_dict)

        # computing the gradients
        dot_state_dict, dot_costate_dict, dot_data_dict = \
            self.compute_gradients(state_dict=state_dict,costate_dict=costate_dict,data_dict=data_dict)

        if self.transpose_state_when_forward:
            # as we transposed the vectors before we need to transpose on the way back
            dot_state_dict = self.transpose_state(dot_state_dict)
            dot_costate_dict = self.transpose_state(dot_costate_dict)
            dot_data_dict = self.transpose_state(dot_data_dict)

        # create a vector out of this to pass to integrator
        output,assembly_plans = self.assemble_tensor(state_dict=dot_state_dict, costate_dict=dot_costate_dict, data_dict=dot_data_dict)

        return output

class AutogradShootingBlockBase(ShootingBlockBase):
    def __init__(self, batch_y0=None, nonlinearity=None, only_random_initialization=False,transpose_state_when_forward=True):
        super(AutogradShootingBlockBase, self).__init__(nonlinearity=nonlinearity,transpose_state_when_forward=transpose_state_when_forward)

    def rhs_advect_costate(self, state_dict, costate_dict, parameter_dict):
        # now that we have the parameters we can get the rhs for the costate using autodiff

        current_lagrangian, current_kinetic_energy, current_potential_energy = \
            self.compute_lagrangian(state_dict=state_dict, costate_dict=costate_dict, parameter_dict=parameter_dict)

        # form a tuple of all the state variables (because this is what we take the derivative of)
        state_tuple = self.compute_tuple_from_generic_dict(state_dict)

        dot_costate_tuple = autograd.grad(current_lagrangian, state_tuple,
                                          grad_outputs=current_lagrangian.data.new(current_lagrangian.shape).fill_(1),
                                          create_graph=True,
                                          retain_graph=True,
                                          allow_unused=True)

        # now we need to put these into a sorted dictionary
        dot_costate_dict = self.extract_dict_from_tuple_based_on_generic_dict(data_tuple=dot_costate_tuple,
                                                                              generic_dict=state_dict, prefix='dot_co_')

        return dot_costate_dict


class LinearInParameterAutogradShootingBlock(AutogradShootingBlockBase):
    def __init__(self, batch_y0=None, nonlinearity=None, only_random_initialization=False,transpose_state_when_forward=True):
        super(LinearInParameterAutogradShootingBlock, self).__init__(nonlinearity=nonlinearity,transpose_state_when_forward=transpose_state_when_forward)

    def compute_parameters_directly(self, state_dict, costate_dict, parameter_weight_dict=None):
        # we assume this is linear here, so we do not need a fixed point iteration, but can just compute the gradient

        parameter_dict = self.create_default_parameter_dict()

        current_lagrangian, current_kinetic_energy, current_potential_energy = \
            self.compute_lagrangian(state_dict=state_dict, costate_dict=costate_dict, parameter_dict=parameter_dict)

        parameter_tuple = self.compute_tuple_from_generic_dict(parameter_dict)

        parameter_grad_tuple = autograd.grad(current_potential_energy,
                                             parameter_tuple,
                                             grad_outputs=current_potential_energy.data.new(
                                                 current_potential_energy.shape).fill_(1),
                                             create_graph=True,
                                             retain_graph=True,
                                             allow_unused=True)

        parameter_grad_dict = self.extract_dict_from_tuple_based_on_generic_dict(data_tuple=parameter_grad_tuple,
                                                                                 generic_dict=parameter_dict,
                                                                                 prefix='grad_')

        parameter_dict = self.negate_divide_and_store_as_parameter_dict(parameter_grad_dict,
                                                                        parameter_dict_for_keys=parameter_dict,
                                                                        parameter_weight_dict=parameter_weight_dict)

        return parameter_dict, current_kinetic_energy

    def compute_parameters(self,state_dict,costate_dict):
        return self.compute_parameters_directly(state_dict=state_dict,costate_dict=costate_dict)


class NonlinearInParameterAutogradShootingBlock(AutogradShootingBlockBase):
    def __init__(self, batch_y0=None, nonlinearity=None, only_random_initialization=False,transpose_state_when_forward=True):
        super(NonlinearInParameterAutogradShootingBlock, self).__init__(nonlinearity=nonlinearity,transpose_state_when_forward=transpose_state_when_forward)

    def compute_parameters_iteratively(self, state_dict, costate_dict):

        learning_rate = 0.5
        nr_of_fixed_point_iterations = 5

        parameter_dict = self.create_default_parameter_dict()

        for n in range(nr_of_fixed_point_iterations):
            current_lagrangian, current_kinectic_energy, current_potential_energy = \
                self.compute_lagrangian(state_dict=state_dict, costate_dict=costate_dict, parameter_dict=parameter_dict)

            parameter_tuple = self.compute_tuple_from_generic_dict(parameter_dict)

            parameter_grad_tuple = autograd.grad(current_lagrangian,
                                                 parameter_tuple,
                                                 grad_outputs=current_lagrangian.data.new(
                                                     current_lagrangian.shape).fill_(1),
                                                 create_graph=True,
                                                 retain_graph=True,
                                                 allow_unused=True)

            parameter_grad_dict = self.extract_dict_from_tuple_based_on_generic_dict(data_tuple=parameter_grad_tuple,
                                                                                     generic_dict=parameter_dict,
                                                                                     prefix='grad_')

            parameter_dict = self.add_multiple_to_parameter_dict(parameter_dict, parameter_grad_dict,
                                                                 multiplier=-learning_rate)

        return parameter_dict, current_kinectic_energy

    def compute_parameters(self,state_dict,costate_dict):
        return self.compute_parameters_iteratively(state_dict=state_dict,costate_dict=costate_dict)


class AutoShootingBlockModel(LinearInParameterAutogradShootingBlock):
    def __init__(self, batch_y0=None, nonlinearity=None, only_random_initialization=False,transpose_state_when_forward=True):
        super(LinearInParameterAutogradShootingBlock, self).__init__(nonlinearity=nonlinearity,transpose_state_when_forward=transpose_state_when_forward)

        state_dict, costate_dict = self.create_initial_state_and_costate_parameters(batch_y0=batch_y0,only_random_initialization=only_random_initialization)
        self.register_state_and_costate_parameters(state_dict=state_dict,costate_dict=costate_dict)

    def create_initial_state_and_costate_parameters(self, batch_y0, only_random_initialization=True):
        # creates these as a sorted dictionary and returns it (need to be in the same order!!)
        state_dict = SortedDict()
        costate_dict = SortedDict()

        rand_mag_q = 0.5
        rand_mag_p = 0.5

        self.k = batch_y0.size()[0]
        self.d = batch_y0.size()[2]

        if only_random_initialization:
            # do a fully random initialization
            state_dict['q1'] = nn.Parameter(rand_mag_q * torch.randn_like(batch_y0))
            state_dict['q2'] = nn.Parameter(rand_mag_q * torch.randn_like(batch_y0))
            costate_dict['p1'] = nn.Parameter(rand_mag_p * torch.randn([self.k, 1, self.d]))
            costate_dict['p2'] = nn.Parameter(rand_mag_p * torch.randn([self.k, 1, self.d]))
        else:
            state_dict['q1'] = nn.Parameter(batch_y0 + rand_mag_q * torch.randn_like(batch_y0))
            state_dict['q2'] = nn.Parameter(batch_y0 + rand_mag_q * torch.randn_like(batch_y0))
            costate_dict['p1'] = nn.Parameter(torch.zeros(self.k, 1, self.d) + rand_mag_p * torch.randn([self.k, 1, self.d]))
            costate_dict['p2'] = nn.Parameter(torch.zeros(self.k, 1, self.d) + rand_mag_p * torch.randn([self.k, 1, self.d]))

        return state_dict,costate_dict

    def create_default_parameter_dict(self):
        parameter_dict = SortedDict()

        parameter_dict['theta1'] = torch.randn(2, 2, requires_grad=True).to(device)
        parameter_dict['bias1'] = torch.randn(2, 1, requires_grad=True).to(device)

        parameter_dict['theta2'] = torch.randn(2, 2, requires_grad=True).to(device)
        parameter_dict['bias2'] = torch.randn(2, 1, requires_grad=True).to(device)

        return parameter_dict

    def rhs_advect_state(self, state_dict, parameter_dict):

        rhs = SortedDict()

        s = state_dict
        p = parameter_dict

        rhs['dot_q1'] = torch.matmul(p['theta1'], self.nl(s['q2'])) + p['bias1']
        rhs['dot_q2'] = torch.matmul(p['theta2'], s['q1']) + p['bias2']

        return rhs

    def get_initial_condition(self, x):
        # Initial condition from given data vector
        # easiest to first build a data dictionary and then call get_initial_conditions_from_data_dict(self,data_dict):
        data_dict = SortedDict()
        data_dict['q1'] = x
        data_dict['q2'] = torch.zeros_like(x)

        initial_conditions = self.get_initial_conditions_from_data_dict(data_dict=data_dict)

        return initial_conditions

    def disassemble(self,input,dim=1):
        state_dict, costate_dict, data_dict = self.disassemble_tensor(input, dim=dim)
        return data_dict['q1']


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

        shooting = AutoShootingBlockModel(batch_y0, only_random_initialization=True, nonlinearity=args.nonlinearity)

        shooting = shooting.to(device)

        #optimizer = optim.RMSprop(shooting.parameters(), lr=5e-3)
        optimizer = optim.Adam(shooting.parameters(), lr=2.5e-2)
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

            if is_higher_order_model:
                z_0 = shooting.get_initial_condition(x=batch_y0)
            else:
                q = (shooting.q_params)
                p = (shooting.p_params)
                z_0 = torch.cat((q,p,batch_y0))

            temp_pred_y = odeint(shooting,z_0 , batch_t, method=args.method, atol=atol, rtol=rtol, options=options)

            # we are actually only interested in the prediction of the batch itself (not the parameterization)
            if is_higher_order_model:
                pred_y = shooting.disassemble(temp_pred_y,dim=1)
                #_,_,_,_,pred_y,_ = shooting.disassemble(temp_pred_y,dim=1)
            else:
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
                    visualize(val_y, val_pred_y, val_t, func, ii, is_odenet=is_odenet, is_higher_order_model=is_higher_order_model)
                    ii += 1

            else:
                ### time clock
                t_1 = time.time()

                print("time",t_1 - t_0)
                t_0 = t_1

                if is_higher_order_model:
                    val_z_0 = shooting.get_initial_condition(x=val_y0)
                else:
                    q = (shooting.q_params)
                    p = (shooting.p_params)
                    val_z_0 = torch.cat((q, p, val_y0))

                tst = shooting(val_t,val_z_0)
                temp_pred_y = odeint(shooting, val_z_0, val_t, method=args.method, atol=atol, rtol=rtol,
                                     options=options)

                # we are actually only interested in the prediction of the batch itself (not the parameterization)
                if is_higher_order_model:
                   # _, _, _, _, val_pred_y, _ = shooting.disassemble(temp_pred_y,dim=1)
                   val_pred_y = shooting.disassemble(temp_pred_y,dim=1)

                else:
                    val_pred_y = temp_pred_y[:, 2 * K:, ...]

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
