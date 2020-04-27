# the goal of this model is to map the input [-2,2] to a desired functional output
# for simplicity we will try to do this with a very simple 1D shooting model

import os
import sys
import time
import random
import argparse
import numpy as np

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torchdiffeq import odeint

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.insert(0, '../neuro_shooting')

import neuro_shooting
import neuro_shooting.shooting_blocks as sblocks
import neuro_shooting.shooting_models as smodels
import neuro_shooting.generic_integrator as gi
import neuro_shooting.parameter_initialization as pi

def setup_cmdline_parsing():
    parser = argparse.ArgumentParser('Shooting spiral')
    parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--niters', type=int, default=4000)

    # shooting model parameters
    parser.add_argument('--shooting_model', type=str, default='updown', choices=['resnet_updown','simple', '2nd_order', 'updown'])
    parser.add_argument('--nonlinearity', type=str, default='relu', choices=['identity', 'relu', 'tanh', 'sigmoid',"softmax"], help='Nonlinearity for shooting.')
    parser.add_argument('--pw', type=float, default=1.0, help='parameter weight')
    parser.add_argument('--nr_of_particles', type=int, default=8, help='Number of particles to parameterize the initial condition')

    # non-shooting networks implemented
    parser.add_argument('--nr_of_layers', type=int, default=30, help='Number of layers for the non-shooting networks')
    parser.add_argument('--use_updown',action='store_true')
    parser.add_argument('--use_double_resnet',action='store_true')
    parser.add_argument('--use_rnn',action='store_true',default = True)
    parser.add_argument('--use_double_resnet_rnn',action="store_true")
    parser.add_argument('--use_simple_resnet',action='store_true')
    parser.add_argument('--use_neural_ode',action='store_true')

    parser.add_argument('--test_freq', type=int, default=100)
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    return args

def get_sample_batch(nr_of_samples=10):

    sample_batch_in = 4*torch.rand([nr_of_samples,1,1])-2 # creates uniform samples in [-2,2]
    sample_batch_out = sample_batch_in**3 # and takes them to the power of three for the output

    return sample_batch_in, sample_batch_out

def replicate_modules(module,nr_of_layers):
    modules = OrderedDict()
    for i in range(nr_of_layers):
        modules['l{}'.format(i)] = module()

    return modules

class SimpleResNetBlock(nn.Module):

    def __init__(self):
        super(SimpleResNetBlock, self).__init__()

        self.l1 = nn.Linear(1,1,bias=True)

    def forward(self, x):
        x = x + self.l1(F.relu(x))

        return x

class UpDownResNetBlock(nn.Module):

    def __init__(self):
        super(UpDownResNetBlock, self).__init__()

        self.l1 = nn.Linear(1,20,bias=True)
        self.l2 = nn.Linear(20,1,bias=True)

    def forward(self, x):
        y = self.l1(F.relu(x))
        z = self.l2(F.relu(y))

        return x + z

class UpDownDoubleResNetBlock(nn.Module):

    def __init__(self):
        super(UpDownDoubleResNetBlock, self).__init__()

        self.l1 = nn.Linear(5,1,bias=True)
        self.l2 = nn.Linear(1,5,bias=False)

    def forward(self, x1x2):
        x1 = x1x2[0]
        x2 = x1x2[1]

        x1 = x1 + self.l1(F.relu(x2))
        x2 = x2 + self.l2(F.relu(x1)) # this is what an integrator would typically do
        #x2 = self.l2(F.relu(x1))

        return x1, x2

class DoubleResNetUpDown(nn.Module):

    def __init__(self, nr_of_layers=30):
        super(DoubleResNetUpDown, self).__init__()
        print("nr_of_layers ",nr_of_layers)
        modules = replicate_modules(module=UpDownDoubleResNetBlock,nr_of_layers=nr_of_layers)
        self.model = nn.Sequential(modules)

    def forward(self, x1x2):
        return self.model(x1x2)

class ResNetUpDown(nn.Module):

    def __init__(self, nr_of_layers=10):
        super(ResNetUpDown, self).__init__()
        modules = replicate_modules(module=UpDownResNetBlock, nr_of_layers=nr_of_layers)
        self.model = nn.Sequential(modules)

    def forward(self, x):
        return self.model(x)

class ResNet(nn.Module): # corresponds to our simple shooting model

    def __init__(self, nr_of_layers=10):
        super(ResNet, self).__init__()

        modules = replicate_modules(module=SimpleResNetBlock, nr_of_layers=nr_of_layers)
        self.model = nn.Sequential(modules)

    def forward(self, x):
        return self.model(x)

class DoubleResNetUpDownRNN(nn.Module): # corresponds to our simple shooting model

    def __init__(self, nr_of_layers=10):
        super(DoubleResNetUpDownRNN, self).__init__()

        self.nr_of_layers = nr_of_layers
        print("use "+str(self) + " with " + str(self.nr_of_layers) + " nr of layers")
        self.l1 = UpDownDoubleResNetBlock()

    def forward(self, x1x2):
        x1 = x1x2[0]
        x2 = x1x2[1]

        for i in range(self.nr_of_layers):
            x1, x2 = self.l1((x1, x2))

        return x1, x2

class ResNetRNN(nn.Module):

    def __init__(self, nr_of_layers=10):
        super(ResNetRNN, self).__init__()
        self.nr_of_layers = nr_of_layers
        print("use "+ str(self) + " nr of layers " + str(self.nr_of_layers))

        self.l1 = nn.Linear(5,1,bias=True)
        self.l2 = nn.Linear(1,5,bias=True)

    def forward(self, x):
        for i in range(self.nr_of_layers):
            x = x + 1./self.nr_of_layers * self.l1(F.relu(self.l2(x)))

        return x

class ODESimpleFunc(nn.Module):

    def __init__(self):
        super(ODESimpleFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 1),
            nn.Tanh(),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)

def print_all_parameters(model):

    print('\n Model parameters:\n')
    for pn,pv in model.named_parameters():
        print('{} = {}\n'.format(pn, pv))

def collect_and_sort_parameter_values_across_layers(model):

    # TODO: in principle I would like to extend this so that we sort the states so it is possible
    # to follow how they change over the layers; this should happen automatically for the continuous
    # models, but states can swap arbitrarily for the resnet models

    print('\n Collecting and sorting parameter values across layers')
    named_pars = list(model.named_parameters())

    nr_of_pars= len(named_pars)

    par_names_per_block = []
    par_size_per_block = []

    # first get the
    for i in range(nr_of_pars):
        cur_par_name = named_pars[i][0]
        cur_par_val = named_pars[i][1]
        names = cur_par_name.split('.')

        if i==0:
            first_layer_name = names[1]

        if first_layer_name==names[1]:
            par_names_per_block.append('.'.join(names[2:]))
            par_size_per_block.append(cur_par_val.size())
        else:
            break

    # nr of pars per block
    nr_of_pars_per_block = len(par_names_per_block)

    # nr of layers
    nr_of_layers = nr_of_pars//nr_of_pars_per_block

    # create the torch arrays where we will store the variables as they change over the layers
    layer_pars = dict()
    for pn,ps in zip(par_names_per_block,par_size_per_block):
        layer_pars[pn] = torch.zeros([nr_of_layers] + list(ps))

    # now we can put the values in these arrays
    layer_nr = 0
    current_layer_name = first_layer_name

    for i in range(nr_of_pars):
        cur_par_name = named_pars[i][0]
        cur_par_val = named_pars[i][1]
        names = cur_par_name.split('.')

        if current_layer_name != names[1]:
            current_layer_name = names[1]
            layer_nr+=1

        layer_par_name = '.'.join(names[2:])
        layer_pars[layer_par_name][layer_nr,...] = cur_par_val

    print('\nCollected (not yet sorted: TODO) block parameters across layers:\n')
    for k in layer_pars:
        print('{} = {}'.format(k,layer_pars[k]))

    import matplotlib.pyplot as plt
    import matplotlib
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from matplotlib import gridspec

    spec = gridspec.GridSpec(ncols=1, nrows=4,
                             width_ratios=[4], height_ratios=[5,1,5,5])

    fig = plt.figure() #(figsize=(12, 4), facecolor='white')
    l1w = fig.add_subplot(spec[0])
    l1b = fig.add_subplot(spec[1])
    l2w = fig.add_subplot(spec[2])
    l2b = fig.add_subplot(spec[3])

    l1w.cla()
    l1w.set_title('l1-weight')
    #l1w.set_xlabel('layers')
    A = layer_pars['l1.weight'].detach().numpy().squeeze()
    A.sort()
    im1 = l1w.imshow(A.transpose(),
                     norm=matplotlib.colors.SymLogNorm(linthresh=0.01, linscale=0.03,vmin=-1.0, vmax=1.0),
                     aspect='auto')

    divider = make_axes_locatable(l1w)
    cax = divider.append_axes('bottom', size='20%', pad=0.25)
    fig.colorbar(im1, cax=cax, orientation='horizontal')

    l1b.cla()
    l1b.set_title('l1-bias')
    l1b.set_xlabel('layers')
    im2 = l1b.imshow(layer_pars['l1.bias'].detach().numpy().transpose(),
                    norm=matplotlib.colors.SymLogNorm(linthresh=0.01, linscale=0.03,vmin=-1.0, vmax=1.0),
                    aspect='auto')

    divider = make_axes_locatable(l1b)
    cax = divider.append_axes('bottom', size='50%', pad=0.25)
    fig.colorbar(im2, cax=cax, orientation='horizontal')

    l2w.cla()
    l2w.set_title('l2-weight')
    #l2w.set_xlabel('layers')
    B = layer_pars['l2.weight'].detach().numpy().squeeze()
    B.sort()
    im3 = l2w.imshow(B.transpose(),
                     #norm=matplotlib.colors.SymLogNorm(linthresh=0.01, linscale=0.03, vmin=-1.0, vmax=1.0),
                     aspect='auto')

    divider = make_axes_locatable(l2w)
    cax = divider.append_axes('bottom', size='20%', pad=0.25)
    fig.colorbar(im3, cax=cax, orientation='horizontal')

    # l2b.cla()
    # l2b.set_title('l2-bias')
    # #l2b.set_xlabel('layers')
    # im4 = l2b.imshow(layer_pars['l2.bias'].detach().numpy().squeeze().transpose(),
    #                  norm=matplotlib.colors.SymLogNorm(linthresh=0.01, linscale=0.03, vmin=-1.0, vmax=1.0),
    #                  aspect='auto')
    #
    # divider = make_axes_locatable(l2b)
    # cax = divider.append_axes('bottom', size='20%', pad=0.25)
    # fig.colorbar(im4, cax=cax, orientation='horizontal')

    fig.show()


if __name__ == '__main__':

    args = setup_cmdline_parsing()
    if args.verbose:
        print(args)


    def parameters_hook(module, t, state_dicts, costate_dicts, data_dict_of_dicts,
                              dot_state_dicts, dot_costate_dicts, dot_data_dict_of_dicts, parameter_objects,
                              custom_hook_data):

        with torch.no_grad():
            custom_hook_data['parameters'].append(
                (
                    t, parameter_objects["l2"]._parameter_dict["weight"]
                ))
        return None

    par_init = pi.VectorEvolutionSampleBatchParameterInitializer(
        only_random_initialization=True,
        random_initialization_magnitude=0.1, 
        sample_batch = torch.linspace(0,1,args.nr_of_particles))


    shootingintegrand_kwargs = {'in_features': 1,
                                'nonlinearity': args.nonlinearity,
                                'nr_of_particles': args.nr_of_particles,
                                'parameter_weight': args.pw,
                                'particle_dimension': 1,
                                'particle_size': 1,
                                "costate_initializer":pi.VectorEvolutionParameterInitializer(random_initialization_magnitude=0.1)}

    if args.shooting_model == 'simple':
        smodel = smodels.AutoShootingIntegrandModelSimple(**shootingintegrand_kwargs,use_analytic_solution=True)
    elif args.shooting_model == '2nd_order':
        smodel = smodels.AutoShootingIntegrandModelSecondOrder(**shootingintegrand_kwargs)
    elif args.shooting_model == 'updown':
        smodel = smodels.AutoShootingIntegrandModelUpDown(**shootingintegrand_kwargs,use_analytic_solution=True)
    elif args.shooting_model == 'resnet_updown':
        smodel = smodels.AutoShootingIntegrandModelResNetUpDown(**shootingintegrand_kwargs)

    sblock = sblocks.ShootingBlockBase(
        name='simple',
        shooting_integrand=smodel,
        integrator_name='rk4',
        use_adjoint_integration=False,
        #integrator_options = {'stepsize':0.1}
    )

    use_shooting = False
    if args.use_rnn:
        weight_decay = 0.0001
        lr = 1e-3
        print('Using ResNetRNN: weight = {}'.format(weight_decay))
        simple_resnet = ResNetRNN(nr_of_layers=args.nr_of_layers)
    elif args.use_double_resnet_rnn:
        weight_decay = 0.025
        lr = 1e-2
        print('Using DoubleResNetRNN: weight = {}'.format(weight_decay))
        simple_resnet = DoubleResNetUpDownRNN(nr_of_layers=args.nr_of_layers)
    elif args.use_updown:
        weight_decay = 0.01
        lr = 1e-2
        print('Using ResNetUpDown: weight = {}'.format(weight_decay))
        simple_resnet = ResNetUpDown(nr_of_layers=args.nr_of_layers)
    elif args.use_simple_resnet:
        weight_decay = 0.0001
        lr = 1e-2
        print('Using ResNet: weight = {}'.format(weight_decay))
        simple_resnet = ResNet(nr_of_layers=args.nr_of_layers)
    elif args.use_double_resnet:
        #weight_decay = 0.025
        weight_decay = 0.01
        lr = 1e-2
        print('Using DoubleResNetUpDown: weight = {}'.format(weight_decay))
        simple_resnet = DoubleResNetUpDown(nr_of_layers=args.nr_of_layers)
    elif args.use_neural_ode:
        print('Using neural ode')
        func = ODESimpleFunc()
        optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
        integrator = gi.GenericIntegrator(integrator_library='odeint',
                                                          integrator_name=args.method,
                                                          use_adjoint_integration=False,
                                                          rtol=1e-8, atol=1e-12)

    else:
        use_shooting = True
        print('Using shooting')


    if not use_shooting:
        if args.use_neural_ode:
            optimizer = optim.RMSprop(func.parameters(), lr=1e-2)
        else:
            optimizer = optim.Adam(simple_resnet.parameters(), lr=lr, weight_decay=weight_decay)
    else:

        sample_batch_in, sample_batch_out = get_sample_batch(nr_of_samples=args.batch_size)
        sblock(x=sample_batch_in)

        # do some parameter freezing
        for pn,pp in sblock.named_parameters():
            if pn=='q1':
                # freeze the locations
                #pp.requires_grad = False
                pass
        optimizer = optim.Adam(sblock.parameters(), lr=1e-2)

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.show(block=False)

    track_loss = []
    for itr in range(1, args.niters + 1):

        # get current batch data
        batch_in, batch_out = get_sample_batch(nr_of_samples=args.batch_size)

        optimizer.zero_grad()

        if not use_shooting:
            if args.use_neural_ode:

                pred_y = integrator.integrate(func=func, x0=batch_in, t=torch.tensor([0, 1]).float())
                pred_y = pred_y[1, :, :, :] # need prediction at time 1

            elif args.use_double_resnet or args.use_double_resnet_rnn:
                x20 = torch.zeros_like(batch_in)
                sz = [1] * len(x20.shape)
                sz[-1] = 5
                x20 = x20.repeat(sz)
                x1x2 = (batch_in, x20)
                pred_y, pred_y2 = simple_resnet(x1x2)
            else:
                pred_y = simple_resnet(x=batch_in)
        else:
            # set integration time
            # sblock.set_integration_time(time_to=1.0) # try to do this mapping in unit time
            pred_y, _, _, _ = sblock(x=batch_in)


        loss = torch.mean((pred_y - batch_out)**2)  # + 1e-2 * sblock.get_norm_penalty()
        loss.backward()

        track_loss.append(loss.item())
        optimizer.step()

        if itr % args.test_freq == 0:
            fig = plt.figure(figsize=(8, 12), facecolor='white')
            ax = fig.add_subplot(111, frameon=False)
            #ax.cla()
            ax.plot(batch_in.detach().numpy().squeeze(),batch_out.detach().numpy().squeeze(),'g+')
            ax.plot(batch_in.detach().numpy().squeeze(),pred_y.detach().numpy().squeeze(),'r*')
            ax.set_xlim(-2, 2)
            ax.set_ylim(-8, 8)
            plt.draw()
            #plt.pause(0.001)
            plt.show()
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

    # now print the paramerers
    if not use_shooting:
        if args.use_neural_ode:
            print_all_parameters(func)
        else:
            print_all_parameters(simple_resnet)
            if args.use_double_resnet:
                collect_and_sort_parameter_values_across_layers(simple_resnet)
    else:
        print_all_parameters(sblock)


    if use_shooting:
        custom_hook_data = {'parameters': []}
        hook = sblock.shooting_integrand.register_lagrangian_gradient_hook(parameters_hook)
        sblock.shooting_integrand.set_custom_hook_data(data=custom_hook_data)

        batch_in, batch_out = get_sample_batch(nr_of_samples=args.batch_size)
        pred_y, _, _, _ = sblock(x=batch_in)
        hook.remove()

        times = np.asarray([t.numpy() for (t,d) in custom_hook_data["parameters"]])
        data = np.asarray([d.detach().numpy() for (t,d) in custom_hook_data["parameters"]])
        plt.figure()
        for i in range(np.shape(data)[1]):
            plt.plot(times, data[:,i,:])
        plt.show()