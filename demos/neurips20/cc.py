import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath

import sys
import random
import argparse
import numpy as np
from collections import defaultdict

from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.append('../../')
from neuro_shooting.shooting_models import AutoShootingIntegrandModelUpDown as UpDown
from neuro_shooting.shooting_models import AutoShootingIntegrandModelUpDownUniversal as Universal
from neuro_shooting.shooting_models import AutoShootingIntegrandModelSimple as Simple
from neuro_shooting.shooting_models import AutoShootingIntegrandModelSecondOrder as SecOrder
from neuro_shooting.shooting_blocks import ShootingBlockBase as Base
import neuro_shooting.parameter_initialization as pi
import neuro_shooting.utils as utils

import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.shooting_models as shooting_models
import neuro_shooting.generic_integrator as generic_integrator

from utils import ConcentricSphere, dataset_to_numpy, sample


class Model(nn.Module):
    def __init__(self, **model_kwargs):
        super(Model, self).__init__()
        
        self.blk = None
        self.cls = nn.Sequential(
            nn.BatchNorm1d(2), 
            nn.Linear(2, 1, bias=False))
        
    def trajectory(self, x, N=10):
        t = torch.linspace(0., 1., N)
        self.blk.set_integration_time_vector(t, suppress_warning=True)
        out = self.blk(x)
        self.blk.set_integration_time(1.)   
        return out

    def set_shooting_block(self, sblock):
        self.blk = sblock

    def forward(self, x):
        z,_,_,_ = self.blk(x)
        x = self.cls(z.view(z.size(0),-1))
        return z, x


def setup_random_seed(seed):
    if seed==-1:
        print('No seed was specified, leaving everthing at random. Use --seed to specify a seed if you want repeatable results.')
    else:
        print('Setting the random seed to {:}'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def setup_shooting_block(integrator=None, 
    in_features=20,
    shooting_model='updown', 
    parameter_weight=1.0, 
    nr_of_particles=10,
    inflation_factor=2, 
    nonlinearity='relu',
    use_particle_rnn_mode=False, 
    use_particle_free_rnn_mode=False,
    optimize_over_data_initial_conditions=False,
    optimize_over_data_initial_conditions_type='linear'):

    if shooting_model=='updown':
        smodel = shooting_models.AutoShootingIntegrandModelUpDown(
            in_features=in_features, 
            nonlinearity=nonlinearity,
            parameter_weight=parameter_weight,
            inflation_factor=inflation_factor,
            nr_of_particles=nr_of_particles, 
            particle_dimension=1,
            particle_size=in_features,
            use_analytic_solution=True,
            use_rnn_mode=use_particle_rnn_mode,
            optimize_over_data_initial_conditions=optimize_over_data_initial_conditions,
            optimize_over_data_initial_conditions_type=optimize_over_data_initial_conditions_type)
    elif shooting_model=='updown_universal':
        smodel = shooting_models.AutoShootingIntegrandModelUpDownUniversal(
            in_features=in_features, 
            nonlinearity=nonlinearity,
            parameter_weight=parameter_weight,
            inflation_factor=inflation_factor,
            nr_of_particles=nr_of_particles, 
            particle_dimension=1,
            particle_size=in_features,
            use_analytic_solution=True,
            optional_weight=0.1,
            use_rnn_mode=use_particle_rnn_mode,
            optimize_over_data_initial_conditions=optimize_over_data_initial_conditions,
            optimize_over_data_initial_conditions_type=optimize_over_data_initial_conditions_type)
    elif shooting_model=='periodic':
        smodel = shooting_models.AutoShootingIntegrandModelUpdownPeriodic(
            in_features=in_features, 
            nonlinearity=nonlinearity,
            parameter_weight=parameter_weight,
            inflation_factor=inflation_factor,
            nr_of_particles=nr_of_particles, particle_dimension=1,
            particle_size=in_features,
            use_analytic_solution=True,
            use_rnn_mode=use_particle_rnn_mode,
            optimize_over_data_initial_conditions=optimize_over_data_initial_conditions,
            optimize_over_data_initial_conditions_type=optimize_over_data_initial_conditions_type)
    elif shooting_model=='simple':
        smodel = shooting_models.AutoShootingIntegrandModelSimple(
            in_features=in_features, 
            nonlinearity=nonlinearity,
            parameter_weight=parameter_weight,
            nr_of_particles=nr_of_particles, 
            particle_dimension=1,
            particle_size=in_features,
            use_analytic_solution=True,
            use_rnn_mode=use_particle_rnn_mode)

    print('Using shooting model {}'.format(shooting_model))

    par_initializer = pi.VectorEvolutionParameterInitializer(
        only_random_initialization=True, 
        random_initialization_magnitude=0.5)
    smodel.set_state_initializer(state_initializer=par_initializer)

    shooting_block = shooting_blocks.ShootingBlockBase(
        name='simple', 
        shooting_integrand=smodel,
        use_particle_free_rnn_mode=use_particle_free_rnn_mode, 
        integrator=integrator)
    
    return shooting_block


def setup_integrator(method, use_adjoint, step_size, rtol=1e-3, atol=1e-5):
    integrator_options = dict()

    if method not in ['dopri5', 'adams']:
        integrator_options  = {'step_size': step_size}
    
    integrator = generic_integrator.GenericIntegrator(
            integrator_library = 'odeint', 
            integrator_name = method,
            use_adjoint_integration=use_adjoint, 
            integrator_options=integrator_options,
            rtol=rtol, atol=atol)

    return integrator


def setup_cmdline_parsing():
    parser = argparse.ArgumentParser('Rotated MNIST')
    parser.add_argument('--method', 
        type=str, 
        choices=['dopri5', 'adams','rk4'], 
        default='rk4', 
        help='Selects the desired integrator')
    parser.add_argument('--shooting_dim', 
        type=int, 
        default=20, 
        help='Dimensionality of latent space where to learn dynamics.')
    parser.add_argument('--cls_weight_decay', 
        type=float, 
        default=1e-4, 
        help='Classifier weight decay.')
    parser.add_argument('--evaluate_ahead', 
        type=int, 
        default=10, 
        help='Evaluate evaluate_ahead timesteps ahead.')
    parser.add_argument('--stepsize', 
        type=float, 
        default=0.05, 
        help='Step size for the integrator (if not adaptive).')
    parser.add_argument('--niters', 
        type=int, 
        default=500, 
        help='Maximum nunber of iterations.')
    parser.add_argument('--lr', 
        type=float, 
        default=1e-2, 
        help='Learning rate.')
    parser.add_argument('--seed', 
        required=False, 
        type=int, 
        default=-1,
        help='Sets the random seed which affects data shuffling.')
    parser.add_argument('--batch_size', 
        type=int, 
        default=25,
        help='Sets the batch size.')
    parser.add_argument('--shooting_model', 
        type=str, 
        default='updown', 
        choices=['simple', 'updown', 'updown_universal'])
    parser.add_argument('--nr_of_particles', 
        type=int, 
        default=25, 
        help='Number of particles to parameterize the initial condition')
    parser.add_argument('--pw', 
        type=float, 
        default=1.0, 
        help='Parameter weight')
    parser.add_argument('--sim_weight', 
        type=float, 
        default=100.0, 
        help='Weight for the similarity measure')
    parser.add_argument('--norm_weight', 
        type=float, 
        default=0.01,
        help='Weight for the similarity measure')
    parser.add_argument('--nonlinearity',
        type=str, choices=['identity', 'relu', 'tanh', 'sigmoid'], 
        default='relu', 
        help='Nonlinearity for shooting.')
    parser.add_argument('--inflation_factor', 
        type=int, 
        default=5,
        help='Multiplier for state dimension for updown shooting model types')
    parser.add_argument('--use_particle_rnn_mode', 
        action='store_true',
        help='When set then parameters are only computed at the initial time and used for the entire evolution; mimicks a particle-based RNN model.')
    parser.add_argument('--use_particle_free_rnn_mode', 
        action='store_true',
        help='This is directly optimizing over the parameters -- no particles here; a la Neural ODE')
    parser.add_argument('--optimize_over_data_initial_conditions', 
        action='store_true', 
        default=False)
    parser.add_argument('--optimize_over_data_initial_conditions_type', 
        type=str, 
        choices=['direct','linear','mini_nn'], 
        default='linear', 
        help='Different ways to predict the initial conditions for higher order models. Currently only supported for updown model.')
    parser.add_argument('--verbose', 
        action='store_true', 
        default=False, 
        help='Enable verbose output')
    parser.add_argument('--save_model', 
        type=str, 
        default='model', 
        help='Save model.')
    parser.add_argument('--gpu', 
        type=int, 
        default=0, 
        help='Enable GPU computation on specified GPU.')
    args = parser.parse_args()
    return args


def generate_data():
    data_dim = 2
    train_data = ConcentricSphere(data_dim, 
                                inner_range=(0., .5), 
                                outer_range=(1., 1.5), 
                                num_points_inner=1000, 
                                num_points_outer=2000)

    test_data = ConcentricSphere(data_dim, 
                                inner_range=(0., .5), 
                                outer_range=(1., 1.5), 
                                num_points_inner=1000, 
                                num_points_outer=2000)

    return train_data, test_data



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        _, output = model(data.unsqueeze(1))            
        loss0 = args.sim_weight * F.binary_cross_entropy_with_logits(output, target)
        #loss1 = args.norm_weight* model.blk.get_norm_penalty()
        loss = loss0 #+ loss1 
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    return epoch_loss / len(train_loader)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, output = model(data.unsqueeze(1))
            test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()
            pred = torch.sigmoid(output).round()
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    return test_loss/len(test_loader.dataset), 100. * correct / len(test_loader.dataset)


def linear_transform_hook(module, t, state_dicts, costate_dicts, data_dict_of_dicts,
            dot_state_dicts, dot_costate_dicts, dot_data_dict_of_dicts, parameter_objects, custom_hook_data):

    with torch.no_grad():
        custom_hook_data['t'].append(t.item())
        custom_hook_data['q1'].append(state_dicts['simple']['q1'].clone())
        custom_hook_data['l1_weight'].append(parameter_objects['l1']._parameter_dict['weight'].clone())
        custom_hook_data['l1_bias'].append(parameter_objects['l1']._parameter_dict['bias'].clone())
        custom_hook_data['l2_weight'].append(parameter_objects['l2']._parameter_dict['weight'].clone())
        custom_hook_data['l2_bias'].append(parameter_objects['l2']._parameter_dict['bias'].clone())


def hook_in(model, device, loader):
    model.eval()
    custom_hook_data = defaultdict(list)
    hook = model.blk.shooting_integrand.register_lagrangian_gradient_hook(linear_transform_hook)
    model.blk.shooting_integrand.set_custom_hook_data(data=custom_hook_data)
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            _, output = model(data.unsqueeze(1))  
            break
    hook.remove()
    return custom_hook_data


def freeze_parameters(shooting_block, parameters_to_freeze):
    pars = shooting_block.state_dict()
    for pn in parameters_to_freeze:
        print('Freezing {}'.format(pn))
        pars[pn].requires_grad = False


if __name__ == '__main__':

    args = setup_cmdline_parsing()
    if args.verbose:
        print(args)
    
    train_data, test_data = generate_data()
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    device = utils.setup_device(desired_gpu=args.gpu)

    # setup reproducibility (if desired)
    setup_random_seed(seed=args.seed)

    # setup integrator
    integrator = setup_integrator(
            method=args.method, 
            step_size=args.stepsize, 
            use_adjoint=False,
            rtol=1e-3,
            atol=1e-5)
    
    # setup shooting block
    shooting_block = setup_shooting_block(
        integrator=integrator,
        in_features=args.shooting_dim,
        shooting_model=args.shooting_model,
        parameter_weight=args.pw,
        nr_of_particles=args.nr_of_particles,
        inflation_factor=args.inflation_factor,
        nonlinearity=args.nonlinearity,
        use_particle_rnn_mode=args.use_particle_rnn_mode,
        use_particle_free_rnn_mode=args.use_particle_free_rnn_mode,
        optimize_over_data_initial_conditions=args.optimize_over_data_initial_conditions,
        optimize_over_data_initial_conditions_type=args.optimize_over_data_initial_conditions_type)

    model = Model()
    model.set_shooting_block(shooting_block)
    model = model.to(device)

    x,_ = next(iter(train_loader))
    x = x.to(device)
    model(x.unsqueeze(1))

    optimizer = torch.optim.Adam([
        {'params': model.blk.parameters()},
        {'params': model.cls.parameters(), 'weight_decay': args.cls_weight_decay}],
        lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.niters)

    tracker = []
    results = []
    for epoch in range(args.niters):
        train_loss = train(
            args,
            model, 
            device, 
            train_loader, 
            optimizer, 
            epoch)

        test_loss, correct = test(
            model, 
            device, 
            test_loader)
        
        tracker.append(hook_in(model, device, train_loader))
        results.append((train_loss,test_loss,correct))
        scheduler.step()
        
        print('{:4d} | train-loss {:.4f} | test-loss {:.4f} | correct: {:.2f} [%]'.format(
            epoch, train_loss, test_loss, correct))

    if args.save_model is not None:
        torch.save(model.cpu(), args.save_model + ".pt")
    
        import pickle
        with open(args.save_model + ".pkl", 'wb') as f: 
            pickle.dump((args, tracker), f)
        
        with open(args.save_model + "_results_.pkl", 'wb') as f: 
            pickle.dump((args, results), f)
