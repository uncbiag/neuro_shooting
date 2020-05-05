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
sys.path.append('../../')

from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from neuro_shooting.shooting_models import AutoShootingIntegrandModelUpDown as UpDown
from neuro_shooting.shooting_models import AutoShootingIntegrandModelSimple as Simple
from neuro_shooting.shooting_models import AutoShootingIntegrandModelSecondOrder as SecOrder
from neuro_shooting.shooting_blocks import ShootingBlockBase as Base
import neuro_shooting.parameter_initialization as pi

from utils import ConcentricSphere, dataset_to_numpy, sample


class Model(nn.Module):
    def __init__(self, **model_kwargs):
        super(Model, self).__init__()
        
        self.int = UpDown(**model_kwargs, use_analytic_solution=True)
        self.blk = Base('Model', 
                        shooting_integrand=self.int,
                        integrator_name = 'rk4',
                        integrator_options = {'step_size': 0.05})
                        #integrator_name = 'dopri5',
                        #integrator_options = {'max_num_steps': 1000})
        
        self.cls = nn.Sequential(
            nn.BatchNorm1d(2), 
            nn.Linear(2, 1, bias=False))
        
    def trajectory(self, x, N=10):
        t = torch.linspace(0., 1., N)
        self.blk.set_integration_time_vector(t, suppress_warning=True)
        out = self.blk(x)
        self.blk.set_integration_time(1.)   
        return out
            
    def forward(self, x):
        z,_,_,_ = self.blk(x)
        x = self.cls(z.view(z.size(0),-1))
        return z, x


def setup_cmdline_parsing():
    parser = argparse.ArgumentParser('Concentric circles')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--nepochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--parameter_weight', type=float, default=5.0, help='parameter weight')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay for classifier')
    parser.add_argument('--nr_of_particles', type=int, default=10, help='Number of particles')
    parser.add_argument('--inflation_factor', type=int, default=5, help='Multiplier for state dimension')
    parser.add_argument('--norm_penalty_weight', type=float, default=0.0, help='Norm penalty weight')
    parser.add_argument('--save_model', type=str, default=None, help='Save model to (.pt) file')
    parser.add_argument('--verbose', action='store_true', default=False)
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
        loss = F.binary_cross_entropy_with_logits(output, target) + \
            args.norm_penalty_weight * model.blk.get_norm_penalty()
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
        custom_hook_data['q1'].append(state_dicts['Model']['q1'].clone())
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
    
    seed = args.seed
    print('Setting the random seed to {:}'.format(seed))
    random.seed(seed)
    torch.manual_seed(seed)

    train_data, test_data = generate_data()
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    device = 'cpu'
    tracker = []

    model_kwargs = {
        'in_features': 2,
        'nonlinearity': 'relu',
        'nr_of_particles': args.nr_of_particles,
        'parameter_weight': args.parameter_weight,
        'inflation_factor': args.inflation_factor,
        'optimize_over_data_initial_conditions': True,
        'costate_initializer' : pi.VectorEvolutionParameterInitializer(
            random_initialization_magnitude=0.5)}
    model = Model(**model_kwargs)
    B = next(iter(train_loader))
    model(B[0].unsqueeze(1))

    optimizer = torch.optim.Adam([
        {'params': model.blk.parameters()},
        {'params': model.cls.parameters(), 'weight_decay': args.wd}],
        lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.nepochs)

    for epoch in range(args.nepochs):
        #freeze_parameters(model.blk, ['q1'])
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
        scheduler.step()
        
        print('{:4d} | train-loss {:.4f} | test-loss {:.4f} | correct: {:.2f} [%]'.format(
            epoch, train_loss, test_loss, correct))

    if args.save_model is not None:
        torch.save(model.cpu(), args.save_model + ".pt")
    
        import pickle
        with open(args.save_model + ".pkl", 'wb') as f: 
            pickle.dump((args, tracker), f)
