import os
import sys
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from collections import defaultdict


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from torch.utils.data import Dataset, DataLoader

sys.path.append('../../')
import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.shooting_models as shooting_models
import neuro_shooting.generic_integrator as generic_integrator
import neuro_shooting.shooting_hooks as sh
import neuro_shooting.vector_visualization as vector_visualization
import neuro_shooting.validation_measures as validation_measures
import neuro_shooting.parameter_initialization as pi
import neuro_shooting.utils as utils


def setup_cmdline_parsing():
    parser = argparse.ArgumentParser('CMU MOCAP (Single)')
    parser.add_argument('--method', 
        type=str, 
        choices=['dopri5', 'adams','rk4'], 
        default='rk4', 
        help='Selects the desired integrator')
    parser.add_argument('--stepsize', 
        type=float, 
        default=0.05, 
        help='Step size for the integrator (if not adaptive).')
    parser.add_argument('--subject_id', 
        type=int, 
        default=0, 
        help='Subject ID of MOCAP data.')
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
        help='Sets the random seed which affects data shuffling')
    parser.add_argument('--shooting_model', 
        type=str, 
        default='updown', 
        choices=['simple', 'updown', 'periodic'])
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
    parser.add_argument('--do_not_use_parameter_penalty_energy', 
        action='store_true', 
        default=False)
    parser.add_argument('--optimize_over_data_initial_conditions', 
        action='store_true', 
        default=False)
    parser.add_argument('--optimize_over_data_initial_conditions_type', 
        type=str, choices=['direct','linear','mini_nn'], 
        default='linear', 
        help='Different ways to predict the initial conditions for higher order models. Currently only supported for updown model.')
    parser.add_argument('--custom_parameter_freezing', 
        action='store_true', 
        default=False, 
        help='Enable custom code for parameter freezing -- development mode')
    parser.add_argument('--custom_parameter_initialization', 
        action='store_true', 
        default=False, 
        help='Enable custom code for parameter initialization -- development mode')
    parser.add_argument('--verbose', 
        action='store_true', 
        default=False, 
        help='Enable verbose output')
    parser.add_argument('--gpu', 
        type=int, 
        default=0, 
        help='Enable GPU computation on specified GPU.')
    parser.add_argument('--save_prefix', 
        type=str, 
        default="subject_",
        help='Prefix to saved files.')
    args = parser.parse_args()
    return args


def setup_random_seed(seed):
    if seed==-1:
        print('No seed was specified, leaving everthing at random. Use --seed to specify a seed if you want repeatable results.')
    else:
        print('Setting the random seed to {:}'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

# taken from ODE^2 VAE repo
class MyBatch:
    def __init__(self,x,y=None):
        self.x = x # high-dim data
        self.y = y # time points
        self.N = x.shape[0]
        
    def next_batch(self,N=None): # draw N samples from the first M samples
        if N is None or N>self.N:
            ids = np.arange(self.N)
        else:
            ids = ss.uniform.rvs(size=N)*self.N
        
        ids = [int(i) for i in ids]
        xs = self.x[ids,:]
        if self.y is None:
            ys = None
        else:
            ys = self.y[ids,:]
        return xs, ys

# taken from ODE^2 VAE repo
class MyDataset:
    def __init__(self,xtr,ytr,xval=None,yval=None,xtest=None,ytest=None):
        self.train = MyBatch(xtr, ytr)
        if xval is not None:
            self.val = MyBatch(xval, yval)
        if xtest is not None:
            self.test = MyBatch(xtest, ytest)


def load_mocap_data_single_walk(data_dir ,subject_id, dt=0.1, plot=False):
    fname = os.path.join(data_dir, 'mocap43.mat')
    mocap_data = loadmat(fname)
    
    Xtest = mocap_data['Ys'][subject_id][0]
    Xtest = np.expand_dims(Xtest,0)
    Ytest = dt*np.arange(0,Xtest.shape[1],dtype=np.float32)
    Ytest = np.tile(Ytest,[Xtest.shape[0],1])
    Xval  = mocap_data['Yobss'][subject_id][0]
    Xval  = np.expand_dims(Xval,0)
    Yval  = dt*np.arange(0,Xval.shape[1],dtype=np.float32)
    Yval  = np.tile(Yval,[Xval.shape[0],1])
    N = Xval.shape[1]
    Xtr   = Xval[:,:4*N//5,:]
    Ytr   = dt*np.arange(0,Xtr.shape[1],dtype=np.float32)
    Ytr   = np.tile(Ytr,[Xtr.shape[0],1])

    dataset = MyDataset(Xtr,Ytr,Xval,Yval,Xtest,Ytest)
    
    if plot:
        x,y = dataset.train.next_batch()
        plt.figure(2,(10,20))
        for i in range(50):
            plt.subplot(10,5,i+1)
            plt.title('sensor-{:d}'.format(i+1))
            plt.plot(x[0,:,i])
            plt.tight_layout()
    return dataset


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


def setup_shooting_block(integrator=None, shooting_model='updown', parameter_weight=1.0, nr_of_particles=10,
                         inflation_factor=2, nonlinearity='relu',
                         use_particle_rnn_mode=False, use_particle_free_rnn_mode=False,
                         optimize_over_data_initial_conditions=False,
                         optimize_over_data_initial_conditions_type='linear'):

    if shooting_model=='updown':
        smodel = shooting_models.AutoShootingIntegrandModelUpDown(
            in_features=50, 
            nonlinearity=nonlinearity,
            parameter_weight=parameter_weight,
            inflation_factor=inflation_factor,
            nr_of_particles=nr_of_particles, 
            particle_dimension=1,
            particle_size=50,
            use_analytic_solution=True,
            use_rnn_mode=use_particle_rnn_mode,
            optimize_over_data_initial_conditions=optimize_over_data_initial_conditions,
            optimize_over_data_initial_conditions_type=optimize_over_data_initial_conditions_type)
    elif shooting_model=='periodic':
        smodel = shooting_models.AutoShootingIntegrandModelUpdownPeriodic(
            in_features=2, 
            nonlinearity=nonlinearity,
            parameter_weight=parameter_weight,
            inflation_factor=inflation_factor,
            nr_of_particles=nr_of_particles, particle_dimension=1,
            particle_size=2,
            use_analytic_solution=True,
            use_rnn_mode=use_particle_rnn_mode,
            optimize_over_data_initial_conditions=optimize_over_data_initial_conditions,
            optimize_over_data_initial_conditions_type=optimize_over_data_initial_conditions_type)
    elif shooting_model=='simple':
        smodel = shooting_models.AutoShootingIntegrandModelSimple(
            in_features=2, nonlinearity=nonlinearity,
            parameter_weight=parameter_weight,
            nr_of_particles=nr_of_particles, particle_dimension=1,
            particle_size=2,
            use_analytic_solution=True,
            use_rnn_mode=use_particle_rnn_mode)

    print('Using shooting model {}'.format(shooting_model))

    par_initializer = pi.VectorEvolutionParameterInitializer(
        only_random_initialization=True, 
        random_initialization_magnitude=0.5)

    #smodel.set_state_initializer(state_initializer=par_initializer)
    shooting_block = shooting_blocks.ShootingBlockBase(
        name='simple', 
        shooting_integrand=smodel,
        use_particle_free_rnn_mode=use_particle_free_rnn_mode, 
        integrator=integrator)
    
    return shooting_block


def setup_optimizer_and_scheduler(args, params):

    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.niters)
    return optimizer, scheduler


def evaluate(data, shooting_block, D):
    shooting_block.eval()
    inp, t = data.next_batch()
    t = torch.tensor(t, dtype=torch.float).squeeze()
    inp = torch.tensor(inp, dtype=torch.float).to(device)
    shooting_block.set_integration_time_vector(t,suppress_warning=True)
    out,_,_,_ = shooting_block(inp[0,0,:].view(1,1,D))
    out_y = out.squeeze().cpu().detach().numpy()
    inp_y = inp.squeeze().cpu().detach().numpy()
    return out_y, inp_y, np.mean(np.power(out_y[:,:]-inp_y[:,:],2.0))


if __name__ == '__main__':

    # setup CMDLINE parsing
    args = setup_cmdline_parsing()

    if args.verbose:
        print(args)

    # setup device
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

    shooting_block = setup_shooting_block(
        integrator=integrator,
        shooting_model=args.shooting_model,
        parameter_weight=args.pw,
        nr_of_particles=args.nr_of_particles,
        inflation_factor=args.inflation_factor,
        nonlinearity=args.nonlinearity,
        use_particle_rnn_mode=args.use_particle_rnn_mode,
        use_particle_free_rnn_mode=args.use_particle_free_rnn_mode,
        optimize_over_data_initial_conditions=args.optimize_over_data_initial_conditions,
        optimize_over_data_initial_conditions_type=args.optimize_over_data_initial_conditions_type)

    # load data
    ds = load_mocap_data_single_walk(
        '../../data/', 
        args.subject_id ,
        0.1, 
        False)

    # get data properties
    [N,T,D] = ds.train.x.shape
    if args.verbose:
        print('{} data points, {} time steps, {} features'.format(
            N,T,D))

    shooting_block = shooting_block.to(device)

    x = torch.tensor(ds.train.x, dtype=torch.float).to(device)
    print(x.view(T,1,D).size())
    shooting_block(x.view(T,1,D))

    optimizer, scheduler = setup_optimizer_and_scheduler(
        args, 
        shooting_block.parameters())
    nr_of_pars = utils.compute_number_of_parameters(model=shooting_block)

    losses = defaultdict(list)

    for it in range(args.niters):
        optimizer.zero_grad()
    
        inp, t = ds.train.next_batch()
        t = torch.tensor(t, dtype=torch.float).squeeze()
        inp = torch.tensor(inp, dtype=torch.float).to(device)
        shooting_block.set_integration_time_vector(t, suppress_warning=True)
        out,_,_,_ = shooting_block(inp[0,0,:].view(1,1,D))

        loss0 = torch.mean(torch.abs(out.squeeze() - inp.squeeze())) 
        loss1 = shooting_block.get_norm_penalty()
        loss =  loss0 + 1.0*loss1
        loss.backward()
        optimizer.step()

        # validation 
        _,_,val_mse = evaluate(ds.val, shooting_block, D)
        
        losses['sim_loss'].append(loss0.item())
        losses['npe_loss'].append(loss1.item())
        losses['val_mse'].append(val_mse.item())

        print('Iter={:04d} | Loss={:.4f} | Sim={:.4f} | Norm-Penalty={:.4f} | Val={:.4f} '.format(
            it, 
            loss.item(),
            loss0.item(),
            loss1.item(),
            val_mse.item()))

    train_pred, train_true, train_mse = evaluate(ds.train, shooting_block, D)
    test_pred, test_true, test_mse    = evaluate(ds.test,  shooting_block, D)
    val_pred, val_true, val_mse       = evaluate(ds.val,   shooting_block, D)

    print('MSEs on subject {}: Train={:.4f} | Val={:.4f} | Test={:4f} '.format(
        int(args.subject_id), train_mse, val_mse, test_mse))
        
    import pickle
    res = {'subject_id': args.subject_id, 
           'train'    :  (train_pred, train_true, train_mse),
           'test'     :  (test_pred, test_true, test_mse),
           'val'      :  (val_pred, val_true, val_mse),
           'losses'   :  losses}

    fname = '{}{}.pkl'.format(args.save_prefix, args.subject_id)
    with open(fname, 'wb') as fid:
        pickle.dump(res, fid)