"""Rotated MNIST experiment."""

import os
import sys
import pickle
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from collections import defaultdict
from torchvision.utils import make_grid

import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append('../../')
import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.shooting_models as shooting_models
import neuro_shooting.generic_integrator as generic_integrator
import neuro_shooting.shooting_hooks as sh
import neuro_shooting.vector_visualization as vector_visualization
import neuro_shooting.validation_measures as validation_measures
import neuro_shooting.parameter_initialization as pi
import neuro_shooting.utils as utils

from utils import load_data, Dataset
from autoencoder import ShootingAE, ShootingAEMasked


def setup_cmdline_parsing():
    parser = argparse.ArgumentParser('Rotated MNIST')
    parser.add_argument('--method', 
        type=str, 
        choices=['dopri5', 'adams','rk4'], 
        default='rk4', 
        help='Selects the desired integrator')
    parser.add_argument('--n_skip', 
        type=int, 
        default=4, 
        help='How many time points to randomly exclude during training.')   
    parser.add_argument('--i_eval', 
        type=int, 
        default=4, 
        help='Timepoint index (16 total) at which to evaluate.')
    parser.add_argument('--shooting_dim', 
        type=int, 
        default=20, 
        help='Dimensionality of latent space where to learn dynamics.')
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
        type=str, choices=['direct','linear','mini_nn'], 
        default='linear', 
        help='Different ways to predict the initial conditions for higher order models. Currently only supported for updown model.')
    parser.add_argument('--verbose', 
        action='store_true', 
        default=False, 
        help='Enable verbose output')
    parser.add_argument('--save', 
        action='store_true', 
        default=False, 
        help='Save model and tracking results to runs/.')
    parser.add_argument('--gpu', 
        type=int, 
        default=0, 
        help='Enable GPU computation on specified GPU.')
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


def setup_optimizer_and_scheduler(args, params):

    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.niters)
    return optimizer, scheduler


def evaluate(args, model, loader, device):
    model.eval()
    errors = []
    ntotal = 0
    for inp in loader:
        inp = inp.to(device)    
        _,_,out = model(inp[:,0,...], use_mask=False)
        R = out.cpu().detach().squeeze().numpy()
        V = inp.cpu().detach().squeeze().numpy()
        for i in range(inp.size(0)):
            error = np.mean(np.power(R[i,args.i_eval,:,:]-V[i,args.i_eval,:,:], 2.0))
            errors.append(error)
            ntotal += 1
    return errors, ntotal


def plot_reconstructions(npbatch, indices, where, prefix):
    for idx in indices:
        img_list = [torch.tensor(npbatch[idx,i,:,:]).unsqueeze(0) for i in range(16)]
        imgsave(
            make_grid(img_list, padding=0, nrow=16, normalize=True), 
            where,
            '{}{}.png'.format(prefix, idx))
    

def imgsave(img, where, fname):
    npimg = img.cpu().numpy()
    plt.figure(figsize=(15,3))
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.savefig(os.path.join(where, fname), bbox_inches='tight', pad_inches=0)
    plt.close('all')


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
    
    Xtr, Xtest, N, T = load_data()
    trainset = Dataset(Xtr)
    trainset = data.DataLoader(trainset, batch_size=25, shuffle=True)
    testset  = Dataset(Xtest)
    testset  = data.DataLoader(testset, batch_size=25, shuffle=False)

    # setup model
    model = ShootingAEMasked(
        n_filt=8, 
        shooting_dim=args.shooting_dim,
        n_skip=args.n_skip,
        i_eval=args.i_eval,
        T=T)

    model.set_shooting_block(shooting_block)
    model.to(device)

    # setup integration time vector
    t = 0.1*torch.arange(T, dtype=torch.float)
    model.set_integration_time_vector(t)

    # setup optimizer/scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(
        args, 
        model.parameters())

    tracker = []
    losses = defaultdict(list) 
    for it in range(args.niters):

        it_sim_loss = 0
        it_npe_loss = 0 
        it_loss = 0

        for inp in trainset:
            optimizer.zero_grad()
            
            """
            Input is a batch of images; we take the FIRST one and run it through the model. This generates, for each image, a trajectory for (T-n_skip-1) timepoints, where n_skip denotes the number of left-out timepoints and the -1 is the timepoint at which we want to evaluate later.
            """
            inp = inp.to(device)
            _, idx, out = model(inp[:,0,:,:], use_mask=True)
            
            """
            Next, we also need to make sure that the loss is only computed from the non-masked timepoints.
            """
            idx = idx[:,:,0].view(
                inp.size(0),T-(args.n_skip+1),1).expand(-1,-1,28*28)            
            inp = inp.view(inp.size(0),inp.size(1),28*28)
            inp = inp.gather(1, idx)
            
            """
            Loss computation: similarity loss + norm penalty for shooting
            """
            diff = inp - out.view(out.size(0),out.size(1),28*28)
            sim_loss = args.sim_weight  * torch.sum(torch.pow(diff,2.0))/float(idx.numel())
            npe_loss = args.norm_weight * model.blk.get_norm_penalty() 
            loss = sim_loss + npe_loss
            loss.backward()
            optimizer.step()

            it_sim_loss += sim_loss.item()
            it_npe_loss += npe_loss.item()
            it_loss += loss.item()
        
            losses['sim_loss'].append(sim_loss.item())
            losses['npe_loss'].append(npe_loss.item())
        
        scheduler.step()

        eval_errors, eval_total = evaluate(args, model, testset, device)
        
        print('Epoch={:04d} | Loss={:.4f} | Sim={:.8f} | NPE={:.8f} | Test={:.4f}'.format(
            it, 
            it_loss/len(trainset), 
            it_sim_loss/len(trainset), 
            it_npe_loss/len(trainset),
            np.sum(eval_errors)/eval_total))

        tracker.append({
          'it': it,
          'it_loss': it_loss/len(trainset),
          'it_sim_loss': it_sim_loss/len(trainset), 
          'it_npe_loss': it_npe_loss/len(trainset),
          'all_eval_error': np.sum(eval_errors)/eval_total,
          'img_eval_errors': eval_errors
        })

    model.eval()
    Rs = []
    Vs = []
    for inp in testset:
        inp = inp.to(device)    
        _,_,out = model(inp[:,0,...], use_mask=False)
        R = out.cpu().detach().squeeze().numpy()
        V = inp.cpu().detach().squeeze().numpy()
        Rs.append(R)
        Vs.append(V)

    #plot_reconstructions(R, list(range(inp.size(0))), png_dirname, 'pred_')
    #plot_reconstructions(V, list(range(inp.size(0))), png_dirname, 'true_')

    if args.save:
        import uuid
        save_dirname = os.path.join('runs',str(uuid.uuid4())) 
        os.makedirs(save_dirname, exist_ok=False)   

        torch.save(model, os.path.join(save_dirname, 'model.pt'))
        with open(os.path.join(save_dirname, 'tracker.pkl'), 'wb') as fid:
            pickle.dump(tracker, fid)
        with open(os.path.join(save_dirname, 'args.pkl'), 'wb') as fid:
            pickle.dump(args, fid)
        with open(os.path.join(save_dirname, 'images.pkl'), 'wb') as fid:
            pickle.dump({'Rs': Rs, 'Vs': Vs}, fid)
            
        # png_dirname = os.path.join(save_dirname, 'png')
        # os.makedirs(png_dirname, exist_ok=True)
        
       