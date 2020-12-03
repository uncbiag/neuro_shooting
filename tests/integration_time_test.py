# the goal of this script is to use determine if integration with multiple time-steps works as intended

import numpy as np
import random
seed = 1234
print('Setting the random seed to {:}'.format(seed))
random.seed(seed)
np.random.seed(seed)
import torch
torch.manual_seed(seed)


import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.shooting_models as shooting_models
import neuro_shooting.generic_integrator as generic_integrator


def zero_grads(pars):
    r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
    for p in pars:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


# particle setup
nonlinearity = 'tanh'
nr_of_particles = 10
parameter_weight = 0.1

# create a simple integrator
stepsize = 0.1
integrator_options = {'step_size': stepsize}
in_features_size = 2

#check_models = ['updown']
#check_models = ['DEBUG']
check_models = ['simple']
#check_models = ['universal']

#check_models = ['updown','DEBUG','simple',"universal"]

number_of_tests_passed = 0
number_of_tests_attempted = 0
tolerance = 5e-3

integrator = generic_integrator.GenericIntegrator(integrator_library = 'odeint', integrator_name = 'rk4',
                                                          use_adjoint_integration=False,
                                                          integrator_options=integrator_options)

for current_model in check_models:

    if current_model=='simple':
        shooting_model = shooting_models.AutoShootingIntegrandModelSimple(in_features=in_features_size,
                                                                          nonlinearity=nonlinearity,
                                                                          nr_of_particles=nr_of_particles,
                                                                          particle_dimension=1,
                                                                          particle_size=in_features_size,
                                                                          parameter_weight=parameter_weight)
    elif current_model == 'universal':
        shooting_model = shooting_models.AutoShootingIntegrandModelUniversal(in_features=in_features_size,
                                                                          nonlinearity=nonlinearity,
                                                                          nr_of_particles=nr_of_particles,
                                                                          particle_dimension=1,
                                                                          particle_size=in_features_size,
                                                                          parameter_weight=parameter_weight,
                                                                            inflation_factor=5)

    elif current_model=='updown':
        shooting_model = shooting_models.AutoShootingIntegrandModelUpDown(in_features=in_features_size, nonlinearity=nonlinearity,
                                                                          nr_of_particles=nr_of_particles,particle_dimension = 1,
                                                                          particle_size = in_features_size,
                                                                          parameter_weight=parameter_weight,
                                                                          inflation_factor=5)

    elif current_model == 'DEBUG':
        shooting_model = shooting_models.DEBUGAutoShootingIntegrandModelSimple(in_features=in_features_size,
                                                                          nonlinearity=nonlinearity,
                                                                          nr_of_particles=nr_of_particles,
                                                                          particle_dimension=1,
                                                                          particle_size=in_features_size,
                                                                          parameter_weight=parameter_weight)
    else:
        raise ValueError('Unknown model to check: {}'.format( current_model ))

    use_analytic_solution = True

    shooting_block = shooting_blocks.ShootingBlockBase(name='test', shooting_integrand=shooting_model, integrator_options=integrator_options)
    shooting_model.use_analytic_solution = use_analytic_solution

    print('\n\nChecking model: {}'.format(current_model))
    print('-------------------------------------\n')

    # create some sample data
    sample_data = torch.randn([1,1,in_features_size])

    # run through the shooting block once to get the necessary parameters
    shooting_block(x=sample_data)

    # create overall time-vector, for simplicity make sure it corresponds to the step-size
    t_np = np.array(range(0,16))*stepsize
    t = torch.from_numpy(t_np)

    # first let's try to integrate this all at once
    shooting_block.set_integration_time_vector(integration_time_vector=t, suppress_warning=True)
    pred,_,_,_ = shooting_block(x=sample_data)

    # now integrate it step by step
    pred_step_by_step = torch.zeros_like(pred)
    pred_step_by_step[0,...] = sample_data

    for ind,ct in enumerate(t[1:]):
        shooting_block.set_integration_time(ct)
        cpred, _, _, _ = shooting_block(x=sample_data)
        pred_step_by_step[ind+1,...] = cpred

    print('Pred = {}\n'.format(pred[:,0,0,:]))

    print('Pred_step_by_step = {}\n'.format(pred_step_by_step[:,0,0,:]))

    print('Pred-pred_step_by_step = {}\n'.format(pred[:,0,0,:]-pred_step_by_step[:,0,0,:]))

    print('diff(pred) = {}\n'.format(pred[1:,0,0,:]-pred[:-1,0,0,:]))

    print('diff(pred_step_by_step) = {}\n'.format(pred_step_by_step[1:,0,0,:]-pred_step_by_step[:-1,0,0,:]))
