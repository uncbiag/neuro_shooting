# the goal of this script is to use a simple affine model and check if the gradients are computed correctly


import random
seed = 1234
print('Setting the random seed to {:}'.format(seed))
random.seed(seed)
import torch
torch.manual_seed(seed)

import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.shooting_models as shooting_models
import neuro_shooting.generic_integrator as generic_integrator
import neuro_shooting.utils as utils

def zero_grads(pars):
    r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
    for p in pars:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


# particle setup
nonlinearity = 'tanh'
nr_of_particles = 8
parameter_weight = 1.0

# create a simple integrator
stepsize = 0.1
integrator_options = {'step_size': stepsize}
in_features_size = 3

#check_models = ['updown']
#check_models = ['DEBUG']
#check_models = ['simple']
#check_models = ['universal']

check_models = ['updown_universal', 'updown','DEBUG','simple','general']

number_of_tests_passed = 0
number_of_tests_attempted = 0
tolerance = 5e-3


# set the default
gpu = 1
utils.setup_device(desired_gpu=gpu)

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
    elif current_model == 'updown_universal':
        shooting_model = shooting_models.AutoShootingIntegrandModelUpDownUniversal(in_features=in_features_size,
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


    elif current_model=="general":
        shooting_model = shooting_models.AutoShootingIntegrandModelGeneralUpDown(in_features=in_features_size, nonlinearity=nonlinearity,
                                                                          nr_of_particles=nr_of_particles,particle_dimension = 1,
                                                                          particle_size = in_features_size,
                                                                          parameter_weight=parameter_weight,
                                                                          inflation_factor=5)




    else:
        raise ValueError('Unknown model to check: {}'.format( current_model ))

    shooting_block = shooting_blocks.ShootingBlockBase(name='test', shooting_integrand=shooting_model)

    print('\n\nChecking model: {}'.format(current_model))
    print('-------------------------------------\n')

    # create some sample data
    sample_data = torch.randn([100,1,in_features_size])
    sample_data_init = torch.randn([100,1,in_features_size])

    autodiff_gradient_results = dict()
    analytic_gradient_results = dict()

    for use_analytic_solution in (True,False):
        shooting_model.use_analytic_solution = use_analytic_solution

        # run through the shooting block once to get the necessary parameters
        shooting_block(x=sample_data_init)

        # zero gradients
        zero_grads(shooting_block.parameters())

        # now run the data through
        pred,_,_,_ = shooting_block(x=sample_data)

        # create a loss
        loss = torch.mean(pred**2) + shooting_block.get_norm_penalty()

        # compute gradient
        loss.backward()

        # now output the gradients with respect to the parameters
        pars = shooting_block.named_parameters()
        for par_name,par_value in pars:
            print('Using analytic solution: {}'.format( use_analytic_solution ))
            print('Name: {}'.format(par_name))
            print('Gradient: {}'.format(par_value.grad))

            if use_analytic_solution:
                analytic_gradient_results[par_name] = par_value.grad.detach().clone()
            else:
                autodiff_gradient_results[par_name] = par_value.grad.detach().clone()

    # now compare them
    for par_name in autodiff_gradient_results:
        autodiff_gradient = autodiff_gradient_results[par_name]
        analytic_gradient = analytic_gradient_results[par_name]
        print('Gradient ratio autodiff/analytic for parameter {}:'.format(par_name))
        print('{}'.format(autodiff_gradient/analytic_gradient))

        number_of_tests_attempted +=1

        nz_autodiff = autodiff_gradient != 0
        nz_analytic = analytic_gradient != 0

        if torch.all(nz_analytic == nz_autodiff).item():
            rel_error = torch.max(torch.abs(autodiff_gradient[nz_autodiff] / analytic_gradient[nz_analytic] - 1))
            if rel_error < tolerance:
                number_of_tests_passed += 1
                print('\nPASSED with relative error of {}\n'.format(rel_error))
            else:
                print('\n-------------------------------------')
                print('FAILED with relative error of {}'.format(rel_error))
                print('-------------------------------------\n')
        else:
            print('\n-------------------------------------')
            print('FAILED due to different ZERO pattern')
            print('-------------------------------------\n')

print('\nOverall summary:')
print('-----------------\n')
print('Passed {}/{} tests.'.format(number_of_tests_passed,number_of_tests_attempted))
print('Failed {}/{} tests.'.format(number_of_tests_attempted-number_of_tests_passed,number_of_tests_attempted))

if number_of_tests_passed==number_of_tests_attempted:
    print('\nCongratulations, all tests passes at a tolerance level of {}'.format(tolerance))

