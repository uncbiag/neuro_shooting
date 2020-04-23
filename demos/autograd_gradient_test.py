# the goal of this script is to use a simple affine model and check if the gradients are computed correctly

import torch
import random

seed = 1234
print('Setting the random seed to {:}'.format(seed))
random.seed(seed)
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
nr_of_particles = 3
parameter_weight = 1.0

# create a simple integrator
stepsize = 0.1
integrator_options = {'step_size': stepsize}

integrator = generic_integrator.GenericIntegrator(integrator_library = 'odeint', integrator_name = 'rk4',
                                                  use_adjoint_integration=False,
                                                  integrator_options=integrator_options)

in_features_size = 5
shooting_model = shooting_models.AutoShootingIntegrandModelUpDown(in_features=in_features_size, nonlinearity=nonlinearity,
                                                                  nr_of_particles=nr_of_particles,particle_dimension = 1,particle_size = in_features_size,
                                                                  parameter_weight=parameter_weight)

shooting_block = shooting_blocks.ShootingBlockBase(name='updown', shooting_integrand=shooting_model)

# create some sample data
sample_data = torch.randn([20,1,in_features_size])

# run through the shooting block once to get the necessary parameters
shooting_block(x=sample_data)

autodiff_gradient_results = dict()
analytic_gradient_results = dict()

for use_analytic_solution in (True,False):
    shooting_model.use_analytic_solution = use_analytic_solution

    # zero gradients
    zero_grads(shooting_block.parameters())

    # now run the data through
    pred,_,_,_ = shooting_block(x=sample_data)

    # create a loss
    loss = torch.mean(pred) + shooting_block.get_norm_penalty()

    # compute gradient
    loss.backward()

    # now output the gradients with respect to the parameters
    pars = shooting_block.named_parameters()
    for par_name,par_value in pars:
        print('Using analytic solution: {}'.format( use_analytic_solution ))
        print('Name: {}'.format(par_name))
        print('Gradient: {}'.format(par_value.grad))

        if use_analytic_solution:
            analytic_gradient_results[par_name] = par_value.grad.clone().detach()
        else:
            autodiff_gradient_results[par_name] = par_value.grad.clone().detach()

# now compare them
for par_name in autodiff_gradient_results:
    autodiff_gradient = autodiff_gradient_results[par_name]
    analytic_gradient = analytic_gradient_results[par_name]
    print('Gradient ratio autodiff/analytic for parameter {}:'.format(par_name))
    print('{}'.format(autodiff_gradient/analytic_gradient))

print('Hello world')
