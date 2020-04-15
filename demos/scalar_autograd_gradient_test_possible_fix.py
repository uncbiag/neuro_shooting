import torch
import torch.autograd as autograd

import random

seed = 1234
print('Setting the random seed to {:}\n'.format(seed))
random.seed(seed)
torch.manual_seed(seed)

import neuro_shooting.activation_functions_and_derivatives as ad

# nonlinearities
nl, dnl = ad.get_nonlinearity(nonlinearity='tanh')

def zero_grad(p):
    r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
    if p.grad is not None:
        p.grad.detach_()
        p.grad.zero_()

def compute_lagrangian(q,p,theta):

    kinetic_energy = 0.5*theta*theta
    potential_energy = p*theta*nl(q)

    L = kinetic_energy - potential_energy

    return L, kinetic_energy, potential_energy

## Here I compute the reduced lagrangian in terms of q and p
## thus it is the composition of the lagrangian in terms of the parameters and a function
## which is here compute_parameter_autograd
def compute_reduced_lagrangian(q,p):
    theta = compute_parameter_autograd(q,p)
    kinetic_energy = 0.5 * theta * theta
    potential_energy = p * theta * nl(q)
    L = kinetic_energy - potential_energy
    return L, kinetic_energy, potential_energy

### hereafter, diffentiation of the reduced lagrangian wrt q
def rhs_costate_autograd_reduced(q,p):
    lagrangian, kinetic_energy, potential_energy = compute_reduced_lagrangian(q=q, p=p)
    rhs_costate = autograd.grad(lagrangian, (q),
                                      grad_outputs=lagrangian.data.new(lagrangian.shape).fill_(1),
                                      create_graph=True,
                                      retain_graph=True,
                                      allow_unused=True)
    return rhs_costate[0]


def rhs_state(q,theta):
    rhs = theta*nl(q)
    return rhs

def rhs_costate(q,p,theta):
    rhs = -dnl(q)*theta*p
    return rhs

def compute_parameter(q,p):
    theta = p*nl(q)
    return theta

def rhs_shooting_analytic(q,p,if_reduced = True):
    theta = compute_parameter(q=q,p=p)
    rhs = dict()
    rhs['q'] = rhs_state(q=q,theta=theta)
    rhs['p'] = rhs_costate(q=q,p=p,theta=theta)
    return rhs

def rhs_costate_autograd(q,p,theta):

    # Change from previous version: I create a new theta (just initialized to 0) with which I compute the Lagrangian
    theta_var = torch.zeros([1], requires_grad=True)
    lagrangian, kinetic_energy, potential_energy = compute_lagrangian(q=q, p=p, theta=theta_var)

    dot_costate_generic_theta = autograd.grad(lagrangian, (q),
                                      grad_outputs=lagrangian.data.new(lagrangian.shape).fill_(1),
                                      create_graph=True,
                                      retain_graph=True,
                                      allow_unused=True)

    # Change from previous version: the costate here is \dot(p) = -dnl(q)*theta*p
    # but the theta is the generic one, so I take the derivative with respect to this genetic theta_var
    # with the goal of replacing it with the actual theta (as this is the one I care about and still has the dependencies)
    dot_costate_dtheta = autograd.grad(dot_costate_generic_theta[0], (theta_var),
                                      grad_outputs=lagrangian.data.new(lagrangian.shape).fill_(1),
                                      create_graph=True,
                                      retain_graph=True,
                                      allow_unused=True)

    # here I post-multiply with the theta I care about
    # Can I always do this, or does this only work for this simple toy example?
    rhs_costate = dot_costate_dtheta[0]*theta
    #rhs_costate = dot_costate[0]
    return rhs_costate

def compute_parameter_autograd(q,p):

    theta_var = torch.zeros([1], requires_grad=True)
    lagrangian, kinetic_energy, potential_energy = compute_lagrangian(q=q, p=p, theta=theta_var)

    theta_grad_potential = autograd.grad(potential_energy, (theta_var),
                                         grad_outputs=potential_energy.data.new(potential_energy.shape).fill_(1),
                                         create_graph=True,
                                         retain_graph=True,
                                         allow_unused=True)

    theta = theta_grad_potential[0]
    return theta

def rhs_shooting_autograd(q,p,if_reduced):

    theta = compute_parameter_autograd(q=q,p=p)
    # don't want to carry around any additional depdendencies when computing the costate via autograd

    # Change from previous version: I removed the detach here
    # theta = theta.detach().requires_grad_(False)

    rhs = dict()
    rhs['q'] = rhs_state(q=q, theta=theta)
    if if_reduced:
        rhs['p'] = rhs_costate_autograd_reduced(q=q, p=p)
    else:
        rhs["p"] = rhs_costate_autograd(q=q,p=p,theta =theta)
    return rhs

def euler_forward(rhs_fcn,q0,p0,dt,nr_of_time_steps,if_reduced = True):
    q=q0
    p=p0
    for n in range(nr_of_time_steps):
        rhs = rhs_fcn(q=q,p=p,if_reduced=if_reduced)
        q = q + dt*rhs['q']
        p = p + dt*rhs['p']
    return q,p

# create input data
q0 = torch.randn([1], requires_grad=True)
p0 = torch.randn([1], requires_grad=True)

# if_reduced == True then use the reduced lagrangian computation
# if_reduced == False then use Marc's trick
if_reduced = True
number_of_time_steps = 8
# first try to compute the rhs analytically and via autograd
rhs_analytic = rhs_shooting_analytic(q=q0, p=p0)
rhs_autograd = rhs_shooting_autograd(q=q0, p=p0,if_reduced=if_reduced)
print("Is reduced ? "+str(if_reduced))
print('analytic: rhs_q={}, rhs_p={}'.format(rhs_analytic['q'], rhs_analytic['p']))
print('autograd: rhs_q={}, rhs_p={}'.format(rhs_autograd['q'], rhs_autograd['p']))
print('analytic/autograd: rhs_q={}, rhs_p={}\n'.format(rhs_analytic['q']/rhs_autograd['q'], rhs_analytic['p']/rhs_autograd['p']))

# now do to some simple euler-forward integration

# do it for the analytic version first
res_analytic_q, res_analytic_p = euler_forward(rhs_fcn=rhs_shooting_analytic, q0=q0,p0=p0,dt=0.1,nr_of_time_steps=number_of_time_steps,if_reduced = if_reduced)
loss_analytic = res_analytic_q**2 + res_analytic_p**2
loss_analytic.backward()

loss_analytic_grad_q0 = q0.grad.item()
loss_analytic_grad_p0 = p0.grad.item()

print('loss_analytic_grad_q0={}'.format(loss_analytic_grad_q0))
print('loss_analytic_grad_p0={}'.format(loss_analytic_grad_p0))

# now do it for the autograd version
# set the gradient to zero
zero_grad(q0)
zero_grad(p0)

res_autograd_q, res_autograd_p = euler_forward(rhs_fcn=rhs_shooting_autograd, q0=q0,p0=p0,dt=0.1,nr_of_time_steps=number_of_time_steps,if_reduced = if_reduced)
loss_autograd = res_autograd_q**2 + res_autograd_p**2
loss_autograd.backward()

loss_autograd_grad_q0 = q0.grad.item()
loss_autograd_grad_p0 = p0.grad.item()

print('loss_autograd_grad_q0={}'.format(loss_autograd_grad_q0))
print('loss_autograd_grad_p0={}'.format(loss_autograd_grad_p0))

# and now print out the ratio
print('loss_analytic/autograd_grad_q0={}'.format(loss_analytic_grad_q0/loss_autograd_grad_q0))
print('loss_analytic/autograd_grad_p0={}\n'.format(loss_autograd_grad_p0/loss_autograd_grad_p0))
print('Should all be the same if it works!')
