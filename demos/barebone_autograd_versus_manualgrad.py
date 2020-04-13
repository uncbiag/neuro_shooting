# the goal of this script is to use a simple affine model and check if the gradients are computed correctly

import torch
import random

seed = 1234
print('Setting the random seed to {:}'.format(seed))
random.seed(seed)
torch.manual_seed(seed)

from sortedcontainers import SortedDict

import neuro_shooting.shooting_models as shooting_models

import neuro_shooting.state_costate_and_data_dictionary_utils as scd_utils
import neuro_shooting.overwrite_classes as oc
import torch.autograd as autograd

# create a shooting integrand

parameter_weight = 1.0
in_features = 3
nr_of_particles = 2

# this is neeeded to determine how many Lagrangian multipliers there are
# as we add them via their mean
nr_of_particle_parameters = in_features*nr_of_particles

shooting_integrand = shooting_models.AutoShootingIntegrandModelSimple(
            in_features=in_features,
            particle_dimension=1,
            particle_size=in_features,
            nonlinearity='tanh',
            nr_of_particles=nr_of_particles,
            parameter_weight=parameter_weight)


def negate_divide_and_store_in_parameter_objects(parameter_objects, generic_dict):
    for o in parameter_objects:
        if o not in generic_dict:
            ValueError('Needs to contain the same objects. Could not find {}.'.format(o))

        current_pars = parameter_objects[o].get_parameter_dict()
        current_pars_from = generic_dict[o]
        current_weights = parameter_objects[o].get_parameter_weight_dict()

        for k, f in zip(current_pars, current_pars_from):
            if k not in current_weights:
                current_pars[k] = -current_pars_from[f]
            else:
                current_pars[k] = -current_pars_from[f] / current_weights[k]


def compute_potential_energy(t, state_dict, costate_dict, parameter_objects):
    """
    Computes the potential energy for the Lagrangian. I.e., it pairs the costates with the right hand sides of the
    state evolution equations. This method is typically not called manually. Everything should happen automatically here.

    :param t: current time-point
    :param state_dict_of_dicts: SortedDict of SortedDicts holding the states
    :param costate_dict_of_dicts: SortedDict of SortedDicts holding the costates
    :param parameter_objects: parameters to compute the current right hand sides, stored as a SortedDict of instances which compute data transformations (for example linear layer or convolutional layer).
    :return: returns the potential energy (as a pytorch variable)
    """

    # this is really only how one propagates through the system given the parameterization

    rhs_state_dict = shooting_integrand.rhs_advect_state(t=t, state_dict_or_dict_of_dicts=state_dict, parameter_objects=parameter_objects)

    potential_energy = 0

    for ks, kcs in zip(rhs_state_dict, costate_dict):
        potential_energy = potential_energy + torch.mean(costate_dict[kcs] * rhs_state_dict[ks])

    return potential_energy


def compute_kinetic_energy(t, parameter_objects):
    """
    Computes the kinetic energy. This is the kinetic energy given the parameters. Will only be relevant if the system is
    nonlinear in its parameters (when a fixed point solution needs to be computed). Otherwise will not be used. By default just
    computes the sum of squares of the parameters which can be weighted via a scalar (given by the instance of the class
    defining the transformation).

    :todo: Add more general weightings to the classes (i.e., more than just scalars)

    :param t: current timepoint
    :param parameter_objects: dictionary holding all the parameters, stored as a SortedDict of instances which compute data transformations (for example linear layer or convolutional layer)
    :return: returns the kinetc energy as a pyTorch variable
    """

    # as a default it just computes the square norms of all of them (overwrite this if it is not the desired behavior)
    # a weight dictionary can be specified for the individual parameters as part of their
    # overwritten classes (default is all uniform weight)

    kinetic_energy = 0

    for o in parameter_objects:

        current_pars = parameter_objects[o].get_parameter_dict()
        current_weights = parameter_objects[o].get_parameter_weight_dict()

        for k in current_pars:
            cpar = current_pars[k]
            if k not in current_weights:
                cpar_penalty = (cpar ** 2).sum()
            else:
                cpar_penalty = current_weights[k] * (cpar ** 2).sum()

            kinetic_energy = kinetic_energy + cpar_penalty

    kinetic_energy = 0.5 * kinetic_energy

    return kinetic_energy


def compute_lagrangian(t, state_dict, costate_dict, parameter_objects):

    kinetic_energy = compute_kinetic_energy(t=t, parameter_objects=parameter_objects)

    potential_energy = compute_potential_energy(t=t, state_dict=state_dict,
                                                     costate_dict=costate_dict,
                                                     parameter_objects=parameter_objects)

    lagrangian = kinetic_energy - potential_energy

    return lagrangian, kinetic_energy, potential_energy


keep_state_parameters_at_zero = False

# get the state and costate dictionaries
state_dict = shooting_integrand.create_initial_state_parameters_if_needed(set_to_zero=keep_state_parameters_at_zero)
costate_dict = shooting_integrand.create_initial_costate_parameters(state_dict=state_dict)

# create some initial data
x = torch.randn([10,1,3])
block_name = 'test_block'


# need to let the integrand know how to go from the vector to the data stuctures
nl = shooting_integrand.nl
dnl = shooting_integrand.dnl

# manually compute the right hand side for the simple model

# the shooting equations should be
#
# dot_qi = A \sigma(qi) + b
# dot_pi = -d\sigma(qi)^T A^T pi
# A = \sum_i pi\sigma(qi)^T
# b = -\sum_i pi

# compute rhs for the state and costate
# first get the state and the costate (for the current block)
s = state_dict
c = costate_dict

# now compute the parameters
qi = s['q1']
pi = c['p_q1']

#particles are saved as rows
At = torch.zeros(in_features,in_features)
for i in range(nr_of_particles):
    At = At -(pi[i,...].t()*nl(qi[i,...])).t()
At = 1/nr_of_particle_parameters*At # because of the mean in the Lagrangian multiplier
bt = -1/nr_of_particle_parameters*pi.sum(dim=0)  # -\sum_i q_i

# we are computing based on the transposed quantities here (this makes the use of torch.matmul possible
dot_qt = torch.matmul(nl(qi),At) + bt

# now we can also compute the rhs of the costate (based on the manually computed shooting equations)
dot_pt = torch.zeros_like(pi)
for i in range(nr_of_particles):
    dot_pt[i,...] = -dnl(qi[0,...])*torch.matmul(pi[0,...],At.t())


# these are the current parameters (we can compare them to what we get by manually computing them below)
parameter_objects = shooting_integrand._parameter_objects
p = parameter_objects


# now try to compute it automatically
current_lagrangian, current_kinetic_energy, current_potential_energy = \
    compute_lagrangian(t=0, state_dict=state_dict, costate_dict=costate_dict,
                            parameter_objects=parameter_objects)

parameter_tuple = scd_utils.compute_tuple_from_parameter_objects(parameter_objects)

parameter_grad_tuple = autograd.grad(current_potential_energy,
                                     parameter_tuple,
                                     grad_outputs=current_potential_energy.data.new(
                                         current_potential_energy.shape).fill_(1),
                                     create_graph=True,
                                     retain_graph=True,
                                     allow_unused=True)

parameter_grad_dict = scd_utils.extract_dict_from_tuple_based_on_parameter_objects(data_tuple=parameter_grad_tuple,
                                                                                   parameter_objects=parameter_objects,
                                                                                   prefix='grad_')

negate_divide_and_store_in_parameter_objects(parameter_objects=parameter_objects, generic_dict=parameter_grad_dict)



# extracting the auto-shooting quantities

as_A = p['l1']._parameter_dict['weight'] # should be the same as At
as_bt = p['l1']._parameter_dict['bias'] # should be the same as bt

#
# as_dot_qt = rhs_state_dict[block_name]['q1'] # should be the same as dot_qt
# as_dot_pt = rhs_costate_dict[block_name]['p_q1'] # should be the same as dot_pt
#
# # check dotqt
# print('dot_qt-as_dot_qt = {}'.format(dot_qt-as_dot_qt))
#
# # check dotpt
# print('dot_pt-as_dot_pt = {}'.format(dot_pt-as_dot_pt))
#
# check A
print('At.t()-as_A = {}'.format(At.t()-as_A))

# check b
print('bt-as_bt = {}'.format(bt-as_bt))

# check A
print('At.t()/as_A = {}'.format(At.t()/as_A))

# check b
print('bt/as_bt = {}'.format(bt/as_bt))

print('Hello world')

