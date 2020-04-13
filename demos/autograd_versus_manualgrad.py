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

# this is the initial conditions code more or less straight from shooting_blocks.py (with some input parameters added to avoid using self
def _get_initial_conditions_from_data_dict_of_dicts(state_parameter_dict, costate_parameter_dict, data_dict_of_dicts, pass_through_state_dict_of_dicts,
                                                    pass_through_costate_dict_of_dicts, zero_pad_new_data_states, shooting_integrand, block_name):
    """
    Given a data dictionary, this method creates a vector which contains the initial condition consisting of state, costate, and the data.
    As a side effect it also stores (caches) the created assembly plan so that it does not need to be specified when calling disassemble_tensor.

    :param data_dict_of_dicts: data dictionary
    :return: vector of initial conditions
    """

    state_dicts = scd_utils._merge_state_costate_or_data_dict_with_generic_dict_of_dicts(
        generic_dict=state_parameter_dict,
        generic_dict_of_dicts=pass_through_state_dict_of_dicts,
        generic_dict_block_name=block_name)

    costate_dicts = scd_utils._merge_state_costate_or_data_dict_with_generic_dict_of_dicts(
        generic_dict=costate_parameter_dict,
        generic_dict_of_dicts=pass_through_costate_dict_of_dicts,
        generic_dict_block_name=block_name)

    if zero_pad_new_data_states and state_parameter_dict is not None:

        data_concatenation_dim = scd_utils.get_data_concatenation_dim(state_dict=state_parameter_dict,
                                                                      data_dict_of_dicts=data_dict_of_dicts,
                                                                      state_concatenation_dim=shooting_integrand.concatenation_dim)

        shooting_integrand.set_data_concatenation_dim(data_concatenation_dim=data_concatenation_dim)

        padded_zeros = scd_utils._get_zero_data_dict_matching_state_dim(state_dict=state_parameter_dict,
                                                                        data_dict_of_dicts=data_dict_of_dicts,
                                                                        state_concatenation_dim=shooting_integrand.concatenation_dim,
                                                                        data_concatenation_dim=data_concatenation_dim)

        data_dicts = scd_utils._merge_state_costate_or_data_dict_with_generic_dict_of_dicts(
            generic_dict=padded_zeros,
            generic_dict_of_dicts=data_dict_of_dicts,
            generic_dict_block_name=block_name)
    else:
        data_dicts = data_dict_of_dicts

    # initialize the second state of x with zero so far
    initial_conditions, assembly_plans = shooting_integrand.assemble_tensor(state_dict_of_dicts=state_dicts,
                                                                            costate_dict_of_dicts=costate_dicts,
                                                                            data_dict_of_dicts=data_dicts)
    return initial_conditions, assembly_plans

# create a shooting integrand

parameter_weight = 1.0
in_features = 3
nr_of_particles = 2

shooting_integrand = shooting_models.AutoShootingIntegrandModelSimple(
            in_features=in_features,
            particle_dimension=1,
            particle_size=in_features,
            nonlinearity='tanh',
            nr_of_particles=nr_of_particles,
            parameter_weight=parameter_weight)

keep_state_parameters_at_zero = False

# get the state and costate dictionaries
state_dict = shooting_integrand.create_initial_state_parameters_if_needed(set_to_zero=keep_state_parameters_at_zero)
costate_dict = shooting_integrand.create_initial_costate_parameters(state_dict=state_dict)

# create some initial data
x = torch.randn([10,1,3])
block_name = 'test_block'
effective_data_dict = shooting_integrand.get_initial_data_dict_from_data_tensor(x)
effective_data_dict_of_dicts = SortedDict({block_name: effective_data_dict})
zero_pad_new_data_states = False

initial_conditions, assembly_plans = _get_initial_conditions_from_data_dict_of_dicts(
    state_parameter_dict=state_dict,
    costate_parameter_dict=costate_dict,
    data_dict_of_dicts=effective_data_dict_of_dicts,
    pass_through_state_dict_of_dicts=None,
    pass_through_costate_dict_of_dicts=None,
    zero_pad_new_data_states=zero_pad_new_data_states,
    shooting_integrand=shooting_integrand,
    block_name=block_name
)

# need to let the integrand know how to go from the vector to the data stuctures
shooting_integrand.set_auto_assembly_plans(assembly_plans=assembly_plans)
nl = shooting_integrand.nl
dnl = shooting_integrand.dnl

# just to test the disassembling
t_state_dict, t_costate_dict, t_data_dict = shooting_integrand.disassemble_tensor(initial_conditions)

# manually compute the right hand side for the simple model

# the shooting equations should be
#
# dot_qi = A \sigma(qi) + b
# dot_pi = -d\sigma(qi)^T A^T pi
# A = \sum_i pi\sigma(qi)^T
# b = -\sum_i qi

# compute rhs for the state and costate
# first get the state and the costate (for the current block)
s = t_state_dict[block_name]
c = t_costate_dict[block_name]

# now compute the parameters
qi = s['q1']
pi = c['p_q1']

#particles are saved as rows
At = torch.zeros(in_features,in_features)
for i in range(nr_of_particles):
    At = At -(pi[i,...].t()*nl(qi[i,...])).t()
bt = -qi.sum(dim=0)  # -\sum_i q_i

# we are computing based on the transposed quantities here (this makes the use of torch.matmul possible
dot_qt = torch.matmul(nl(qi),At) + bt

# now we can also compute the rhs of the costate (based on the manually computed shooting equations)
dot_pt = torch.zeros_like(pi)
for i in range(nr_of_particles):
    dot_pt[i,...] = -dnl(qi[0,...])*torch.matmul(pi[0,...],At.t())

initial_time = torch.tensor([0.0]).float()
rhs = shooting_integrand(initial_time,initial_conditions)

# transform back into actual dictionary
rhs_state_dict, rhs_costate_dict, rhs_data_dict = shooting_integrand.disassemble_tensor(rhs)

# these are the current parameters (we can compare them to what we get by manually computing them below)
parameter_objects = shooting_integrand._parameter_objects
p = parameter_objects

# extracting the auto-shooting quantities

as_At = p['l1']._parameter_dict['weight'] # should be the same as At
as_bt = p['l1']._parameter_dict['bias'] # should be the same as bt

as_dot_qt = rhs_state_dict[block_name]['q1'] # should be the same as dot_qt
as_dot_pt = rhs_costate_dict[block_name]['p_q1'] # should be the same as dot_pt

# check dotqt
print('dot_qt-as_dot_qt = {}'.format(dot_qt-as_dot_qt))

# check dotpt
print('dot_pt-as_dot_pt = {}'.format(dot_pt-as_dot_pt))

# check A
print('At-as_At = {}'.format(At-as_At))

# check b
print('bt-as_bt = {}'.format(bt-as_bt))


print('Hello world')

