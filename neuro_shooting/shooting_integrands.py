import torch
import torch.nn as nn
import torch.autograd as autograd

from abc import ABCMeta, abstractmethod
# may require conda install sortedcontainers
from sortedcontainers import SortedDict
from collections import OrderedDict, defaultdict, namedtuple
import torch.utils.hooks as hooks

import neuro_shooting.activation_functions_and_derivatives as ad
import neuro_shooting.state_costate_and_data_dictionary_utils as scd_utils
import neuro_shooting.parameter_initialization as parameter_initialization

class ShootingIntegrandBase(nn.Module):
    """
    Base class for shooting based neural ODE approaches
    """
    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                nr_of_particles=10, particle_dimension=1, particle_size=2, parameter_weight=None, use_analytic_solution=False,
                 use_rnn_mode=False,
                 *args, **kwargs):
        """
        Constructor
        :param nonlinearity: desired nonlinearity to be used tanh, sigmoid, relu, ...
        :param transpose_state_when_forward: if set to true states get transposed (1,2) at the beginng and end of processing
        :param parameter_weight: weight that gets associated to the kinetic energy for the parameters
        :param use_analytic_solution: if True, then the evolution euqations are not being inferred via autograd, but need to be defined as part of the model (for costate evolution and parameter computation)
        """
        super(ShootingIntegrandBase, self).__init__()

        #todo: see how we can make sure these initializers get defined in the derived classes
        self._state_initializer = None
        self._costate_initializer = None

        self.nl, self.dnl = ad.get_nonlinearity(nonlinearity=nonlinearity)
        """Nonlinearity and its derivative"""

        self.in_features = in_features
        """Number of features (for convolutional filters, or dimensions of linear transforms, etc.)"""

        self.nr_of_particles = nr_of_particles
        """Number of particles used for the representation"""
        self.particle_dimension = particle_dimension
        """Dimension of the particle (i.e., how many channels)"""
        self.particle_size = particle_size
        """Size of the particle, corresponds to vector dimension for a vector-evolution, should be a tuple for convolution."""
        self.parameter_weight = parameter_weight
        """This weight should be associated with the model parameters when the different layers are created"""
        self.use_analytic_solution = use_analytic_solution
        """If set to True the shooting evolution equations have to be entirely specified as part of the model"""

        self._lagrangian_gradient_hooks = OrderedDict()
        """Hooks called at any integration step"""

        self._custom_hook_data = None
        """Custom data that is passed to a hook (set via set_custom_hook_data)"""

        self._check_for_availability_of_analytic_shooting_equations = True
        """If called with autodiff option, i.e., use_analytic_solution=False, checks upon first call if an analytic solution has been specified"""

        self.transpose_state_when_forward = transpose_state_when_forward

        # norm penalty
        self.current_norm_penalty = None
        """Will hold the last computed norm penality"""

        self.concatenate_parameters = concatenate_parameters
        """If set to true, parameters will be concatenated (need to have the same dimension)"""

        self.auto_assembly_plans = None
        """Keeps track of how data stuctures are assembled and disassembled"""

        self.use_rnn_mode = use_rnn_mode
        self._rnn_parameters = None
        self._externally_managed_rnn_parameters = False

        # todo: can we force somehow that these will be defined in the derived classes
        self.concatenation_dim = None
        self.data_concatenation_dim = None
        self.enlargement_dimensions = None

    def reset(self):
        self.current_norm_penalty = None
        if not self._externally_managed_rnn_parameters:
            self._rnn_parameters = None

    def set_externally_managed_rnn_parameters(self,rnn_parameters):
        self._rnn_parameters = rnn_parameters
        self.use_rnn_mode = True
        self._externally_managed_rnn_parameters = True

    def set_data_concatenation_dim(self,data_concatenation_dim):
        self.data_concatenation_dim = data_concatenation_dim

    def get_data_concatenation_dim(self):
        return self.data_concatenation_dim

    def set_state_initializer(self,state_initializer):
        self._state_initializer = state_initializer

    def get_state_initializer(self):
        return self._state_initializer

    def set_costate_initializer(self,costate_initializer):
        self._costate_initializer = costate_initializer

    def get_costate_initializer(self):
        return self._costate_initializer

    def concatenate_parameters_on(self):
        self.concatenate_parameters = True

    def concatenate_parameters_off(self):
        self.concatenate_parameters = False

    def set_custom_hook_data(self,data):
        """
        Custom data that is passed to the shooting integrand hooks. In this way a hook can modify its behavior. For
        example, the current batch and epoch could be passed to the hook and the hook could react to it.

        :param data: data to be passed to the hooks (should be a dictionary, but can otherwise be arbitrary as long as the hooks know about the format)
        :return: n/a
        """

        if type(data)!=dict and type(data)!=defaultdict:
            print('WARNING: Expected a dictionary for custom hook data, but got {} instead. Proceeding, but there may be trouble ahead.'.format(type(data)))

        self._custom_hook_data = data

    def register_lagrangian_gradient_hook(self, hook):
        r"""Registers an integration hook on the module.

        The hook will be called every time at the end of the gradient computation in :func:`compute_gradients`.
        It should have the following signature::

            hook(module, t, state_dicts, costate_dicts, data_dict,
            dot_state_dicts, dot_costate_dicts, dot_data_dict, parameter_objects, custom_hook_data) -> None

        The hook should not modify any of its inputs.
        custom_hook_data can be set by the user, via self.set_custom_hook_data. Use case would for example
        to pass a batch and an epoch value so that the hook only is doing something in specific cases
        (for example to only log with tensorboard for the first batch for a given epoch)

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._lagrangian_gradient_hooks)
        self._lagrangian_gradient_hooks[handle.id] = hook
        return handle

    def _apply(self, fn):
        """
        Applies a function to all the parameters of the shooting block.

        :param fn: function which will be applied.
        :return: returns self
        """
        super(ShootingIntegrandBase, self)._apply(fn)
        # make sure that all the filters that were created get moved

        # TODO: remove?
        # for k in self._parameter_objects:
        #     #print('Applying _apply, to {}'.format(k))
        #     self._parameter_objects[k]._apply(fn)

        return self

    def to(self, *args, **kwargs):
        """
        Convenience function to allow moving the block to a different device. Calls .to() on all parameters
        of the shooting block.

        :param args:
        :param kwargs:
        :return: returns self
        """
        super(ShootingIntegrandBase, self).to(*args, **kwargs)

        # make sure that all the filters that were created get moved
        # TODO: remove?
        # for k in self._parameter_objects:
        #     print('Applying to, to {}'.format(k))
        #     self._parameter_objects[k].to(*args, **kwargs)

        return self

    def get_current_norm_penalty(self):
        """
        Returns the last computed norm penalty

        :return: scalar, last computed norm penalty
        """
        return self.current_norm_penalty

    def get_norm_penalty(self):
        """
        Currently mapped to get_current_norm_penalty, buy one could envision computing it here from scratch

        :return:
        """
        current_norm_penalty = self.get_current_norm_penalty()
        if current_norm_penalty is None:
            print('WARNING: current norm penalty is None. Make sure your integration time started at zero. As this is when it is computed.')
        return current_norm_penalty

    def set_auto_assembly_plans(self,assembly_plans):
        self.auto_assembly_plans = assembly_plans

    def assemble_tensor(self, state_dict_of_dicts, costate_dict_of_dicts, data_dict_of_dicts):
        """
        Vectorize all dictionaries together (state, costate, and data). Also returns all their assembly plans.

        :param state_dict: SortedDict holding the SortedDict's of the states
        :param costate_dict: SortedDict holding the SortedDict's of the costate
        :param data_dict: SortedDict holding the SortedDict's for the transported data
        :return: vectorized dictonaries (as one vecctor) and their assembly plans
        """

        ret_vec, assembly_plans = scd_utils.assemble_tensor(state_dict_of_dicts=state_dict_of_dicts,
                                                            costate_dict_of_dicts=costate_dict_of_dicts,
                                                            data_dict_of_dicts=data_dict_of_dicts)
        return ret_vec, assembly_plans

    def disassemble_tensor(self, input, assembly_plans=None, dim=0):
        """
        Disassembles an input vector into state, data, and costate directories.

        :param input: input tensor (vector)
        :param assembly_plans: assembly_plans (does not need to be specified if previously computed -- will be cached)
        :param dim: integrator may add a 0-th dimension to keep track of time. In this case use dim=1, otherwise dim=0 should be fine.
        :return: tuple holding the state, costate, and data dictionaries
        """

        if assembly_plans is None:
            assembly_plans = self.auto_assembly_plans
        else:
            raise ValueError('No assembly plan specified to disassemble tensor. Either explicity specify when calling this method or set one via set_auto_assembly_plans')

        return scd_utils.disassemble_tensor(input=input, assembly_plans=assembly_plans, dim=dim)

    @abstractmethod
    def disassemble(self, input):
        """
        Abstract emthod which needs to be implemented. Takes an input vector and shoud disassemble so that it can return
        the desired part of the state (for example only the position). Implementation will likely make use of
        disassmble_tensor (in state_costate_and_data_dictionary_utils) to implement this.

        :param input: input tensor
        :return: desired part of the state vector
        """

        #Is supposed to return the desired data state (possibly only one) from an input vector
        pass

    def compute_potential_energy(self,t,state_dict_of_dicts,costate_dict_of_dicts,parameter_objects):
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

        rhs_state_dict_of_dicts = self.rhs_advect_state_dict_of_dicts(t=t,state_dict_of_dicts=state_dict_of_dicts,parameter_objects=parameter_objects,concatenation_dim=self.concatenation_dim)

        potential_energy = 0

        for d_ks,d_kcs in zip(rhs_state_dict_of_dicts,costate_dict_of_dicts):
            c_rhs_state_dict = rhs_state_dict_of_dicts[d_ks]
            c_costate_dict = costate_dict_of_dicts[d_kcs]
            for ks,kcs in zip(c_rhs_state_dict,c_costate_dict):
                potential_energy = potential_energy + torch.sum(torch.mean(c_costate_dict[kcs]*c_rhs_state_dict[ks],dim=0))
        return potential_energy

    def compute_kinetic_energy(self,t,parameter_objects):
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
                    cpar_penalty = current_weights[k]*(cpar**2).sum()

                kinetic_energy = kinetic_energy + cpar_penalty

        kinetic_energy = 0.5*kinetic_energy

        return kinetic_energy

    def compute_reduced_lagrangian(self, t, state_dict_of_dicts, costate_dict_of_dicts):
        """
        Computes the lagrangian. Note that this is the Lagrangian in the sense of optimal control, i.e.,

        L = T - U,

        where T is the kinetic energy (here some norm on the parameters governing the state propagation/advection) and
        U is the potential energy (which amounts to the costates paired with the right hand sides of the state advection equations),
        i,e. <p,dot_x>

        Returns a triple of scalars. The value of the Lagrangian as well as of the kinetic and the potential energies.

        :param t: current timepoint
        :param state_dict_of_dicts: SortedDict of SortedDict's containing the states
        :param costate_dict_of_dicts: SortedDict of SortedDict's containing the costates
        :return: triple (value of lagrangian, value of the kinetic energy, value of the potential energy)
        """

        if self.use_rnn_mode:
            # we only compute it the first time
            if self._rnn_parameters is None:
                if self._externally_managed_rnn_parameters:
                    raise ValueError('Externally managed RNN parameters cannot be created here. Should have been assigned before.')

                parameter_objects = self.compute_parameters(t=t, state_dict_of_dicts=state_dict_of_dicts,
                                                            costate_dict_of_dicts=costate_dict_of_dicts)
                self._rnn_parameters = parameter_objects
            else:
                parameter_objects = self._rnn_parameters

        else:
            # first compute the current parameters
            # current parameters are computed via autodiff
            parameter_objects = self.compute_parameters(t=t, state_dict_of_dicts=state_dict_of_dicts,
                                                             costate_dict_of_dicts=costate_dict_of_dicts)


        kinetic_energy = self.compute_kinetic_energy(t=t, parameter_objects=parameter_objects)
        potential_energy = self.compute_potential_energy(t=t, state_dict_of_dicts=state_dict_of_dicts,
                                                         costate_dict_of_dicts=costate_dict_of_dicts,
                                                         parameter_objects=parameter_objects)

        lagrangian = kinetic_energy-potential_energy

        return self.nr_of_particles * lagrangian,self.nr_of_particles * kinetic_energy, self.nr_of_particles *potential_energy

    def compute_lagrangian(self, t, state_dict_of_dicts, costate_dict_of_dicts, parameter_objects):
        """
        Computes the lagrangian. Note that this is the Lagrangian in the sense of optimal control, i.e.,

        L = T - U,

        where T is the kinetic energy (here some norm on the parameters governing the state propagation/advection) and
        U is the potential energy (which amounts to the costates paired with the right hand sides of the state advection equations),
        i,e. <p,dot_x>

        Returns a triple of scalars. The value of the Lagrangian as well as of the kinetic and the potential energies.

        :param t: current timepoint
        :param state_dict_of_dicts: SortedDict of SortedDict's containing the states
        :param costate_dict_of_dicts: SortedDict of SortedDict's containing the costates
        :param parameter_objects: SortedDict with all the parameters for the advection equation, stored as a SortedDict of instances which compute data transformations (for example linear layer or convolutional layer)
        :return: triple (value of lagrangian, value of the kinetic energy, value of the potential energy)
        """

        kinetic_energy = self.compute_kinetic_energy(t=t, parameter_objects=parameter_objects)
        potential_energy = self.compute_potential_energy(t=t, state_dict_of_dicts=state_dict_of_dicts,
                                                         costate_dict_of_dicts=costate_dict_of_dicts,
                                                         parameter_objects=parameter_objects)

        lagrangian = kinetic_energy-potential_energy

        return lagrangian, kinetic_energy, potential_energy

    @abstractmethod
    def create_initial_state_parameters(self, set_to_zero, *argv,**kwargs):
        # todo: maybe a better design is possible here. for now left as part of the integrands as it makes the definitions much easier (in one place)
        # use self._state_initializer.create_parameters() or .create_parameters_like() to initialize the state
        pass

    def create_initial_state_parameters_if_needed(self, set_to_zero, *argv, **kwargs):

        if self.nr_of_particles is None or self.particle_size is None or self.particle_dimension is None:
            return None
        else:
            return self.create_initial_state_parameters(set_to_zero=set_to_zero, *argv, **kwargs)

    def create_initial_costate_parameters(self, state_dict=None, *args, **kwargs):
        """
        By default this function automatically creates the costates for the given states (and does not need to be called
        manually). Costates are names with the prefix \'p\_\', i.e., if a state was called \'q\' the corresponding costate
        will be called \'p\_q\'. Overwrite this method if you want to do something custom for the costates, though this
        should only rarely be necessary.

        :param batch_y0: data batch passed in (but not currently used) that allows to create state-specific costates.
        :param only_random_initialization: to indicate if the costates should be randomized (also currently not used; will be random by default).
        :param state_dict: state dictionary which can be used to create the corresponding costate
        :return: returns a SortedDict containing the costate.
        """

        if state_dict is None:
            return None
        else:

            costate_dict = SortedDict()
            for k in state_dict:
                costate_dict['p_' + str(k)] = self._costate_initializer.create_parameters_like(state_dict[k])

            return costate_dict

    @abstractmethod
    def create_default_parameter_objects(self):
        """
        Abstract method which should return a SortedDict which contains instances of the objects which are used to compute
        the state equations. These objects in turn contain the parameters. For example, overwritten convolutional or linear layer.

        :return: returns a SortedDict of parameter objects
        """
        raise ValueError('Not implemented. Needs to return a SortedDict of parameter objects')

    @abstractmethod
    def get_initial_data_dict_from_data_tensor(self, x):
        """
        Abstract method to obtain an intial data dictionary from a data tensor. Needs to be a SortedDict(). Needs to have
        the same state variable names as for the states themselves. Needed to make sure what to do with higher-order states
        (typically set to zero in implementations). Part of the integrands as they know about the states, but will be called
        by the shooting block to build an initial condition.

        :param x: data vector (tensor)
        :return: SortedDict() for the data
        """

        pass

    def rhs_advect_state_dict_of_dicts(self,t,state_dict_of_dicts,parameter_objects,concatenation_dim):
        if self.concatenate_parameters:
            state_dicts_concatenated = scd_utils._concatenate_dict_of_dicts(generic_dict_of_dicts=state_dict_of_dicts,concatenation_dim=concatenation_dim)
            rhs_state_dict = self.rhs_advect_state(t=t,state_dict_or_dict_of_dicts=state_dicts_concatenated, parameter_objects=parameter_objects)
            rhs_state_dict_of_dicts = scd_utils._deconcatenate_based_on_generic_dict_of_dicts(rhs_state_dict,generic_dict_of_dicts=state_dict_of_dicts,concatenation_dim=concatenation_dim)
            #rhs_state_dict_of_dicts = SortedDict({self._block_name:rhs_state_dict})
            return rhs_state_dict_of_dicts
        else:
            return self.rhs_advect_state(t=t,state_dict_or_dict_of_dicts=state_dict_of_dicts,
                                         parameter_objects=parameter_objects)

    def optional_rhs_advect_costate_dict_of_dicts_analytic(self,t,state_dict_of_dicts,costate_dict_of_dicts,parameter_objects,concatenation_dim):
        if self.concatenate_parameters:
            state_dicts_concatenated = scd_utils._concatenate_dict_of_dicts(generic_dict_of_dicts=state_dict_of_dicts,concatenation_dim=concatenation_dim)
            costate_dicts_concatenated = scd_utils._concatenate_dict_of_dicts(generic_dict_of_dicts=costate_dict_of_dicts,concatenation_dim=concatenation_dim)
            rhs_costate_dict = self.optional_rhs_advect_costate_analytic(t=t,state_dict_or_dict_of_dicts=state_dicts_concatenated, costate_dict_or_dict_of_dicts=costate_dicts_concatenated, parameter_objects=parameter_objects)
            rhs_costate_dict_of_dicts = scd_utils._deconcatenate_based_on_generic_dict_of_dicts(rhs_costate_dict,generic_dict_of_dicts=costate_dict_of_dicts,concatenation_dim=concatenation_dim)
            #rhs_state_dict_of_dicts = SortedDict({self._block_name:rhs_state_dict})
            return rhs_costate_dict_of_dicts
        else:
            return self.optional_rhs_advect_costate_analytic(t=t,state_dict_or_dict_of_dicts=state_dict_of_dicts,
                                                             costate_dict_or_dict_of_dicts=costate_dict_of_dicts,
                                                             parameter_objects=parameter_objects)


    @abstractmethod
    def rhs_advect_state(self, t, state_dict_or_dict_of_dicts, parameter_objects):
        pass

    def rhs_advect_data_dict_of_dicts(self,t,data_dict_of_dicts,parameter_objects):
        ret_dict_of_dicts = self.rhs_advect_state_dict_of_dicts(t=t,state_dict_of_dicts=data_dict_of_dicts,parameter_objects=parameter_objects,concatenation_dim=self.data_concatenation_dim)
        return ret_dict_of_dicts

    @abstractmethod
    def rhs_advect_costate_reduced_dict_of_dicts(self, t, state_dict_of_dicts, costate_dict_of_dicts):
        # now that we have the parameters we can get the rhs for the costate using autodiff
        # returns a dictionary of the RHS of the costate
        pass

    def optional_rhs_advect_costate_analytic(self, t, state_dict_of_dicts, costate_dict_of_dicts, parameter_objects):
        """
        We can prescribe an analytic costate evolution (where we do not need to do this via autodiff).
        This is optional, but can be used for testing.

        :param t:
        :param state_dict_of_dicts:
        :param costate_dict_of_dicts:
        :param parameter_objects:
        :return:
        """

        print('To use this functionality, overwrite optional_compute_parameters_analytic and optional_rhs_advect_costate_analytic in your model.')
        raise ValueError('Not implemented.')

    def add_multiple_to_parameter_objects(self,parameter_objects,pd_from,multiplier=1.0):

        for o in parameter_objects:
            if o not in pd_from:
                ValueError('Needs to contain the same objects. Could not find {}.'.format(o))

            current_pars = parameter_objects[o].get_parameter_dict()
            current_from_pars = pd_from[o]

            for p,p_from in zip(current_pars,current_from_pars):
                current_pars[p] = current_pars[p]+multiplier*current_from_pars[p_from]


    def divide_and_store_in_parameter_objects(self,parameter_objects,generic_dict):

        for o in parameter_objects:
            if o not in generic_dict:
                ValueError('Needs to contain the same objects. Could not find {}.'.format(o))

            current_pars = parameter_objects[o].get_parameter_dict()
            current_pars_from = generic_dict[o]
            current_weights = parameter_objects[o].get_parameter_weight_dict()

            for k,f in zip(current_pars,current_pars_from):
                if k not in current_weights:
                    current_pars[k] = current_pars_from[f]
                else:
                    current_pars[k] = current_pars_from[f]/current_weights[k]

        return parameter_objects

    @abstractmethod
    def compute_parameters(self,t,state_dict,costate_dict):
        """
        Computes parameters and stores them in parameter_objects. Returns the current kinectic energy (i.e., penalizer on parameters)
        :param state_dict:
        :param costate_dict:
        :return:
        """
        pass

    def optional_compute_parameters_analytic(self,t,state_dict,costate_dict):
        """
        We can prescribe an analytic computation of the parameters (where we do not need to do this via autodiff).
        This is optional, but can be used for testing.

        :param t:
        :param state_dict:
        :param costate_dict:
        :return:
        """
        print('To use this functionality, overwrite optional_compute_parameters_analytic and optional_rhs_advect_costate_analytic in your model.')
        raise ValueError('Not implemented.')

    def detach_and_require_gradients_for_parameter_objects(self,parameter_objects):

        # TODO: check if this method is really needed
        for o in parameter_objects:
            current_pars = parameter_objects[o].get_parameter_dict()
            for k in current_pars:
                current_pars[k] = current_pars[k].detach().requires_grad_(True)

    def compute_gradients_analytic(self, t, state_dict_of_dicts, costate_dict_of_dicts, data_dict_of_dicts):

        # TODO: Fix this method, not yet converted. First loop looks suspicious

        # here we compute the rhs of the equations via their analytic solutions
        # assumes that optional_rhs_advect_costate_analytic and optimal_compute_parameters_analytic have been defined in the model

        if len(state_dict_of_dicts)!=1 or len(costate_dict_of_dicts)!=1:
            raise ValueError('Analytic computation does not currently support multiple blocks.')

        state_dict = state_dict_of_dicts.values()[0]
        costate_dict = costate_dict_of_dicts.values()[0]

        if self.use_rnn_mode:
            # we only compute it the first time
            if self._rnn_parameters is None:
                if self._externally_managed_rnn_parameters:
                    raise ValueError(
                        'Externally managed RNN parameters cannot be created here. Should have been assigned before.')

                parameter_objects = self.optional_compute_parameters_analytic(t=t,state_dict=state_dict,costate_dict=costate_dict)
                self._rnn_parameters = parameter_objects
            else:
                parameter_objects = self._rnn_parameters

        else:
            # we compute these parameters every time
            parameter_objects = self.optional_compute_parameters_analytic(t=t,
                                                                        state_dict=state_dict,
                                                                        costate_dict=costate_dict)

        if t == 0:
            # we only need to compute the kinetic energy here
            current_kinetic_energy = self.compute_kinetic_energy(t=t, parameter_objects=parameter_objects)

            # we only want it at the initial condition
            self.current_norm_penalty = current_kinetic_energy

        # evolution of state equation is always known (so the same as for the autodiff approach)
        dot_state_dict_of_dicts = self.rhs_advect_state_dict_of_dicts(t=t, state_dict_of_dicts=state_dict_of_dicts,
                                                                      parameter_objects=parameter_objects,
                                                                      concatenation_dim=self.concatenation_dim)
        dot_data_dict_of_dicts = self.rhs_advect_data_dict_of_dicts(t=t, data_dict_of_dicts=data_dict_of_dicts,
                                                                    parameter_objects=parameter_objects)

        # costate evolution is automatically obtained in the autodiff solution, but here specified explicitly
        dot_costate_dict_of_dicts = self.optional_rhs_advect_costate_dict_of_dicts_analytic(t=t, state_dict_of_dicts=state_dict_of_dicts,
                                                                                            costate_dict_of_dicts=costate_dict_of_dicts,
                                                                                            parameter_objects=parameter_objects,
                                                                                            concatenation_dim=self.data_concatenation_dim)

        return dot_state_dict_of_dicts, dot_costate_dict_of_dicts, dot_data_dict_of_dicts, parameter_objects

    def compute_gradients_autodiff(self,t,state_dict_of_dicts,costate_dict_of_dicts,data_dict_of_dicts):

        # here we compute the rhs of the equations via automatic differentiation

        if self.use_rnn_mode:
            # we only compute it the first time
            if self._rnn_parameters is None:
                if self._externally_managed_rnn_parameters:
                    raise ValueError(
                        'Externally managed RNN parameters cannot be created here. Should have been assigned before.')

                parameter_objects = self.compute_parameters(t=t, state_dict_of_dicts=state_dict_of_dicts,
                                                    costate_dict_of_dicts=costate_dict_of_dicts)
                self._rnn_parameters = parameter_objects
            else:
                parameter_objects = self._rnn_parameters

        else:
            # we compute these parameters every time
            # current parameters are computed via autodiff
            parameter_objects = self.compute_parameters(t=t, state_dict_of_dicts=state_dict_of_dicts,
                                                        costate_dict_of_dicts=costate_dict_of_dicts)

        if t == 0:
            # we only want it at the initial condition
            self.current_norm_penalty = self.compute_kinetic_energy(t=0, parameter_objects=parameter_objects)

        dot_state_dict_of_dicts = self.rhs_advect_state_dict_of_dicts(t=t, state_dict_of_dicts=state_dict_of_dicts,
                                                                      parameter_objects=parameter_objects,
                                                                      concatenation_dim=self.concatenation_dim)
        dot_data_dict_of_dicts = self.rhs_advect_data_dict_of_dicts(t=t, data_dict_of_dicts=data_dict_of_dicts,
                                                                    parameter_objects=parameter_objects)

        # costate evolution is obtained via autodiff
        dot_costate_dict_of_dicts = self.rhs_advect_costate_reduced_dict_of_dicts(t=t, state_dict_of_dicts=state_dict_of_dicts,
                                                                          costate_dict_of_dicts=costate_dict_of_dicts)

        return dot_state_dict_of_dicts, dot_costate_dict_of_dicts, dot_data_dict_of_dicts, parameter_objects

    def compute_gradients(self,t,state_dict_of_dicts,costate_dict_of_dicts,data_dict_of_dicts):

        if self.use_analytic_solution:
            dot_state_dict_of_dicts, dot_costate_dict_of_dicts, dot_data_dict_of_dicts, parameter_objects = \
                self.compute_gradients_analytic(t,state_dict_of_dicts,costate_dict_of_dicts,data_dict_of_dicts)

        else:
            if self._check_for_availability_of_analytic_shooting_equations:
                if (type(self).optional_compute_parameters_analytic is not ShootingIntegrandBase.optional_compute_parameters_analytic) and \
                    (type(self).optional_rhs_advect_costate_analytic is not ShootingIntegrandBase.optional_rhs_advect_costate_analytic):
                    print('Analytic shooting equations appear to be defined. You can turn them on by instantiating the class with option use_analytic_solution=True')
                    print('Shooting equations will be computed via autodiff')
                    self._check_for_availability_of_analytic_shooting_equations = False

            dot_state_dict_of_dicts, dot_costate_dict_of_dicts, dot_data_dict_of_dicts, parameter_objects = \
                self.compute_gradients_autodiff(t,state_dict_of_dicts,costate_dict_of_dicts,data_dict_of_dicts)

        # run the hooks so we can get parameters, states, etc.; for example, to create tensorboard output
        for hook in self._lagrangian_gradient_hooks.values():
            hook(self, t, state_dict_of_dicts, costate_dict_of_dicts, data_dict_of_dicts,
                 dot_state_dict_of_dicts, dot_costate_dict_of_dicts, dot_data_dict_of_dicts,
                 parameter_objects, self._custom_hook_data)

        return dot_state_dict_of_dicts,dot_costate_dict_of_dicts,dot_data_dict_of_dicts, parameter_objects


    def transpose_state_dict_of_dicts(self,generic_dict_dict_of_dicts):
        # there might be multiple dictionaries that constitute the state dictionary, so transpose each of them
        ret_dicts = SortedDict()
        for k in generic_dict_dict_of_dicts:
            ret_dicts[k] = self.transpose_state(ret_dicts[k])
        return ret_dicts

    def transpose_state(self,generic_dict):
        ret = SortedDict()
        for k in generic_dict:
            ret[k] = (generic_dict[k]).transpose(1,2)
        return ret


    def forward(self, t, input):

        with torch.enable_grad():

            if input.requires_grad==False:
                # we always need this to compute the gradient, because we derive the shooting equations automatically
                # if the gradient computation is on by default nothing should change here
                # however, when the adjoint integrators are used, these variables will not have the gradient on
                input = input.clone().detach().requires_grad_(True)

            state_dict_of_dicts,costate_dict_of_dicts,data_dict_of_dicts = self.disassemble_tensor(input)

            if self.transpose_state_when_forward:
                state_dict_of_dicts = self.transpose_state_dict_of_dicts(state_dict_of_dicts)
                costate_dict_of_dicts = self.transpose_state_dict_of_dicts(costate_dict_of_dicts)
                data_dict_of_dicts = self.transpose_state_dict_of_dicts(data_dict_of_dicts)

            # computing the gradients
            dot_state_dict_of_dicts, dot_costate_dict_of_dicts, dot_data_dict_of_dicts, parameter_objects = \
                self.compute_gradients(t=t, state_dict_of_dicts=state_dict_of_dicts,costate_dict_of_dicts=costate_dict_of_dicts,data_dict_of_dicts=data_dict_of_dicts)

            if self.transpose_state_when_forward:
                # as we transposed the vectors before we need to transpose on the way back
                dot_state_dict_of_dicts = self.transpose_state_dict_of_dicts(dot_state_dict_of_dicts)
                dot_costate_dict_of_dicts = self.transpose_state_dict_of_dicts(dot_costate_dict_of_dicts)
                dot_data_dict_of_dicts = self.transpose_state(dot_data_dict_of_dicts)

            # create a vector out of this to pass to integrator
            output,assembly_plans = self.assemble_tensor(state_dict_of_dicts=dot_state_dict_of_dicts, costate_dict_of_dicts=dot_costate_dict_of_dicts, data_dict_of_dicts=dot_data_dict_of_dicts)

        return output


class AutogradShootingIntegrandBase(ShootingIntegrandBase):
    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False,
                 concatenate_parameters=True,
                 nr_of_particles=10, particle_dimension=1, particle_size=2, parameter_weight=None,
                 *args,**kwargs):
        super(AutogradShootingIntegrandBase, self).__init__(in_features=in_features,
                                                            nonlinearity=nonlinearity,
                                                            transpose_state_when_forward=transpose_state_when_forward,
                                                            concatenate_parameters=concatenate_parameters,
                                                            nr_of_particles=nr_of_particles,
                                                            particle_dimension=particle_dimension,
                                                            particle_size=particle_size,
                                                            parameter_weight=parameter_weight,
                                                            *args, **kwargs)

    def rhs_advect_costate_reduced_dict_of_dicts(self, t, state_dict_of_dicts, costate_dict_of_dicts):
        # now that we have the parameters we can get the rhs for the costate using autodiff

        current_lagrangian, current_kinetic_energy, current_potential_energy = \
            self.compute_reduced_lagrangian(t=t, state_dict_of_dicts=state_dict_of_dicts,
                                    costate_dict_of_dicts=costate_dict_of_dicts)

        # form a tuple of all the state variables (because this is what we take the derivative of)
        state_tuple = scd_utils.compute_tuple_from_generic_dict_of_dicts(state_dict_of_dicts)

        dot_costate_tuple = autograd.grad(current_lagrangian, state_tuple,
                                          grad_outputs=current_lagrangian.data.new(current_lagrangian.shape).fill_(1.0),
                                          create_graph=True,
                                          retain_graph=True,
                                          allow_unused=True)

        # now we need to put these into a sorted dictionary
        dot_costate_dict_of_dicts = scd_utils.extract_dict_of_dicts_from_tuple_based_on_generic_dict_of_dicts(
            data_tuple=dot_costate_tuple,
            generic_dict_of_dicts=state_dict_of_dicts, prefix='dot_p_')

        return dot_costate_dict_of_dicts


class LinearInParameterAutogradShootingIntegrand(AutogradShootingIntegrandBase):
    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False,
                 concatenate_parameters=True,
                 nr_of_particles=10, particle_dimension=1, particle_size=2, parameter_weight=None,
                 *args,**kwargs):
        super(LinearInParameterAutogradShootingIntegrand, self).__init__(in_features=in_features,
                                                                         nonlinearity=nonlinearity,
                                                                         transpose_state_when_forward=transpose_state_when_forward,
                                                                         concatenate_parameters=concatenate_parameters,
                                                                         nr_of_particles=nr_of_particles,
                                                                         particle_dimension=particle_dimension,
                                                                         particle_size=particle_size,
                                                                         parameter_weight=parameter_weight,
                                                                         *args, **kwargs)

    def compute_parameters_directly(self, t, state_dict_of_dicts, costate_dict_of_dicts):
        # we assume this is linear here, so we do not need a fixed point iteration, but can just compute the gradient

        parameter_objects = self.create_default_parameter_objects()

        current_lagrangian, current_kinetic_energy, current_potential_energy = \
            self.compute_lagrangian(t=t, state_dict_of_dicts=state_dict_of_dicts, costate_dict_of_dicts=costate_dict_of_dicts, parameter_objects=parameter_objects)

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

        parameter_objects = self.divide_and_store_in_parameter_objects(parameter_objects=parameter_objects,generic_dict=parameter_grad_dict)

        return parameter_objects

    def compute_parameters(self,t, state_dict_of_dicts,costate_dict_of_dicts):

        return self.compute_parameters_directly(t=t, state_dict_of_dicts=state_dict_of_dicts,costate_dict_of_dicts=costate_dict_of_dicts)


class NonlinearInParameterAutogradShootingIntegrand(AutogradShootingIntegrandBase):
    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False,
                 concatenate_parameters=True,
                 nr_of_particles=10, particle_dimension=1, particle_size=2, parameter_weight=None,
                 *args,**kwargs):
        super(NonlinearInParameterAutogradShootingIntegrand, self).__init__(in_features=in_features,
                                                                            nonlinearity=nonlinearity,
                                                                            transpose_state_when_forward=transpose_state_when_forward,
                                                                            concatenate_parameters=concatenate_parameters,
                                                                            nr_of_particles=nr_of_particles,
                                                                            particle_dimension=particle_dimension,
                                                                            particle_size=particle_size,
                                                                            parameter_weight=parameter_weight,
                                                                            *args, **kwargs)

    def compute_parameters_iteratively(self, t, state_dict_of_dicts, costate_dict_of_dicts):

        parameter_objects = self.create_default_parameter_objects()

        learning_rate = 0.5
        nr_of_fixed_point_iterations = 5

        for n in range(nr_of_fixed_point_iterations):
            current_lagrangian, current_kinetic_energy, current_potential_energy = \
                self.compute_lagrangian(t=t, state_dict_of_dicts=state_dict_of_dicts, costate_dict_of_dicts=costate_dict_of_dicts, parameter_objects=parameter_objects)

            parameter_tuple = scd_utils.compute_tuple_from_parameter_objects(parameter_objects)

            parameter_grad_tuple = autograd.grad(current_lagrangian,
                                                 parameter_tuple,
                                                 grad_outputs=current_lagrangian.data.new(
                                                     current_lagrangian.shape).fill_(1),
                                                 create_graph=True,
                                                 retain_graph=True,
                                                 allow_unused=True)

            parameter_grad_dict = scd_utils.extract_dict_from_tuple_based_on_parameter_objects(data_tuple=parameter_grad_tuple,
                                                                                     parameter_objects=parameter_objects,
                                                                                     prefix='grad_')

            self.add_multiple_to_parameter_objects(parameter_objects=parameter_objects,
                                                   pd_from=parameter_grad_dict, multiplier=-learning_rate)

        return parameter_objects, current_kinetic_energy

    def compute_parameters(self,t, parameter_objects,state_dict_of_dicts,costate_dict_of_dicts):

        return self.compute_parameters_iteratively(t=t, parameter_objects=parameter_objects,state_dict_of_dicts=state_dict_of_dicts,costate_dict_of_dicts=costate_dict_of_dicts)


class ShootingLinearInParameterVectorIntegrand(LinearInParameterAutogradShootingIntegrand):
    """
    Base class for shooting based neural ODE approaches
    """
    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                nr_of_particles=10,particle_dimension=1,particle_size=2,parameter_weight=None,
                 state_initializer=None,costate_initializer=None,
                 *args, **kwargs):

        super(ShootingLinearInParameterVectorIntegrand, self).__init__(in_features=in_features,
                                                                       nonlinearity=nonlinearity,
                                                                       transpose_state_when_forward=transpose_state_when_forward,
                                                                       concatenate_parameters=concatenate_parameters,
                                                                       nr_of_particles=nr_of_particles,
                                                                       particle_dimension=particle_dimension,particle_size=particle_size,
                                                                       parameter_weight=parameter_weight,
                                                                       *args, **kwargs)

        if state_initializer is not None:
            self._state_initializer = state_initializer
        else:
            self._state_initializer = parameter_initialization.VectorEvolutionParameterInitializer()

        if costate_initializer is not None:
            self._costate_initializer = costate_initializer
        else:
            self._costate_initializer = parameter_initialization.VectorEvolutionParameterInitializer()

        self.concatenation_dim = 2
        self.data_concatenation_dim = self.concatenation_dim
        self.enlargement_dimensions = None


class ShootingNonlinearInParameterVectorIntegrand(NonlinearInParameterAutogradShootingIntegrand):
    """
    Base class for shooting based neural ODE approaches
    """
    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                nr_of_particles=10,particle_dimension=1,particle_size=2,
                 parameter_weight=None,
                 state_initializer=None,costate_initializer=None,
                 *args, **kwargs):

        super(ShootingNonlinearInParameterVectorIntegrand, self).__init__(in_features=in_features,
                                                                          nonlinearity=nonlinearity,
                                                                          transpose_state_when_forward=transpose_state_when_forward,
                                                                          concatenate_parameters=concatenate_parameters,
                                                                          nr_of_particles=nr_of_particles,
                                                                          particle_dimension=particle_dimension,particle_size=particle_size,
                                                                          parameter_weight=parameter_weight,
                                                                          *args, **kwargs)

        if state_initializer is not None:
            self._state_initializer = state_initializer
        else:
            self._state_initializer = parameter_initialization.VectorEvolutionParameterInitializer()

        if costate_initializer is not None:
            self._costate_initializer = costate_initializer
        else:
            self._costate_initializer = parameter_initialization.VectorEvolutionParameterInitializer()

        self.concatenation_dim = 2
        self.data_concatenation_dim = self.concatenation_dim
        self.enlargement_dimensions = None


class ShootingLinearInParameterConvolutionIntegrand(LinearInParameterAutogradShootingIntegrand):
    """
    Base class for shooting based neural ODE approaches
    """

    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                 nr_of_particles=10, particle_dimension=1, particle_size=2, parameter_weight=None,
                 state_initializer=None,costate_initializer=None,
                 *args, **kwargs):

        super(ShootingLinearInParameterConvolutionIntegrand, self).__init__(in_features=in_features,
                                                                            nonlinearity=nonlinearity,
                                                                            transpose_state_when_forward=transpose_state_when_forward,
                                                                            concatenate_parameters=concatenate_parameters,
                                                                            nr_of_particles=nr_of_particles,
                                                                            particle_dimension=particle_dimension, particle_size=particle_size,
                                                                            parameter_weight=parameter_weight,
                                                                            *args, **kwargs)

        if state_initializer is not None:
            self._state_initializer = state_initializer
        else:
            self._state_initializer = parameter_initialization.ConvolutionEvolutionParameterInitializer()

        if costate_initializer is not None:
            self._costate_initializer = costate_initializer
        else:
            self._costate_initializer = parameter_initialization.ConvolutionEvolutionParameterInitializer()

        self.concatenation_dim = 1
        self.data_concatenation_dim = self.concatenation_dim


class ShootingNonlinearInParameterConvolutionIntegrand(NonlinearInParameterAutogradShootingIntegrand):
    """
    Base class for shooting based neural ODE approaches
    """

    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                 nr_of_particles=10, particle_dimension=1, particle_size=2,parameter_weight=None,
                 state_initializer=None,costate_initializer=None,
                 *args, **kwargs):

        super(ShootingNonlinearInParameterConvolutionIntegrand, self).__init__(in_features=in_features,
                                                                               nonlinearity=nonlinearity,
                                                                               transpose_state_when_forward=transpose_state_when_forward,
                                                                               concatenate_parameters=concatenate_parameters,
                                                                               nr_of_particles=nr_of_particles,
                                                                               particle_dimension=particle_dimension, particle_size=particle_size,
                                                                               parameter_weight=parameter_weight,
                                                                               *args, **kwargs)

        if state_initializer is not None:
            self._state_initializer = state_initializer
        else:
            self._state_initializer = parameter_initialization.ConvolutionEvolutionParameterInitializer()

        if costate_initializer is not None:
            self._costate_initializer = costate_initializer
        else:
            self._costate_initializer = parameter_initialization.ConvolutionEvolutionParameterInitializer()

        self.concatenation_dim = 1
        self.data_concatenation_dim = self.concatenation_dim


class OptimalTransportNonLinearInParameter(NonlinearInParameterAutogradShootingIntegrand):
    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                 nr_of_particles=10, particle_dimension=1, particle_size=2,
                 parameter_weight=None,
                 state_initializer=None, costate_initializer=None,
                 *args, **kwargs):
        super(OptimalTransportNonLinearInParameter, self).__init__(in_features=in_features,
                                                                   nonlinearity=nonlinearity,
                                                                   transpose_state_when_forward=transpose_state_when_forward,
                                                                   concatenate_parameters=concatenate_parameters,
                                                                   nr_of_particles=nr_of_particles,
                                                                   particle_dimension=particle_dimension,
                                                                   particle_size=particle_size,
                                                                   parameter_weight=parameter_weight,
                                                                   *args, **kwargs)

        if state_initializer is not None:
            self._state_initializer = state_initializer
        else:
            self._state_initializer = parameter_initialization.VectorEvolutionParameterInitializer()

        if costate_initializer is not None:
            self._costate_initializer = costate_initializer
        else:
            self._costate_initializer = parameter_initialization.VectorEvolutionParameterInitializer()

        self.concatenation_dim = 2
        self.data_concatenation_dim = self.concatenation_dim
        self.enlargement_dimensions = None

    def compute_kinetic_energy(self, t, state_dict_of_dicts, costate_dict_of_dicts, parameter_objects):
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

        kinetic_energy = 0
        rhs_state_dict_of_dicts = self.rhs_advect_state_dict_of_dicts(t=t, state_dict_of_dicts=state_dict_of_dicts,
                                                                      parameter_objects=parameter_objects,
                                                                      concatenation_dim=self.concatenation_dim)

        for d_ks in rhs_state_dict_of_dicts:
            c_rhs_state_dict = rhs_state_dict_of_dicts[d_ks]
            for ks in c_rhs_state_dict:
                kinetic_energy += torch.sum(c_rhs_state_dict[ks]**2)
        #todo : introduce a penalty parameter.
        kinetic_energy = 0.5 * self.parameter_weight * kinetic_energy
        return kinetic_energy

    def compute_parameters_iteratively(self, t, state_dict_of_dicts, costate_dict_of_dicts):

        parameter_objects = self.create_default_parameter_objects()

        learning_rate = 0.5
        nr_of_fixed_point_iterations = 10

        for n in range(nr_of_fixed_point_iterations):
            current_lagrangian, current_kinetic_energy, current_potential_energy = \
                self.compute_lagrangian(t=t, state_dict_of_dicts=state_dict_of_dicts, costate_dict_of_dicts=costate_dict_of_dicts, parameter_objects=parameter_objects)

            parameter_tuple = scd_utils.compute_tuple_from_parameter_objects(parameter_objects)

            parameter_grad_tuple = autograd.grad(current_lagrangian,
                                                 parameter_tuple,
                                                 grad_outputs=current_lagrangian.data.new(
                                                     current_lagrangian.shape).fill_(1),
                                                 create_graph=True,
                                                 retain_graph=True,
                                                 allow_unused=True)

            parameter_grad_dict = scd_utils.extract_dict_from_tuple_based_on_parameter_objects(data_tuple=parameter_grad_tuple,
                                                                                     parameter_objects=parameter_objects,
                                                                                     prefix='grad_')

            self.add_multiple_to_parameter_objects(parameter_objects=parameter_objects,
                                                   pd_from=parameter_grad_dict, multiplier=-learning_rate)

        return parameter_objects, current_kinetic_energy

    def compute_lagrangian(self, t, state_dict_of_dicts, costate_dict_of_dicts, parameter_objects):
        """
        Computes the lagrangian. Note that this is the Lagrangian in the sense of optimal control, i.e.,

        L = T - U,

        where T is the kinetic energy (here some norm on the parameters governing the state propagation/advection) and
        U is the potential energy (which amounts to the costates paired with the right hand sides of the state advection equations),
        i,e. <p,dot_x>

        Returns a triple of scalars. The value of the Lagrangian as well as of the kinetic and the potential energies.

        :param t: current timepoint
        :param state_dict_of_dicts: SortedDict of SortedDict's containing the states
        :param costate_dict_of_dicts: SortedDict of SortedDict's containing the costates
        :param parameter_objects: SortedDict with all the parameters for the advection equation, stored as a SortedDict of instances which compute data transformations (for example linear layer or convolutional layer)
        :return: triple (value of lagrangian, value of the kinetic energy, value of the potential energy)
        """

        kinetic_energy = self.compute_kinetic_energy(t=t, state_dict_of_dicts=state_dict_of_dicts,
                                                         costate_dict_of_dicts=costate_dict_of_dicts,
                                                         parameter_objects=parameter_objects)
        potential_energy = self.compute_potential_energy(t=t, state_dict_of_dicts=state_dict_of_dicts,
                                                         costate_dict_of_dicts=costate_dict_of_dicts,
                                                         parameter_objects=parameter_objects)

        lagrangian = kinetic_energy-potential_energy

        return lagrangian, kinetic_energy, potential_energy