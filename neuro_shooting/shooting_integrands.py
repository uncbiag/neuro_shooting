import torch
import torch.nn as nn
import torch.autograd as autograd

from abc import ABCMeta, abstractmethod
# may require conda install sortedcontainers
from sortedcontainers import SortedDict
from collections import OrderedDict, namedtuple
import torch.utils.hooks as hooks

import neuro_shooting.activation_functions_and_derivatives as ad
import neuro_shooting.state_costate_and_data_dictionary_utils as scd_utils

class ShootingIntegrandBase(nn.Module):
    """
    Base class for shooting based neural ODE approaches
    """
    def __init__(self, name, batch_y0=None, nonlinearity=None, only_random_initialization=False, transpose_state_when_forward=False,
                 concatenate_parameters=True,*args, **kwargs):
        """
        Constructor
        :param unique name of this block (needed to keep track of the parameters when there is pass through)
        :param batch_y0: example batch, can be used to construct initial conditions for patches
        :param nonlinearity: desired nonlinearity to be used tanh, sigmoid, relu, ...
        :param only_random_initialization: just a flag passed on to the initialization of the state and costate
        :param transpose_state_when_forward: if set to true states get transposed (1,2) at the beginng and end of processing
        """
        super(ShootingIntegrandBase, self).__init__()

        self._block_name = name
        """Name of the shooting block"""

        self.nl, self.dnl = self._get_nonlinearity(nonlinearity=nonlinearity)
        """Nonlinearity and its derivative"""

        self._state_parameter_dict = None
        """Dictionary holding the state variables"""
        self._costate_parameter_dict = None
        """Dictionary holding the costates (i.e., adjoints/duals)"""

        self._pass_through_state_parameter_dict_of_dicts = None
        """State parameters that are passed in externally, but are not parameters to be optimized over"""
        self._pass_through_costate_parameter_dict_of_dicts = None
        """Costate parameters that are passed in externally, but are not parameters to be optimized over"""

        self.concatenate_parameters = concatenate_parameters
        """If set to True all state and costate dictionaries will be merged into one; only possible when dimension is consistent"""

        self._parameter_objects = None
        """Hierarchical dictionary for the parameters (stored within the repspective nn.Modules)"""

        self._lagrangian_gradient_hooks = OrderedDict()
        """Hooks called at any integration step"""

        self._custom_hook_data = None
        """Custom data that is passed to a hook (set via set_custom_hook_data)"""

        self.transpose_state_when_forward = transpose_state_when_forward

        # this allows to define one at some point and it will then be used going forward until it is reset
        self.auto_assembly_plans = None
        """Assembly plans, i.e., how to go from vectorized data to the named data structure"""

        # norm penalty
        self.current_norm_penalty = None
        """Will hold the last computed norm penality"""

        state_dict, costate_dict = self.create_initial_state_and_costate_parameters(batch_y0=batch_y0,
                                                                                    only_random_initialization=only_random_initialization,
                                                                                    *args,**kwargs)
        self.register_state_and_costate_parameters(state_dict=state_dict, costate_dict=costate_dict)

        if self._parameter_objects is None:
            self._parameter_objects = self.create_default_parameter_objects()

    def set_custom_hook_data(self,data):
        """
        Custom data that is passed to the shooting integrand hooks. In this way a hook can modify its behavior. For
        example, the current batch and epoch could be passed to the hook and the hook could react to it.

        :param data: data to be passed to the hooks (should be a dictionary, but can otherwise be arbitrary as long as the hooks know about the format)
        :return: n/a
        """

        if type(data)!=dict:
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
        handle = hooks.RemovableHandle(self._forward_hooks)
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
        for k in self._parameter_objects:
            print('Applying _apply, to {}'.format(k))
            self._parameter_objects[k]._apply(fn)
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
        for k in self._parameter_objects:
            print('Applying to, to {}'.format(k))
            self._parameter_objects[k].to(*args, **kwargs)
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
        return self.get_current_norm_penalty()

    def _get_nonlinearity(self, nonlinearity):
        """
        Returns the desired nonlinearity and its derivative as a tuple. Currently supported nonlinearities are:
        identity, relu, tanh, sigmoid, and softmax.

        :param nonlinearity: as a string: 'identity', 'relu', 'tanh', 'sigmoid', 'softmax'
        :return: tuple (nonlinearity,derivative of nonlinearity)
        """

        supported_nonlinearities = ['identity', 'relu', 'tanh', 'sigmoid', 'softmax']

        if nonlinearity is None:
            use_nonlinearity = 'identity'
        else:
            use_nonlinearity = nonlinearity.lower()

        if use_nonlinearity not in supported_nonlinearities:
            raise ValueError('Unsupported nonlinearity {}'.format(use_nonlinearity))

        if use_nonlinearity == 'relu':
            nl = nn.functional.relu
            dnl = ad.drelu
        elif use_nonlinearity == 'tanh':
            nl = torch.tanh
            dnl = ad.dtanh
        elif use_nonlinearity == 'identity':
            nl = ad.identity
            dnl = ad.didentity
        elif use_nonlinearity == 'sigmoid':
            nl = torch.sigmoid
            dnl = torch.sigmoid
        elif use_nonlinearity == 'softmax':
            nl = ad.softmax
            dnl = ad.dsoftmax
        else:
            raise ValueError('Unknown nonlinearity {}'.format(use_nonlinearity))

        return nl,dnl

    @abstractmethod
    def create_initial_state_parameters(self,batch_y0,only_random_initialization,*args,**kwargs):
        """
        Abstract method. Needs to be defined and needs to create and return a SortedDict containing the state variables.

        :param batch_y0: sample batch as input, can be used to initialize
        :param only_random_initialization: to indicate if purely random initialization is desired
        :return: SortedDict containing the state variables (e.g., SortedDict({'q1': torch.zeros(20,10,5)})
        """
        pass

    def set_initial_pass_through_state_parameters(self,state_dict_of_dicts):
        """
        If the parameters are not optimized over, this function can be used to set the initial conditions for the state variables.
        It is possible to pass through states and to create new ones (in this case they will be combined).

        :param state_dict_of_dicts: SortedDict of possibly multiple SortedDict's that hold the states.
        :return: n/a
        """
        self._pass_through_state_parameter_dict_of_dicts = state_dict_of_dicts

    def set_initial_pass_through_costate_parameters(self,costate_dict_of_dicts):
        """
        If the parameters are not optimized over, this function can be used to set the initial conditions for the costate variables.
        It is possible to pass through co-states and to create new ones (in this case they will be combined).

        :param costate_dict_of_dicts: SortedDict of possibly multiple SortedDict's that hold the costates.
        :return: n/a
        """
        self._pass_through_costate_parameter_dict_of_dicts = costate_dict_of_dicts

    def set_initial_pass_through_state_and_costate_parameters(self,state_dict_of_dicts,costate_dict_of_dicts):
        """
        Convenience function that allows setting the pass-through states and costates at the same time.
        Same as calling set_initial_pass_through_state_parameters and set_initial_pass_through_costate_parameters separately.

        :param state_dict_of_dicts: SortedDict of possibly multiple SortedDict's that hold the states.
        :param costate_dict_of_dicts: SortedDict of possibly multiple SortedDict's that hold the costates.
        :return: n/a
        """
        # if the parameters are not optimized over, this function can be used to set the initial conditions
        self.set_initial_pass_through_state_parameters(state_dict_of_dicts=state_dict_of_dicts)
        self.set_initial_pass_through_costate_parameters(costate_dict_of_dicts=costate_dict_of_dicts)


    def create_initial_costate_parameters(self,batch_y0=None,only_random_initialization=True,state_dict=None,*args,**kwargs):
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
            ValueError('This default method to create the initial costate requires to specify state_dict')

        rand_mag_p = 0.5

        costate_dict = SortedDict()
        for k in state_dict:
            costate_dict['p_' + str(k)] = nn.Parameter(rand_mag_p*torch.randn_like(state_dict[k]))

        return costate_dict

    def create_initial_state_and_costate_parameters(self,batch_y0,only_random_initialization,*args,**kwargs):
        """
        Creates the intial state and costate parameters. Should typically *not* be necessary to call this manually.

        :param batch_y0: sample batch as input, can be used to initialize
        :param only_random_initialization: to indicate if purely random initialization is desired
        :return: tuple of SortedDicts holding the state dictionary and the cotate dictionary
        """

        state_dict = self.create_initial_state_parameters(batch_y0=batch_y0,only_random_initialization=only_random_initialization,*args,**kwargs)
        costate_dict = self.create_initial_costate_parameters(batch_y0=batch_y0,only_random_initialization=only_random_initialization,state_dict=state_dict,*args,**kwargs)

        return state_dict,costate_dict

    def assemble_tensor(self, state_dict_of_dicts, costate_dict_of_dicts, data_dict):
        """
        Vectorize all dictionaries together (state, costate, and data). Also returns all their assembly plans.

        :param state_dict: SortedDict holding the SortedDict's of the states
        :param costate_dict: SortedDict holding the SortedDict's of the costate
        :param data_dict: SortedDict holding the state for the transported data
        :return: vectorized dictonaries (as one vecctor) and their assembly plans
        """

        ret_vec, assembly_plans = scd_utils.assemble_tensor(state_dict_of_dicts=state_dict_of_dicts,
                                                            costate_dict_of_dicts=costate_dict_of_dicts,
                                                            data_dict=data_dict)

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

        rhs_state_dict_of_dicts = self.rhs_advect_state_dict_of_dicts(t=t,state_dict_of_dicts=state_dict_of_dicts,parameter_objects=parameter_objects)

        potential_energy = 0

        for d_ks,d_kcs in zip(rhs_state_dict_of_dicts,costate_dict_of_dicts):
            c_rhs_state_dict = rhs_state_dict_of_dicts[d_ks]
            c_costate_dict = costate_dict_of_dicts[d_kcs]
            for ks,kcs in zip(c_rhs_state_dict,c_costate_dict):
                potential_energy = potential_energy + torch.mean(c_costate_dict[kcs]*c_rhs_state_dict[ks])

        return potential_energy

    def compute_kinetic_energy(self,t,parameter_objects):
        """
        Computes the kinetic energy. This is the kinetic energy given the parameters. Will only be relevant if the system is
        nonlinear in its parameters (when a fixed point solution needs to be computed). Otherwise will bot be used. By default just
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
    def get_initial_condition(self,x):
        """
        Abstract method to obtain the intial condition (as a vector) from a given data vector. It is likely easiest to
        implement by first building a data dictionary and then calling get_initial_conditions_from_data_dict(self,data_dict).

        :param x: data vector (tensor)
        :return: initial conditions as a vector (which can then be fed into a general integrator)
        """

        pass


    def get_initial_conditions_from_data_dict(self,data_dict):
        """
        Given a data dictionary, this method creates a vector which contains the initial condition consisting of state, costate, and the data.
        As a side effect it also stores (caches) the created assembly plan so that it does not need to be specified when calling disassemble_tensor.

        :param data_dict: data dictionary
        :return: vector of initial conditions
        """

        state_dicts = scd_utils._merge_state_or_costate_dict_with_generic_dict_of_dicts(generic_dict=self._state_parameter_dict,
                                                                                        generic_dict_of_dicts=self._pass_through_state_parameter_dict_of_dicts,
                                                                                        generic_dict_block_name=self._block_name)

        costate_dicts = scd_utils._merge_state_or_costate_dict_with_generic_dict_of_dicts(generic_dict=self._costate_parameter_dict,
                                                                                          generic_dict_of_dicts=self._pass_through_state_parameter_dict_of_dicts,
                                                                                          generic_dict_block_name=self._block_name)

        # initialize the second state of x with zero so far
        initial_conditions,assembly_plans = self.assemble_tensor(state_dict_of_dicts=state_dicts, costate_dict_of_dicts=costate_dicts, data_dict=data_dict)
        self.auto_assembly_plans = assembly_plans
        return initial_conditions

    @abstractmethod
    def create_default_parameter_objects(self):
        """
        Abstract method which should return a SortedDict which contains instances of the objects which are used to compute
        the state equations. These objects in turn contain the parameters. For example, overwritten convolutional or linear layer.

        :return: returns a SortedDict of parameter objects
        """
        raise ValueError('Not implemented. Needs to return a SortedDict of parameter objects')



    def rhs_advect_state_dict_of_dicts(self,t,state_dict_of_dicts,parameter_objects):
        if self.concatenate_parameters:
            state_dicts_concatenated = scd_utils._concatenate_dict_of_dicts(generic_dict_of_dicts=state_dict_of_dicts)
            rhs_state_dict = self.rhs_advect_state(t=t,state_dict_or_dict_of_dicts=state_dicts_concatenated, parameter_objects=parameter_objects)
            rhs_state_dict_of_dicts = scd_utils._deconcatenate_based_on_generic_dict_of_dicts(rhs_state_dict,generic_dict_of_dicts=state_dict_of_dicts)
            #rhs_state_dict_of_dicts = SortedDict({self._block_name:rhs_state_dict})
            return rhs_state_dict_of_dicts
        else:
            return self.rhs_advect_state(t=t,state_dict_or_dict_of_dicts=state_dict_of_dicts,
                                         parameter_objects=parameter_objects)

    @abstractmethod
    def rhs_advect_state(self, t, state_dict_or_dict_of_dicts, parameter_objects):
        pass

    def rhs_advect_data(self,t,data_dict,parameter_objects):
        # wrap it
        c_data_state_dicts = SortedDict({'data':data_dict})
        ret_dict_of_dicts = self.rhs_advect_state_dict_of_dicts(t=t,state_dict_of_dicts=c_data_state_dicts,parameter_objects=parameter_objects)
        # unwrap it
        ret = ret_dict_of_dicts['data']
        return ret

    @abstractmethod
    def rhs_advect_costate(self, t, state_dict_of_dicts, costate_dict_of_dicts, parameter_objects):
        # now that we have the parameters we can get the rhs for the costate using autodiff
        # returns a dictionary of the RHS of the costate
        pass

    def add_multiple_to_parameter_objects(self,parameter_objects,pd_from,multiplier=1.0):

        for o in parameter_objects:
            if o not in pd_from:
                ValueError('Needs to contain the same objects. Could not find {}.'.format(o))

            current_pars = parameter_objects[o].get_parameter_dict()
            current_from_pars = pd_from[o]

            for p,p_from in zip(current_pars,current_from_pars):
                current_pars[p] = current_pars[p]+multiplier*current_from_pars[p_from]


    def negate_divide_and_store_in_parameter_objects(self,parameter_objects,generic_dict):

        for o in parameter_objects:
            if o not in generic_dict:
                ValueError('Needs to contain the same objects. Could not find {}.'.format(o))

            current_pars = parameter_objects[o].get_parameter_dict()
            current_pars_from = generic_dict[o]
            current_weights = parameter_objects[o].get_parameter_weight_dict()

            for k,f in zip(current_pars,current_pars_from):
                if k not in current_weights:
                    current_pars[k] = -current_pars_from[f]
                else:
                    current_pars[k] = -current_weights[k]*current_pars_from[f]

    @abstractmethod
    def compute_parameters(self,t,parameter_objects,state_dict,costate_dict):
        """
        Computes parameters and stores them in parameter_objects. Returns the current kinectic energy (i.e., penalizer on parameters)
        :param state_dict:
        :param costate_dict:
        :return:
        """
        pass

    def detach_and_require_gradients_for_parameter_objects(self,parameter_objects):

        for o in parameter_objects:
            current_pars = parameter_objects[o].get_parameter_dict()
            for k in current_pars:
                current_pars[k] = current_pars[k].detach().requires_grad_(True)

    def compute_gradients(self,t,state_dict_of_dicts,costate_dict_of_dicts,data_dict):

        if self._parameter_objects is None:
            self._parameter_objects = self.create_default_parameter_objects()
        else:
            # detaching here is important as otherwise the gradient is accumulated and we only want this to store
            # the current parameters
            self.detach_and_require_gradients_for_parameter_objects(self._parameter_objects)

        current_kinetic_energy = self.compute_parameters(t=t,parameter_objects=self._parameter_objects,state_dict_of_dicts=state_dict_of_dicts,costate_dict_of_dicts=costate_dict_of_dicts)

        self.current_norm_penalty = current_kinetic_energy

        dot_state_dict_of_dicts = self.rhs_advect_state_dict_of_dicts(t=t,state_dict_of_dicts=state_dict_of_dicts,parameter_objects=self._parameter_objects)
        dot_data_dict = self.rhs_advect_data(t=t,data_dict=data_dict,parameter_objects=self._parameter_objects)
        dot_costate_dict_of_dicts = self.rhs_advect_costate(t=t,state_dict_of_dicts=state_dict_of_dicts,costate_dict_of_dicts=costate_dict_of_dicts,parameter_objects=self._parameter_objects)

        # run the hooks so we can get parameters, states, etc.; for example, to create tensorboard output
        for hook in self._lagrangian_gradient_hooks.values():
            hook(self, t, state_dict_of_dicts, costate_dict_of_dicts, data_dict,
                 dot_state_dict_of_dicts, dot_costate_dict_of_dicts, dot_data_dict,
                 self._parameter_objects, self._custom_hook_data)

        return dot_state_dict_of_dicts,dot_costate_dict_of_dicts,dot_data_dict


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

    def register_state_and_costate_parameters(self,state_dict,costate_dict):

        self._state_parameter_dict = state_dict

        if self._state_parameter_dict is not None:

            if type(self._state_parameter_dict) != SortedDict:
                raise ValueError('state parameter dictionrary needs to be an SortedDict and not {}'.format(
                    type(self._state_parameter_dict)))

            for k in self._state_parameter_dict:
                self.register_parameter(k,self._state_parameter_dict[k])

        self._costate_parameter_dict = costate_dict

        if self._costate_parameter_dict is not None:

            if type(self._costate_parameter_dict) != SortedDict:
                raise ValueError('costate parameter dictionrary needs to be an SortedDict and not {}'.format(
                    type(self._costate_parameter_dict)))

            for k in self._costate_parameter_dict:
                self.register_parameter(k,self._costate_parameter_dict[k])

    # todo: add possibility for time-dependency to everything here (currently it is just ignored)

    def forward(self, t, input):

        state_dict_of_dicts,costate_dict_of_dicts,data_dict = self.disassemble_tensor(input)

        if self.transpose_state_when_forward:
            state_dict_of_dicts = self.transpose_state_dict_of_dicts(state_dict_of_dicts)
            costate_dict_of_dicts = self.transpose_state_dict_of_dicts(costate_dict_of_dicts)
            data_dict = self.transpose_state(data_dict)

        # computing the gradients
        dot_state_dict_of_dicts, dot_costate_dict_of_dicts, dot_data_dict = \
            self.compute_gradients(t=t, state_dict_of_dicts=state_dict_of_dicts,costate_dict_of_dicts=costate_dict_of_dicts,data_dict=data_dict)

        if self.transpose_state_when_forward:
            # as we transposed the vectors before we need to transpose on the way back
            dot_state_dict_of_dicts = self.transpose_state_dict_of_dicts(dot_state_dict_of_dicts)
            dot_costate_dict_of_dicts = self.transpose_state_dict_of_dicts(dot_costate_dict_of_dicts)
            dot_data_dict = self.transpose_state(dot_data_dict)

        # create a vector out of this to pass to integrator
        output,assembly_plans = self.assemble_tensor(state_dict_of_dicts=dot_state_dict_of_dicts, costate_dict_of_dicts=dot_costate_dict_of_dicts, data_dict=dot_data_dict)

        return output

class AutogradShootingIntegrandBase(ShootingIntegrandBase):
    def __init__(self, name, batch_y0=None, nonlinearity=None, only_random_initialization=False,transpose_state_when_forward=False,
                 concatenate_parameters=True,*args,**kwargs):
        super(AutogradShootingIntegrandBase, self).__init__(name=name, batch_y0=batch_y0, nonlinearity=nonlinearity,
                                                            only_random_initialization=only_random_initialization,
                                                            transpose_state_when_forward=transpose_state_when_forward,
                                                            concatenate_parameters=concatenate_parameters,
                                                            *args, **kwargs)

    def rhs_advect_costate(self, t, state_dict_of_dicts, costate_dict_of_dicts, parameter_objects):
        # now that we have the parameters we can get the rhs for the costate using autodiff

        current_lagrangian, current_kinetic_energy, current_potential_energy = \
            self.compute_lagrangian(t=t, state_dict_of_dicts=state_dict_of_dicts, costate_dict_of_dicts=costate_dict_of_dicts, parameter_objects=parameter_objects)

        # form a tuple of all the state variables (because this is what we take the derivative of)
        state_tuple = scd_utils.compute_tuple_from_generic_dict_of_dicts(state_dict_of_dicts)

        dot_costate_tuple = autograd.grad(current_lagrangian, state_tuple,
                                          grad_outputs=current_lagrangian.data.new(current_lagrangian.shape).fill_(1),
                                          create_graph=True,
                                          retain_graph=True,
                                          allow_unused=True)

        # now we need to put these into a sorted dictionary
        dot_costate_dict_of_dicts = scd_utils.extract_dict_of_dicts_from_tuple_based_on_generic_dict_of_dicts(data_tuple=dot_costate_tuple,
                                                                              generic_dict_of_dicts=state_dict_of_dicts, prefix='dot_p_')

        return dot_costate_dict_of_dicts


class LinearInParameterAutogradShootingIntegrand(AutogradShootingIntegrandBase):
    def __init__(self, name, batch_y0=None, nonlinearity=None, only_random_initialization=False,transpose_state_when_forward=False,
                 concatenate_parameters=True,*args,**kwargs):
        super(LinearInParameterAutogradShootingIntegrand, self).__init__(name=name, batch_y0=batch_y0, nonlinearity=nonlinearity,
                                                                         only_random_initialization=only_random_initialization,
                                                                         transpose_state_when_forward=transpose_state_when_forward,
                                                                         concatenate_parameters=concatenate_parameters,
                                                                         *args, **kwargs)

    def compute_parameters_directly(self, t, parameter_objects, state_dict_of_dicts, costate_dict_of_dicts):
        # we assume this is linear here, so we do not need a fixed point iteration, but can just compute the gradient

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

        self.negate_divide_and_store_in_parameter_objects(parameter_objects=parameter_objects,generic_dict=parameter_grad_dict)

        return current_kinetic_energy

    def compute_parameters(self,t, parameter_objects,state_dict_of_dicts,costate_dict_of_dicts):
        return self.compute_parameters_directly(t=t, parameter_objects=parameter_objects,state_dict_of_dicts=state_dict_of_dicts,costate_dict_of_dicts=costate_dict_of_dicts)


class NonlinearInParameterAutogradShootingIntegrand(AutogradShootingIntegrandBase):
    def __init__(self, name, batch_y0=None, nonlinearity=None, only_random_initialization=False,transpose_state_when_forward=False,
                 concatenate_parameters=True,*args,**kwargs):
        super(NonlinearInParameterAutogradShootingIntegrand, self).__init__(name=name, batch_y0=batch_y0, nonlinearity=nonlinearity,
                                                                            only_random_initialization=only_random_initialization,
                                                                            transpose_state_when_forward=transpose_state_when_forward,
                                                                            concatenate_parameters=concatenate_parameters,
                                                                            *args, **kwargs)

    def compute_parameters_iteratively(self, t, parameter_objects, state_dict_of_dicts, costate_dict_of_dicts):

        learning_rate = 0.5
        nr_of_fixed_point_iterations = 5

        for n in range(nr_of_fixed_point_iterations):
            current_lagrangian, current_kinectic_energy, current_potential_energy = \
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

        return current_kinectic_energy

    def compute_parameters(self,t, parameter_objects,state_dict_of_dicts,costate_dict_of_dicts):
        return self.compute_parameters_iteratively(t=t, parameter_objects=parameter_objects,state_dict_of_dicts=state_dict_of_dicts,costate_dict_of_dicts=costate_dict_of_dicts)


