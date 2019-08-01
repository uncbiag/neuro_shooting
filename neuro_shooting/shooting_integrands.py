import torch
import torch.nn as nn
import torch.autograd as autograd

from abc import ABCMeta, abstractmethod
# may require conda install sortedcontainers
from sortedcontainers import SortedDict

import neuro_shooting.activation_functions_and_derivatives as ad

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

        self._parameter_objects = None
        """Hierarchical dictionary for the parameters (stored within the repspective nn.Modules)"""

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


    def _assemble_generic_dict(self,d):
        """
        Given a SortedDict returns its vectorized version and a plan (a dictionary of sizes) on how to reassemble it.
        Given an empty directory it will return None, None

        :param d: sorted dictionary to vectorize
        :return: tuple: vectorized dictionary, assembly plan
        """

        # d is a sorted dictionary
        # first test that this assumption is true
        if type(d)!=SortedDict:
            raise ValueError('Expected a SortedDict, but got {}'.format(type(d)))

        d_list = []
        assembly_plan = SortedDict()
        for k in d:
            d_list.append(d[k].view(-1)) # entirely flatten is (shape is stored by assembly plan)
            assembly_plan[k] = d[k].shape

        if len(d_list)>0:
            ret = torch.cat(tuple(d_list))
        else:
            # was an empty directory
            ret = None
            assembly_plan = None

        return ret, assembly_plan

    def _assemble_generic_dict_of_dicts(self,d):
        """
        Similar to _assemble_generic_dict, but works for SortedDict's which contain other SortedDicts (as needed for the
        state and the costate).

        :param d: Input dictionary containing a SortedDict (which itself contains SortedDict entries).
        :return: returns a tuple of a vectorized dictionary and the associated assembly plan
        """

        assembly_plans = dict()
        ret = None
        for dk in d:
            current_ret,current_plan = self._assemble_generic_dict(d[dk])
            if current_ret is not None:
                assembly_plans[dk] = current_plan
                if ret is None:
                    ret = current_ret
                else:
                    ret = torch.cat((ret,current_ret))
        return ret, assembly_plans

    def assemble_tensor(self,state_dict_of_dicts,costate_dict_of_dicts,data_dict):
        """
        Vectorize all dictionaries together (state, costate, and data). Also returns all their assembly plans.

        :param state_dict: SortedDict holding the SortedDict's of the states
        :param costate_dict: SortedDict holding the SortedDict's of the costate
        :param data_dict: SortedDict holding the state for the transported data
        :return: vectorized dictonaries (as one vecctor) and their assembly plans
        """

        # these are all ordered dictionaries, will assemble all into a big torch vector
        state_vector,state_assembly_plans = self._assemble_generic_dict_of_dicts(state_dict_of_dicts)
        costate_vector,costate_assembly_plans = self._assemble_generic_dict_of_dicts(costate_dict_of_dicts)
        data_vector,data_assembly_plan = self._assemble_generic_dict(data_dict)

        assembly_plans = dict()
        assembly_plans['state_dicts'] = state_assembly_plans
        assembly_plans['costate_dicts'] = costate_assembly_plans
        assembly_plans['data_dict'] = data_assembly_plan

        return torch.cat((state_vector,costate_vector,data_vector)),assembly_plans

    @abstractmethod
    def disassemble(self, input):
        """
        Abstract emthod which needs to be implemented. Takes an input vector and shoud disassemble so that it can return
        the desired part of the state (for example only the position). Implementation will likely make use of
        disassmble_tensor to implement this.

        :param input: input tensor
        :return: desired part of the state vector
        """

        #Is supposed to return the desired data state (possibly only one) from an input vector
        pass


    def _disassemble_dict(self, input, assembly_plan, dim, incr):
        """
        Disassembles an input vector into its corresponding dictionary structure, given an assembly plan, a dimension,
        and an increment (as to where to start in the input vector).

        :param input: Input vector
        :param assembly_plan: Assembly plan to disassemble the vector into the dictionary (as created by assemble_tensor).
        :param dim: dimension (should typically be set to 0, set it to 1 in case there are for example multiple time-points stored in the zero dimension.
        :param incr: offset (specifying where to start in the vector)
        :return: tuple of the disassembled dictionary (as a SortedDict) and the increment (incr) indicating a new starting location (for the next call to _dissassemble)
        """

        ret_dict = SortedDict()

        for k in assembly_plan:
            current_shape = assembly_plan[k]
            len_shape = torch.prod(torch.tensor(current_shape)).item()

            if dim == 0:
                ret_dict[k] = (input[incr:incr + len_shape]).view(current_shape)
            elif dim == 1:
                first_dim = input.shape[0]
                all_shape = torch.Size([first_dim] + list(current_shape))
                ret_dict[k] = (input[:, incr:incr + len_shape]).view(all_shape)
            else:
                raise ValueError('Currently only supported for dims 0 or 1, but got dim = {}'.format(dim))

            incr += len_shape

        return ret_dict, incr


    def _disassemble_dict_of_dicts(self, input, assembly_plans, dim, incr):
        """
        Similar to _disassemble_dict, but applies to a dictionary of dictionaries that is supposed to be disassembled.

        :param input: Input vector
        :param assembly_plans: Assembly plan to disassemble the vector into the dictionary of dictionaries (as created by assemble_tensor).
        :param dim: dimension (should typically be set to 0, set it to 1 in case there are for example multiple time-points stored in the zero dimension.
        :param incr: offset (specifying where to start in the vector)
        :return: tuple of the disassembled dictionary of dictionaries (as a SortedDict) and the increment (incr) indicating a new starting location (for the next call to _dissassemble)
        """

        ret_dict_of_dicts = SortedDict()

        for cp in assembly_plans:
            assembly_plan = assembly_plans[cp]
            ret_dict_of_dicts[cp], incr = self._disassemble_dict(input=input, assembly_plan=assembly_plan, dim=dim,
                                                                 incr=incr)

        return ret_dict_of_dicts, incr


    def disassemble_tensor(self, input, assembly_plans=None, dim=0):
        """
        Disassembles an input vector into state, data, and costate directories.

        :param input: input tensor (vector)
        :param assembly_plans: assembly_plans (does not need to be specified if previously computed -- will be cached)
        :param dim: integrator may add a 0-th dimension to keep track of time. In this case use dim=1, otherwise dim=0 should be fine.
        :return: tuple holding the state, costate, and data dictionaries
        """

        # will create sorted dictionaries for state, costate and data based on the assembly plans

        supported_dims = [0,1]
        if dim not in supported_dims:
            raise ValueError('Only supports dimensions 0 and 1; if 1, then the 0-th dimension is time')


        if assembly_plans is None:
            if self.auto_assembly_plans is None:
                raise ValueError('No assembly plan specified and none was previously stored automatically (for example by calling get_initial_conditions_from_data_dict).')
            else:
                assembly_plans = self.auto_assembly_plans

        state_dicts = None
        costate_dicts = None
        data_dict = None

        incr = 0
        for ap in ['state_dicts','costate_dicts','data_dict']:

            if ap=='state_dicts':
                state_dicts, incr = self._disassemble_dict_of_dicts(input=input, assembly_plans = assembly_plans[ap], dim=dim, incr=incr)
            elif ap=='costate_dicts':
                costate_dicts, incr = self._disassemble_dict_of_dicts(input=input, assembly_plans = assembly_plans[ap], dim=dim, incr=incr)
            elif ap=='data_dict':
                data_dict, incr = self._disassemble_dict(input=input, assembly_plan = assembly_plans[ap], dim=dim, incr=incr)
            else:
                raise ValueError('Unknown dictionary assembly plan kind {}'.format(ap))

        return state_dicts,costate_dicts,data_dict


    def compute_potential_energy(self,state_dict_of_dicts,costate_dict_of_dicts,parameter_objects):
        """
        Computes the potential energy for the Lagrangian. I.e., it pairs the costates with the right hand sides of the
        state evolution equations. This method is typically not called manually. Everything should happen automatically here.

        :param state_dict_of_dicts: SortedDict of SortedDicts holding the states
        :param costate_dict_of_dicts: SortedDict of SortedDicts holding the costates
        :param parameter_objects: parameters to compute the current right hand sides, stored as a SortedDict of instances which compute data transformations (for example linear layer or convolutional layer).
        :return: returns the potential energy (as a pytorch variable)
        """

        # this is really only how one propagates through the system given the parameterization

        rhs_state_dict_of_dicts = self.rhs_advect_state_dict_of_dicts(state_dict_of_dicts=state_dict_of_dicts,parameter_objects=parameter_objects)

        potential_energy = 0

        for d_ks,d_kcs in zip(rhs_state_dict_of_dicts,costate_dict_of_dicts):
            c_rhs_state_dict = rhs_state_dict_of_dicts[d_ks]
            c_costate_dict = costate_dict_of_dicts[d_kcs]
            for ks,kcs in zip(c_rhs_state_dict,c_costate_dict):
                potential_energy = potential_energy + torch.mean(c_costate_dict[kcs]*c_rhs_state_dict[ks])

        return potential_energy

    def compute_kinetic_energy(self,parameter_objects):
        """
        Computes the kinetic energy. This is the kinetic energy given the parameters. Will only be relevant if the system is
        nonlinear in its parameters (when a fixed point solution needs to be computed). Otherwise will bot be used. By default just
        computes the sum of squares of the parameters which can be weighted via a scalar (given by the instance of the class
        defining the transformation).

        :todo: Add more general weightings to the classes (i.e., more than just scalars)

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

    def compute_lagrangian(self, state_dict_of_dicts, costate_dict_of_dicts, parameter_objects):
        """
        Computes the lagrangian. Note that this is the Lagrangian in the sense of optimal control, i.e.,

        L = T - U,

        where T is the kinetic energy (here some norm on the parameters governing the state propagation/advection) and
        U is the potential energy (which amounts to the costates paired with the right hand sides of the state advection equations),
        i,e. <p,dot_x>

        Returns a triple of scalars. The value of the Lagrangian as well as of the kinetic and the potential energies.

        :param state_dict_of_dicts: SortedDict of SortedDict's containing the states
        :param costate_dict_of_dicts: SortedDict of SortedDict's containing the costates
        :param parameter_objects: SortedDict with all the parameters for the advection equation, stored as a SortedDict of instances which compute data transformations (for example linear layer or convolutional layer)
        :return: triple (value of lagrangian, value of the kinetic energy, value of the potential energy)
        """

        kinetic_energy = self.compute_kinetic_energy(parameter_objects=parameter_objects)
        potential_energy = self.compute_potential_energy(state_dict_of_dicts=state_dict_of_dicts,
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


    def _merge_state_or_costate_dict_with_generic_dict_of_dicts(self,generic_dict,generic_dict_of_dicts):
        """
        To keep the interface reasonably easy it is often desired to add a state or costate dictionary
        to a dictionary of dictionaries (which already contains various state or costate dictionaries) to obtain
        a combined dictionary of dictionaries. As the entries of the dictonary of dictionaries are named (based on
        what block created them) the dictionary is added based on the name of the current block. It can only be
        addded if the name has not been used before. Hence it is essential to use unique names when using multiple
        blocks in a system that are being chained together.

        :param generic_dict: a Sorted Dict
        :param generic_dict_of_dicts: a SortedDict of SortedDicts's
        :return: merges both and returns a SortedDict of SortedDict's with the combined entries
        """

        if generic_dict_of_dicts is not None:
            if self._block_name in generic_dict_of_dicts:
                raise ValueError('Block name {} already taken. Cannot be added to dict of dicts.'.format(self._block_name))

        ret_dict_of_dicts = SortedDict()
        # now add the individual one
        if generic_dict is not None:
            ret_dict_of_dicts[self._block_name] = generic_dict

        if generic_dict_of_dicts is not None:
            # todo: maybe there is an easier way to copy these dictionary entries instead of always looping over the entries (copy? deepcopy?)
            # first create the same key structure as in the generic_dict_of_dicts (we are not copying as we do not want to change the keys of generic_dict_of_dicts)
            for dk in generic_dict_of_dicts:
                ret_dict_of_dicts[dk] = SortedDict()
                c_generic_dict = generic_dict_of_dicts[dk]
                c_ret_dict = ret_dict_of_dicts[dk]
                for k in c_generic_dict:
                    c_ret_dict[k] = c_generic_dict[k]

        return ret_dict_of_dicts

    def get_initial_conditions_from_data_dict(self,data_dict):
        """
        Given a data dictionary, this method creates a vector which contains the initial condition consisting of state, costate, and the data.
        As a side effect it also stores (caches) the created assembly plan so that it does not need to be specified when calling disassemble_tensor.

        :param data_dict: data dictionary
        :return: vector of initial conditions
        """

        state_dicts = self._merge_state_or_costate_dict_with_generic_dict_of_dicts(generic_dict=self._state_parameter_dict,
                                                                                   generic_dict_of_dicts=self._pass_through_state_parameter_dict_of_dicts)

        costate_dicts = self._merge_state_or_costate_dict_with_generic_dict_of_dicts(generic_dict=self._costate_parameter_dict,
                                                                                     generic_dict_of_dicts=self._pass_through_state_parameter_dict_of_dicts)

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

    def extract_dict_from_tuple_based_on_generic_dict(self,data_tuple,generic_dict,prefix=''):
        """
        Autodiff's autograd requires specifying variables to differentiate with respect to as tuples. Hence, this is a convenience
        function which takes such a tuple and generates a dictionary from it based on the dictionary structure defined via
        generic_dict (it's values are not used). If desired a prefix for the generated dictionary keys can be specified.

        :param data_tuple: tuple of variables
        :param generic_dict: SortedDict providing the desired dictionary structure (ideally the one used to create the tuple in the first place)
        :param prefix: text prefix for the generated keys
        :return: returns a SortedDict containing the data from the tuple
        """

        extracted_dict = SortedDict()
        indx = 0
        for k in generic_dict:
            extracted_dict[prefix+k] = data_tuple[indx]
            indx += 1

        return extracted_dict

    def extract_dict_of_dicts_from_tuple_based_on_generic_dict_of_dicts(self,data_tuple,generic_dict_of_dicts,prefix=''):
        """
        Similar to extract_dict_from_tuple_based_on_generic_dict, but creates dict_of_dicts from the tuple. I.e., the tuple
        must have been created from a SortedDict containing SortedDict's.

        :param data_tuple: tuple of variables
        :param generic_dict_of_dicts: SortedDict of SortedDict's providing the desired dictionary structure (ideally the one used to create the tuple in the first place)
        :param prefix: text prefix for the generated keys
        :return: returns a SortedDict containing the data from the tuple
        """

        extracted_dicts = SortedDict()
        indx = 0
        for k in generic_dict_of_dicts:
            extracted_dicts[k] = SortedDict()
            c_extracted_dict = extracted_dicts[k]
            c_generic_dict = generic_dict_of_dicts[k]
            for m in c_generic_dict:
                c_extracted_dict[prefix+m] = data_tuple[indx]
                indx +=1
        return extracted_dicts

    def extract_dict_from_tuple_based_on_parameter_objects(self,data_tuple,parameter_objects,prefix=''):
        """
        Similar to extract_dict_from_tuple_based_on_generic_dict, but is based on the SortedDict of parameter objects
        (which contain all the parameters required to evolve the states; e.g., convolutional filter coefficients.)

        :param data_tuple: tuple of variables
        :param parameter_objects: SortedDict of parameter objects
        :param prefix: text prefix for the generated keys
        :return: returns a SortedDict containing the data from the tuple
        """

        extracted_dict = SortedDict()
        indx = 0

        for o in parameter_objects:
            extracted_dict[o] = SortedDict()
            current_extracted_dict = extracted_dict[o]
            current_pars = parameter_objects[o].get_parameter_dict()

            for k in current_pars:
                current_extracted_dict[prefix+k] = data_tuple[indx]
                indx += 1

        return extracted_dict

    def compute_tuple_from_generic_dict_of_dicts(self,generic_dict_of_dicts):
        """
        Given a SortedDict of SortedDict's (e.g., for the states or the costates) this method returns a tuple of its entries.

        :param generic_dict_of_dicts: SortedDict of SortedDict's (for example holding the states or the costates)
        :return: Returns a tuple of the dictionary entries
        """

        # form a tuple of all the state variables (because this is what we take the derivative of)
        sv_list = []
        for k in generic_dict_of_dicts:
            c_generic_dict = generic_dict_of_dicts[k]
            for m in c_generic_dict:
                sv_list.append(c_generic_dict[m])

        return tuple(sv_list)

    def compute_tuple_from_generic_dict(self,generic_dict):
        """

        :param generic_dict:
        :return:
        """

        # form a tuple of all the state variables (because this is what we take the derivative of)
        sv_list = []
        for k in generic_dict:
            sv_list.append(generic_dict[k])

        return tuple(sv_list)

    def compute_tuple_from_parameter_objects(self,parameter_objects):
        # form a tuple of all the variables (because this is what we take the derivative of)

        sv_list = []
        for o in parameter_objects:
            current_pars = parameter_objects[o].get_parameter_dict()
            for k in current_pars:
                sv_list.append((current_pars[k]))

        return tuple(sv_list)

    def _concatenate_dict_of_dicts(self,generic_dict_of_dicts):
        concatenated_dict = SortedDict()
        for dk in generic_dict_of_dicts:
            c_generic_dict = generic_dict_of_dicts[dk]
            for k in c_generic_dict:
                if k not in concatenated_dict:
                    concatenated_dict[k] = c_generic_dict[k]
                else:
                    # should be concatenated along the feature channel

                    t_shape1 = concatenated_dict[k].size()
                    t_shape2 = c_generic_dict[k].size()

                    try:
                        concatenated_dict[k] = torch.cat((concatenated_dict[k],c_generic_dict[k]),dim=1)
                    except:
                        raise ValueError('Dimension mismatch when trying to concatenate tensor of shape {} and {} along dimension 1.'.format(t_shape1,t_shape2))

        # lastly check that we have the same number of keys, otherwise throw an error
        nr_of_resulting_keys = len(concatenated_dict.keys())
        nr_of_expected_keys = len((generic_dict_of_dicts.peekitem(0)[1]).keys())

        if nr_of_resulting_keys!=nr_of_expected_keys:
            raise ValueError('Expected {} different keys, but got {}.'.format(nr_of_expected_keys,nr_of_resulting_keys))

        return concatenated_dict

    def _deconcatenate_based_on_generic_dict_of_dicts(self,concatenated_dict,generic_dict_of_dicts):
        # deconcatenate along dimension 1
        ret = SortedDict()
        indx = dict()
        for dk in generic_dict_of_dicts:
            ret[dk] = SortedDict()
            c_ret = ret[dk]
            c_generic_dict = generic_dict_of_dicts[dk]
            for kc,k in zip(concatenated_dict,c_generic_dict):
                if k not in indx:
                    indx[k]=0
                t_shape = c_generic_dict[k].size()
                c_ret[kc] = concatenated_dict[kc][:,indx[k]:indx[k]+t_shape[1],...]
                indx[k] += t_shape[1]
        return ret

    def rhs_advect_state_dict_of_dicts(self,state_dict_of_dicts,parameter_objects):
        if self.concatenate_parameters:
            state_dicts_concatenated = self._concatenate_dict_of_dicts(generic_dict_of_dicts=state_dict_of_dicts)
            rhs_state_dict = self.rhs_advect_state(state_dict_or_dict_of_dicts=state_dicts_concatenated, parameter_objects=parameter_objects)
            rhs_state_dict_of_dicts = self._deconcatenate_based_on_generic_dict_of_dicts(rhs_state_dict,generic_dict_of_dicts=state_dict_of_dicts)
            #rhs_state_dict_of_dicts = SortedDict({self._block_name:rhs_state_dict})
            return rhs_state_dict_of_dicts
        else:
            return self.rhs_advect_state(state_dict_or_dict_of_dicts=state_dict_of_dicts,
                                         parameter_objects=parameter_objects)

    @abstractmethod
    def rhs_advect_state(self, state_dict_or_dict_of_dicts, parameter_objects):
        pass

    def rhs_advect_data(self,data_dict,parameter_objects):
        # wrap it
        c_data_state_dicts = SortedDict({'data':data_dict})
        ret_dict_of_dicts = self.rhs_advect_state_dict_of_dicts(state_dict_of_dicts=c_data_state_dicts,parameter_objects=parameter_objects)
        # unwrap it
        ret = ret_dict_of_dicts['data']
        return ret

    @abstractmethod
    def rhs_advect_costate(self, state_dict_of_dicts, costate_dict_of_dicts, parameter_objects):
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
    def compute_parameters(self,parameter_objects,state_dict,costate_dict):
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

    def compute_gradients(self,state_dict_of_dicts,costate_dict_of_dicts,data_dict):

        if self._parameter_objects is None:
            self._parameter_objects = self.create_default_parameter_objects()
        else:
            # detaching here is important as otherwise the gradient is accumulated and we only want this to store
            # the current parameters
            self.detach_and_require_gradients_for_parameter_objects(self._parameter_objects)

        current_kinetic_energy = self.compute_parameters(parameter_objects=self._parameter_objects,state_dict_of_dicts=state_dict_of_dicts,costate_dict_of_dicts=costate_dict_of_dicts)

        self.current_norm_penalty = current_kinetic_energy

        dot_state_dict_of_dicts = self.rhs_advect_state_dict_of_dicts(state_dict_of_dicts=state_dict_of_dicts,parameter_objects=self._parameter_objects)
        dot_data_dict = self.rhs_advect_data(data_dict=data_dict,parameter_objects=self._parameter_objects)
        dot_costate_dict_of_dicts = self.rhs_advect_costate(state_dict_of_dicts=state_dict_of_dicts,costate_dict_of_dicts=costate_dict_of_dicts,parameter_objects=self._parameter_objects)

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

    def forward(self, t, input):

        state_dict_of_dicts,costate_dict_of_dicts,data_dict = self.disassemble_tensor(input)

        if self.transpose_state_when_forward:
            state_dict_of_dicts = self.transpose_state_dict_of_dicts(state_dict_of_dicts)
            costate_dict_of_dicts = self.transpose_state_dict_of_dicts(costate_dict_of_dicts)
            data_dict = self.transpose_state(data_dict)

        # computing the gradients
        dot_state_dict_of_dicts, dot_costate_dict_of_dicts, dot_data_dict = \
            self.compute_gradients(state_dict_of_dicts=state_dict_of_dicts,costate_dict_of_dicts=costate_dict_of_dicts,data_dict=data_dict)

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

    def rhs_advect_costate(self, state_dict_of_dicts, costate_dict_of_dicts, parameter_objects):
        # now that we have the parameters we can get the rhs for the costate using autodiff

        current_lagrangian, current_kinetic_energy, current_potential_energy = \
            self.compute_lagrangian(state_dict_of_dicts=state_dict_of_dicts, costate_dict_of_dicts=costate_dict_of_dicts, parameter_objects=parameter_objects)

        # form a tuple of all the state variables (because this is what we take the derivative of)
        state_tuple = self.compute_tuple_from_generic_dict_of_dicts(state_dict_of_dicts)

        dot_costate_tuple = autograd.grad(current_lagrangian, state_tuple,
                                          grad_outputs=current_lagrangian.data.new(current_lagrangian.shape).fill_(1),
                                          create_graph=True,
                                          retain_graph=True,
                                          allow_unused=True)

        # now we need to put these into a sorted dictionary
        dot_costate_dict_of_dicts = self.extract_dict_of_dicts_from_tuple_based_on_generic_dict_of_dicts(data_tuple=dot_costate_tuple,
                                                                              generic_dict_of_dicts=state_dict_of_dicts, prefix='dot_co_')

        return dot_costate_dict_of_dicts


class LinearInParameterAutogradShootingIntegrand(AutogradShootingIntegrandBase):
    def __init__(self, name, batch_y0=None, nonlinearity=None, only_random_initialization=False,transpose_state_when_forward=False,
                 concatenate_parameters=True,*args,**kwargs):
        super(LinearInParameterAutogradShootingIntegrand, self).__init__(name=name, batch_y0=batch_y0, nonlinearity=nonlinearity,
                                                                         only_random_initialization=only_random_initialization,
                                                                         transpose_state_when_forward=transpose_state_when_forward,
                                                                         concatenate_parameters=concatenate_parameters,
                                                                         *args, **kwargs)

    def compute_parameters_directly(self, parameter_objects, state_dict_of_dicts, costate_dict_of_dicts):
        # we assume this is linear here, so we do not need a fixed point iteration, but can just compute the gradient

        current_lagrangian, current_kinetic_energy, current_potential_energy = \
            self.compute_lagrangian(state_dict_of_dicts=state_dict_of_dicts, costate_dict_of_dicts=costate_dict_of_dicts, parameter_objects=parameter_objects)

        parameter_tuple = self.compute_tuple_from_parameter_objects(parameter_objects)

        parameter_grad_tuple = autograd.grad(current_potential_energy,
                                             parameter_tuple,
                                             grad_outputs=current_potential_energy.data.new(
                                                 current_potential_energy.shape).fill_(1),
                                             create_graph=True,
                                             retain_graph=True,
                                             allow_unused=True)

        parameter_grad_dict = self.extract_dict_from_tuple_based_on_parameter_objects(data_tuple=parameter_grad_tuple,
                                                                                 parameter_objects=parameter_objects,
                                                                                 prefix='grad_')

        self.negate_divide_and_store_in_parameter_objects(parameter_objects=parameter_objects,generic_dict=parameter_grad_dict)

        return current_kinetic_energy

    def compute_parameters(self,parameter_objects,state_dict_of_dicts,costate_dict_of_dicts):
        return self.compute_parameters_directly(parameter_objects=parameter_objects,state_dict_of_dicts=state_dict_of_dicts,costate_dict_of_dicts=costate_dict_of_dicts)


class NonlinearInParameterAutogradShootingIntegrand(AutogradShootingIntegrandBase):
    def __init__(self, name, batch_y0=None, nonlinearity=None, only_random_initialization=False,transpose_state_when_forward=False,
                 concatenate_parameters=True,*args,**kwargs):
        super(NonlinearInParameterAutogradShootingIntegrand, self).__init__(name=name, batch_y0=batch_y0, nonlinearity=nonlinearity,
                                                                            only_random_initialization=only_random_initialization,
                                                                            transpose_state_when_forward=transpose_state_when_forward,
                                                                            concatenate_parameters=concatenate_parameters,
                                                                            *args, **kwargs)

    def compute_parameters_iteratively(self, parameter_objects, state_dict_of_dicts, costate_dict_of_dicts):

        learning_rate = 0.5
        nr_of_fixed_point_iterations = 5

        for n in range(nr_of_fixed_point_iterations):
            current_lagrangian, current_kinectic_energy, current_potential_energy = \
                self.compute_lagrangian(state_dict_of_dicts=state_dict_of_dicts, costate_dict_of_dicts=costate_dict_of_dicts, parameter_objects=parameter_objects)

            parameter_tuple = self.compute_tuple_from_parameter_objects(parameter_objects)

            parameter_grad_tuple = autograd.grad(current_lagrangian,
                                                 parameter_tuple,
                                                 grad_outputs=current_lagrangian.data.new(
                                                     current_lagrangian.shape).fill_(1),
                                                 create_graph=True,
                                                 retain_graph=True,
                                                 allow_unused=True)

            parameter_grad_dict = self.extract_dict_from_tuple_based_on_parameter_objects(data_tuple=parameter_grad_tuple,
                                                                                     parameter_objects=parameter_objects,
                                                                                     prefix='grad_')

            self.add_multiple_to_parameter_objects(parameter_objects=parameter_objects,
                                                   pd_from=parameter_grad_dict, multiplier=-learning_rate)

        return current_kinectic_energy

    def compute_parameters(self,parameter_objects,state_dict_of_dicts,costate_dict_of_dicts):
        return self.compute_parameters_iteratively(parameter_objects=parameter_objects,state_dict_of_dicts=state_dict_of_dicts,costate_dict_of_dicts=costate_dict_of_dicts)


