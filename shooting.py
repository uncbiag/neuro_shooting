import torch
import torch.nn as nn
import torch.autograd as autograd

from abc import ABCMeta, abstractmethod
# may require conda install sortedcontainers
from sortedcontainers import SortedDict

def softmax(x,epsilon = 1.0):
  return x*(torch.ones_like(x))/(torch.exp(-x*epsilon) + torch.ones_like(x))


def dsoftmax(x,epsilon = 1.0):
  return epsilon*softmax(x,epsilon)*(torch.ones_like(x))/(torch.exp(epsilon*x) + torch.ones_like(x)) + (torch.ones_like(x))/(torch.exp(-x*epsilon) + torch.ones_like(x))

def drelu(x):
    # derivative of relu
    res = (x>=0)
    res = res.type(x.type())
    return res

def dtanh(x):
    # derivative of tanh
    return 1.0-torch.tanh(x)**2

def identity(x):
    return x

def didentity(x):
    return torch.ones_like(x)


class ShootingBlockBase(nn.Module):
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
        super(ShootingBlockBase, self).__init__()

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
        """Hierarchical dictionary for the parameters (stored within the repspective nn.Modules"""

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
        super(ShootingBlockBase, self)._apply(fn)
        # make sure that all the filters that were created get moved
        for k in self._parameter_objects:
            print('Applying _apply, to {}'.format(k))
            self._parameter_objects[k]._apply(fn)
        return self

    def to(self, *args, **kwargs):
        super(ShootingBlockBase,self).to(*args, **kwargs)
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
        Returns the desired nonlinearity and its derivative as a tuple

        :param nonlinearity:
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
            dnl = drelu
        elif use_nonlinearity == 'tanh':
            nl = torch.tanh
            dnl = dtanh
        elif use_nonlinearity == 'identity':
            nl = identity
            dnl = didentity
        elif use_nonlinearity == 'sigmoid':
            nl = torch.sigmoid
            dnl = torch.sigmoid
        elif use_nonlinearity == 'softmax':
            nl = softmax
            dnl = dsoftmax
        else:
            raise ValueError('Unknown nonlinearity {}'.format(use_nonlinearity))

        return nl,dnl

    @abstractmethod
    def create_initial_state_parameters(self,batch_y0,only_random_initialization,*args,**kwargs):
        """
        Abstract method. Needs to be defined and needs to create and return a SortedDict as a tuple (state_dict,costate_dict)

        :param batch_y0: sample batch as input, can be used to initialize
        :param only_random_initialization: to indicate if purely random initialization is desired
        :return:
        """
        pass

    def set_initial_pass_through_state_parameters(self,state_dict_of_dicts):
        # if the parameters are not optimized over, this function can be used to set the initial conditions
        # are combined with parameters specified 
        self._pass_through_state_parameter_dict_of_dicts = state_dict_of_dicts

    def set_initial_pass_through_costate_parameters(self,costate_dict_of_dicts):
        # if the parameters are not optimized over, this function can be used to set the initial conditions
        self._pass_through_costate_parameter_dict_of_dicts = costate_dict_of_dicts

    def set_initial_pass_through_state_and_costate_parameters(self,state_dict_of_dicts,costate_dict_of_dicts):
        # if the parameters are not optimized over, this function can be used to set the initial conditions
        self.set_initial_pass_through_state_parameters(state_dict_of_dicts=state_dict_of_dicts)
        self.set_initial_pass_through_costate_parameters(costate_dict_of_dicts=costate_dict_of_dicts)


    def create_initial_costate_parameters(self,batch_y0,only_random_initialization,state_dict=None,*args,**kwargs):
        """
        Overwrite this method if you want to do something custom

        :param batch_y0:
        :param only_random_initialization:
        :param state_dict: state dictionary which can be used to create the corresponding costate
        :return:
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
        Abstract method. Needs to be defined and needs to create and return a SortedDict as a tuple (state_dict,costate_dict)

        :param batch_y0: sample batch as input, can be used to initialize
        :param only_random_initialization: to indicate if purely random initialization is desired
        :return:
        """

        state_dict = self.create_initial_state_parameters(batch_y0=batch_y0,only_random_initialization=only_random_initialization,*args,**kwargs)
        costate_dict = self.create_initial_costate_parameters(batch_y0=batch_y0,only_random_initialization=only_random_initialization,state_dict=state_dict,*args,**kwargs)

        return state_dict,costate_dict


    def _assemble_generic_dict(self,d):
        """
        Given a SortedDict returns its vectorized version and a plan (a dictionary of sizes) on how to reassemble it.

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

        ret = torch.cat(tuple(d_list))

        return ret, assembly_plan

    def _assemble_generic_dict_of_dicts(self,d):
        assembly_plans = dict()
        ret = None
        for dk in d:
            current_ret,assembly_plans[dk] = self._assemble_generic_dict(d[dk])
            if ret is None:
                ret = current_ret
            else:
                ret = torch.cat((ret,current_ret))
        return ret, assembly_plans

    def assemble_tensor(self,state_dict_of_dicts,costate_dict_of_dicts,data_dict):
        """
        Vectorize all dictionaries togehter (state, costatem and data). Also returns all their assembly plans.

        :param state_dict: SortedDict holding the state
        :param costate_dict: SortedDict holding the costate
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
        Abstract emthod which needs to be implemented. Takes an input and shoud disassemble so that it can return
        the desired part of the state (for example only the position). Implementation will likely make use of
        disassmble_tensor to implement this.

        :param input: input tensor
        :return: desired part of the state vector
        """
        #Is supposed to return the desired data state (possibly only one) from an input vector
        pass


    def _disassemble_dict(self, input, assembly_plan, dim, incr):

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

        kinetic_energy = self.compute_kinetic_energy(parameter_objects=parameter_objects)
        potential_energy = self.compute_potential_energy(state_dict_of_dicts=state_dict_of_dicts,
                                                         costate_dict_of_dicts=costate_dict_of_dicts,
                                                         parameter_objects=parameter_objects)

        lagrangian = kinetic_energy-potential_energy

        return lagrangian, kinetic_energy, potential_energy

    @abstractmethod
    def get_initial_condition(self,x):
        # Initial condition from given data vector
        # easiest to first build a data dictionary and then call get_initial_conditions_from_data_dict(self,data_dict):
        pass


    def _merge_state_or_costate_dict_with_generic_dict_of_dicts(self,generic_dict,generic_dict_of_dicts):

        if generic_dict_of_dicts is not None:
            if self._block_name in generic_dict_of_dicts:
                raise ValueError('Block name {} already taken. Cannot be added to dict of dicts.'.format(self._block_name))

        ret_dict_of_dicts = SortedDict()
        # now add the individual one
        if generic_dict is not None:
            ret_dict_of_dicts[self._block_name] = generic_dict

        if generic_dict_of_dicts is not None:
            # first create the same key structure as in the generic_dict_of_dicts (we are not copying as we do not want to change the keys of generic_dict_of_dicts)
            for dk in generic_dict_of_dicts:
                ret_dict_of_dicts[dk] = SortedDict()
                c_generic_dict = generic_dict_of_dicts[dk]
                c_ret_dict = ret_dict_of_dicts[dk]
                for k in c_generic_dict:
                    c_ret_dict[k] = c_generic_dict[k]

        return ret_dict_of_dicts

    def get_initial_conditions_from_data_dict(self,data_dict):

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
        raise ValueError('Not implemented. Needs to return a SortedDict of parameters')

    def extract_dict_from_tuple_based_on_generic_dict(self,data_tuple,generic_dict,prefix=''):
        extracted_dict = SortedDict()
        indx = 0
        for k in generic_dict:
            extracted_dict[prefix+k] = data_tuple[indx]
            indx += 1

        return extracted_dict

    def extract_dict_of_dicts_from_tuple_based_on_generic_dict_of_dicts(self,data_tuple,generic_dict_of_dicts,prefix=''):
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
        # form a tuple of all the state variables (because this is what we take the derivative of)
        sv_list = []
        for k in generic_dict_of_dicts:
            c_generic_dict = generic_dict_of_dicts[k]
            for m in c_generic_dict:
                sv_list.append(c_generic_dict[m])

        return tuple(sv_list)

    def compute_tuple_from_generic_dict(self,generic_dict):
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
        indx = 0
        for dk in generic_dict_of_dicts:
            ret[dk] = SortedDict()
            c_ret = ret[dk]
            c_generic_dict = generic_dict_of_dicts[dk]
            for kc,k in zip(concatenated_dict,c_generic_dict):
                t_shape = c_generic_dict[k].size()
                c_ret[kc] = concatenated_dict[kc][:,indx:indx+t_shape[1],...]
                indx += t_shape[1]
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

class AutogradShootingBlockBase(ShootingBlockBase):
    def __init__(self, name, batch_y0=None, nonlinearity=None, only_random_initialization=False,transpose_state_when_forward=False,
                 concatenate_parameters=True,*args,**kwargs):
        super(AutogradShootingBlockBase, self).__init__(name=name, batch_y0=batch_y0,nonlinearity=nonlinearity,
                                                        only_random_initialization=only_random_initialization,
                                                        transpose_state_when_forward=transpose_state_when_forward,
                                                        concatenate_parameters=concatenate_parameters,
                                                        *args,**kwargs)

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


class LinearInParameterAutogradShootingBlock(AutogradShootingBlockBase):
    def __init__(self, name, batch_y0=None, nonlinearity=None, only_random_initialization=False,transpose_state_when_forward=False,
                 concatenate_parameters=True,*args,**kwargs):
        super(LinearInParameterAutogradShootingBlock, self).__init__(name=name, batch_y0=batch_y0,nonlinearity=nonlinearity,
                                                                     only_random_initialization=only_random_initialization,
                                                                     transpose_state_when_forward=transpose_state_when_forward,
                                                                     concatenate_parameters=concatenate_parameters,
                                                                     *args,**kwargs)

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


class NonlinearInParameterAutogradShootingBlock(AutogradShootingBlockBase):
    def __init__(self, name, batch_y0=None, nonlinearity=None, only_random_initialization=False,transpose_state_when_forward=False,
                 concatenate_parameters=True,*args,**kwargs):
        super(NonlinearInParameterAutogradShootingBlock, self).__init__(name=name, batch_y0=batch_y0,nonlinearity=nonlinearity,
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
