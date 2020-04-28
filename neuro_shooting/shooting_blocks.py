import torch
import torch.nn as nn
from sortedcontainers import SortedDict

from abc import ABCMeta, abstractmethod

import neuro_shooting.generic_integrator as generic_integrator
import neuro_shooting.state_costate_and_data_dictionary_utils as scd_utils


class ShootingBlockBase(nn.Module):

    def __init__(self, name, shooting_integrand=None, shooting_integrand_name=None,
                 integrator_library='odeint', integrator_name='rk4',
                 use_adjoint_integration=False, integrator_options=None,
                 keep_initial_state_parameters_at_zero=False,
                 enlarge_pass_through_states_and_costates=True,
                 *args, **kwargs):
        """

        :param name: unique name of this block (needed to keep track of the parameters when there is pass through)
        :param shooting_integrand:
        :param shooting_integrand_name:
        :param integrator_library:
        :param integrator_name:
        :param use_adjoint_integration:
        :param integrator_options:
        :param keep_initial_state_parameters_at_zero: If set to true than all the newly created initial state parameters are kept at zero (and not optimized over); this includes state parameters created via state/costate enlargement.
        :param enlarge_pass_through_states_and_costates: all the pass through states/costates are enlarged so they match the dimensions of the states/costates. This assures that parameters can be concatenated.
        :param args:
        :param kwargs:
        """

        super(ShootingBlockBase, self).__init__()

        self._block_name = name
        """Name of the shooting block"""

        self._forward_not_yet_executed = True
        """This is to keep track if it has been run forward. To assure parameters() is not called before one forward pass has been accomplished.
        As for convenience the parameters are constructed dynamically."""

        if shooting_integrand is None and shooting_integrand_name is None:
            raise ValueError('Either shooting_integrand or shooting_integrand_name need to be specified')

        if shooting_integrand is not None and shooting_integrand_name is not None:
            raise ValueError('Either shooting_integrand or shooting_integrand_name need to be specified. You specified both. Pick one option.')

        if shooting_integrand is not None:
            self.shooting_integrand = shooting_integrand
        else:
            self.shooting_integrand = self._get_shooting_integrand_by_name(shooting_integrand_name=shooting_integrand_name)

        #self.add_module(name='shooting_integrand', module=self.shooting_integrand) # probably remove

        self.integration_time = torch.tensor([0, 1]).float()
        self.integration_time_vector = None

        self.integrator_name = integrator_name
        self.integrator_library = integrator_library
        self.integrator_options = integrator_options
        self.use_adjoint_integration = use_adjoint_integration

        self.integrator = generic_integrator.GenericIntegrator(integrator_library = self.integrator_library,
                                                               integrator_name = self.integrator_name,
                                                               use_adjoint_integration=self.use_adjoint_integration,
                                                               integrator_options=self.integrator_options)

        self._state_parameter_dict = None
        """Dictionary holding the state variables"""

        self.keep_state_parameters_at_zero = keep_initial_state_parameters_at_zero
        """
        If set to true one only optimizes over the costate and the state parameters are kept at zero (i.e., are no parameters).
        This is for example useful when mimicking ResNet style dimension increase. 
        """

        self.enlarge_pass_through_states_and_costates = enlarge_pass_through_states_and_costates
        """If set to true the pass through states and costates will be enlarged so they are compatible in size with the states and costates and can be concatenated"""

        self._costate_parameter_dict = None
        """Dictionary holding the costates (i.e., adjoints/duals)"""

        # TODO: see if we really need these variables, seems like they are no longer needed
        # self._pass_through_state_parameter_dict_of_dicts = None
        # """State parameters that are passed in externally, but are not parameters to be optimized over"""
        # self._pass_through_costate_parameter_dict_of_dicts = None
        # """Costate parameters that are passed in externally, but are not parameters to be optimized over"""

        self._pass_through_state_dict_of_dicts_enlargement_parameters = None
        self._pass_through_costate_dict_of_dicts_enlargement_parameters = None

        state_dict, costate_dict = self.create_initial_state_and_costate_parameters(*args, **kwargs)

        self._state_parameter_dict, self._costate_parameter_dict = self.register_state_and_costate_parameters(
            state_dict=state_dict, costate_dict=costate_dict,
            keep_state_parameters_at_zero=keep_initial_state_parameters_at_zero)


    def parameters(self, recurse=True):
        if self._forward_not_yet_executed:
            raise ValueError('Parameters are created dynamically. Please execute your entire pipeline once first before calling parameters()!')
        # we overwrite this to assure that one forward pass has been done (so we can collect parameters)
        return super(ShootingBlockBase, self).parameters(recurse=recurse)

    def get_norm_penalty(self):
        return self.shooting_integrand.get_norm_penalty()

    def _get_shooting_integrand_by_name(self,shooting_integrand_name):
        raise ValueError('Not yet implemented')

    def set_integration_time(self,time_to):
        self.integration_time = torch.tensor([0,time_to]).float()
        self.integration_time_vector = None

    def set_integration_time_vector(self,integration_time_vector,suppress_warning=False):
        if not suppress_warning:
            print('WARNING: Typically the integration time is [0,1] and should not be set manually. Setting it to a vector will change the output behavior. All times will be output.')
        self.integration_time_vector = integration_time_vector
        self.integration_time =  None

    def _apply(self, fn):
        super(ShootingBlockBase, self)._apply(fn)
        self.shooting_integrand._apply(fn)
        self.integration_time = fn(self.integration_time)
        return self

    def to(self, *args, **kwargs):
        super(ShootingBlockBase,self).to(*args, **kwargs)
        self.shooting_integrand.to(*args, **kwargs)
        self.integration_time = self.integration_time.to(*args, **kwargs)
        return self

    def create_initial_state_and_costate_parameters(self, *args, **kwargs):
        """
        Creates the intial state and costate parameters. Should typically *not* be necessary to call this manually.
        :return: tuple of SortedDicts holding the state dictionary and the cotate dictionary
        """

        state_dict = self.shooting_integrand.create_initial_state_parameters_if_needed(set_to_zero=self.keep_state_parameters_at_zero, *args, **kwargs)
        costate_dict = self.shooting_integrand.create_initial_costate_parameters(state_dict=state_dict, *args, **kwargs)

        return state_dict, costate_dict

    def _create_raw_enlargement_parameters(self, desired_size, current_size, data_type, dtype, device):
        # todo: make sure the device and dtype is correct here (happens implicit in the initializer)
        vol_d = desired_size.prod()
        vol_c = current_size.prod()
        if vol_d < vol_c:
            raise ValueError('Cannot be enlarged. Desired volume is smaller than current volume.')

        if data_type=='state':
            ep = self.shooting_integrand._state_initializer.create_parameters_of_size(vol_d-vol_c, set_to_zero=self.keep_state_parameters_at_zero)
        elif data_type=='costate':
            ep = self.shooting_integrand._costate_initializer.create_parameters_of_size(vol_d-vol_c)
        else:
            raise ValueError('Unknown data_type {} for initialization. Needs to be state or costate.'.format(data_type))

        return ep

    def _create_generic_dict_of_dicts_enlargement_parameters(self, generic_dict_of_dicts, dict_for_desired_size, data_type='state', enlargement_dimensions = None):
        """

        :param generic_dict_of_dicts:
        :param dict_for_desired_size:
        :param data_type:
        :param enlargement_dimensions:
        :return: returns a three-tuple: first element are the new parameters; second one is the size of the enlargement; third one is the target size
        """

        if dict_for_desired_size is None or enlargement_dimensions is None:
            # for example if this block does not have a state of its own, but is pure pass-through
            # or if it s a vector-valued block which does not support enlargement
            return None

        dict_of_dicts_enlargement = SortedDict()

        # todo: check that we are not considering the batch and the channel size for enlargement here

        for k in dict_for_desired_size:
            desired_size_incorrect_channel = torch.tensor(
                dict_for_desired_size[k].size())  # skip batch and channel size
            # go through all the dictionaries and find the same keys
            for dk in generic_dict_of_dicts:
                if dk not in dict_of_dicts_enlargement:
                    dict_of_dicts_enlargement[dk] = SortedDict()

                current_dict_enlargement = dict_of_dicts_enlargement[dk]

                current_dict = generic_dict_of_dicts[dk]
                current_size = torch.tensor(current_dict[k].size())
                desired_size = desired_size_incorrect_channel
                desired_size[1] = current_size[1]  # keep the same number of channels (but we are for example at liberty to increase the batch size here)

                dtype = current_dict[k].dtype
                device = current_dict[k].device

                current_diff = [0] * len(current_size)
                found_greater_than_zero = False

                for i in range(len(current_diff)):
                    current_diff[i] = desired_size[i] - current_size[i]
                    if current_diff[i] > 0:
                        found_greater_than_zero = True
                    if current_diff[i] < 0:
                        raise ValueError(
                            'State size is smaller than pass through state size. Currently not supported. Aborting.')

                if not found_greater_than_zero:
                    current_dict_enlargement[k] = None
                else:
                    current_dict_enlargement[k] = \
                        (self._create_raw_enlargement_parameters(desired_size=desired_size, current_size=current_size, data_type=data_type,
                                                                 dtype=dtype, device=device), current_diff, desired_size)

        return dict_of_dicts_enlargement

    def _create_pass_through_state_and_costate_enlargement_parameters(self,
                                                                      pass_through_state_parameter_dict_of_dicts,
                                                                      pass_through_costate_parameter_dict_of_dicts,
                                                                      state_parameter_dict,
                                                                      costate_parameter_dict,
                                                                      *args, **kwargs):

        pass_through_state_dict_of_dicts_enlargement = \
            self._create_generic_dict_of_dicts_enlargement_parameters(
                generic_dict_of_dicts=pass_through_state_parameter_dict_of_dicts,
                dict_for_desired_size=state_parameter_dict, data_type='state',
                enlargement_dimensions=self.shooting_integrand.enlargement_dimensions,
                *args, **kwargs)

        pass_through_costate_dict_of_dicts_enlargement = \
            self._create_generic_dict_of_dicts_enlargement_parameters(
                generic_dict_of_dicts=pass_through_costate_parameter_dict_of_dicts,
                enlargement_dimensions=self.shooting_integrand.enlargement_dimensions,
                dict_for_desired_size=costate_parameter_dict, data_type='costate', *args, **kwargs)

        return pass_through_state_dict_of_dicts_enlargement, pass_through_costate_dict_of_dicts_enlargement

    def _enlarge_tensor(self, current_tensor, enlargement):
        e_pars, e_diff, e_desired_size = enlargement

        # first assure that the channel dimension is the same
        if e_diff[1] != 0:
            raise ValueError('Should have the same number of channels. Aborting')

        current_size = torch.tensor(current_tensor.size())

        ret = current_tensor

        indx = 0

        # grows out the hyperrectangle
        for c_dim, c_diff in enumerate(e_diff):
            if c_diff > 0:
                # if there is something to grow, grow c_dim
                beg_add = c_diff // 2  # add at the beginning (add less at the beginning if we need to add something odd)
                end_add = c_diff - beg_add  # add at the end

                beg_size = current_size.clone()
                beg_size[c_dim] = beg_add
                indx_end = indx + beg_size.prod()
                beg_pars = e_pars[indx:indx_end]
                indx = indx_end

                end_size = current_size.clone()
                end_size[c_dim] = end_add
                indx_end = indx + end_size.prod()
                end_pars = e_pars[indx:indx_end]
                indx = indx_end

                ret = torch.cat((beg_pars.view(tuple(beg_size)),
                                 ret,
                                 end_pars.view(tuple(end_size))), dim=c_dim)

                # keep track of current size so we know how to grow the next dimension
                current_size = torch.tensor(ret.size())

        return ret

    def _enlarge_generic_dict_of_dicts(self, generic_dict_of_dicts, enlargement_parameters):

        if enlargement_parameters is None:
            # if no enlargement was necessary, just return the orignal dictionary
            return generic_dict_of_dicts

        enlarged_dict_of_dicts = SortedDict()

        for gd, epd in zip(generic_dict_of_dicts, enlargement_parameters):

            enlarged_dict_of_dicts[gd] = SortedDict()

            c_dict = generic_dict_of_dicts[gd]
            c_enlargement_parameter_dict = enlargement_parameters[epd]
            c_enlarged_dict = enlarged_dict_of_dicts[gd]

            for kd, kep in zip(c_dict, c_enlargement_parameter_dict):
                if c_enlargement_parameter_dict[kep] is not None:
                    c_enlarged_dict[kd] = self._enlarge_tensor(c_dict[kd], c_enlargement_parameter_dict[kep])
                else:
                    c_enlarged_dict[kd] = c_dict[kd]

        return enlarged_dict_of_dicts

    def _get_initial_conditions_from_data_dict_of_dicts(self, data_dict_of_dicts, pass_through_state_dict_of_dicts, pass_through_costate_dict_of_dicts, zero_pad_new_data_states, *args, **kwargs):
        """
        Given a data dictionary, this method creates a vector which contains the initial condition consisting of state, costate, and the data.
        As a side effect it also stores (caches) the created assembly plan so that it does not need to be specified when calling disassemble_tensor.

        :param data_dict_of_dicts: data dictionary
        :return: vector of initial conditions
        """

        state_dicts = scd_utils._merge_state_costate_or_data_dict_with_generic_dict_of_dicts(
            generic_dict=self._state_parameter_dict,
            generic_dict_of_dicts=pass_through_state_dict_of_dicts,
            generic_dict_block_name=self._block_name)

        costate_dicts = scd_utils._merge_state_costate_or_data_dict_with_generic_dict_of_dicts(
            generic_dict=self._costate_parameter_dict,
            generic_dict_of_dicts=pass_through_costate_dict_of_dicts,
            generic_dict_block_name=self._block_name)

        if zero_pad_new_data_states and self._state_parameter_dict is not None:

            data_concatenation_dim = scd_utils.get_data_concatenation_dim(state_dict=self._state_parameter_dict,
                                                                          data_dict_of_dicts=data_dict_of_dicts,
                                                                          state_concatenation_dim=self.shooting_integrand.concatenation_dim)

            self.shooting_integrand.set_data_concatenation_dim(data_concatenation_dim=data_concatenation_dim)

            padded_zeros = scd_utils._get_zero_data_dict_matching_state_dim(state_dict=self._state_parameter_dict,
                                                                            data_dict_of_dicts=data_dict_of_dicts,
                                                                            state_concatenation_dim=self.shooting_integrand.concatenation_dim,
                                                                            data_concatenation_dim=data_concatenation_dim)

            data_dicts = scd_utils._merge_state_costate_or_data_dict_with_generic_dict_of_dicts(
                generic_dict=padded_zeros,
                generic_dict_of_dicts=data_dict_of_dicts,
                generic_dict_block_name=self._block_name)
        else:
            data_dicts = data_dict_of_dicts

        # initialize the second state of x with zero so far
        initial_conditions, assembly_plans = self.shooting_integrand.assemble_tensor(state_dict_of_dicts=state_dicts,
                                                                                     costate_dict_of_dicts=costate_dicts,
                                                                                     data_dict_of_dicts=data_dicts)
        return initial_conditions,assembly_plans


    def register_state_and_costate_parameters(self,state_dict,costate_dict,keep_state_parameters_at_zero):

        if state_dict is not None:

            if type(state_dict) != SortedDict:
                raise ValueError('state parameter dictionrary needs to be an SortedDict and not {}'.format(
                    type(state_dict)))

            # only register if we want to optimize over them, otherwise force them to zero
            if keep_state_parameters_at_zero:
                print('INFO: Keeping new state parameters at zero for {}'.format(self._block_name))
                for k in state_dict:
                    state_dict[k].zero_()
            else:
                for k in state_dict:
                    self.register_parameter(k,state_dict[k])

        if costate_dict is not None:

            if type(costate_dict) != SortedDict:
                raise ValueError('costate parameter dictionrary needs to be an SortedDict and not {}'.format(
                    type(costate_dict)))

            for k in costate_dict:
                self.register_parameter(k,costate_dict[k])

        return state_dict,costate_dict

    def register_pass_through_state_and_costate_enlargement_parameters(self,
                                                                       pass_through_state_dict_of_dicts_enlargement_parameters,
                                                                       pass_through_costate_dict_of_dicts_enlargement_parameters,
                                                                       keep_state_parameters_at_zero):

        if pass_through_state_dict_of_dicts_enlargement_parameters is not None:

            for dk in pass_through_state_dict_of_dicts_enlargement_parameters:
                c_state_dict = pass_through_state_dict_of_dicts_enlargement_parameters[dk]
                if c_state_dict is not None:
                    for k in c_state_dict:
                        if c_state_dict[k] is not None:
                            if keep_state_parameters_at_zero:
                                print('INFO: Keeping new state enlargement parameters at zero for {}'.format(self._block_name))
                                c_state_dict[k][0]._zero()
                            self.register_parameter('pt_enlarge_' + k, c_state_dict[k][0])
                            print('INFO: registering enlargement state parameters {} of size {} for block {}'.format(k,c_state_dict[k][0].numel(),self._block_name))

        if pass_through_costate_dict_of_dicts_enlargement_parameters is not None:

            for dk in pass_through_costate_dict_of_dicts_enlargement_parameters:
                c_costate_dict = pass_through_costate_dict_of_dicts_enlargement_parameters[dk]
                if c_costate_dict is not None:
                    for k in c_costate_dict:
                        if c_costate_dict[k] is not None:
                            self.register_parameter('pt_enlarge_' + k, c_costate_dict[k][0])
                            print('INFO: registering enlargement costate parameters {} of size {} for block {}'.format(k,c_costate_dict[k][0].numel(),self._block_name))

    def _get_or_create_and_register_pass_through_enlargement_parameters(self,pass_through_state_dict_of_dicts,pass_through_costate_dict_of_dicts,*args,**kwargs):

        if self._forward_not_yet_executed:
            # first compute how much we need to enlarge

            if self._pass_through_state_dict_of_dicts_enlargement_parameters is not None or self._pass_through_costate_dict_of_dicts_enlargement_parameters:
                raise ValueError(
                    'Enlargement parameters have previously been created. Aborting. Something went wrong. Should only be created once!')

            pass_through_state_dict_of_dicts_enlargement_parameters, pass_through_costate_dict_of_dicts_enlargement_parameters = \
                self._create_pass_through_state_and_costate_enlargement_parameters(
                    pass_through_state_parameter_dict_of_dicts=pass_through_state_dict_of_dicts,
                    pass_through_costate_parameter_dict_of_dicts=pass_through_costate_dict_of_dicts,
                    state_parameter_dict=self._state_parameter_dict,
                    costate_parameter_dict=self._costate_parameter_dict,
                    *args, **kwargs)

            # now we register the parameters used for the enlargement so we can optimize over them
            self.register_pass_through_state_and_costate_enlargement_parameters(
                pass_through_state_dict_of_dicts_enlargement_parameters=pass_through_state_dict_of_dicts_enlargement_parameters,
                pass_through_costate_dict_of_dicts_enlargement_parameters=pass_through_costate_dict_of_dicts_enlargement_parameters,
                keep_state_parameters_at_zero=self.keep_state_parameters_at_zero
            )

            return pass_through_state_dict_of_dicts_enlargement_parameters,pass_through_costate_dict_of_dicts_enlargement_parameters

        else:
            # just return what we computed previously
            return self._pass_through_state_dict_of_dicts_enlargement_parameters,self._pass_through_costate_dict_of_dicts_enlargement_parameters

    def _enlarge_pass_through_state_and_costate_dict_of_dicts(self,pass_through_state_dict_of_dicts,pass_through_costate_dict_of_dicts):

        enlarged_pass_through_state_parameter_dict_of_dicts = self._enlarge_generic_dict_of_dicts(
            generic_dict_of_dicts=pass_through_state_dict_of_dicts,
            enlargement_parameters=self._pass_through_state_dict_of_dicts_enlargement_parameters)

        enlarged_pass_through_costate_parameter_dict_of_dicts = self._enlarge_generic_dict_of_dicts(
            generic_dict_of_dicts=pass_through_costate_dict_of_dicts,
            enlargement_parameters=self._pass_through_costate_dict_of_dicts_enlargement_parameters)

        return enlarged_pass_through_state_parameter_dict_of_dicts,enlarged_pass_through_costate_parameter_dict_of_dicts

    def forward(self,x=None,pass_through_state_dict_of_dicts=None,pass_through_costate_dict_of_dicts=None,data_dict_of_dicts=None,*args,**kwargs):

        if x is None and pass_through_state_dict_of_dicts is None and pass_through_costate_dict_of_dicts is None:
            raise ValueError('At least one input needs to be specified')

        # first we need to check if we
        # a) need to enlarge the pass through states and costates
        # b) if we do, we need to register their corresponding parameters

        if (pass_through_state_dict_of_dicts is None and pass_through_costate_dict_of_dicts is not None) or \
                (pass_through_state_dict_of_dicts is not None and pass_through_costate_dict_of_dicts is None):
            raise ValueError('State and costate need to be specified simultaneously')

        if pass_through_state_dict_of_dicts is not None and pass_through_costate_dict_of_dicts is not None:

            if self.enlarge_pass_through_states_and_costates:

                self._pass_through_state_dict_of_dicts_enlargement_parameters,self._pass_through_costate_dict_of_dicts_enlargement_parameters = \
                    self._get_or_create_and_register_pass_through_enlargement_parameters(pass_through_state_dict_of_dicts=pass_through_state_dict_of_dicts,
                                                                                         pass_through_costate_dict_of_dicts=pass_through_costate_dict_of_dicts,*args,**kwargs)

                # now we can take the pass-through parameters and create their enlarged versions
                # then compute the actual enlargement

                effective_pass_through_state_dict_of_dicts,effective_pass_through_costate_dict_of_dicts = \
                    self._enlarge_pass_through_state_and_costate_dict_of_dicts(pass_through_state_dict_of_dicts=pass_through_state_dict_of_dicts,
                                                                               pass_through_costate_dict_of_dicts=pass_through_costate_dict_of_dicts)

            else:
                # if they are not getting enlarged, simply pass them through
                effective_pass_through_state_dict_of_dicts = pass_through_state_dict_of_dicts
                effective_pass_through_costate_dict_of_dicts = pass_through_costate_dict_of_dicts

        else:
            effective_pass_through_state_dict_of_dicts = None
            effective_pass_through_costate_dict_of_dicts = None


        if data_dict_of_dicts is not None and x is not None:
            raise ValueError('Cannot specify x and data_dict_of_dicts at the same time. When at all possible specify the data_dict_of_dicts instead as it will contain the full state.')

        if data_dict_of_dicts is not None:
            effective_data_dict_of_dicts = data_dict_of_dicts
            zero_pad_new_data_states = True
        elif x is not None:
            effective_data_dict = self.shooting_integrand.get_initial_data_dict_from_data_tensor(x)
            effective_data_dict_of_dicts = SortedDict({self._block_name: effective_data_dict})
            zero_pad_new_data_states = False
        else:
            raise ValueError('Neither x or a data_dict is specified, cannot compute initial condition. Aborting.')

        initial_conditions,assembly_plans = self._get_initial_conditions_from_data_dict_of_dicts(data_dict_of_dicts=effective_data_dict_of_dicts,
                                                                                                 pass_through_state_dict_of_dicts=effective_pass_through_state_dict_of_dicts,
                                                                                                 pass_through_costate_dict_of_dicts=effective_pass_through_costate_dict_of_dicts,
                                                                                                 zero_pad_new_data_states=zero_pad_new_data_states)

        # need to let the integrand know how to go from the vector to the data stuctures
        self.shooting_integrand.set_auto_assembly_plans(assembly_plans=assembly_plans)

        # reset shooting integrand, this should be called at the beginning of a new integration
        self.shooting_integrand.reset()

        #integrate
        if self.integration_time_vector is not None:
            res_all_times = self.integrator.integrate(func=self.shooting_integrand, x0=initial_conditions, t=self.integration_time_vector)
            state_dict_of_dicts, costate_dict_of_dicts, data_dict_of_dicts = self.shooting_integrand.disassemble_tensor(res_all_times,dim=1)
            res = self.shooting_integrand.disassemble(res_all_times,dim=1)
        else:
            res_all_times = self.integrator.integrate(func=self.shooting_integrand, x0=initial_conditions, t=self.integration_time)
            res_final = res_all_times[-1, ...]
            state_dict_of_dicts, costate_dict_of_dicts, data_dict_of_dicts = self.shooting_integrand.disassemble_tensor(res_final)
            # and get what should typically be returned (the transformed data)
            res = self.shooting_integrand.disassemble(res_final,dim=0)

        if self.shooting_integrand.concatenate_parameters:
            # we also concatenate the output
            res_state_dict_of_dicts = scd_utils._concatenate_named_dict_of_dicts_keeping_structure(generic_dict_of_dicts=state_dict_of_dicts,
                                                                                                   concatenation_dim=self.shooting_integrand.concatenation_dim,
                                                                                                   block_name=self._block_name)
            res_costate_dict_of_dicts = scd_utils._concatenate_named_dict_of_dicts_keeping_structure(generic_dict_of_dicts=costate_dict_of_dicts,
                                                                                                     concatenation_dim=self.shooting_integrand.concatenation_dim,
                                                                                                     block_name=self._block_name)
            res_data_dict_of_dicts = scd_utils._concatenate_named_dict_of_dicts_keeping_structure(generic_dict_of_dicts=data_dict_of_dicts,
                                                                                                  concatenation_dim=self.shooting_integrand.data_concatenation_dim,
                                                                                                  block_name=self._block_name)
            # todo: find a more elegant solution to deal with the output
            if type(res)==torch.Tensor:
                res_res = res
            else:
                res_res = scd_utils._concatenate_dict_of_dicts(generic_dict_of_dicts=res,
                                                               concatenation_dim=self.shooting_integrand.data_concatenation_dim)
        else:
            res_state_dict_of_dicts = state_dict_of_dicts
            res_costate_dict_of_dicts = costate_dict_of_dicts
            res_data_dict_of_dicts = data_dict_of_dicts
            res_res = res

        # from now on it will be save to call parameters() for this block
        self._forward_not_yet_executed = False

        # we return the typical return value (in case this is the last block and we want to easily integrate,
        # but also all the dictionaries so these blocks can be easily chained together

        return res_res,res_state_dict_of_dicts,res_costate_dict_of_dicts,res_data_dict_of_dicts

