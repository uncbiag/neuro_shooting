import torch
import torch.nn as nn
from sortedcontainers import SortedDict

import neuro_shooting.generic_integrator as generic_integrator

class ShootingStridingBlock(nn.Module):
    """
    Allows striding. Will take as an input a SortedDict of SortedDict's for the state and the costate as well as
    a dict holding the state for the data and will subsample as desired. This basically amounts to striding.
    """

    # todo: padding should upsample it going forward or we introduce new variables in the surrounding
    # todo: maybe start with padding as it is easier (zero padding and optimize over dual variables)

    def __init__(self,stride=2,dim=2):
        super(ShootingStridingBlock, self).__init__()

        if type(stride)==tuple:
            if len(stride)!=dim:
                raise ValueError('Stride tuple needs to be of the same dimension as the dimension for the striding. Got {}, but should be {}'.format(len(stride),dim))
            else:
                self.stride = stride
        else:
            if type(stride)==int:
                self.stide = tuple([stride]*dim)
            else:
                raise ValueError('Unsupported stride type {}'.format(type(stride)))

        self.dim = dim

    def _stride_tensor(self,input,stride):

        dim_input = len(input.shape)
        if dim_input!=self.dim+2:
            raise ValueError('Dimension mismatch. Expected tensor dimension {} for batch x channel x ..., but got {}.'.format(self.dim+2,dim_input))

        # compute stride offsets to make sure we pick the center element if we have an odd number of elements
        offsets = [0]*self.dim
        for i,v in stride:
            offsets[i] = v%2

        if self.dim==1:
            if dim_input[2]<stride[0]:
                return None
            else:
                return input[:,:,offsets[0]::stride[0]]
        elif self.dim==2:
            if (dim_input[2]<stride[0]) or (dim_input[3]<stride[1]):
                return  None
            else:
                return input[:,:,offsets[0]::stride[0],offsets[1]::stride[1]]
        elif self.dim==3:
            if (dim_input[2]<stride[0]) or (dim_input[3]<stride[1]) or (dim_input[4]<stride[2]):
                return None
            else:
                return input[:, :, offsets[0]::stride[0], offsets[1]::stride[1], offsets[2]::stride[2]]
        else:
            raise ValueError('Unsupported dimension {}'.format(self.dim))

    def _stride_dict_of_dicts(self,generic_dict_of_dicts,stride=None):

        ret = SortedDict()
        for dk in generic_dict_of_dicts:
            c_generic_dict = generic_dict_of_dicts[dk]
            ret[dk] = self._stride_dict(generic_dict=c_generic_dict,stride=stride)
        return ret

    def _stride_dict(self,generic_dict,stride=None):

        ret = SortedDict()
        for k in generic_dict:
            ret[k] = self._stride_tensor(generic_dict[k],stride=stride)
        return ret


    def forward(self, state_dict_of_dicts=None,costate_dict_of_dicts=None,data_dict=None):

        # compute strided versions of all of these dictionaries and then return them
        strided_state_dict_of_dicts = self._stride_dict_of_dicts(generic_dict_of_dicts=state_dict_of_dicts,stride=self.stride)
        strided_costate_dict_of_dicts = self._stride_dict_of_dicts(generic_dict_of_dicts=costate_dict_of_dicts,stride=self.stride)
        strided_data_dict = self._stride_dict(generic_dict=data_dict,stride=self.stride)

        return strided_state_dict_of_dicts,strided_costate_dict_of_dicts,strided_data_dict

class ShootingBlockBase(nn.Module):

    def __init__(self,shooting_integrand=None,shooting_integrand_name=None,
                 integrator_library='odeint',integrator_name='rk4',use_adjoint_integration=False, integrator_options=None):
        super(ShootingBlockBase, self).__init__()

        if shooting_integrand is None and shooting_integrand_name is None:
            raise ValueError('Either shooting_integrand or shooting_integrand_name need to be specified')

        if shooting_integrand is not None and shooting_integrand_name is not None:
            raise ValueError('Either shooting_integrand or shooting_integrand_name need to be specified. You specified both. Pick one option.')

        if shooting_integrand is not None:
            self.shooting_integrand = shooting_integrand
        else:
            self.shooting_integrand = self._get_shooting_integrand_by_name(shooting_integrand_name=shooting_integrand_name)

        self.add_module(name='shooting_integrand', module=self.shooting_integrand)

        self.integration_time = torch.tensor([0, 1]).float()

        self.integrator_name = integrator_name
        self.integrator_library = integrator_library
        self.integrator_options = integrator_options
        self.use_adjoint_integration = use_adjoint_integration

        self.integrator = generic_integrator.GenericIntegrator(integrator_library = self.integrator_library,
                                                               integrator_name = self.integrator_name,
                                                               use_adjoint_integration=self.use_adjoint_integration,
                                                               integrator_options=self.integrator_options)


    def _get_shooting_integrand_by_name(self,shooting_integrand_name):
        raise ValueError('Not yet implemented')

    def set_integration_time(self,time_to):
        self.integration_time = torch.tensor([0,time_to]).float()

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

    def forward(self,x=None,pass_through_state_dict_of_dicts=None,pass_through_costate_dict_of_dicts=None,data_dict=None):

        if x is None and pass_through_state_dict_of_dicts is None and pass_through_costate_dict_of_dicts is None:
            raise ValueError('At least one input needs to be specified')

        if data_dict is not None and x is not None:
            raise ValueError('Cannot specify x and data_dict at the same time. When at all possible specify the data_dict instead as it will contain the full state.')

        if data_dict is not None:
            initial_conditions = self.shooting_integrand.get_initial_conditions_from_data_dict(data_dict=data_dict)
        else:
            if x is not None:
                initial_conditions = self.shooting_integrand.get_initial_condition(x=x)
            else:
               raise ValueError('Neither x or a data_dict is specified, cannot compute initial condition. Aborting.')

        if (pass_through_state_dict_of_dicts is None and pass_through_costate_dict_of_dicts is not None) or \
                (pass_through_state_dict_of_dicts is None and pass_through_costate_dict_of_dicts is None):
            raise ValueError('State and costate need to be specified simultaneously')

        if pass_through_state_dict_of_dicts is not None and pass_through_costate_dict_of_dicts is not None:
            self.shooting_integrand.set_initial_pass_through_state_and_costate_parameters(state_dict_of_dicts=pass_through_state_dict_of_dicts,
                                                                                          costate_dict_of_dicts=pass_through_costate_dict_of_dicts)
        #integrate
        res_all_times = self.integrator.integrate(func=self.shooting_integrand, x0=initial_conditions, t=self.integration_time)
        res_final = res_all_times[-1, ...]

        state_dict_of_dicts, costate_dict_of_dicts, data_dict = self.shooting_integrand.disassemble_tensor(res_final)

        # and get what should typically be returned (the transformed data)
        res = self.shooting_integrand.disassemble(res_final)

        # we return the typical return value (in case this is the last block and we want to easily integrate,
        # but also all the dictionaries so these blocks can be easily chained together

        return res,state_dict_of_dicts,costate_dict_of_dicts,data_dict

