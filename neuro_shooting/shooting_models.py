import torch
import torch.nn as nn
import neuro_shooting.shooting_integrands as shooting
import neuro_shooting.overwrite_classes as oc
from sortedcontainers import SortedDict
from torchdiffeq import odeint
import neuro_shooting.state_costate_and_data_dictionary_utils as scd_utils

class AutoShootingIntegrandModelSecondOrder(shooting.ShootingLinearInParameterVectorIntegrand):
    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                nr_of_particles=10, particle_dimension=1, particle_size=2,*args, **kwargs):

        super(AutoShootingIntegrandModelSecondOrder, self).__init__(in_features=in_features,
                                                                    nonlinearity=nonlinearity,
                                                                    transpose_state_when_forward=transpose_state_when_forward,
                                                                    concatenate_parameters=concatenate_parameters,
                                                                    nr_of_particles=nr_of_particles,
                                                                    particle_dimension=particle_dimension,
                                                                    particle_size=particle_size,
                                                                    *args, **kwargs)

    def create_initial_state_parameters(self, set_to_zero, *args, **kwargs):
        # creates these as a sorted dictionary and returns it (need to be in the same order!!)
        state_dict = SortedDict()

        state_dict['q1'] = self._state_initializer.create_parameters(nr_of_particles=self.nr_of_particles,
                                                                     particle_size=self.particle_size,
                                                                     particle_dimension=self.particle_dimension,
                                                                     set_to_zero=set_to_zero)

        state_dict['q2'] = self._state_initializer.create_parameters(nr_of_particles=self.nr_of_particles,
                                                                     particle_size=self.particle_size,
                                                                     particle_dimension=self.particle_dimension,
                                                                     set_to_zero=set_to_zero)

        return state_dict

    def create_default_parameter_objects(self):

        parameter_objects = SortedDict()

        linear1 = oc.SNN_Linear(in_features=self.in_features, out_features=self.in_features)
        linear2 = oc.SNN_Linear(in_features=self.in_features, out_features=self.in_features)

        parameter_objects['l1'] = linear1
        parameter_objects['l2'] = linear2

        return parameter_objects

    def rhs_advect_state(self, t, state_dict_or_dict_of_dicts, parameter_objects):

        rhs = SortedDict()

        s = state_dict_or_dict_of_dicts
        p = parameter_objects

        rhs['dot_q1'] = p['l1'](self.nl(s['q2']))
        rhs['dot_q2'] = p['l2'](s['q1'])

        return rhs

    def get_initial_data_dict_from_data_tensor(self, x):
        # intial data dict from given data tensor
        data_state_dict = SortedDict()
        data_costate_dict = SortedDict()
        data_state_dict['q1'] = x
        data_costate_dict['p_q1'] = torch.zeros_like(data_state_dict['q1'])
        data_state_dict['q2'] = torch.zeros_like(x)
        data_costate_dict['p_q2'] = torch.zeros_like(data_state_dict['q2'])
        return data_state_dict,data_costate_dict

    def disassemble(self,input,dim=1):
        state_dict, costate_dict, data_state_dicts, data_costate_dicts = self.disassemble_tensor(input, dim=dim)
        return scd_utils.extract_key_from_dict_of_dicts(data_state_dicts, 'q1')

class AutoShootingIntegrandModelUpDown(shooting.ShootingLinearInParameterVectorIntegrand):

    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                nr_of_particles=10, particle_dimension=1, particle_size=2,*args, **kwargs):

        super(AutoShootingIntegrandModelUpDown, self).__init__(in_features=in_features,
                                                               nonlinearity=nonlinearity,
                                                               transpose_state_when_forward=transpose_state_when_forward,
                                                               concatenate_parameters=concatenate_parameters,
                                                               nr_of_particles=nr_of_particles,
                                                               particle_dimension=particle_dimension,
                                                               particle_size=particle_size,
                                                               *args, **kwargs)

        self.inflation_factor = 5

    def create_initial_state_parameters(self, set_to_zero, *args, **kwargs):
        # creates these as a sorted dictionary and returns it (need to be in the same order!!)
        state_dict = SortedDict()

        state_dict['q1'] = self._state_initializer.create_parameters(nr_of_particles=self.nr_of_particles,
                                                                     particle_size=self.particle_size,
                                                                     particle_dimension=self.particle_dimension,
                                                                     set_to_zero=set_to_zero)

        # make the dimension of this state 5 times bigger
        state_dict['q2'] = self._state_initializer.create_parameters(nr_of_particles=self.nr_of_particles,
                                                                     particle_size=self.particle_size*self.inflation_factor,
                                                                     particle_dimension=self.particle_dimension,
                                                                     set_to_zero=set_to_zero)

        return state_dict

    def create_default_parameter_objects(self):

        parameter_objects = SortedDict()

        linear1 = oc.SNN_Linear(in_features=self.in_features*self.inflation_factor,out_features=self.in_features)
        linear2 = oc.SNN_Linear(in_features=self.in_features,out_features=self.in_features*self.inflation_factor)

        parameter_objects['l1'] = linear1
        parameter_objects['l2'] = linear2

        return parameter_objects

    def rhs_advect_state(self, t, state_dict_or_dict_of_dicts, parameter_objects):

        rhs = SortedDict()

        s = state_dict_or_dict_of_dicts
        p = parameter_objects

        rhs['dot_q1'] = p['l1'](input=self.nl(s['q2']))
        rhs['dot_q2'] = p['l2'](input=s['q1'])

        return rhs

    def get_initial_data_dict_from_data_tensor(self, x):
        # Initial data dict from given data tensor
        data_state_dict = SortedDict()
        data_costate_dict = SortedDict()
        data_state_dict['q1'] = x
        data_costate_dict['p_q1'] = torch.zeros_like(x)

        z = torch.zeros_like(x)
        sz = [1]*len(z.shape)
        sz[-1] = self.inflation_factor

        data_state_dict['q2'] = z.repeat(sz)
        data_costate_dict['p_q2'] = torch.zeros_like(data_state_dict["q2"])

        return data_state_dict,data_costate_dict

    def disassemble(self,input,dim=1):
        state_dict, costate_dict, data_state_dicts,data_costate_dicts = self.disassemble_tensor(input, dim=dim)
        return scd_utils.extract_key_from_dict_of_dicts(data_state_dicts,'q1')

class AutoShootingIntegrandModelSimple(shooting.ShootingLinearInParameterVectorIntegrand):

    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                nr_of_particles=10, particle_dimension=1, particle_size=2,*args, **kwargs):

        super(AutoShootingIntegrandModelSimple, self).__init__(in_features=in_features,
                                                               nonlinearity=nonlinearity,
                                                               transpose_state_when_forward=transpose_state_when_forward,
                                                               concatenate_parameters=concatenate_parameters,
                                                               nr_of_particles=nr_of_particles,
                                                               particle_dimension=particle_dimension,
                                                               particle_size=particle_size,
                                                               *args, **kwargs)


    def create_initial_state_parameters(self, set_to_zero, *args, **kwargs):
        # creates these as a sorted dictionary and returns it (need to be in the same order!!)
        state_dict = SortedDict()

        state_dict['q1'] = self._state_initializer.create_parameters(nr_of_particles=self.nr_of_particles,
                                                                     particle_size=self.particle_size,
                                                                     particle_dimension=self.particle_dimension,
                                                                     set_to_zero=set_to_zero)

        return state_dict

    def create_default_parameter_objects(self):

        parameter_objects = SortedDict()

        # todo: make this more generic again
        dim = 2
        #linear = oc.SNN_Linear(in_features=self.d, out_features=self.d)
        linear = oc.SNN_Linear(in_features=self.in_features, out_features=self.in_features)
        parameter_objects['l1'] = linear

        return parameter_objects

    def rhs_advect_state(self, t, state_dict_or_dict_of_dicts, parameter_objects):

        rhs = SortedDict()

        s = state_dict_or_dict_of_dicts
        p = parameter_objects

        rhs['dot_q1'] = p['l1'](input=self.nl(s['q1']))

        return rhs

    def get_initial_data_dict_from_data_tensor(self, x):
        # Initial data_dict for given initial data tensor
        data_state_dict = SortedDict()
        data_costate_dict = SortedDict()
        data_state_dict['q1'] = x
        data_costate_dict['p_q1'] = torch.zeros_like(x)

        return data_state_dict,data_costate_dict

    def disassemble(self,input,dim=1):
        state_dict, costate_dict, data_state_dicts,data_costate_dicts = self.disassemble_tensor(input, dim=dim)
        return scd_utils.extract_key_from_dict_of_dicts(data_state_dicts,'q1')

class AutoShootingIntegrandModelSimpleConv2D(shooting.ShootingLinearInParameterConvolutionIntegrand):

    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                nr_of_particles=10, particle_dimension=1, particle_size=2, filter_size=3, *args, **kwargs):

        super(AutoShootingIntegrandModelSimpleConv2D, self).__init__(in_features=in_features,
                                                                     nonlinearity=nonlinearity,
                                                                     transpose_state_when_forward=transpose_state_when_forward,
                                                                     concatenate_parameters=concatenate_parameters,
                                                                     nr_of_particles=nr_of_particles,
                                                                     particle_dimension=particle_dimension,
                                                                     particle_size=particle_size,
                                                                     *args, **kwargs)

        self.filter_size = filter_size
        self.enlargement_dimensions = [2,3]


    def create_initial_state_parameters(self,set_to_zero, *args, **kwargs):
        # creates these as a sorted dictionary and returns it (need to be in the same order!!)
        state_dict = SortedDict()

        state_dict['q1'] = self._state_initializer.create_parameters(nr_of_particles=self.nr_of_particles,
                                                                     particle_size=self.particle_size,
                                                                     particle_dimension=self.particle_dimension,
                                                                     set_to_zero=set_to_zero)

        state_dict['q2'] = self._state_initializer.create_parameters(nr_of_particles=self.nr_of_particles,
                                                                     particle_size=self.particle_size,
                                                                     particle_dimension=self.particle_dimension,
                                                                     set_to_zero=set_to_zero)

        return state_dict

    def create_default_parameter_objects(self):

        parameter_objects = SortedDict()

        conv1 = oc.SNN_Conv2d(in_channels=self.in_features,out_channels=self.in_features,kernel_size=self.filter_size,padding = 1)
        conv2 = oc.SNN_Conv2d(in_channels=self.in_features,out_channels=self.in_features,kernel_size=self.filter_size,padding = 1)

        parameter_objects['conv1'] = conv1
        parameter_objects['conv2'] = conv2

        return parameter_objects

    def rhs_advect_state(self, t, state_dict_or_dict_of_dicts, parameter_objects):

        rhs = SortedDict()

        s = state_dict_or_dict_of_dicts
        p = parameter_objects

        rhs['dot_q1'] = p['conv1'](self.nl(s['q2']))
        rhs['dot_q2'] = p['conv2'](s['q1'])

        return rhs

    def get_initial_data_dict_from_data_tensor(self, x):
        # Initial data dict from given data tensor
        data_state_dict = SortedDict()
        data_costate_dict = SortedDict()
        data_state_dict['q1'] = x
        data_costate_dict['p_q1'] = torch.zeros_like(x)
        data_state_dict['q2'] = torch.zeros_like(x)
        data_costate_dict['p_q2'] = torch.zeros_like(x)

        return data_state_dict,data_costate_dict

    def disassemble(self,input,dim=1):
        state_dict, costate_dict, data_state_dicts,data_costate_dicts = self.disassemble_tensor(input, dim=dim)
        return scd_utils.extract_key_from_dict_of_dicts(data_state_dicts,'q1')

class AutoShootingIntegrandModelConv2DBatch(shooting.ShootingLinearInParameterConvolutionIntegrand):

    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                 nr_of_particles=10, particle_dimension=1, particle_size=2, filter_size=3, *args, **kwargs):

        super(AutoShootingIntegrandModelConv2DBatch, self).__init__(in_features=in_features,
                                                                    nonlinearity=nonlinearity,
                                                                    transpose_state_when_forward=transpose_state_when_forward,
                                                                    concatenate_parameters=concatenate_parameters,
                                                                    nr_of_particles=nr_of_particles,
                                                                    particle_dimension=particle_dimension,
                                                                    particle_size=particle_size,
                                                                    *args, **kwargs)

        self.filter_size = filter_size
        self.enlargement_dimensions = [2,3]

    def create_initial_state_parameters(self,set_to_zero,*args,**kwargs):
        # creates these as a sorted dictionary and returns it (need to be in the same order!!)

        state_dict = SortedDict()

        state_dict['q1'] = self._state_initializer.create_parameters(nr_of_particles=self.nr_of_particles,
                                                                     particle_size=self.particle_size,
                                                                     particle_dimension=self.particle_dimension,
                                                                     set_to_zero=set_to_zero)

        state_dict['q2'] = self._state_initializer.create_parameters(nr_of_particles=self.nr_of_particles,
                                                                     particle_size=self.particle_size,
                                                                     particle_dimension=self.particle_dimension,
                                                                     set_to_zero=set_to_zero)

        return state_dict

    def create_default_parameter_objects(self):

        parameter_objects = SortedDict()

        conv1 = oc.SNN_Conv2d(in_channels=self.in_features,out_channels=self.in_features,kernel_size=self.filter_size,padding = 1)
        conv2 = oc.SNN_Conv2d(in_channels=self.in_features,out_channels=self.in_features,kernel_size=self.filter_size,padding = 1)
        #group_norm = oc.SNN_GroupNorm(self.channel_number,self.channel_number,affine = False)
        group_norm = nn.GroupNorm(self.in_features,self.in_features,affine = False)
        parameter_objects['conv1'] = conv1
        parameter_objects['conv2'] = conv2
        self.group_norm = group_norm

        return parameter_objects


    def rhs_advect_state(self, t, state_dict_or_dict_of_dicts, parameter_objects):

        rhs = SortedDict()

        s = state_dict_or_dict_of_dicts
        p = parameter_objects

        rhs['dot_q1'] = p['conv1'](self.nl(s['q2']))
        rhs['dot_q2'] = p["conv2"](self.group_norm(s['q1']))

        return rhs

    def get_initial_data_dict_from_data_tensor(self, x):
        # Initial data dict from given data tensor
        data_state_dict = SortedDict()
        data_state_dict['q1'] = x
        data_state_dict['q2'] = torch.zeros_like(x)

        data_costate_dict = SortedDict()
        data_costate_dict['q1'] = torch.zeros_like(data_state_dict['q1'])
        data_costate_dict['q2'] = torch.zeros_like(data_state_dict['q2'])

        return data_state_dict,data_costate_dict

    def disassemble(self,input,dim=1):
        state_dict, costate_dict, data_state_dicts,data_costate_dicts = self.disassemble_tensor(input, dim=dim)
        return scd_utils.extract_key_from_dict_of_dicts(data_state_dicts,'q1')


