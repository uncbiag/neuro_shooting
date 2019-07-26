import torch
import torch.nn as nn
import shooting
import overwrite_classes as oc
from sortedcontainers import SortedDict
from torchdiffeq import odeint

class AutoShootingBlockModelSecondOrder(shooting.LinearInParameterAutogradShootingBlock):
    def __init__(self, batch_y0=None, nonlinearity=None, only_random_initialization=False,transpose_state_when_forward=False):
        super(AutoShootingBlockModelSecondOrder, self).__init__(batch_y0=batch_y0,nonlinearity=nonlinearity,
                                                                only_random_initialization=only_random_initialization,
                                                                transpose_state_when_forward=transpose_state_when_forward)

    def create_initial_state_parameters(self, batch_y0, only_random_initialization=True):
        # creates these as a sorted dictionary and returns it (need to be in the same order!!)
        state_dict = SortedDict()

        rand_mag_q = 0.5

        self.k = batch_y0.size()[0]
        self.d = batch_y0.size()[2]

        if only_random_initialization:
            # do a fully random initialization
            state_dict['q1'] = nn.Parameter(rand_mag_q * torch.randn_like(batch_y0))
            state_dict['q2'] = nn.Parameter(rand_mag_q * torch.randn_like(batch_y0))
        else:
            state_dict['q1'] = nn.Parameter(batch_y0 + rand_mag_q * torch.randn_like(batch_y0))
            state_dict['q2'] = nn.Parameter(batch_y0 + rand_mag_q * torch.randn_like(batch_y0))

        return state_dict

    def create_default_parameter_objects(self):

        parameter_objects = SortedDict()

        linear1 = oc.SNN_Linear(in_features=2, out_features=2)
        linear2 = oc.SNN_Linear(in_features=2, out_features=2)

        parameter_objects['l1'] = linear1
        parameter_objects['l2'] = linear2

        return parameter_objects

    def rhs_advect_state(self, state_dict, parameter_objects):

        rhs = SortedDict()

        s = state_dict
        p = parameter_objects

        rhs['dot_q1'] = p['l1'](self.nl(s['q2']))
        rhs['dot_q2'] = p['l2'](s['q1'])

        return rhs

    def get_initial_condition(self, x):
        # Initial condition from given data vector
        # easiest to first build a data dictionary and then call get_initial_conditions_from_data_dict(self,data_dict):
        data_dict = SortedDict()
        data_dict['q1'] = x
        data_dict['q2'] = torch.zeros_like(x)

        initial_conditions = self.get_initial_conditions_from_data_dict(data_dict=data_dict)

        return initial_conditions

    def disassemble(self,input,dim=1):
        state_dict, costate_dict, data_dict = self.disassemble_tensor(input, dim=dim)
        return data_dict['q1']

class AutoShootingBlockModelUpDown(shooting.LinearInParameterAutogradShootingBlock):
    def __init__(self, batch_y0=None, nonlinearity=None, only_random_initialization=False,transpose_state_when_forward=False):
        super(AutoShootingBlockModelUpDown, self).__init__(batch_y0=batch_y0,nonlinearity=nonlinearity,
                                                           only_random_initialization=only_random_initialization,
                                                           transpose_state_when_forward=transpose_state_when_forward)

    def create_initial_state_parameters(self, batch_y0, only_random_initialization=True):
        # creates these as a sorted dictionary and returns it (need to be in the same order!!)
        state_dict = SortedDict()

        rand_mag_q = 0.5

        self.k = batch_y0.size()[0]
        self.d = batch_y0.size()[2]

        if only_random_initialization:
            # do a fully random initialization
            state_dict['q1'] = nn.Parameter(rand_mag_q * torch.randn([self.k, 1, self.d]))
            state_dict['q2'] = nn.Parameter(rand_mag_q * torch.randn([self.k, 1, self.d*5]))
        else:
            raise ValueError('Not yet implemented')

        return state_dict

    def create_default_parameter_objects(self):

        parameter_objects = SortedDict()

        linear1 = oc.SNN_Linear(in_features=10,out_features=2)
        linear2 = oc.SNN_Linear(in_features=2,out_features=10)

        parameter_objects['l1'] = linear1
        parameter_objects['l2'] = linear2

        return parameter_objects

    def rhs_advect_state(self, state_dict, parameter_objects):

        rhs = SortedDict()

        s = state_dict
        p = parameter_objects

        rhs['dot_q1'] = p['l1'](input=self.nl(s['q2']))
        rhs['dot_q2'] = p['l2'](input=s['q1'])

        return rhs

    def get_initial_condition(self, x):
        # Initial condition from given data vector
        # easiest to first build a data dictionary and then call get_initial_conditions_from_data_dict(self,data_dict):
        data_dict = SortedDict()
        data_dict['q1'] = x

        max_dim = len(x.shape)-1

        data_dict['q2'] = torch.cat((torch.zeros_like(x),
                                     torch.zeros_like(x),
                                     torch.zeros_like(x),
                                     torch.zeros_like(x),
                                     torch.zeros_like(x)), dim=max_dim)

        initial_conditions = self.get_initial_conditions_from_data_dict(data_dict=data_dict)

        return initial_conditions

    def disassemble(self,input,dim=1):
        state_dict, costate_dict, data_dict = self.disassemble_tensor(input, dim=dim)
        return data_dict['q1']

class AutoShootingBlockModelSimple(shooting.LinearInParameterAutogradShootingBlock):
    def __init__(self, batch_y0=None, nonlinearity=None, only_random_initialization=False,transpose_state_when_forward=False):
        super(AutoShootingBlockModelSimple, self).__init__(batch_y0=batch_y0,nonlinearity=nonlinearity,
                                                           only_random_initialization=only_random_initialization,
                                                           transpose_state_when_forward=transpose_state_when_forward)

    def create_initial_state_parameters(self, batch_y0, only_random_initialization=True):
        # creates these as a sorted dictionary and returns it (need to be in the same order!!)
        state_dict = SortedDict()

        rand_mag_q = 0.5

        self.k = batch_y0.size()[0]
        self.d = batch_y0.size()[2]

        if only_random_initialization:
            # do a fully random initialization
            state_dict['q1'] = nn.Parameter(rand_mag_q * torch.randn_like(batch_y0))
        else:
            state_dict['q1'] = nn.Parameter(batch_y0 + rand_mag_q * torch.randn_like(batch_y0))

        return state_dict

    def create_default_parameter_objects(self):

        parameter_objects = SortedDict()

        linear = oc.SNN_Linear(in_features=2, out_features=2)
        parameter_objects['l1'] = linear

        return parameter_objects

    def rhs_advect_state(self, state_dict, parameter_objects):

        rhs = SortedDict()

        s = state_dict
        p = parameter_objects

        rhs['dot_q1'] = p['l1'](input=self.nl(s['q1']))

        return rhs

    def get_initial_condition(self, x):
        # Initial condition from given data vector
        # easiest to first build a data dictionary and then call get_initial_conditions_from_data_dict(self,data_dict):
        data_dict = SortedDict()
        data_dict['q1'] = x

        initial_conditions = self.get_initial_conditions_from_data_dict(data_dict=data_dict)

        return initial_conditions

    def disassemble(self,input,dim=1):
        state_dict, costate_dict, data_dict = self.disassemble_tensor(input, dim=dim)
        return data_dict['q1']


class AutoShootingBlockModelSimpleConv2d(shooting.LinearInParameterAutogradShootingBlock):
    def __init__(self, batch_y0=None, nonlinearity=None, only_random_initialization=True,transpose_state_when_forward=False, channel_number=64):
        super(AutoShootingBlockModelSimpleConv2d, self).__init__(batch_y0=batch_y0,nonlinearity=nonlinearity,
                                                                 only_random_initialization=only_random_initialization,
                                                                 transpose_state_when_forward=transpose_state_when_forward,
                                                                 channel_number=channel_number)

    def create_initial_state_parameters(self,channel_number, batch_y0, only_random_initialization=True,filter_size = 3,particle_size = 6,particle_number = 10):
        # creates these as a sorted dictionary and returns it (need to be in the same order!!)
        state_dict = SortedDict()

        rand_mag_q = 0.5

        self.filter_size = filter_size
        self.particle_size = particle_size
        self.particle_number = particle_number
        self.channel_number = channel_number

        if only_random_initialization:
            # do a fully random initialization
            state_dict['q1'] = nn.Parameter(rand_mag_q * torch.randn([self.particle_number,self.channel_number,self.particle_size,self.particle_size]))
            state_dict['q2'] = nn.Parameter(rand_mag_q * torch.randn([self.particle_number,self.channel_number,self.particle_size,self.particle_size]))
        else:
            raise ValueError('Not yet implemented')

        return state_dict

    def create_default_parameter_objects(self):

        parameter_objects = SortedDict()

        conv1 = oc.SNN_Conv2d(in_channels=self.channel_number,out_channels=self.channel_number,kernel_size=self.filter_size,padding = 1)
        conv2 = oc.SNN_Conv2d(in_channels=self.channel_number,out_channels=self.channel_number,kernel_size=self.filter_size,padding = 1)

        parameter_objects['conv1'] = conv1
        parameter_objects['conv2'] = conv2

        return parameter_objects

    def rhs_advect_state(self, state_dict, parameter_objects):

        rhs = SortedDict()

        s = state_dict
        p = parameter_objects

        rhs['dot_q1'] = p['conv1'](self.nl(s['q2']))
        rhs['dot_q2'] = p['conv2'](s['q1'])

        return rhs

    def get_initial_condition(self, x):
        # Initial condition from given data vector
        # easiest to first build a data dictionary and then call get_initial_conditions_from_data_dict(self,data_dict):
        data_dict = SortedDict()
        data_dict['q1'] = x
        data_dict['q2'] = torch.zeros_like(x)

        initial_conditions = self.get_initial_conditions_from_data_dict(data_dict=data_dict)

        return initial_conditions

    def disassemble(self,input,dim=1):
        state_dict, costate_dict, data_dict = self.disassemble_tensor(input, dim=dim)
        return data_dict['q1']


class ShootingModule(nn.Module):

    def __init__(self,shooting,method = 'rk4',rtol = 1e-8,atol = 1e-10,stepsize = 0.1):
        super(ShootingModule, self).__init__()
        self.shooting = shooting
        self.integration_time = torch.tensor([0, 1]).float()
        self.rtol = rtol
        self.atol = atol
        self.method = method

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        self.initial_condition = self.shooting.get_initial_condition(x)
        out = odeint(self.shooting, self.initial_condition, self.integration_time,method = self.method, rtol = self.rtol,atol = self.atol)
        out1 = self.shooting.disassemble(out, dim=1)
        return out1[1,...]


