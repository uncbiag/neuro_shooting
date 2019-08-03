import torch
import torch.nn as nn
import neuro_shooting.shooting_integrands as shooting
import neuro_shooting.overwrite_classes as oc
from sortedcontainers import SortedDict
from torchdiffeq import odeint

class AutoShootingIntegrandModelSecondOrder(shooting.LinearInParameterAutogradShootingIntegrand):
    def __init__(self, name, batch_y0=None, nonlinearity=None, only_random_initialization=False,transpose_state_when_forward=False):
        super(AutoShootingIntegrandModelSecondOrder, self).__init__(name=name, batch_y0=batch_y0, nonlinearity=nonlinearity,
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

    def rhs_advect_state(self, t, state_dict_or_dict_of_dicts, parameter_objects):

        rhs = SortedDict()

        s = state_dict_or_dict_of_dicts
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

class AutoShootingIntegrandModelUpDown(shooting.LinearInParameterAutogradShootingIntegrand):
    def __init__(self, name, batch_y0=None, nonlinearity=None, only_random_initialization=False,transpose_state_when_forward=False):
        super(AutoShootingIntegrandModelUpDown, self).__init__(name=name, batch_y0=batch_y0, nonlinearity=nonlinearity,
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

    def rhs_advect_state(self, t, state_dict_or_dict_of_dicts, parameter_objects):

        rhs = SortedDict()

        s = state_dict_or_dict_of_dicts
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

class AutoShootingIntegrandModelSimple(shooting.LinearInParameterAutogradShootingIntegrand):
    def __init__(self, name, batch_y0=None, nonlinearity=None, only_random_initialization=False,transpose_state_when_forward=False):
        super(AutoShootingIntegrandModelSimple, self).__init__(name=name, batch_y0=batch_y0, nonlinearity=nonlinearity,
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

        linear = oc.SNN_Linear(in_features=self.d, out_features=self.d)
        parameter_objects['l1'] = linear

        return parameter_objects

    def rhs_advect_state(self, t, state_dict_or_dict_of_dicts, parameter_objects):

        rhs = SortedDict()

        s = state_dict_or_dict_of_dicts
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


class ChainedShootingBlockExample(nn.Module):
    def __init__(self,batch_y0=None, nonlinearity=None, only_random_initialization=False,transpose_state_when_forward=False):
        super(ChainedShootingBlockExample, self).__init__()

        self.block1 = ChainableAutoShootingIntegrandModelSimple(name='simple1', batch_y0=batch_y0, only_random_initialization=True, nonlinearity=nonlinearity)
        self.block2 = ChainableAutoShootingIntegrandModelSimple(name='simple2', batch_y0=batch_y0, only_random_initialization=True, nonlinearity=nonlinearity, pass_through=True)

        #self.integration_time = torch.tensor([0, 1]).float()

        self.method = 'rk4'

        self.rtol = 1e-8
        self.atol = 1e-10
        self.step_size = 0.25

        self.add_module(name='block1', module=self.block1)
        self.add_module(name='block2', module=self.block2)

        self.integrator_options = dict()
        self.integrator_options['step_size'] = self.step_size

    def get_initial_condition(self, x):
        return self.block1.get_initial_condition(x)

    def assemble_tensor(self,state_dict_of_dicts,costate_dict_of_dicts,data_dict):
        return self.block1.assemble_tensor(state_dict_of_dicts==state_dict_of_dicts,costate_dict_of_dicts=costate_dict_of_dicts,data_dict=data_dict)

    def disassemble(self, input, dim=0):
        return self.block2.disassemble(input=input, dim=dim)

    def get_norm_penalty(self):
        return self.block1.get_norm_penalty()

    def forward(self, t, x):

        out1 = odeint(self.block1, x, t, method=self.method, rtol=self.rtol,
                     atol=self.atol, options=self.integrator_options)

        out1_final = out1[-1,...]

        state_dict_of_dicts, costate_dict_of_dicts, data_dict = self.block1.disassemble_tensor(out1_final)

        self.block2.set_initial_pass_through_state_and_costate_parameters(state_dict_of_dicts=state_dict_of_dicts,
                                                                          costate_dict_of_dicts=costate_dict_of_dicts)

        block2_init = self.block2.get_initial_conditions_from_data_dict(data_dict=data_dict)

        out2 = odeint(self.block2, block2_init, t, method=self.method, rtol=self.rtol,
                      atol=self.atol, options=self.integrator_options)

        #return out2[-1, ...]
        return out2


class ChainableAutoShootingIntegrandModelSimple(shooting.LinearInParameterAutogradShootingIntegrand):
    def __init__(self, name, batch_y0=None, nonlinearity=None, only_random_initialization=False,transpose_state_when_forward=False, pass_through=False):
        self.pass_through = pass_through

        super(ChainableAutoShootingIntegrandModelSimple, self).__init__(name=name, batch_y0=batch_y0, nonlinearity=nonlinearity,
                                                                        only_random_initialization=only_random_initialization,
                                                                        transpose_state_when_forward=transpose_state_when_forward)


    def create_initial_state_parameters(self, batch_y0, only_random_initialization=True):
        # creates these as a sorted dictionary and returns it (need to be in the same order!!)
        state_dict = SortedDict()

        self.k = batch_y0.size()[0]
        self.d = batch_y0.size()[2]

        if not self.pass_through:

            rand_mag_q = 0.5

            if only_random_initialization:
                # do a fully random initialization
                state_dict['q1'] = nn.Parameter(rand_mag_q * torch.randn_like(batch_y0))
            else:
                state_dict['q1'] = nn.Parameter(batch_y0 + rand_mag_q * torch.randn_like(batch_y0))

        return state_dict

    def create_default_parameter_objects(self):

        parameter_objects = SortedDict()

        linear = oc.SNN_Linear(in_features=self.d, out_features=self.d)
        parameter_objects['l1'] = linear

        return parameter_objects

    def rhs_advect_state(self, t, state_dict_or_dict_of_dicts, parameter_objects):

        rhs = SortedDict()

        s = state_dict_or_dict_of_dicts
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


class AutoShootingIntegrandModelSimpleConv2D(shooting.LinearInParameterAutogradShootingIntegrand):
    def __init__(self, name, batch_y0=None, nonlinearity=None, only_random_initialization=True,transpose_state_when_forward=False,
                 channel_number=64,
                 filter_size=3,
                 particle_size=9,
                 particle_number=25):
        self.filter_size = filter_size
        self.particle_size = particle_size
        self.particle_number = particle_number
        self.channel_number = channel_number

        super(AutoShootingIntegrandModelSimpleConv2D, self).__init__(name=name,
                                                                     batch_y0=batch_y0, nonlinearity=nonlinearity,
                                                                     only_random_initialization=only_random_initialization,
                                                                     transpose_state_when_forward=transpose_state_when_forward,
                                                                     channel_number=channel_number,
                                                                     filter_size=filter_size,
                                                                     particle_size=particle_size,
                                                                     particle_number=particle_number)


    def create_initial_state_parameters(self,channel_number, batch_y0, only_random_initialization=True,filter_size = 3,particle_size = 8,particle_number = 10):
        # creates these as a sorted dictionary and returns it (need to be in the same order!!)
        state_dict = SortedDict()

        rand_mag_q = 1.

        #self.filter_size = filter_size
        #self.particle_size = particle_size
        #self.particle_number = particle_number
        #self.channel_number = channel_number

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

    def rhs_advect_state(self, t, state_dict_or_dict_of_dicts, parameter_objects):

        rhs = SortedDict()

        s = state_dict_or_dict_of_dicts
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

class AutoShootingIntegrandModelConv2DBatch(shooting.LinearInParameterAutogradShootingIntegrand):
    def __init__(self, name, batch_y0=None, nonlinearity=None, only_random_initialization=True,transpose_state_when_forward=False,
                 channel_number=64,
                 filter_size=3,
                 particle_size=6,
                 particle_number=25):

        super(AutoShootingIntegrandModelConv2DBatch, self).__init__(name=name,
                                                                    batch_y0=batch_y0, nonlinearity=nonlinearity,
                                                                    only_random_initialization=only_random_initialization,
                                                                    transpose_state_when_forward=transpose_state_when_forward,
                                                                    channel_number=channel_number,
                                                                    filter_size=filter_size,
                                                                    particle_size=particle_size,
                                                                    particle_number=particle_number)

    def create_initial_state_parameters(self,channel_number, batch_y0, only_random_initialization=True,filter_size = 3,particle_size = 8,particle_number = 10):
        # creates these as a sorted dictionary and returns it (need to be in the same order!!)
        state_dict = SortedDict()

        rand_mag_q = 1.

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
        #group_norm = oc.SNN_GroupNorm(self.channel_number,self.channel_number,affine = False)
        group_norm = nn.GroupNorm(self.channel_number,self.channel_number,affine = False)
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

    def __init__(self,shooting,method = 'rk4',rtol = 1e-8,atol = 1e-10, step_size = None, max_num_steps=None):
        super(ShootingModule, self).__init__()
        self.shooting = shooting
        self.integration_time = torch.tensor([0, 1]).float()
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.add_module(name='shooting',module=self.shooting)

        self.integrator_options = dict()
        if step_size is not None:
            self.integrator_options['step_size'] = step_size
        if max_num_steps is not None:
            self.integrator_options['max_num_steps'] = max_num_steps

    def _apply(self, fn):
        super(ShootingModule, self)._apply(fn)
        self.shooting._apply(fn)
        self.integration_time = fn(self.integration_time)
        return self

    def to(self, *args, **kwargs):
        super(ShootingModule,self).to(*args, **kwargs)
        self.shooting.to(*args, **kwargs)
        self.integration_time = self.integration_time.to(*args, **kwargs)
        return self

    def forward(self, x):
        self.initial_condition = self.shooting.get_initial_condition(x)
        out = odeint(self.shooting, self.initial_condition, self.integration_time,method = self.method, rtol = self.rtol,atol = self.atol, options=self.integrator_options)
        out1 = self.shooting.disassemble(out, dim=1)
        return out1[1,...]



