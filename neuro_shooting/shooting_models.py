import torch
import torch.nn as nn
import neuro_shooting.shooting_integrands as shooting
import neuro_shooting.overwrite_classes as oc
import neuro_shooting.state_costate_and_data_dictionary_utils as scd_utils

from torch.nn.parameter import Parameter

from sortedcontainers import SortedDict

class AutoShootingIntegrandModelDampenedUpDown(shooting.ShootingLinearInParameterVectorIntegrand):

    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                nr_of_particles=10, particle_dimension=1, particle_size=2, parameter_weight=None, inflation_factor=5,
                *args, **kwargs):

        super(AutoShootingIntegrandModelDampenedUpDown, self).__init__(in_features=in_features,
                                                               nonlinearity=nonlinearity,
                                                               transpose_state_when_forward=transpose_state_when_forward,
                                                               concatenate_parameters=concatenate_parameters,
                                                               nr_of_particles=nr_of_particles,
                                                               particle_dimension=particle_dimension,
                                                               particle_size=particle_size,
                                                               parameter_weight=parameter_weight,
                                                               *args, **kwargs)

        print('WARNING: This model is currently not functional. Play around with it at your own risk.')

        self.inflation_factor = inflation_factor

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

        state_dict['c1'] = self._state_initializer.create_parameters(nr_of_particles=self.nr_of_particles,
                                                                     particle_size=1,
                                                                     particle_dimension=1,
                                                                     set_to_constant=0.1)

        state_dict['c2'] = self._state_initializer.create_parameters(nr_of_particles=self.nr_of_particles,
                                                                     particle_size=1,
                                                                     particle_dimension=1,
                                                                     set_to_constant=0.1)

        return state_dict

    def create_default_parameter_objects(self):

        parameter_objects = SortedDict()

        #linear1 = oc.SNN_LinearDampening(in_features=self.in_features*self.inflation_factor,out_features=self.in_features,weight=self.parameter_weight,bias=True)
        #linear2 = oc.SNN_LinearDampening(in_features=self.in_features,out_features=self.in_features*self.inflation_factor,weight=self.parameter_weight,bias=True)

        linear1 = oc.SNN_Linear(in_features=self.in_features * self.inflation_factor,out_features=self.in_features, weight=self.parameter_weight, bias=True)
        linear2 = oc.SNN_Linear(in_features=self.in_features,out_features=self.in_features * self.inflation_factor, weight=self.parameter_weight, bias=True)

        parameter_objects['l1'] = linear1
        parameter_objects['l2'] = linear2

        return parameter_objects

    def rhs_advect_state(self, t, state_dict_or_dict_of_dicts, parameter_objects):

        rhs = SortedDict()

        s = state_dict_or_dict_of_dicts
        p = parameter_objects

        # # evolution equation is
        # # \dot{q}_1 = A_1 \sigma( q_2 ) + b_1 - d_1*q1
        # # \dot{q}_2 = A_2 \sigma( q_1 ) + b_2 - d_2*q2  (last term is dampening)
        #
        # rhs['dot_q1'] = p['l1'](input=self.nl(s['q2']),dampening_input=s['q1'])
        # # TODO: maybe make this 0.1 factor a learnable parameter
        # rhs['dot_q2'] = p['l2'](input=self.nl(s['q1']),dampening_input=s['q2'])

        # evolution equation is
        # \dot{q}_1 = A_1 \sigma( q_2 ) + b_1 - d_1*q1
        # \dot{q}_2 = A_2 \sigma( q_1 ) + b_2 - d_2*q2  (last term is dampening)

        # rhs['dot_q1'] = p['l1'](input=self.nl(s['q2'])) -0.1*s['q1']
        # # TODO: maybe make this 0.1 factor a learnable parameter
        # rhs['dot_q2'] = p['l2'](input=self.nl(s['q1'])) -0.1*s['q2']

        rhs['dot_q1'] = p['l1'](input=self.nl(s['q2'])) - self.nl(s['c1']) * s['q1']
        # TODO: maybe make this 0.1 factor a learnable parameter
        rhs['dot_q2'] = p['l2'](input=self.nl(s['q1'])) - self.nl(s['c2']) * s['q2']

        rhs['dot_c1'] = torch.zeros_like(s['c1'])
        rhs['dot_c2'] = torch.zeros_like(s['c2'])

        return rhs

    def get_initial_data_dict_from_data_tensor(self, x):
        # Initial data dict from given data tensor
        data_dict = SortedDict()
        data_dict['q1'] = x

        z = torch.zeros_like(x)
        sz = [1]*len(z.shape)
        sz[-1] = self.inflation_factor

        data_dict['q2'] = z.repeat(sz)

        sz_orig = [1]*len(x.shape)
        sz_orig[0] = x.shape[0]
        data_dict['c1'] = torch.zeros(sz_orig)
        data_dict['c2'] = torch.zeros(sz_orig)

        return data_dict

    def disassemble(self,input,dim=1):
        state_dict, costate_dict, data_dicts = self.disassemble_tensor(input, dim=dim)
        return scd_utils.extract_key_from_dict_of_dicts(data_dicts,'q1')

class AutoShootingIntegrandModelSecondOrder(shooting.ShootingLinearInParameterVectorIntegrand):
    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                nr_of_particles=10, particle_dimension=1, particle_size=2,parameter_weight=None,
                *args, **kwargs):

        super(AutoShootingIntegrandModelSecondOrder, self).__init__(in_features=in_features,
                                                                    nonlinearity=nonlinearity,
                                                                    transpose_state_when_forward=transpose_state_when_forward,
                                                                    concatenate_parameters=concatenate_parameters,
                                                                    nr_of_particles=nr_of_particles,
                                                                    particle_dimension=particle_dimension,
                                                                    particle_size=particle_size,
                                                                    parameter_weight=parameter_weight,
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

        linear1 = oc.SNN_Linear(in_features=self.in_features, out_features=self.in_features, weight=self.parameter_weight)
        linear2 = oc.SNN_Linear(in_features=self.in_features, out_features=self.in_features, weight=self.parameter_weight)

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
        data_dict = SortedDict()
        data_dict['q1'] = x
        data_dict['q2'] = torch.zeros_like(x)

        return data_dict

    def disassemble(self,input,dim=1):
        state_dict, costate_dict, data_dicts = self.disassemble_tensor(input, dim=dim)
        return scd_utils.extract_key_from_dict_of_dicts(data_dicts, 'q1')

class AutoShootingIntegrandModelUpDown(shooting.ShootingLinearInParameterVectorIntegrand):

    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                nr_of_particles=10, particle_dimension=1, particle_size=2, parameter_weight=None, inflation_factor=5,
                *args, **kwargs):

        super(AutoShootingIntegrandModelUpDown, self).__init__(in_features=in_features,
                                                               nonlinearity=nonlinearity,
                                                               transpose_state_when_forward=transpose_state_when_forward,
                                                               concatenate_parameters=concatenate_parameters,
                                                               nr_of_particles=nr_of_particles,
                                                               particle_dimension=particle_dimension,
                                                               particle_size=particle_size,
                                                               parameter_weight=parameter_weight,
                                                               *args, **kwargs)

        self.inflation_factor = inflation_factor

        if self.optimize_over_data_initial_conditions:
            self.data_q20 = Parameter(torch.zeros([particle_dimension,particle_size*inflation_factor]))
        else:
            self.data_q20 = None


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

        linear1 = oc.SNN_Linear(in_features=self.in_features*self.inflation_factor,out_features=self.in_features,weight=self.parameter_weight)
        linear2 = oc.SNN_Linear(in_features=self.in_features,out_features=self.in_features*self.inflation_factor,weight=self.parameter_weight)

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
        data_dict = SortedDict()
        data_dict['q1'] = x

        if self.data_q20 is None:
            z = torch.zeros_like(x)
            sz = [1]*len(z.shape)
            sz[-1] = self.inflation_factor

            data_dict['q2'] = z.repeat(sz)
        else:
            # just repeat it for all the data
            szq20 = list(self.data_q20.shape)
            szx = list(x.shape)
            dim_diff = len(szx)-len(szq20)
            data_q20 = self.data_q20.view([1]*dim_diff+szq20)
            data_q20_replicated = data_q20.expand(szx[0:dim_diff]+[-1]*len(szq20))

            data_dict['q2'] = data_q20_replicated

        return data_dict

    def disassemble(self,input,dim=1):
        state_dict, costate_dict, data_dicts = self.disassemble_tensor(input, dim=dim)
        return scd_utils.extract_key_from_dict_of_dicts(data_dicts,'q1')

    def optional_rhs_advect_costate_analytic(self, t, state_dict_or_dict_of_dicts, costate_dict_or_dict_of_dicts, parameter_objects):
        """
        This is optional. We do not need to define this. But if we do, we can sidestep computing the co-state evolution via
        auto-diff. We can use this for example to test if the autodiff shooting approach properly recovers the analytic evolution equations.

        :param t:
        :param state_dict_of_dicts:
        :param costate_dict_of_dicts:
        :param parameter_objects:
        :return:
        """

        rhs = SortedDict()

        s = state_dict_or_dict_of_dicts
        c = costate_dict_or_dict_of_dicts
        p = parameter_objects

        par_dict1 = p['l1'].get_parameter_dict()
        par_dict2 = p['l2'].get_parameter_dict()
        l1 = par_dict1['weight']
        l2 = par_dict2['weight']


        # now compute the parameters

        q2i = s['q2']
        p1i = c['p_q1']
        p2i = c['p_q2']

        # we are computing based on the transposed quantities here (this makes the use of torch.matmul possible
        #dot_qt = torch.matmul(self.nl(qi), A.t()) + bt

        # now we can also compute the rhs of the costate (based on the manually computed shooting equations)
        # for i in range(self.nr_of_particles):
        #     dot_pt[i, ...] = -self.dnl(qi[i, ...]) * torch.matmul(pi[i, ...], A)
        dot_p2t = - self.dnl(q2i) * torch.matmul(p1i,l1)
        dot_p1t = - torch.matmul(p2i,l2)
        rhs['dot_p_q1'] = dot_p1t
        rhs['dot_p_q2'] = dot_p2t

        return rhs


    def optional_compute_parameters_analytic(self,t,state_dict, costate_dict):
        """
        This is optional. We can prescribe an analytic computation of the parameters (where we do not need to do this via autodiff).
        This is optional, but can be used for testing.

        :param t:
        :param parameter_objects:
        :param state_dict:
        :param costate_dict:
        :return:
        """

        s = state_dict
        c = costate_dict
        p = self.create_default_parameter_objects_on_consistent_device()

        # now compute the parameters
        q1i = s['q1']
        p1i = c['p_q1']
        q2i = s['q2'].transpose(1,2)
        p2i = c['p_q2'].transpose(1,2)

        temp = torch.matmul(self.nl(q2i),p1i)
        l1 = torch.mean(temp,dim = 0).t()

        temp2 = torch.matmul(p2i,q1i)
        l2 = torch.mean(temp2,dim = 0)
        # particles are saved as rows
        #At = torch.zeros(self.in_features, self.in_features)
        #for i in range(self.nr_of_particles):
        #    At = At + (pi[i, ...].t() * self.nl(qi[i, ...])).t()
        #At = 1 / self._overall_number_of_state_parameters * At  # because of the mean in the Lagrangian multiplier
        #bt = 1 / self._overall_number_of_state_parameters * pi.sum(dim=0)  # -\sum_i q_i

        # results need to be written in the respective parameter variables
        par_dict = p['l1'].get_parameter_dict()
        weight_dict = p['l1'].get_parameter_weight_dict()
        par_dict['weight'] = l1/weight_dict['weight']
        par_dict['bias'] = torch.mean(p1i,dim = 0)/weight_dict['bias']

        par_dict2 = p['l2'].get_parameter_dict()
        weight_dict2 = p['l2'].get_parameter_weight_dict()
        par_dict2['weight'] = l2/weight_dict2['weight']
        par_dict2['bias'] = (torch.mean(p2i,dim = 0).t())/weight_dict2['bias']

        return p

class DEBUGAutoShootingIntegrandModelSimple(shooting.ShootingLinearInParameterVectorIntegrand):

    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                nr_of_particles=10, particle_dimension=1, particle_size=2, parameter_weight=None,
                *args, **kwargs):

        super(DEBUGAutoShootingIntegrandModelSimple, self).__init__(in_features=in_features,
                                                               nonlinearity=nonlinearity,
                                                               transpose_state_when_forward=transpose_state_when_forward,
                                                               concatenate_parameters=concatenate_parameters,
                                                               nr_of_particles=nr_of_particles,
                                                               particle_dimension=particle_dimension,
                                                               particle_size=particle_size,
                                                               parameter_weight=parameter_weight,
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

        # TODO: make sure parameter objects are created on the right device

        parameter_objects = SortedDict()

        # todo: make this more generic again
        # dim = 2
        #linear = oc.SNN_Linear(in_features=self.d, out_features=self.d)
        linear = oc.SNN_Linear(in_features=self.in_features, out_features=self.in_features, weight=self.parameter_weight)
        parameter_objects['l1'] = linear

        linear = oc.SNN_Linear(in_features=self.in_features, out_features=self.in_features,
                               weight=self.parameter_weight)
        parameter_objects['l2'] = linear

        return parameter_objects

    def rhs_advect_state(self, t, state_dict_or_dict_of_dicts, parameter_objects):

        rhs = SortedDict()

        s = state_dict_or_dict_of_dicts
        p = parameter_objects

        rhs['dot_q1'] = p['l1'](input=self.nl(s['q1']))
        rhs['dot_q2'] = p['l2'](input=self.nl(s['q2']))

        return rhs

    def optional_rhs_advect_costate_analytic(self, t, state_dict_or_dict_of_dicts, costate_dict_or_dict_of_dicts, parameter_objects):
        """
        This is optional. We do not need to define this. But if we do, we can sidestep computing the co-state evolution via
        auto-diff. We can use this for example to test if the autodiff shooting approach properly recovers the analytic evolution equations.

        :param t:
        :param state_dict_of_dicts:
        :param costate_dict_of_dicts:
        :param parameter_objects:
        :return:
        """

        rhs = SortedDict()

        s = state_dict_or_dict_of_dicts
        c = costate_dict_or_dict_of_dicts
        p = parameter_objects

        # -----------------------

        par_dict = p['l1'].get_parameter_dict()
        A = par_dict['weight']
        bt = par_dict['bias']

        # now compute the parameters
        qi = s['q1']
        pi = c['p_q1']

        # we are computing based on the transposed quantities here (this makes the use of torch.matmul possible
        dot_qt = torch.matmul(self.nl(qi), A.t()) + bt

        # now we can also compute the rhs of the costate (based on the manually computed shooting equations)
        dot_pt = torch.zeros_like(pi)
        for i in range(self.nr_of_particles):
            dot_pt[i, ...] = -self.dnl(qi[i, ...]) * torch.matmul(pi[i, ...], A)

        rhs['dot_p_q1'] = dot_pt
        rhs['dot_p_q1'] = -self.dnl(qi)*torch.matmul(pi,A)

        # -----------------------

        par_dict = p['l2'].get_parameter_dict()
        A = par_dict['weight']
        bt = par_dict['bias']

        # now compute the parameters
        qi = s['q2']
        pi = c['p_q2']

        # we are computing based on the transposed quantities here (this makes the use of torch.matmul possible
        dot_qt = torch.matmul(self.nl(qi), A.t()) + bt

        # now we can also compute the rhs of the costate (based on the manually computed shooting equations)
        dot_pt = torch.zeros_like(pi)
        for i in range(self.nr_of_particles):
            dot_pt[i, ...] = -self.dnl(qi[i, ...]) * torch.matmul(pi[i, ...], A)

        rhs['dot_p_q2'] = dot_pt
        rhs['dot_p_q2'] = -self.dnl(qi) * torch.matmul(pi,A)

        return rhs


    def optional_compute_parameters_analytic(self,t,state_dict, costate_dict):
        """
        This is optional. We can prescribe an analytic computation of the parameters (where we do not need to do this via autodiff).
        This is optional, but can be used for testing.

        :param t:
        :param parameter_objects:
        :param state_dict:
        :param costate_dict:
        :return:
        """

        s = state_dict
        c = costate_dict
        p = self.create_default_parameter_objects_on_consistent_device()

        # ------------------

        # now compute the parameters
        qi = s['q1']
        pi = c['p_q1']

        par_dict = p['l1'].get_parameter_dict()
        device = par_dict['weight'].device
        dtype = par_dict['weight'].dtype

        # particles are saved as rows
        At = torch.zeros(self.in_features, self.in_features, device=device, dtype=dtype)
        for i in range(self.nr_of_particles):
            At = At + (pi[i, ...].t() * self.nl(qi[i, ...])).t()
        At = 1 / self.nr_of_particles * At  # because of the mean in the Lagrangian multiplier
        bt = 1 / self.nr_of_particles * pi.sum(dim=0)  # -\sum_i q_i

        # results need to be written in the respective parameter variables
        par_dict = p['l1'].get_parameter_dict()
        weight_dict = p['l1'].get_parameter_weight_dict()
        par_dict['weight'] = (At.t())/weight_dict['weight']
        par_dict['bias'] = (bt)/weight_dict['bias']

        # ------------------

        # now compute the parameters
        qi = s['q2']
        pi = c['p_q2']

        # particles are saved as rows
        At = torch.zeros(self.in_features, self.in_features, device=device, dtype=dtype)
        for i in range(self.nr_of_particles):
            At = At + (pi[i, ...].t() * self.nl(qi[i, ...])).t()
        At = 1 / self.nr_of_particles * At  # because of the mean in the Lagrangian multiplier
        bt = 1 / self.nr_of_particles * pi.sum(dim=0)  # -\sum_i q_i

        # results need to be written in the respective parameter variables
        par_dict = p['l2'].get_parameter_dict()
        weight_dict = p['l2'].get_parameter_weight_dict()
        par_dict['weight'] = (At.t())/weight_dict['weight']
        par_dict['bias'] = (bt)/weight_dict['bias']

        return p


    def get_initial_data_dict_from_data_tensor(self, x):
        # Initial data_dict for given initial data tensor
        data_dict = SortedDict()
        data_dict['q1'] = x
        data_dict['q2'] = torch.zeros_like(x)

        return data_dict

    def disassemble(self,input,dim=1):
        state_dict, costate_dict, data_dicts = self.disassemble_tensor(input, dim=dim)
        return scd_utils.extract_key_from_dict_of_dicts(data_dicts,'q1')

class AutoShootingIntegrandModelSimple(shooting.ShootingLinearInParameterVectorIntegrand):

    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                nr_of_particles=10, particle_dimension=1, particle_size=2, parameter_weight=None,
                *args, **kwargs):

        super(AutoShootingIntegrandModelSimple, self).__init__(in_features=in_features,
                                                               nonlinearity=nonlinearity,
                                                               transpose_state_when_forward=transpose_state_when_forward,
                                                               concatenate_parameters=concatenate_parameters,
                                                               nr_of_particles=nr_of_particles,
                                                               particle_dimension=particle_dimension,
                                                               particle_size=particle_size,
                                                               parameter_weight=parameter_weight,
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

        # TODO: make sure parameter objects are created on the right device

        parameter_objects = SortedDict()

        # todo: make this more generic again
        # dim = 2
        #linear = oc.SNN_Linear(in_features=self.d, out_features=self.d)
        linear = oc.SNN_Linear(in_features=self.in_features, out_features=self.in_features, weight=self.parameter_weight)
        parameter_objects['l1'] = linear

        return parameter_objects

    def rhs_advect_state(self, t, state_dict_or_dict_of_dicts, parameter_objects):

        rhs = SortedDict()

        s = state_dict_or_dict_of_dicts
        p = parameter_objects

        rhs['dot_q1'] = p['l1'](input=self.nl(s['q1']))

        return rhs

    def optional_rhs_advect_costate_analytic(self, t, state_dict_or_dict_of_dicts, costate_dict_or_dict_of_dicts, parameter_objects):
        """
        This is optional. We do not need to define this. But if we do, we can sidestep computing the co-state evolution via
        auto-diff. We can use this for example to test if the autodiff shooting approach properly recovers the analytic evolution equations.

        :param t:
        :param state_dict_of_dicts:
        :param costate_dict_of_dicts:
        :param parameter_objects:
        :return:
        """

        rhs = SortedDict()

        s = state_dict_or_dict_of_dicts
        c = costate_dict_or_dict_of_dicts
        p = parameter_objects

        par_dict = p['l1'].get_parameter_dict()
        A = par_dict['weight']
        bt = par_dict['bias']

        # now compute the parameters
        qi = s['q1']
        pi = c['p_q1']

        # we are computing based on the transposed quantities here (this makes the use of torch.matmul possible
        dot_qt = torch.matmul(self.nl(qi), A.t()) + bt

        # now we can also compute the rhs of the costate (based on the manually computed shooting equations)
        dot_pt = torch.zeros_like(pi)
        for i in range(self.nr_of_particles):
            dot_pt[i, ...] = -self.dnl(qi[i, ...]) * torch.matmul(pi[i, ...], A)

        rhs['dot_p_q1'] = dot_pt

        return rhs


    def optional_compute_parameters_analytic(self,t,state_dict, costate_dict):
        """
        This is optional. We can prescribe an analytic computation of the parameters (where we do not need to do this via autodiff).
        This is optional, but can be used for testing.

        :param t:
        :param parameter_objects:
        :param state_dict:
        :param costate_dict:
        :return:
        """

        s = state_dict
        c = costate_dict
        p = self.create_default_parameter_objects_on_consistent_device()

        # now compute the parameters
        qi = s['q1']
        pi = c['p_q1']

        par_dict = p['l1'].get_parameter_dict()
        weight_dict = p['l1'].get_parameter_weight_dict()
        device = par_dict['weight'].device
        dtype = par_dict['weight'].dtype

        # particles are saved as rows
        At = torch.zeros(self.in_features, self.in_features, device=device, dtype=dtype)
        for i in range(self.nr_of_particles):
            At = At + (pi[i, ...].t() * self.nl(qi[i, ...])).t()
        At = 1 / self.nr_of_particles * At  # because of the mean in the Lagrangian multiplier
        bt = 1 / self.nr_of_particles * pi.sum(dim=0)  # -\sum_i q_i

        # results need to be written in the respective parameter variables
        par_dict['weight'] = (At.t())/weight_dict['weight']
        par_dict['bias'] = (bt)/weight_dict['bias']

        return p


    def get_initial_data_dict_from_data_tensor(self, x):
        # Initial data_dict for given initial data tensor
        data_dict = SortedDict()
        data_dict['q1'] = x

        return data_dict

    def disassemble(self,input,dim=1):
        state_dict, costate_dict, data_dicts = self.disassemble_tensor(input, dim=dim)
        return scd_utils.extract_key_from_dict_of_dicts(data_dicts,'q1')

class AutoShootingIntegrandModelSimpleConv2D(shooting.ShootingLinearInParameterConvolutionIntegrand):

    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                nr_of_particles=10, particle_dimension=1, particle_size=2, filter_size=3, parameter_weight=None,
                *args, **kwargs):

        super(AutoShootingIntegrandModelSimpleConv2D, self).__init__(in_features=in_features,
                                                                     nonlinearity=nonlinearity,
                                                                     transpose_state_when_forward=transpose_state_when_forward,
                                                                     concatenate_parameters=concatenate_parameters,
                                                                     nr_of_particles=nr_of_particles,
                                                                     particle_dimension=particle_dimension,
                                                                     particle_size=particle_size,
                                                                     parameter_weight=parameter_weight,
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

        conv1 = oc.SNN_Conv2d(in_channels=self.in_features,out_channels=self.in_features,kernel_size=self.filter_size,padding = 1, weight=self.parameter_weight)
        conv2 = oc.SNN_Conv2d(in_channels=self.in_features,out_channels=self.in_features,kernel_size=self.filter_size,padding = 1, weight=self.parameter_weight)

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
        data_dict = SortedDict()
        data_dict['q1'] = x
        data_dict['q2'] = torch.zeros_like(x)

        return data_dict

    def disassemble(self,input,dim=1):
        state_dict, costate_dict, data_dicts = self.disassemble_tensor(input, dim=dim)
        return scd_utils.extract_key_from_dict_of_dicts(data_dicts,'q1')

class AutoShootingIntegrandModelConv2DBatch(shooting.ShootingLinearInParameterConvolutionIntegrand):

    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                 nr_of_particles=10, particle_dimension=1, particle_size=2, filter_size=3, parameter_weight=None,
                 *args, **kwargs):

        super(AutoShootingIntegrandModelConv2DBatch, self).__init__(in_features=in_features,
                                                                    nonlinearity=nonlinearity,
                                                                    transpose_state_when_forward=transpose_state_when_forward,
                                                                    concatenate_parameters=concatenate_parameters,
                                                                    nr_of_particles=nr_of_particles,
                                                                    particle_dimension=particle_dimension,
                                                                    particle_size=particle_size,
                                                                    parameter_weight=parameter_weight,
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

        conv1 = oc.SNN_Conv2d(in_channels=self.in_features,out_channels=self.in_features,kernel_size=self.filter_size,padding = 1, weight=self.parameter_weight)
        conv2 = oc.SNN_Conv2d(in_channels=self.in_features,out_channels=self.in_features,kernel_size=self.filter_size,padding = 1, weight=self.parameter_weight)
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
        data_dict = SortedDict()
        data_dict['q1'] = x
        data_dict['q2'] = torch.zeros_like(x)

        return data_dict

    def disassemble(self,input,dim=1):
        state_dict, costate_dict, data_dicts = self.disassemble_tensor(input, dim=dim)
        return scd_utils.extract_key_from_dict_of_dicts(data_dicts,'q1')

class AutoShootingOptimalTransportSimple(shooting.OptimalTransportNonLinearInParameter):
    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                nr_of_particles=10, particle_dimension=1, particle_size=2, parameter_weight=None, inflation_factor=1,
                *args, **kwargs):

        super(AutoShootingOptimalTransportSimple, self).__init__(in_features=in_features,
                                                               nonlinearity=nonlinearity,
                                                               transpose_state_when_forward=transpose_state_when_forward,
                                                               concatenate_parameters=concatenate_parameters,
                                                               nr_of_particles=nr_of_particles,
                                                               particle_dimension=particle_dimension,
                                                               particle_size=particle_size,
                                                               parameter_weight=parameter_weight,
                                                               *args, **kwargs)
        self.inflation_factor = inflation_factor

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

        # todo: make this more generic again
        dim = 2
        #linear = oc.SNN_Linear(in_features=self.d, out_features=self.d)
        linear = oc.SNN_Linear(in_features=self.in_features, out_features=self.inflation_factor * self.in_features, weight=self.parameter_weight)
        linear2 = oc.SNN_Linear(in_features=self.inflation_factor * self.in_features, out_features=self.in_features, weight=self.parameter_weight)
        parameter_objects['l1'] = linear
        parameter_objects["l2"] = linear2
        return parameter_objects

    def rhs_advect_state(self, t, state_dict_or_dict_of_dicts, parameter_objects):
        rhs = SortedDict()
        s = state_dict_or_dict_of_dicts
        p = parameter_objects
        rhs['dot_q1'] = p['l1'](self.nl(s["q2"]))
        rhs['dot_q2'] = p['l2'](s["q1"])
        return rhs

    def get_initial_data_dict_from_data_tensor(self, x):
        # Initial data_dict for given initial data tensor
        data_dict = SortedDict()
        data_dict['q1'] = x
        data_dict['q2'] = torch.zeros_like(x)
        return data_dict

    def disassemble(self,input,dim=1):
        state_dict, costate_dict, data_dicts = self.disassemble_tensor(input, dim=dim)
        return scd_utils.extract_key_from_dict_of_dicts(data_dicts,'q1')

class AutoShootingIntegrandModelUniversal(shooting.ShootingLinearInParameterVectorIntegrand):

    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                nr_of_particles=10, particle_dimension=1, particle_size=2, parameter_weight=None, inflation_factor=5,
                *args, **kwargs):

        super(AutoShootingIntegrandModelUniversal, self).__init__(in_features=in_features,
                                                               nonlinearity=nonlinearity,
                                                               transpose_state_when_forward=transpose_state_when_forward,
                                                               concatenate_parameters=concatenate_parameters,
                                                               nr_of_particles=nr_of_particles,
                                                               particle_dimension=particle_dimension,
                                                               particle_size=particle_size,
                                                               parameter_weight=parameter_weight,
                                                               *args, **kwargs)

        self.inflation_factor = inflation_factor
        self.dampening_factor = -1.0

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

        linear1 = oc.SNN_Linear(in_features=self.in_features*self.inflation_factor,out_features=self.in_features,weight=self.parameter_weight)
        linear2 = oc.SNN_Linear(in_features=self.in_features,out_features=self.in_features*self.inflation_factor,weight=self.parameter_weight)

        parameter_objects['l1'] = linear1
        parameter_objects['l2'] = linear2

        return parameter_objects

    def rhs_advect_state(self, t, state_dict_or_dict_of_dicts, parameter_objects):

        rhs = SortedDict()

        s = state_dict_or_dict_of_dicts
        p = parameter_objects

        rhs['dot_q1'] = p['l1'](input=self.nl(s['q2'])) #- self.dampening_factor * s["q1"]
        rhs['dot_q2'] = p['l2'](input=s['q1']) - self.dampening_factor * s['q2']

        return rhs

    def get_initial_data_dict_from_data_tensor(self, x):
        # Initial data dict from given data tensor
        data_dict = SortedDict()
        data_dict['q1'] = x

        z = torch.zeros_like(x)
        #z = x.clone()
        sz = [1]*len(z.shape)
        sz[-1] = self.inflation_factor

        data_dict['q2'] = z.repeat(sz)

        return data_dict

    def disassemble(self,input,dim=1):
        state_dict, costate_dict, data_dicts = self.disassemble_tensor(input, dim=dim)
        return scd_utils.extract_key_from_dict_of_dicts(data_dicts,'q1')

    def optional_rhs_advect_costate_analytic(self, t, state_dict_or_dict_of_dicts, costate_dict_or_dict_of_dicts, parameter_objects):
        """
        This is optional. We do not need to define this. But if we do, we can sidestep computing the co-state evolution via
        auto-diff. We can use this for example to test if the autodiff shooting approach properly recovers the analytic evolution equations.

        :param t:
        :param state_dict_of_dicts:
        :param costate_dict_of_dicts:
        :param parameter_objects:
        :return:
        """

        rhs = SortedDict()

        s = state_dict_or_dict_of_dicts
        c = costate_dict_or_dict_of_dicts
        p = parameter_objects

        par_dict1 = p['l1'].get_parameter_dict()
        par_dict2 = p['l2'].get_parameter_dict()
        l1 = par_dict1['weight']
        l2 = par_dict2['weight']


        # now compute the parameters

        q2i = s['q2']
        p1i = c['p_q1']
        p2i = c['p_q2']

        # we are computing based on the transposed quantities here (this makes the use of torch.matmul possible
        #dot_qt = torch.matmul(self.nl(qi), A.t()) + bt

        # now we can also compute the rhs of the costate (based on the manually computed shooting equations)
        # for i in range(self.nr_of_particles):
        #     dot_pt[i, ...] = -self.dnl(qi[i, ...]) * torch.matmul(pi[i, ...], A)
        dot_p2t = - self.dnl(q2i) * torch.matmul(p1i,l1)
        dot_p1t = - torch.matmul(p2i,l2)
        rhs['dot_p_q1'] = dot_p1t #+ self.dampening_factor * p1i
        rhs['dot_p_q2'] = dot_p2t #+ self.dampening_factor * p2i

        return rhs


    def optional_compute_parameters_analytic(self,t,state_dict, costate_dict):
        """
        This is optional. We can prescribe an analytic computation of the parameters (where we do not need to do this via autodiff).
        This is optional, but can be used for testing.

        :param t:
        :param parameter_objects:
        :param state_dict:
        :param costate_dict:
        :return:
        """

        s = state_dict
        c = costate_dict
        p = self.create_default_parameter_objects_on_consistent_device()

        # now compute the parameters
        q1i = s['q1']
        p1i = c['p_q1']
        q2i = s['q2'].transpose(1,2)
        p2i = c['p_q2'].transpose(1,2)

        temp = torch.matmul(self.nl(q2i),p1i)
        l1 = torch.mean(temp,dim = 0).t()

        temp2 = torch.matmul(p2i,q1i)
        l2 = torch.mean(temp2,dim = 0)
        # particles are saved as rows
        #At = torch.zeros(self.in_features, self.in_features)
        #for i in range(self.nr_of_particles):
        #    At = At + (pi[i, ...].t() * self.nl(qi[i, ...])).t()
        #At = 1 / self._overall_number_of_state_parameters * At  # because of the mean in the Lagrangian multiplier
        #bt = 1 / self._overall_number_of_state_parameters * pi.sum(dim=0)  # -\sum_i q_i

        # results need to be written in the respective parameter variables
        par_dict = p['l1'].get_parameter_dict()
        weight_dict = p['l1'].get_parameter_weight_dict()
        par_dict['weight'] = l1/weight_dict['weight']
        par_dict['bias'] = torch.mean(p1i,dim = 0)/weight_dict['bias']

        par_dict2 = p['l2'].get_parameter_dict()
        weight_dict2 = p['l2'].get_parameter_weight_dict()
        par_dict2['weight'] = l2/weight_dict2['weight']
        par_dict2['bias'] = (torch.mean(p2i,dim = 0).t())/weight_dict2['bias']

        return p

class AutoShootingIntegrandModelUpdownSymmetrized(shooting.ShootingLinearInParameterVectorIntegrand):

    def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
                nr_of_particles=10, particle_dimension=1, particle_size=2, parameter_weight=None, inflation_factor=5,
                *args, **kwargs):

        super(AutoShootingIntegrandModelUpdownSymmetrized, self).__init__(in_features=in_features,
                                                               nonlinearity=nonlinearity,
                                                               transpose_state_when_forward=transpose_state_when_forward,
                                                               concatenate_parameters=concatenate_parameters,
                                                               nr_of_particles=nr_of_particles,
                                                               particle_dimension=particle_dimension,
                                                               particle_size=particle_size,
                                                               parameter_weight=parameter_weight,
                                                               *args, **kwargs)

        self.inflation_factor = inflation_factor
        self.dampening_factor = -1.0

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
        state_dict['q3'] = self._state_initializer.create_parameters(nr_of_particles=self.nr_of_particles,
                                                                     particle_size=self.particle_size,
                                                                     particle_dimension=self.particle_dimension,
                                                                     set_to_zero=set_to_zero)
        return state_dict

    def create_default_parameter_objects(self):

        parameter_objects = SortedDict()

        linear1 = oc.SNN_Linear(in_features=self.in_features*self.inflation_factor,out_features=self.in_features,weight=self.parameter_weight)
        linear2 = oc.SNN_Linear(in_features=self.in_features,out_features=self.in_features*self.inflation_factor,weight=self.parameter_weight)
        linear3 = oc.SNN_Linear(in_features=self.in_features,out_features=self.in_features*self.inflation_factor,weight=self.parameter_weight)
        parameter_objects['l1'] = linear1
        parameter_objects['l2'] = linear2
        parameter_objects['l3'] = linear3

        return parameter_objects

    def rhs_advect_state(self, t, state_dict_or_dict_of_dicts, parameter_objects):

        rhs = SortedDict()

        s = state_dict_or_dict_of_dicts
        p = parameter_objects

        rhs['dot_q1'] = p['l1'](input=self.nl(s['q2']))
        rhs['dot_q2'] = p['l2'](input = self.nl(s['q3'])) #- self.dampening_factor * s['q2']
        rhs["dot_q3"] = p["l3"](input = s["q1"]) #- self.dampening_factor * s['q3']
        return rhs

    def get_initial_data_dict_from_data_tensor(self, x):
        # Initial data dict from given data tensor
        data_dict = SortedDict()
        data_dict['q1'] = x

        z = torch.zeros_like(x)
        z = x.clone()
        sz = [1]*len(z.shape)
        sz[-1] = self.inflation_factor

        data_dict['q2'] = z.repeat(sz)
        data_dict["q3"] = x
        return data_dict

    def disassemble(self,input,dim=1):
        state_dict, costate_dict, data_dicts = self.disassemble_tensor(input, dim=dim)
        return scd_utils.extract_key_from_dict_of_dicts(data_dicts,'q1')

    def optional_rhs_advect_costate_analytic(self, t, state_dict_or_dict_of_dicts, costate_dict_or_dict_of_dicts, parameter_objects):
        """
        This is optional. We do not need to define this. But if we do, we can sidestep computing the co-state evolution via
        auto-diff. We can use this for example to test if the autodiff shooting approach properly recovers the analytic evolution equations.

        :param t:
        :param state_dict_of_dicts:
        :param costate_dict_of_dicts:
        :param parameter_objects:
        :return:
        """

        rhs = SortedDict()

        s = state_dict_or_dict_of_dicts
        c = costate_dict_or_dict_of_dicts
        p = parameter_objects

        par_dict1 = p['l1'].get_parameter_dict()
        par_dict2 = p['l2'].get_parameter_dict()
        par_dict3 = p['l3'].get_parameter_dict()
        l1 = par_dict1['weight']
        l2 = par_dict2['weight']
        l3 = par_dict3['weight']

        # now compute the parameters

        q2i = s['q2']
        q1i = s['q1']
        q3i = s['q3']
        p1i = c['p_q1']
        p2i = c['p_q2']
        p3i = c['p_q3']

        # we are computing based on the transposed quantities here (this makes the use of torch.matmul possible
        #dot_qt = torch.matmul(self.nl(qi), A.t()) + bt

        # now we can also compute the rhs of the costate (based on the manually computed shooting equations)
        # for i in range(self.nr_of_particles):
        #     dot_pt[i, ...] = -self.dnl(qi[i, ...]) * torch.matmul(pi[i, ...], A)
        dot_p2t = - self.dnl(q2i) * torch.matmul(p1i,l1)
        dot_p3t = - self.dnl(q3i) * torch.matmul(p2i,l2)
        dot_p1t = - torch.matmul(p3i,l3)
        rhs['dot_p_q1'] = dot_p1t
        rhs['dot_p_q2'] = dot_p2t
        rhs['dot_p_q3'] = dot_p3t

        return rhs


    def optional_compute_parameters_analytic(self,t,state_dict, costate_dict):
        """
        This is optional. We can prescribe an analytic computation of the parameters (where we do not need to do this via autodiff).
        This is optional, but can be used for testing.

        :param t:
        :param parameter_objects:
        :param state_dict:
        :param costate_dict:
        :return:
        """

        s = state_dict
        c = costate_dict
        p = self.create_default_parameter_objects_on_consistent_device()

        # now compute the parameters
        q1i = s['q1']
        p1i = c['p_q1']
        q2i = s['q2'].transpose(1,2)
        q3i = s['q3']
        p3i = c['p_q3']
        p2i = c['p_q2'].transpose(1,2)

        temp = torch.matmul(self.nl(q2i),p1i)
        l1 = torch.mean(temp,dim = 0).t()

        temp2 = torch.matmul(p2i,self.nl(q3i))
        l2 = torch.mean(temp2,dim = 0)

        temp3 = torch.matmul(p3i,q1i)
        l3 = torch.mean(temp3,dim = 0)
        # particles are saved as rows
        #At = torch.zeros(self.in_features, self.in_features)
        #for i in range(self.nr_of_particles):
        #    At = At + (pi[i, ...].t() * self.nl(qi[i, ...])).t()
        #At = 1 / self._overall_number_of_state_parameters * At  # because of the mean in the Lagrangian multiplier
        #bt = 1 / self._overall_number_of_state_parameters * pi.sum(dim=0)  # -\sum_i q_i

        # results need to be written in the respective parameter variables
        par_dict = p['l1'].get_parameter_dict()
        weight_dict = p['l1'].get_parameter_weight_dict()
        par_dict['weight'] = l1/weight_dict['weight']
        par_dict['bias'] = torch.mean(p1i,dim = 0)/weight_dict['bias']

        par_dict2 = p['l2'].get_parameter_dict()
        weight_dict2 = p['l2'].get_parameter_weight_dict()
        par_dict2['weight'] = l2/weight_dict2['weight']
        par_dict2['bias'] = (torch.mean(p2i,dim = 0).t())/weight_dict2['bias']

        par_dict3 = p['l3'].get_parameter_dict()
        weight_dict3 = p['l3'].get_parameter_weight_dict()
        par_dict3['weight'] = l3 / weight_dict['weight']
        par_dict3['bias'] = torch.mean(p3i, dim=0) / weight_dict3['bias']
        return p


# class AutoShootingIntegrandModelUniversal(shooting.ShootingLinearInParameterVectorIntegrand):
#     def __init__(self, in_features, nonlinearity=None, transpose_state_when_forward=False, concatenate_parameters=True,
#                 nr_of_particles=10, particle_dimension=1, particle_size=2, parameter_weight=None, inflation_factor=5,
#                 *args, **kwargs):
#
#         super(AutoShootingIntegrandModelUniversal, self).__init__(in_features=in_features,
#                                                                nonlinearity=nonlinearity,
#                                                                transpose_state_when_forward=transpose_state_when_forward,
#                                                                concatenate_parameters=concatenate_parameters,
#                                                                nr_of_particles=nr_of_particles,
#                                                                particle_dimension=particle_dimension,
#                                                                particle_size=particle_size,
#                                                                parameter_weight=parameter_weight,
#                                                                *args, **kwargs)
#
#         self.inflation_factor = inflation_factor
#         self.damping_factor = -2.0
#
#     def create_initial_state_parameters(self, set_to_zero, *args, **kwargs):
#         # creates these as a sorted dictionary and returns it (need to be in the same order!!)
#         state_dict = SortedDict()
#
#         state_dict['q1'] = self._state_initializer.create_parameters(nr_of_particles=self.nr_of_particles,
#                                                                      particle_size=self.particle_size,
#                                                                      particle_dimension=self.particle_dimension,
#                                                                      set_to_zero=set_to_zero)
#
#         # make the dimension of this state 5 times bigger
#         state_dict['q2'] = self._state_initializer.create_parameters(nr_of_particles=self.nr_of_particles,
#                                                                      particle_size=self.particle_size*self.inflation_factor,
#                                                                      particle_dimension=self.particle_dimension,
#                                                                      set_to_zero=set_to_zero)
#
#         return state_dict
#
#     def create_default_parameter_objects(self):
#
#         parameter_objects = SortedDict()
#
#         linear1 = oc.SNN_Linear(in_features=self.in_features*self.inflation_factor,out_features=self.in_features,weight=self.parameter_weight)
#         linear2 = oc.SNN_Linear(in_features=self.in_features,out_features=self.in_features*self.inflation_factor,weight=self.parameter_weight)
#         linear3 = oc.SNN_Linear(in_features=self.in_features*self.inflation_factor,out_features=self.in_features*self.inflation_factor,weight=self.parameter_weight,bias = False)
#
#         parameter_objects['l1'] = linear1
#         parameter_objects['l2'] = linear2
#         parameter_objects['l3'] = linear3
#
#         return parameter_objects
#
#     def rhs_advect_state(self, t, state_dict_or_dict_of_dicts, parameter_objects):
#
#         rhs = SortedDict()
#
#         s = state_dict_or_dict_of_dicts
#         p = parameter_objects
#         #rhs['dot_q1'] = p['l1'](input=self.nl(s['q2']))
#         rhs['dot_q1'] = p['l1'](input=self.nl(s['q2'])) #- self.damping_factor * s['q1']
#         rhs['dot_q2'] = p['l2'](input=s['q1']) - self.damping_factor * s['q2'] #+ p["l3"](input=self.nl(s['q2']))
#         return rhs
#
#     def get_initial_data_dict_from_data_tensor(self, x):
#         # Initial data dict from given data tensor
#         data_dict = SortedDict()
#         data_dict['q1'] = x
#
#         z = torch.zeros_like(x)
#         #z = x.clone()
#         sz = [1]*len(z.shape)
#         sz[-1] = self.inflation_factor
#
#         data_dict['q2'] = z.repeat(sz)
#
#         return data_dict
#
#     def disassemble(self,input,dim=1):
#         state_dict, costate_dict, data_dicts = self.disassemble_tensor(input, dim=dim)
#         return scd_utils.extract_key_from_dict_of_dicts(data_dicts,'q1')
#
#     def optional_rhs_advect_costate_analytic(self, t, state_dict_or_dict_of_dicts, costate_dict_or_dict_of_dicts, parameter_objects):
#         """
#         This is optional. We do not need to define this. But if we do, we can sidestep computing the co-state evolution via
#         auto-diff. We can use this for example to test if the autodiff shooting approach properly recovers the analytic evolution equations.
#
#         :param t:
#         :param state_dict_of_dicts:
#         :param costate_dict_of_dicts:
#         :param parameter_objects:
#         :return:
#         """
#
#         rhs = SortedDict()
#
#         s = state_dict_or_dict_of_dicts
#         c = costate_dict_or_dict_of_dicts
#         p = parameter_objects
#
#         par_dict1 = p['l1'].get_parameter_dict()
#         par_dict2 = p['l2'].get_parameter_dict()
#         par_dict3 = p['l3'].get_parameter_dict()
#         l1 = par_dict1['weight']
#         l2 = par_dict2['weight']
#         l3 = par_dict3["weight"]
#         # now compute the parameters
#         q2i = s['q2']
#         p1i = c['p_q1']
#         p2i = c['p_q2']
#
#         # we are computing based on the transposed quantities here (this makes the use of torch.matmul possible
#         #dot_qt = torch.matmul(self.nl(qi), A.t()) + bt
#
#         # now we can also compute the rhs of the costate (based on the manually computed shooting equations)
#         # for i in range(self.nr_of_particles):
#         #     dot_pt[i, ...] = -self.dnl(qi[i, ...]) * torch.matmul(pi[i, ...], A)
#         dot_p2t = - self.dnl(q2i) * torch.matmul(p1i,l1)
#         temp_dot_p2 = - self.dnl(q2i) * torch.matmul(p2i,l3)
#
#         dot_p1t = - torch.matmul(p2i,l2)
#         rhs['dot_p_q1'] = dot_p1t #- self.damping_factor * p1i
#         rhs['dot_p_q2'] = dot_p2t #+ self.damping_factor * p2i  # +temp_dot_p2
#         return rhs
#     #
#     #
#     def optional_compute_parameters_analytic(self,t,state_dict, costate_dict):
#         """
#         This is optional. We can prescribe an analytic computation of the parameters (where we do not need to do this via autodiff).
#         This is optional, but can be used for testing.
#
#         :param t:
#         :param parameter_objects:
#         :param state_dict:
#         :param costate_dict:
#         :return:
#         """
#
#         s = state_dict
#         c = costate_dict
#         p = self.create_default_parameter_objects_on_consistent_device()
#
#         # now compute the parameters
#         q1i = s['q1']
#         p1i = c['p_q1']
#         q2i = s['q2'].transpose(1,2)
#         p2i = c['p_q2'].transpose(1,2)
#
#         temp = torch.matmul(self.nl(q2i),p1i)
#         l1 = torch.mean(temp,dim = 0).t()
#
#         temp2 = torch.matmul(p2i,q1i)
#         l2 = torch.mean(temp2,dim = 0)
#
#         temp = torch.matmul(p2i,self.nl(q2i).transpose(1,2))
#         l3  = torch.mean(temp,dim = 0)
#         # results need to be written in the respective parameter variables
#         par_dict = p['l1'].get_parameter_dict()
#         weight_dict = p['l1'].get_parameter_weight_dict()
#         par_dict['weight'] = l1/weight_dict['weight']
#         par_dict['bias'] = torch.mean(p1i,dim = 0)/weight_dict['bias']
#
#         par_dict2 = p['l2'].get_parameter_dict()
#         weight_dict2 = p['l2'].get_parameter_weight_dict()
#         par_dict2['weight'] = l2/weight_dict2['weight']
#         par_dict2['bias'] = (torch.mean(p2i,dim = 0).t())/weight_dict2['bias']
#
#         #par_dict3 = p['l3'].get_parameter_dict()
#         #weight_dict3 = p['l3'].get_parameter_weight_dict()
#         #par_dict3['weight'] = l3/weight_dict3['weight']
#
#         return p

# class ShootingModule(nn.Module):
#
#     def __init__(self,shooting,method = 'rk4',rtol = 1e-8,atol = 1e-10, step_size = None, max_num_steps=None):
#         super(ShootingModule, self).__init__()
#         self.shooting = shooting
#         self.integration_time = torch.tensor([0, 1]).float()
#         self.rtol = rtol
#         self.atol = atol
#         self.method = method
#         self.add_module(name='shooting',module=self.shooting)
#
#         self.integrator_options = dict()
#         if step_size is not None:
#             self.integrator_options['step_size'] = step_size
#         if max_num_steps is not None:
#             self.integrator_options['max_num_steps'] = max_num_steps
#
#     def _apply(self, fn):
#         super(ShootingModule, self)._apply(fn)
#         self.shooting._apply(fn)
#         self.integration_time = fn(self.integration_time)
#         return self
#
#     def to(self, *args, **kwargs):
#         super(ShootingModule,self).to(*args, **kwargs)
#         self.shooting.to(*args, **kwargs)
#         self.integration_time = self.integration_time.to(*args, **kwargs)
#         return self
#
#     def forward(self, x):
#         self.initial_condition = self.shooting.get_initial_condition(x)
#         out = odeint(self.shooting, self.initial_condition, self.integration_time,method = self.method, rtol = self.rtol,atol = self.atol, options=self.integrator_options)
#         out1 = self.shooting.disassemble(out, dim=1)
#         return out1[1,...]





