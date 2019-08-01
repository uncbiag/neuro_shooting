import torch
import torch.nn as nn

import neuro_shooting.generic_integrator as generic_integrator

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

        if x is not None:
            initial_conditions = self.shooting_integrand.get_initial_condition(x=x)

        if (pass_through_state_dict_of_dicts is None and pass_through_costate_dict_of_dicts is not None) or \
                (pass_through_state_dict_of_dicts is None and pass_through_costate_dict_of_dicts is None):
            raise ValueError('State and costate need to be specified simultaneously')

        if pass_through_state_dict_of_dicts is not None and pass_through_costate_dict_of_dicts is not None:
            self.shooting_integrand.set_initial_pass_through_state_and_costate_parameters(state_dict_of_dicts=pass_through_state_dict_of_dicts,
                                                                                          costate_dict_of_dicts=pass_through_costate_dict_of_dicts)
        #integrate
        res_all_times = self.integrator.integrate(func=self.shooting_integrand, x0=self.initial_condition, t=self.integration_time)
        res_final = res_all_times[-1, ...]

        state_dict_of_dicts, costate_dict_of_dicts, data_dict = self.shooting_integrand.disassemble_tensor(res_final)

        # and get what should typically be returned (the transformed data)
        res = self.shooting_integrand.disassemble(res_final)

        # we return the typical return value (in case this is the last block and we want to easily integrate,
        # but also all the dictionaries so these blocks can be easily chained together

        return res,state_dict_of_dicts,costate_dict_of_dicts,data_dict

