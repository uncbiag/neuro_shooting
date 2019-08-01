import torch
import torch.nn as nn

import torchdiffeq

from torchdiffeq import odeint_adjoint as odeintadjoint
from torchdiffeq import odeint

class GenericIntegrator(object):
    def __init__(self,integrator_library = None, integrator_name = None, use_adjoint_integration=False, integrator_options=None, **kwargs):
        super(GenericIntegrator, self).__init__()

        self._available_integrator_libraries = ['odeint','anode']

        if integrator_library is None:
            self.integrator_library = 'odeint'
        else:
            self.integrator_library = integrator_library

        if self.integrator_library not in self._available_integrator_libraries:
            raise ValueError('Unknown integrator library {}'.format(self.integrator_library))

        self._available_integrators = dict()
        self._available_integrators['odeint'] = ['rk4','dopri5']
        self._available_integrators['anode'] = ['rk2','rk4']

        if integrator_name is None:
            self.integrator_name = self._available_integrators[self.integrator_library][0]
        else:
            self.integrator_name = integrator_name

        if self.integrator_name not in self._available_integrators[self.integrator_library]:
            raise ValueError('Integrator {} not available for {}'.format(self.integrator_name, self.integrator_library))

        self.use_adjoint_integration = use_adjoint_integration
        self.integrator_options = integrator_options

        self.kwargs = kwargs

        # self.rtol = rtol
        # self.atol = atol
        # self.method = method

        # self.integrator_options = dict()
        # if step_size is not None:
        #     self.integrator_options['step_size'] = step_size
        # if max_num_steps is not None:
        #     self.integrator_options['max_num_steps'] = max_num_steps


    def _integrate_odeint(self,func,x0,t):
        if self.use_adjoint_integration:
            return odeintadjoint(func=func,y0=x0,t=t,method=self.integrator_name,options=self.integrator_options,**self.kwargs)
        else:
            return odeint(func=func,y0=x0,t=t,method=self.integrator_name,options=self.integrator_options,**self.kwargs)

    def _integrate_anode(self,func,x0,t):
        raise ValueError('ANODE support not yet implemented.')

    def integrate(self,func,x0,t):
        if self.integrator_library=='odeint':
            return self._integrate_odeint(func=func,x0=x0,t=t)
        elif self.integrator_library=='anode':
            return self._integrate_anode(func=func,x0=x0,t=t)

