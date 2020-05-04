import torch
import torch.nn as nn


try:
    from torchdiffeq import odeint_adjoint as odeintadjoint
    from torchdiffeq import odeint
    odeint_available = True
except:
    odeint_available = False

try:
    from .external.anode.adjoint import odesolver_adjoint as odesolver
    anode_available = True
except:
    anode_available = False


class GenericIntegrator(object):
    def __init__(self,integrator_library = None, integrator_name = None, use_adjoint_integration=False,
                 integrator_options=None, step_size=0.5, rtol=1e-3, atol=1e-5, **kwargs):
        """
        Generic integrator class

        :param integrator_library: 'odeint' (for NODE integrators) or 'anode' (for ANODE integrators)
        :param integrator_name: string of integrator name ('odeint': 'dopri5','adams','euler','midpoint','rk4','explicit_adams','fixed_adams; 'anode': 'rk2', 'rk4'
        :param use_adjoint_integration: if true, then ODE is solved backward to compute gradient (only for odeint)
        :param integrator_options: dictionary with additional integrator option (passed to the integrators)
        :param step_size: integrator step size (only for fixed steps-size solvers, i.e., not for dopri5/adams)
        :param rtol: relative integration tolerance (only for adaptive solvers (dopri5/adams))
        :param atol: absolute integration tolerance (only for adaptive solvers (dopri5/adams))
        :param kwargs: optional arguments passed directly to the integrator

        """
        super(GenericIntegrator, self).__init__()

        self._available_integrator_libraries = ['odeint','anode']

        if integrator_library is None:
            self.integrator_library = 'odeint'
        else:
            self.integrator_library = integrator_library

        if self.integrator_library not in self._available_integrator_libraries:
            raise ValueError('Unknown integrator library {}'.format(self.integrator_library))

        self._available_integrators = dict()
        self._available_integrators['odeint'] = ['dopri5','adams','euler','midpoint','rk4','explicit_adams','fixed_adams']
        self._available_integrators['anode'] = ['rk2','rk4']

        if integrator_name is None:
            self.integrator_name = self._available_integrators[self.integrator_library][0]
        else:
            self.integrator_name = integrator_name

        if self.integrator_name not in self._available_integrators[self.integrator_library]:
            raise ValueError('Integrator {} not available for {}'.format(self.integrator_name, self.integrator_library))

        self.use_adjoint_integration = use_adjoint_integration
        self.integrator_options = integrator_options
        if self.integrator_options is None:
            self.integrator_options = dict()

        self.kwargs = kwargs

        self.rtol = rtol
        self.atol = atol
        self.step_size = step_size

        if step_size is not None:
            if self.integrator_library == 'odeint':
                if self.integrator_name not in ['dopri5', 'adams']:
                    if 'step_size' not in self.integrator_options:
                        self.integrator_options['step_size'] = step_size

        # if max_num_steps is not None:
        #     self.integrator_options['max_num_steps'] = max_num_steps

    def _integrate_odeint(self,func,x0,t):
        if self.use_adjoint_integration:
            # odeint(func, y0, t, rtol=1e-7, atol=1e-9, method=None, options=None)
            res = odeintadjoint(func=func,y0=x0,t=t,rtol=self.rtol, atol=self.atol, method=self.integrator_name,options=self.integrator_options,**self.kwargs)
            return res
        else:
            # odeint(func, y0, t, rtol=1e-7, atol=1e-9, method=None, options=None)
            res = odeint(func=func,y0=x0,t=t, rtol=self.rtol, atol=self.atol, method=self.integrator_name,options=self.integrator_options,**self.kwargs)
            return res

    def _integrate_anode(self,func,x0,t=None):
        # todo: provide more options for stepsize-control here

        if self.integrator_options is None:
            options = dict()
        else:
            options = self.integrator_options

        # check that this is called correctly
        if t is not None:
            if len(t)==1:
                if t!=1.0:
                    raise ValueError('Warning: ANODE always integates to one. Aborting.')
            elif len(t)>2 or ((t[-1]-t[0])!=1.0):
                raise ValueError('Warning: ANODE always integrates to unit time and does not provide any intermediate values. Expect trouble when calling it this way. Aborting.')

        # everything okay, so we can proceed

        Nt = 10
        options.update({'Nt': int(Nt)})
        options.update({'method': (self.integrator_name).upper()})

        res = odesolver(func=func,z0=x0,options=options)
        # to conform with odeint, the first dimension should be time, here it only produces one time-point
        res_reshaped = res.unsqueeze(dim=0)

        return res_reshaped


    def integrate(self,func,x0,t):
        if self.integrator_library=='odeint':
            return self._integrate_odeint(func=func,x0=x0,t=t)
        elif self.integrator_library=='anode':
            return self._integrate_anode(func=func,x0=x0,t=t)

