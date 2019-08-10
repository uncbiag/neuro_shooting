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
            res = odeintadjoint(func=func,y0=x0,t=t,method=self.integrator_name,options=self.integrator_options,**self.kwargs)
            return res
        else:
            res = odeint(func=func,y0=x0,t=t,method=self.integrator_name,options=self.integrator_options,**self.kwargs)
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


class HamiltonianFlowAndCustomBackward(torch.autograd.Function):

    def __init__(self):
        super(HamiltonianFlowAndCustomBackward, self).__init__()

    @staticmethod
    def forward(ctx,input,integrator,shooting,times_integration):
        initial_condition = shooting.get_initial_condition(input)
        ctx.shooting = shooting
        ctx.integrator = integrator
        ctx.init_cond = initial_condition
        ctx.times_integration = times_integration
        out = integrator.integrate(shooting.forward,initial_condition,times_integration)
        ctx.final_condition = out
        return out


    @staticmethod
    def backward(ctx,grad_output):
        grad_input = grad_output.clone()
        integration_time_backward = -ctx.times_integration
        init_cond = ctx.init_cond
        final_condition = ctx.final_condition
        eps = 0.001
        tensor_initial_conditions = shooting.get_initial_conditions_for_backward(grad_input,final_condition,dim = 0)
        temp = final_condition + eps * shooting.symplectic_map(tensor_initial_conditions)
        out = integrator.integrate(shooting.forward,temp,integration_time_backward)
        y = out - init_cond
        grad = -shooting.symplectic_map(y) / eps
        return grad, None, None



