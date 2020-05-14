import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.utils.checkpoint as checkpoint


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


class IntegrationWrapper(nn.Module):
    def __init__(self,integration_fcn,func):
        super(IntegrationWrapper, self).__init__()
        self.func = func
        self.integration_fcn = integration_fcn

    def forward(self,t,x0, dummy_arg=None):
        # this dummy arg needs to be such that it requires gradient, does not need to be used, but otherwise the autograd tape breaks
        # See
        # here: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
        # and here: https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
        assert dummy_arg is not None
        assert dummy_arg.requires_grad is True

        return self.integration_fcn(func=self.func,x0=x0,t=t)

class GenericIntegrator(object):
    def __init__(self,integrator_library = None, integrator_name = None, use_adjoint_integration=False,
                 integrator_options=None, step_size=0.5, rtol=1e-3, atol=1e-5, nr_of_checkpoints=None, checkpointing_time_interval=None, **kwargs):
        """
        Generic integrator class

        :param integrator_library: 'odeint' (for NODE integrators) or 'anode' (for ANODE integrators)
        :param integrator_name: string of integrator name ('odeint': 'dopri5','adams','euler','midpoint','rk4','explicit_adams','fixed_adams; 'anode': 'rk2', 'rk4'
        :param use_adjoint_integration: if true, then ODE is solved backward to compute gradient (only for odeint)
        :param integrator_options: dictionary with additional integrator option (passed to the integrators)
        :param step_size: integrator step size (only for fixed steps-size solvers, i.e., not for dopri5/adams)
        :param rtol: relative integration tolerance (only for adaptive solvers (dopri5/adams))
        :param atol: absolute integration tolerance (only for adaptive solvers (dopri5/adams))
        :param nr_of_checkpoints: supports more memory efficient inregration by adding checkpoints uniform in time.
        :param checkpointing_time_interval: intead of defining the number of checkpoints (which dynamically adapts to the integration time, we can also define the desired time-interval between checkpoints
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

        self.nr_of_checkpoints = nr_of_checkpoints
        self.checkpointing_time_interval = checkpointing_time_interval

        # dummy tensor to support checkpointing
        self._dummy_tensor = torch.ones(1, requires_grad=True)

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


    def create_integration_time_intervals(self,t):

        if len(t)<2:
            raise ValueError('Expected a range of time-points, but only got {}'.format(t))

        if (self.nr_of_checkpoints is not None) and (self.checkpointing_time_interval is not None):
            raise ValueError('nr_of_checkpoints and checkpointing_time_interval cannot both be set. Set one or the other')

        t_from = t[0]
        t_to = t[-1]

        if self.nr_of_checkpoints is not None:
            checkpointing_time_points = torch.linspace(t_from,t_to,self.nr_of_checkpoints+2) # if we want one checkpoint we need three points, hence +2
        elif self.checkpointing_time_interval is not None:
            checkpointing_time_points = torch.arange(t_from,t_to,self.checkpointing_time_interval)
            if checkpointing_time_points[-1]!=t_to:
                # append it
                checkpointing_time_points = torch.cat((checkpointing_time_points, torch.tensor([t_to])), dim=0)
        else:
            raise ValueError('Either nr_of_checkpoints or checkpointing_time_interval needs to be set.')

        # force the last time-points to numericall agree
        checkpointing_time_points[-1] = t_to

        # now we need to create the intervals (i.e., match the integration time-points we want to hit, to the checkpoints)
        time_intervals = []
        output_time_points = []

        idx_t = 0
        nr_t = len(t)
        idx_checkpoint_t = 0
        nr_checkpoint_t = len(checkpointing_time_points)

        keep_accumulating = True

        # always starts with the first-timepoint
        current_time_interval = torch.tensor([t[idx_t]])
        current_output_time_point = torch.tensor([True])

        if t[idx_t]!=checkpointing_time_points[idx_checkpoint_t]:
            raise ValueError('Need to start with the same time.')

        idx_t += 1
        idx_checkpoint_t += 1

        while keep_accumulating:

            next_t = t[idx_t]
            next_cp_t = checkpointing_time_points[idx_checkpoint_t]

            if next_cp_t>next_t:
                # case: just keep on adding this time-point to the current time-interval and retain it for the output
                current_time_interval = torch.cat((current_time_interval,torch.tensor([next_t])))
                current_output_time_point = torch.cat((current_output_time_point,torch.tensor([True])))
                idx_t += 1
            elif next_cp_t<next_t:
                # case: this is the checkpoint we want, so finalize it and move on to the next one
                current_time_interval = torch.cat((current_time_interval,torch.tensor([next_cp_t])))
                current_output_time_point = torch.cat((current_output_time_point,torch.tensor([False])))
                time_intervals.append(current_time_interval)
                output_time_points.append(current_output_time_point)
                current_time_interval = torch.tensor([next_cp_t])
                current_output_time_point = torch.tensor([False])
                idx_checkpoint_t += 1
            else: # the same
                # case: they conincide, so move on to the next for both, but only keep one
                current_time_interval = torch.cat((current_time_interval,torch.tensor([next_cp_t])))
                current_output_time_point = torch.cat((current_output_time_point,torch.tensor([True])))
                time_intervals.append(current_time_interval)
                output_time_points.append(current_output_time_point)
                current_time_interval = torch.tensor([next_cp_t])
                current_output_time_point = torch.tensor([False])
                idx_t += 1
                idx_checkpoint_t += 1

            # let's see if we are at the end
            if (idx_t>=nr_t) or (idx_checkpoint_t>=nr_checkpoint_t):
                keep_accumulating = False

        return time_intervals, output_time_points


    def _integrate_checkpointed(self,func,x0,t):

        # first get the checkpointed time-inntervals
        integration_times, output_time_points = self.create_integration_time_intervals(t=t)

        current_x0 = x0
        overall_integration_results = None

        # now let's chunk the solutions together

        integrator = IntegrationWrapper(integration_fcn=self._integrate_direct,func=func)

        for current_integration_times, current_output_time_points in zip(integration_times,output_time_points):

            current_res = checkpoint.checkpoint(integrator, current_integration_times, current_x0, self._dummy_tensor)
            current_x0 = current_res[-1,...]

            if overall_integration_results is None:
                overall_integration_results = current_res[current_output_time_points,...]
            else:
                overall_integration_results = torch.cat((overall_integration_results,current_res[current_output_time_points,...]),dim=0)

        return overall_integration_results

    def _integrate_direct(self, func, x0, t):
        if self.integrator_library == 'odeint':
            return self._integrate_odeint(func=func, x0=x0, t=t)
        elif self.integrator_library == 'anode':
            return self._integrate_anode(func=func, x0=x0, t=t)

    def integrate(self,func,x0,t):

        if (self.nr_of_checkpoints is None) and (self.checkpointing_time_interval is None):
            return self._integrate_direct(func=func,x0=x0,t=t)
        else:
            # do chunk-based-integration
            return self._integrate_checkpointed(func=func,x0=x0,t=t)

