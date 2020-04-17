import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _single, _pair, _triple

from sortedcontainers import SortedDict

# todo: implement convenience functions so we can move datastructures to the GPU if desired

class RemoveParameters(object):
    def __init__(self,weight=None):
        """
        Removes the parameters and replaces them by variables.
        :param weight: if specified this scalar is used to weight this parameter in the energy. \
                    Can also be specified individually via the add_weight method (if different weights for the different parameters are desired)
        """
        super(RemoveParameters, self).__init__()
        self._parameter_dict = None
        self._parameter_weight_dict = SortedDict()
        self._weight = weight

    def _remove_parameters(self):
        new_parameter_dict = SortedDict()
        for k in self._parameters:
            #new_parameter_dict[k] = self._parameters[k]

            if self._parameters[k] is not None:
                new_parameter_dict[k] = torch.zeros_like(self._parameters[k],requires_grad=True)

            # # gets rid of the variables like self.bias or self.weight (so there is no confusion afterwards)
            # setattr(self,k,None)
            # delattr(self,k)

        self._parameters.clear()
        self._parameter_dict = new_parameter_dict

        # if there is a global weight specified, associate it with all the parameters
        if self._weight is not None:
            self._parameter_weight_dict = SortedDict()
            for k in self._parameter_dict:
                self._parameter_weight_dict[k] = self._weight

    def add_weight(self,parameter_name,parameter_weight):
        if parameter_name not in self._parameter_dict:
            raise ValueError('Key {} not found in parameter_dict, cannot assign a weight'.format(parameter_name))
        self._parameter_weight_dict[parameter_name] = parameter_weight

    def get_parameter_dict(self):
        return self._parameter_dict

    def get_parameter_weight_dict(self):
        return self._parameter_weight_dict

    def _apply(self,fn):

        for k in self._parameter_dict:
            self._parameter_dict[k] = fn(self._parameter_dict[k])

        return self

    def to(self, *args, **kwargs):

        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)

        if dtype is not None:
            if not dtype.is_floating_point:
                raise TypeError('nn.Module.to only accepts floating point '
                                'dtypes, but got desired dtype={}'.format(dtype))

        def convert(t):
            return t.to(device, dtype if t.is_floating_point() else None, non_blocking)

        for k in self._parameter_dict:
            self._parameter_dict[k] = convert(self._parameter_dict[k])

        for k in self._parameter_weight_dict:
            if type(self._parameter_weight_dict[k]) is torch.Tensor:
                self._parameter_weight_dict[k] = convert(self._parameter_weight_dict[k])

        return self

class SNN_Linear(nn.Linear,RemoveParameters):
    def __init__(self, in_features, out_features, bias=True, weight=None):
        super(SNN_Linear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        RemoveParameters.__init__(self, weight=weight)
        self._remove_parameters()

    def _apply(self, fn):
        nn.Linear._apply(self,fn)
        RemoveParameters._apply(self,fn)
        return self

    def to(self, *args, **kwargs):
        nn.Linear.to(self,*args, **kwargs)
        RemoveParameters.to(self,*args, **kwargs)
        return self

    def forward(self, input):
        if 'bias' in self._parameter_dict:
            return F.linear(input, self._parameter_dict['weight'], self._parameter_dict['bias'])
        else:
            return F.linear(input, self._parameter_dict['weight'])

class SNN_Conv2d(nn.Conv2d,RemoveParameters):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', weight=None):
        super(SNN_Conv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups,
                                         bias=bias, padding_mode=padding_mode)
        RemoveParameters.__init__(self, weight=weight)
        self._remove_parameters()

    def _apply(self, fn):
        nn.Conv2d._apply(self,fn)
        RemoveParameters._apply(self,fn)
        return self

    def to(self, *args, **kwargs):
        nn.Conv2d.to(self,*args, **kwargs)
        RemoveParameters.to(self,*args, **kwargs)
        return self

    def forward(self, input):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            self._parameter_dict['weight'], self._parameter_dict['bias'], self.stride,
                            _pair(0), self.dilation, self.groups)

        return F.conv2d(input, self._parameter_dict['weight'], self._parameter_dict['bias'], self.stride,
                        self.padding, self.dilation, self.groups)


class SNN_GroupNorm(nn.GroupNorm,RemoveParameters):
    def __init__(self, num_groups, num_channels, affine = True,
                 eps=1e-05, weight=None):
        super(SNN_GroupNorm, self).__init__(num_groups=num_groups, num_channels=num_channels,
                                         affine=affine, eps=eps)
        RemoveParameters.__init__(self, weight=weight)
        if affine:
            self._remove_parameters()
        else:
            self._parameter_dict  = SortedDict()

    def _apply(self, fn):
        nn.GroupNorm._apply(self,fn)
        RemoveParameters._apply(self,fn)
        return self

    def to(self, *args, **kwargs):
        nn.GroupNorm.to(self,*args, **kwargs)
        RemoveParameters.to(self,*args, **kwargs)
        return self


    def forward(self, input):
        return F.group_norm(
            input, self.num_groups, self._parameter_dict['weight'], self._parameter_dict['bias'], self.eps)



