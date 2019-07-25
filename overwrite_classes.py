import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _single, _pair, _triple

from collections import OrderedDict
from sortedcontainers import SortedDict

# todo: implement convenience functions so we can move datastructures to the GPU if desired

class RemoveParameters():
    def __init__(self):
        self._parameter_dict = None

    def _remove_parameters(self,name_prefix='',name_postfix=''):
        new_parameter_dict = SortedDict()
        for k in self._parameters:
            new_parameter_dict[name_prefix + str(k) + name_postfix] = self._parameters[k]
        self._parameters.clear()
        self._parameter_dict = new_parameter_dict

    def get_parameter_dict(self):
        return self._parameter_dict

class SNN_Linear(nn.Linear,RemoveParameters):
    def __init__(self, in_features, out_features, bias=True, name_prefix='',name_postfix=''):
        super(SNN_Linear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)

        self._remove_parameters(name_prefix=name_prefix,name_postfix=name_postfix)

    def forward(self, input, weight, bias):
        return F.linear(input, weight, bias)

class SNN_Conv2d(nn.Conv2d,RemoveParameters):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',name_prefix='',name_postfix=''):
        super(SNN_Conv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups,
                                         bias=bias, padding_mode=padding_mode)

        self._remove_parameters(name_prefix=name_prefix,name_postfix=name_postfix)

    def forward(self, input, weight, bias):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

def merge_parameter_dicts(parameter_dicts):

    res_dict = SortedDict()

    if (type(parameter_dicts)!=tuple) and (type(parameter_dicts)!=list):
        raise ValueError('Expected tuple or list of parameter dictionaries.')

    for current_dict in parameter_dicts:
        for k in current_dict:
            if k in res_dict:
                raise ValueError('Key {} already contained in the dictionary.'.format(k))
            res_dict[k] = current_dict[k]

    return res_dict


if __name__ == '__main__':
    # Example on how to use it

    my_conv_1 = SNN_Conv2d(2,2,3,name_postfix='_1')
    my_conv_2 = SNN_Conv2d(2,2,3,name_postfix='_2')

    parameter_dict = merge_parameter_dicts((my_conv_1.get_parameter_dict(),my_conv_2.get_parameter_dict()))

    print(parameter_dict)
