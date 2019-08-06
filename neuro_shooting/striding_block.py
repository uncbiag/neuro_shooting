import torch.nn as nn
from sortedcontainers import SortedDict

class ShootingStridingBlock(nn.Module):
    """
    Allows striding. Will take as an input a SortedDict of SortedDict's for the state and the costate as well as
    a dict holding the state for the data and will subsample as desired. This basically amounts to striding.
    """

    def __init__(self,stride=2,dim=2):
        super(ShootingStridingBlock, self).__init__()

        if type(stride)==tuple:
            if len(stride)!=dim:
                raise ValueError('Stride tuple needs to be of the same dimension as the dimension for the striding. Got {}, but should be {}'.format(len(stride),dim))
            else:
                self.stride = stride
        else:
            if type(stride)==int:
                self.stide = tuple([stride]*dim)
            else:
                raise ValueError('Unsupported stride type {}'.format(type(stride)))

        self.dim = dim

    def _stride_tensor(self,input,stride):

        dim_input = len(input.shape)
        if dim_input!=self.dim+2:
            raise ValueError('Dimension mismatch. Expected tensor dimension {} for batch x channel x ..., but got {}.'.format(self.dim+2,dim_input))

        # compute stride offsets to make sure we pick the center element if we have an odd number of elements
        offsets = [0]*self.dim
        for i,v in stride:
            offsets[i] = v%2

        if self.dim==1:
            if dim_input[2]<stride[0]:
                return None
            else:
                return input[:,:,offsets[0]::stride[0]]
        elif self.dim==2:
            if (dim_input[2]<stride[0]) or (dim_input[3]<stride[1]):
                return  None
            else:
                return input[:,:,offsets[0]::stride[0],offsets[1]::stride[1]]
        elif self.dim==3:
            if (dim_input[2]<stride[0]) or (dim_input[3]<stride[1]) or (dim_input[4]<stride[2]):
                return None
            else:
                return input[:, :, offsets[0]::stride[0], offsets[1]::stride[1], offsets[2]::stride[2]]
        else:
            raise ValueError('Unsupported dimension {}'.format(self.dim))

    def _stride_dict_of_dicts(self,generic_dict_of_dicts,stride=None):

        ret = SortedDict()
        for dk in generic_dict_of_dicts:
            c_generic_dict = generic_dict_of_dicts[dk]
            ret[dk] = self._stride_dict(generic_dict=c_generic_dict,stride=stride)
        return ret

    def _stride_dict(self,generic_dict,stride=None):

        ret = SortedDict()
        for k in generic_dict:
            ret[k] = self._stride_tensor(generic_dict[k],stride=stride)
        return ret


    def forward(self, state_dict_of_dicts=None,costate_dict_of_dicts=None,data_dict=None):

        # compute strided versions of all of these dictionaries and then return them
        strided_state_dict_of_dicts = self._stride_dict_of_dicts(generic_dict_of_dicts=state_dict_of_dicts,stride=self.stride)
        strided_costate_dict_of_dicts = self._stride_dict_of_dicts(generic_dict_of_dicts=costate_dict_of_dicts,stride=self.stride)
        strided_data_dict = self._stride_dict(generic_dict=data_dict,stride=self.stride)

        return strided_state_dict_of_dicts,strided_costate_dict_of_dicts,strided_data_dict
