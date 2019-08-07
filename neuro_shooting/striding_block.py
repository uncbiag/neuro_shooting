import torch.nn as nn
from sortedcontainers import SortedDict
import neuro_shooting.state_costate_and_data_dictionary_utils as scd_utils

class ShootingStridingBlock(nn.Module):
    """
    Allows striding. Will take as an input a SortedDict of SortedDict's for the state and the costate as well as
    a dict holding the state for the data and will subsample as desired. This basically amounts to striding.
    """

    def __init__(self,stride,stride_dims):
        super(ShootingStridingBlock, self).__init__()

        if type(stride_dims)==int:
            self.stride_dims = tuple([stride_dims])
        elif type(stride_dims)==tuple:
            self.stride_dims = stride_dims
        elif type(stride_dims)==list:
            self.stride_dims = tuple(stride_dims)
        else:
            raise ValueError('Unsupported stride dim type {}. Stride dims need to be specified either as an int or as a tuple or a list of ints.'.format(type(stride_dims)))

        nr_of_stride_dims = len(self.stride_dims)

        if type(stride)==int:
            self.stride = tuple([stride]*nr_of_stride_dims)
        elif type(stride)==tuple:
            if len(stride)!=nr_of_stride_dims:
                raise ValueError('Stride tuple needs to be of the same dimension as the dimension for the striding. Got {}, but should be {}'.format(len(stride),nr_of_stride_dims))
            else:
                self.stride = stride
        elif type(stride)==list:
            if len(stride)!=nr_of_stride_dims:
                raise ValueError('Stride tuple needs to be of the same dimension as the dimension for the striding. Got {}, but should be {}'.format(len(stride),nr_of_stride_dims))
            else:
                self.stride = tuple(stride)
        else:
            raise ValueError('Unsupported stride type {}. Strides need to be specified either as an int or as a tuple or a list of ints.'.format(type(stride)))

    def _stride_tensor_multiple_dimensions(self,input,stride,stride_dims,stride_dim_offset):

        current_tensor = input
        for s,d in zip(stride,stride_dims):
            current_tensor = self._stride_tensor_single_dimension(current_tensor,stride=s,stride_dim=d+stride_dim_offset)
        return current_tensor

    def _stride_tensor_single_dimension(self,input,stride,stride_dim):

        dim_input = len(input.shape)
        if stride_dim>dim_input-1:
            raise ValueError('Dimension mismatch. Stride dimension is too large. Expected stride dimension at most {}, but got {}.'.format(dim_input-1,stride_dim))

        if stride_dim<2:
            raise ValueError('Striding in batch or channel dimension is not supported.')

        # compute stride offsets to make sure we pick the center element if we have an odd number of elements
        # todo: check that this offset calculation makes sense also for strides that are not 2
        offset = input.shape[stride_dim]%stride

        if stride_dim==2:
            return input[:,:,offset::stride,...]
        elif stride_dim==3:
            return input[:,:,:,offset::stride,...]
        elif stride_dim==4:
            return input[:,:,:,:,offset::stride,...]
        elif stride_dim==5:
            return input[:,:,:,:,:,offset::stride,...]
        else:
            raise ValueError('Unsupported dimension {}'.format(stride_dim))

    def _stride_dict_of_dicts(self,generic_dict_of_dicts,stride=None,stride_dims=None,stride_dim_offset=0):

        ret = SortedDict()
        for dk in generic_dict_of_dicts:
            c_generic_dict = generic_dict_of_dicts[dk]
            ret[dk] = self._stride_dict(generic_dict=c_generic_dict,stride=stride,stride_dims=stride_dims,stride_dim_offset=stride_dim_offset)
        return ret

    def _stride_dict(self,generic_dict,stride=None,stride_dims=None,stride_dim_offset=None):

        ret = SortedDict()
        for k in generic_dict:
            ret[k] = self._stride_tensor_multiple_dimensions(generic_dict[k],stride=stride,stride_dims=stride_dims,stride_dim_offset=stride_dim_offset)
        return ret


    def forward(self, state_dict_of_dicts=None,costate_dict_of_dicts=None,data_dict_of_dicts=None):

        # compute strided versions of all of these dictionaries and then return them
        strided_state_dict_of_dicts = self._stride_dict_of_dicts(generic_dict_of_dicts=state_dict_of_dicts,stride=self.stride,stride_dims=self.stride_dims)
        strided_costate_dict_of_dicts = self._stride_dict_of_dicts(generic_dict_of_dicts=costate_dict_of_dicts,stride=self.stride,stride_dims=self.stride_dims)

        sz_state = scd_utils.extract_size_from_dict_of_dicts(state_dict_of_dicts)
        sz_data = scd_utils.extract_size_from_dict_of_dicts(data_dict_of_dicts)

        # this is to account for cases where there are many time-samples of the data (for linear layers)
        # todo: check that this works properly for the convolution models
        stride_dim_offset = len(sz_data)-len(sz_state)
        strided_data_dict_of_dicts = self._stride_dict_of_dicts(generic_dict_of_dicts=data_dict_of_dicts,stride=self.stride,stride_dims=self.stride_dims,stride_dim_offset=stride_dim_offset)

        return strided_state_dict_of_dicts,strided_costate_dict_of_dicts,strided_data_dict_of_dicts
