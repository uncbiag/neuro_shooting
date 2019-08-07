"""
Various convenience functions to assemble and disassemble state, costate, and data dictionaries.
"""

import torch
from sortedcontainers import SortedDict

def _assemble_generic_dict(d):
    """
    Given a SortedDict returns its vectorized version and a plan (a dictionary of sizes) on how to reassemble it.
    Given an empty directory it will return None, None

    :param d: sorted dictionary to vectorize
    :return: tuple: vectorized dictionary, assembly plan
    """

    # d is a sorted dictionary
    # first test that this assumption is true
    if type(d) != SortedDict:
        raise ValueError('Expected a SortedDict, but got {}'.format(type(d)))

    d_list = []
    assembly_plan = SortedDict()
    for k in d:
        d_list.append(d[k].contiguous().view(-1))  # entirely flatten is (shape is stored by assembly plan)
        assembly_plan[k] = d[k].shape

    if len(d_list) > 0:
        ret = torch.cat(tuple(d_list))
    else:
        # was an empty directory
        ret = None
        assembly_plan = None

    return ret, assembly_plan


def _assemble_generic_dict_of_dicts(d):
    """
    Similar to _assemble_generic_dict, but works for SortedDict's which contain other SortedDicts (as needed for the
    state and the costate).

    :param d: Input dictionary containing a SortedDict (which itself contains SortedDict entries).
    :return: returns a tuple of a vectorized dictionary and the associated assembly plan
    """

    assembly_plans = dict()
    ret = None
    for dk in d:
        current_ret, current_plan = _assemble_generic_dict(d[dk])
        if current_ret is not None:
            assembly_plans[dk] = current_plan
            if ret is None:
                ret = current_ret
            else:
                ret = torch.cat((ret, current_ret))
    return ret, assembly_plans


def assemble_tensor(state_dict_of_dicts, costate_dict_of_dicts, data_dict_of_dicts):
    """
    Vectorize all dictionaries together (state, costate, and data). Also returns all their assembly plans.

    :param state_dict: SortedDict holding the SortedDict's of the states
    :param costate_dict: SortedDict holding the SortedDict's of the costate
    :param data_dict: SortedDict holding SortedDict's of the state for the transported data
    :return: vectorized dictonaries (as one vecctor) and their assembly plans
    """

    # these are all ordered dictionaries, will assemble all into a big torch vector
    state_vector, state_assembly_plans = _assemble_generic_dict_of_dicts(state_dict_of_dicts)
    costate_vector, costate_assembly_plans = _assemble_generic_dict_of_dicts(costate_dict_of_dicts)
    data_vector, data_assembly_plan = _assemble_generic_dict_of_dicts(data_dict_of_dicts)

    assembly_plans = dict()
    assembly_plans['state_dicts'] = state_assembly_plans
    assembly_plans['costate_dicts'] = costate_assembly_plans
    assembly_plans['data_dicts'] = data_assembly_plan

    return torch.cat((state_vector, costate_vector, data_vector)), assembly_plans

def _disassemble_dict(input, assembly_plan, dim, incr):
    """
    Disassembles an input vector into its corresponding dictionary structure, given an assembly plan, a dimension,
    and an increment (as to where to start in the input vector).

    :param input: Input vector
    :param assembly_plan: Assembly plan to disassemble the vector into the dictionary (as created by assemble_tensor).
    :param dim: dimension (should typically be set to 0, set it to 1 in case there are for example multiple time-points stored in the zero dimension.
    :param incr: offset (specifying where to start in the vector)
    :return: tuple of the disassembled dictionary (as a SortedDict) and the increment (incr) indicating a new starting location (for the next call to _dissassemble)
    """

    ret_dict = SortedDict()

    for k in assembly_plan:
        current_shape = assembly_plan[k]
        len_shape = torch.prod(torch.tensor(current_shape)).item()

        if dim == 0:
            ret_dict[k] = (input[incr:incr + len_shape]).view(current_shape)
        elif dim == 1:
            first_dim = input.shape[0]
            all_shape = torch.Size([first_dim] + list(current_shape))
            ret_dict[k] = (input[:, incr:incr + len_shape]).view(all_shape)
        else:
            raise ValueError('Currently only supported for dims 0 or 1, but got dim = {}'.format(dim))

        incr += len_shape

    return ret_dict, incr


def _disassemble_dict_of_dicts(input, assembly_plans, dim, incr):
    """
    Similar to _disassemble_dict, but applies to a dictionary of dictionaries that is supposed to be disassembled.

    :param input: Input vector
    :param assembly_plans: Assembly plan to disassemble the vector into the dictionary of dictionaries (as created by assemble_tensor).
    :param dim: dimension (should typically be set to 0, set it to 1 in case there are for example multiple time-points stored in the zero dimension.
    :param incr: offset (specifying where to start in the vector)
    :return: tuple of the disassembled dictionary of dictionaries (as a SortedDict) and the increment (incr) indicating a new starting location (for the next call to _dissassemble)
    """

    ret_dict_of_dicts = SortedDict()

    for cp in assembly_plans:
        assembly_plan = assembly_plans[cp]
        ret_dict_of_dicts[cp], incr = _disassemble_dict(input=input, assembly_plan=assembly_plan, dim=dim,incr=incr)

    return ret_dict_of_dicts, incr


def disassemble_tensor(input, assembly_plans=None, dim=0):
    """
    Disassembles an input vector into state, data, and costate directories.

    :param input: input tensor (vector)
    :param assembly_plans: assembly_plans (does not need to be specified if previously computed -- will be cached)
    :param dim: integrator may add a 0-th dimension to keep track of time. In this case use dim=1, otherwise dim=0 should be fine.
    :return: tuple holding the state, costate, and data dictionaries
    """

    # will create sorted dictionaries for state, costate and data based on the assembly plans

    supported_dims = [0,1]
    if dim not in supported_dims:
        raise ValueError('Only supports dimensions 0 and 1; if 1, then the 0-th dimension is time')


    if assembly_plans is None:
        raise ValueError('No assembly plan specified and none was previously stored automatically (for example by calling get_initial_conditions_from_data_dict).')

    state_dicts = None
    costate_dicts = None
    data_dicts = None

    incr = 0
    for ap in ['state_dicts','costate_dicts','data_dicts']:

        if ap=='state_dicts':
            state_dicts, incr = _disassemble_dict_of_dicts(input=input, assembly_plans = assembly_plans[ap], dim=dim, incr=incr)
        elif ap=='costate_dicts':
            costate_dicts, incr = _disassemble_dict_of_dicts(input=input, assembly_plans = assembly_plans[ap], dim=dim, incr=incr)
        elif ap=='data_dicts':
            data_dicts, incr = _disassemble_dict_of_dicts(input=input, assembly_plans = assembly_plans[ap], dim=dim, incr=incr)
        else:
            raise ValueError('Unknown dictionary assembly plan kind {}'.format(ap))

    return state_dicts,costate_dicts,data_dicts


def _merge_state_costate_or_data_dict_with_generic_dict_of_dicts(generic_dict, generic_dict_of_dicts, generic_dict_block_name):
    """
    To keep the interface reasonably easy it is often desired to add a state or costate dictionary
    to a dictionary of dictionaries (which already contains various state or costate dictionaries) to obtain
    a combined dictionary of dictionaries. As the entries of the dictionary of dictionaries are named (based on
    what block created them) the dictionary is added based on the name of the current block. It can only be
    addded if the name has not been used before. Hence it is essential to use unique names when using multiple
    blocks in a system that are being chained together.

    :param generic_dict: a Sorted Dict
    :param generic_dict_of_dicts: a SortedDict of SortedDicts's
    :param generic_dict_block_name: name of the generic_dict block
    :return: merges both and returns a SortedDict of SortedDict's with the combined entries
    """

    if generic_dict_of_dicts is not None:
        if generic_dict_block_name in generic_dict_of_dicts:
            raise ValueError('Block name {} already taken. Cannot be added to dict of dicts.'.format(generic_dict_block_name))

    ret_dict_of_dicts = SortedDict()
    # now add the individual one
    if generic_dict is not None:
        ret_dict_of_dicts[generic_dict_block_name] = generic_dict

    if generic_dict_of_dicts is not None:
        # todo: maybe there is an easier way to copy these dictionary entries instead of always looping over the entries (copy? deepcopy?)
        # first create the same key structure as in the generic_dict_of_dicts (we are not copying as we do not want to change the keys of generic_dict_of_dicts)
        for dk in generic_dict_of_dicts:
            ret_dict_of_dicts[dk] = SortedDict()
            c_generic_dict = generic_dict_of_dicts[dk]
            c_ret_dict = ret_dict_of_dicts[dk]
            for k in c_generic_dict:
                c_ret_dict[k] = c_generic_dict[k]

    return ret_dict_of_dicts


def extract_dict_from_tuple_based_on_generic_dict(data_tuple, generic_dict, prefix=''):
    """
    Autodiff's autograd requires specifying variables to differentiate with respect to as tuples. Hence, this is a convenience
    function which takes such a tuple and generates a dictionary from it based on the dictionary structure defined via
    generic_dict (it's values are not used). If desired a prefix for the generated dictionary keys can be specified.

    :param data_tuple: tuple of variables
    :param generic_dict: SortedDict providing the desired dictionary structure (ideally the one used to create the tuple in the first place)
    :param prefix: text prefix for the generated keys
    :return: returns a SortedDict containing the data from the tuple
    """

    extracted_dict = SortedDict()
    indx = 0
    for k in generic_dict:
        extracted_dict[prefix + k] = data_tuple[indx]
        indx += 1

    return extracted_dict


def extract_dict_of_dicts_from_tuple_based_on_generic_dict_of_dicts(data_tuple, generic_dict_of_dicts, prefix=''):
    """
    Similar to extract_dict_from_tuple_based_on_generic_dict, but creates dict_of_dicts from the tuple. I.e., the tuple
    must have been created from a SortedDict containing SortedDict's.

    :param data_tuple: tuple of variables
    :param generic_dict_of_dicts: SortedDict of SortedDict's providing the desired dictionary structure (ideally the one used to create the tuple in the first place)
    :param prefix: text prefix for the generated keys
    :return: returns a SortedDict containing the data from the tuple
    """

    extracted_dicts = SortedDict()
    indx = 0
    for k in generic_dict_of_dicts:
        extracted_dicts[k] = SortedDict()
        c_extracted_dict = extracted_dicts[k]
        c_generic_dict = generic_dict_of_dicts[k]
        for m in c_generic_dict:
            c_extracted_dict[prefix + m] = data_tuple[indx]
            indx += 1
    return extracted_dicts


def extract_dict_from_tuple_based_on_parameter_objects(data_tuple, parameter_objects, prefix=''):
    """
    Similar to extract_dict_from_tuple_based_on_generic_dict, but is based on the SortedDict of parameter objects
    (which contain all the parameters required to evolve the states; e.g., convolutional filter coefficients.)

    :param data_tuple: tuple of variables
    :param parameter_objects: SortedDict of parameter objects
    :param prefix: text prefix for the generated keys
    :return: returns a SortedDict containing the data from the tuple
    """

    extracted_dict = SortedDict()
    indx = 0

    for o in parameter_objects:
        extracted_dict[o] = SortedDict()
        current_extracted_dict = extracted_dict[o]
        current_pars = parameter_objects[o].get_parameter_dict()

        for k in current_pars:
            current_extracted_dict[prefix + k] = data_tuple[indx]
            indx += 1

    return extracted_dict


def compute_tuple_from_generic_dict_of_dicts(generic_dict_of_dicts):
    """
    Given a SortedDict of SortedDict's (e.g., for the states or the costates) this method returns a tuple of its entries.

    :param generic_dict_of_dicts: SortedDict of SortedDict's (for example holding the states or the costates)
    :return: Returns a tuple of the dictionary entries
    """

    # form a tuple of all the state variables (because this is what we take the derivative of)
    sv_list = []
    for k in generic_dict_of_dicts:
        c_generic_dict = generic_dict_of_dicts[k]
        for m in c_generic_dict:
            sv_list.append(c_generic_dict[m])

    return tuple(sv_list)


def compute_tuple_from_generic_dict(generic_dict):
    """

    :param generic_dict:
    :return:
    """

    # form a tuple of all the state variables (because this is what we take the derivative of)
    sv_list = []
    for k in generic_dict:
        sv_list.append(generic_dict[k])

    return tuple(sv_list)


def compute_tuple_from_parameter_objects(parameter_objects):
    # form a tuple of all the variables (because this is what we take the derivative of)

    sv_list = []
    for o in parameter_objects:
        current_pars = parameter_objects[o].get_parameter_dict()
        for k in current_pars:
            sv_list.append((current_pars[k]))

    return tuple(sv_list)


def _concatenate_dict_of_dicts(generic_dict_of_dicts,concatenation_dim):
    concatenated_dict = SortedDict()
    for dk in generic_dict_of_dicts:
        c_generic_dict = generic_dict_of_dicts[dk]
        for k in c_generic_dict:
            if k not in concatenated_dict:
                concatenated_dict[k] = c_generic_dict[k]
            else:
                # should be concatenated along the feature channel

                t_shape1 = concatenated_dict[k].size()
                t_shape2 = c_generic_dict[k].size()

                try:
                    concatenated_dict[k] = torch.cat((concatenated_dict[k], c_generic_dict[k]), dim=concatenation_dim)
                except:
                    raise ValueError(
                        'Dimension mismatch when trying to concatenate tensor of shape {} and {} along dimension 1.'.format(
                            t_shape1, t_shape2))

    # lastly check that we have the same number of keys, otherwise throw an error
    nr_of_resulting_keys = len(concatenated_dict.keys())
    nr_of_expected_keys = len((generic_dict_of_dicts.peekitem(0)[1]).keys())

    if nr_of_resulting_keys != nr_of_expected_keys:
        raise ValueError('Expected {} different keys, but got {}.'.format(nr_of_expected_keys, nr_of_resulting_keys))

    return concatenated_dict


def _deconcatenate_based_on_generic_dict_of_dicts(concatenated_dict, generic_dict_of_dicts, concatenation_dim):
    # deconcatenate along the specified dimension
    ret = SortedDict()
    indx = dict()
    for dk in generic_dict_of_dicts:
        ret[dk] = SortedDict()
        c_ret = ret[dk]
        c_generic_dict = generic_dict_of_dicts[dk]
        for kc, k in zip(concatenated_dict, c_generic_dict):
            if k not in indx:
                indx[k] = 0
            t_shape = c_generic_dict[k].size()
            if concatenation_dim==0:
                c_ret[kc] = concatenated_dict[kc][indx[k]:indx[k] + t_shape[0], ...]
                indx[k] += t_shape[0]
            elif concatenation_dim==1:
                c_ret[kc] = concatenated_dict[kc][:, indx[k]:indx[k] + t_shape[1], ...]
                indx[k] += t_shape[1]
            elif concatenation_dim==2:
                c_ret[kc] = concatenated_dict[kc][:, :, indx[k]:indx[k] + t_shape[2], ...]
                indx[k] += t_shape[2]
            elif concatenation_dim==3:
                c_ret[kc] = concatenated_dict[kc][:, :, :, indx[k]:indx[k] + t_shape[3], ...]
                indx[k] += t_shape[3]
            elif concatenation_dim==4:
                c_ret[kc] = concatenated_dict[kc][:, :, :, :, indx[k]:indx[k] + t_shape[4], ...]
                indx[k] += t_shape[4]
            else:
                raise ValueError('Deconcatenation is only supported for dimensions 0-4; you requested {} instead.'.format(concatenation_dim))

    return ret

def _get_zero_dict_like(dict_like):
    ret = SortedDict()
    for k in dict_like:
        ret[k] = torch.zeros_like(dict_like[k])
    return ret

def extract_batch_size_from_dict_of_dicts(dict_of_dicts):
    sz = extract_size_from_dict_of_dicts(dict_of_dicts)
    return sz[0]

def extract_channel_size_from_dict_of_dicts(dict_of_dicts):
    sz = extract_size_from_dict_of_dicts(dict_of_dicts)
    return sz[1]

def extract_size_from_dict_of_dicts(dict_of_dicts):
    sample_tensor = dict_of_dicts.peekitem(0)[1].peekitem(0)[1]
    sz = sample_tensor.size()
    return sz

def extract_size_from_dict(d):
    sample_tensor = d.peekitem(0)[1]
    sz = sample_tensor.size()
    return sz

def _get_zero_dict_like_with_matched_batch_size(dict_like,batch_size):
    ret = SortedDict()
    for k in dict_like:
        sz = list(dict_like[k].size())
        sz[0] = batch_size
        ret[k] = torch.zeros(tuple(sz),device=dict_like[k].device,dtype=dict_like[k].dtype)
    return ret

def _get_zero_dict_like_with_matched_batch_and_channel_size(dict_like,batch_size,channel_size):
    ret = SortedDict()
    for k in dict_like:
        sz = list(dict_like[k].size())
        sz[0] = batch_size
        sz[1] = channel_size
        ret[k] = torch.zeros(tuple(sz),device=dict_like[k].device,dtype=dict_like[k].dtype)
    return ret

def _get_zero_data_dict_matching_state_dim(state_dict,data_dict_of_dicts,state_concatenation_dim,data_concatenation_dim):

    # todo: not clear if this would work if we have differently sized elements in the dictionary (for now likely okay, because we auto-enlarge)
    sample_size_data = list(extract_size_from_dict_of_dicts(data_dict_of_dicts))

    ret = SortedDict()
    for k in state_dict:
        current_tensor = state_dict[k]
        current_size = list(current_tensor.size())

        target_sz = sample_size_data
        target_sz[data_concatenation_dim] = current_size[state_concatenation_dim]

        ret[k] = torch.zeros(tuple(target_sz),device=state_dict[k].device,dtype=state_dict[k].dtype)

    return ret

def get_data_concatenation_dim(state_dict,data_dict_of_dicts,state_concatenation_dim):

    if state_dict is None:
        return None

    sample_size_data = list(extract_size_from_dict_of_dicts(data_dict_of_dicts))
    sample_size_state = list(extract_size_from_dict(state_dict))

    diff_dim = len(sample_size_data) - len(sample_size_state)
    if diff_dim < 0:
        raise ValueError('Data dimension is expected to be at least as large as the state dimension')

    data_concatenation_dim = state_concatenation_dim + diff_dim

    return data_concatenation_dim


def extract_key_from_dict_of_dicts(dict_of_dicts,key):
    ret = SortedDict()
    for dk in dict_of_dicts:
        ret[dk] = SortedDict()
        c_dict = dict_of_dicts[dk]
        c_ret = ret[dk]
        if key in c_dict:
            c_ret[key] = c_dict[key]

    if len(ret.values())==1:
        # only one block, then just return this one as a SortedDict and not as a SortedDict of SortedDict's
        return ret.peekitem(0)[1].peekitem(0)[1]
    else:
        return ret