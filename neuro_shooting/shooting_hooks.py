import torch
from collections import defaultdict

def record_generic_dict_of_dicts(custom_hook_data, d, d_name):
    for block_name in d:
        cur_block = d[block_name]
        for cur_state_name in cur_block:
            cur_key = '{}.{}.{}'.format(block_name, d_name, cur_state_name)
            custom_hook_data[cur_key].append(cur_block[cur_state_name].detach().cpu().numpy())


def parameter_evolution_hook(module, t, state_dicts, costate_dicts, data_dict_of_dicts,
                             dot_state_dicts, dot_costate_dicts, dot_data_dict_of_dicts, parameter_objects,
                             custom_hook_data):

    if type(custom_hook_data) is not type(defaultdict(list)):
        raise ValueError('Expected custom_hook_data to be of type {}, instead got type {}'.format(type(defaultdict(list)),type(custom_hook_data)))

    with torch.no_grad():

        # record time
        custom_hook_data['t'].append(t.item())

        current_energy = torch.zeros(1)
        # record all parameters
        for k in parameter_objects:
            cur_par_dict = parameter_objects[k]._parameter_dict
            for p in cur_par_dict:
                cur_key = '{}.{}'.format(k, p)
                custom_hook_data[cur_key].append(cur_par_dict[p].detach().cpu().numpy())
                # add to current energy
                current_energy += 0.5 * torch.sum(cur_par_dict[p] ** 2)

        # record the current energy
        custom_hook_data['energy'].append(current_energy.item())

        # now record all the states
        record_generic_dict_of_dicts(custom_hook_data=custom_hook_data, d=state_dicts, d_name='state')
        # now record all the costates
        record_generic_dict_of_dicts(custom_hook_data=custom_hook_data, d=costate_dicts, d_name='costate')
        # now record all the data states
        record_generic_dict_of_dicts(custom_hook_data=custom_hook_data, d=data_dict_of_dicts, d_name='data')

        # now record all the current derivatives
        record_generic_dict_of_dicts(custom_hook_data=custom_hook_data, d=dot_state_dicts, d_name='dot_state')
        record_generic_dict_of_dicts(custom_hook_data=custom_hook_data, d=dot_costate_dicts, d_name='dot_costate')
        record_generic_dict_of_dicts(custom_hook_data=custom_hook_data, d=dot_data_dict_of_dicts, d_name='dot_data')

    return None