import torch
from collections import defaultdict

def record_generic_dict_of_dicts(custom_hook_data, d, d_name):
    for block_name in d:
        cur_block = d[block_name]
        for cur_state_name in cur_block:
            cur_key = '{}.{}.{}'.format(block_name, d_name, cur_state_name)
            custom_hook_data[cur_key].append(cur_block[cur_state_name].detach().cpu().numpy())

def compute_parameter_norms(parameter_objects):

    parameter_norms = dict()

    # TODO: this currently will only work for vector-valued evolutions; add something so we can also assess CNNs

    # we do not want to compute norms over biases
    parameters_to_ignore = ['bias']

    sum_sqr_frobenius_norm = 0
    sum_sqr_nuclear_norm = 0
    sum_sqr_spectral_norm = 0

    for k in parameter_objects:
        par_dict = parameter_objects[k].get_parameter_dict()

        for p in par_dict:
            if p not in parameters_to_ignore:
                cur_name = 'sqr_frobenius_norm_{}.{}'.format(k,p)
                cur_val = (torch.norm(par_dict[p].detach(), p='fro')**2).cpu().numpy()
                parameter_norms[cur_name] = cur_val
                sum_sqr_frobenius_norm += cur_val

                cur_name = 'sqr_nuclear_norm_{}.{}'.format(k,p)
                cur_val = (torch.norm(par_dict[p].detach(), p='nuc')**2).cpu().numpy()
                parameter_norms[cur_name] = cur_val
                sum_sqr_nuclear_norm += cur_val

                cur_name = 'sqr_spectral_norm_{}.{}'.format(k,p)
                _, S, _ = par_dict[p].detach().svd()
                cur_val = (S[0]**2).cpu().numpy()
                parameter_norms[cur_name] = cur_val
                sum_sqr_spectral_norm += cur_val

    parameter_norms['sum_sqr_frobenius_norm'] = sum_sqr_frobenius_norm
    parameter_norms['sum_sqr_nuclear_norm'] = sum_sqr_nuclear_norm
    parameter_norms['sum_sqr_spectral_norm'] = sum_sqr_spectral_norm

    return parameter_norms

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
            cur_par_dict = parameter_objects[k].get_parameter_dict()
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

        # record norms of the parameters
        current_norms = compute_parameter_norms(parameter_objects=parameter_objects)
        for k in current_norms:
            custom_hook_data[k].append(current_norms[k])

    return None