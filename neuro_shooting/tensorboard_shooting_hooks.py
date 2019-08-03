# defines a number of different hooks which can be used to create tensorboar output from the shooting blocks

import torch
from torch.utils.tensorboard import SummaryWriter

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

def linear_transform_hook(module, t, state_dicts, costate_dicts, data_dict,
            dot_state_dicts, dot_costate_dicts, dot_data_dict, parameter_objects, custom_hook_data):

    if 'epoch' in custom_hook_data:
        epoch = custom_hook_data['epoch']
    else:
        epoch = 0

    if 'batch' in custom_hook_data:
        batch = custom_hook_data['batch']
    else:
        batch = 0

    if epoch%100==0: # only every 10th

        pars = parameter_objects['l1'].get_parameter_dict()

        weight = pars['weight']
        bias = pars['bias']

        writer.add_scalar('bias0', bias[0].item(), global_step=epoch)
        writer.add_scalar('bias1', bias[1].item(), global_step=epoch)

        writer.add_image('weight', weight, global_step=epoch, dataformats='HW')
