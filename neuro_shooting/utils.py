import torch

def setup_device(desired_gpu=None):
    print('Device setup:')
    print('-------------')
    if torch.cuda.is_available() and (desired_gpu is not None):
        device = torch.device('cuda:' + str(desired_gpu))
        print('Setting the default tensor type to torch.cuda.FloatTensor')
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print('Setting the cude device to {}'.format(desired_gpu))
        torch.cuda.set_device(desired_gpu)
    else:
        device = 'cpu'
        print('Setting the default tensor type to torch.FloatTensor')
        torch.set_default_tensor_type(torch.FloatTensor)
        print('Device is {}'.format(device))

    print('\n')

    return device

def compute_number_of_parameters(model):

    nr_of_fixed_parameters = 0
    nr_of_optimized_parameters = 0
    print('\nModel parameters:\n')
    print('-----------------')
    for pn, pv in model.named_parameters():
        #print('{} = {}'.format(pn, pv))
        current_number_of_parameters = np.prod(list(pv.size()))
        print('{}: # of parameters = {}\n'.format(pn,current_number_of_parameters))
        if pv.requires_grad:
            nr_of_optimized_parameters += current_number_of_parameters
        else:
            nr_of_fixed_parameters += current_number_of_parameters

    print('Number of fixed parameters = {}'.format(nr_of_fixed_parameters))
    print('Number of optimized parameters = {}'.format(nr_of_optimized_parameters))
    overall_nr_of_parameters = nr_of_fixed_parameters + nr_of_optimized_parameters
    print('Overall number of parameters = {}\n'.format(overall_nr_of_parameters))

    nr_of_pars = dict()
    nr_of_pars['fixed'] = nr_of_fixed_parameters
    nr_of_pars['optimized'] = nr_of_optimized_parameters
    nr_of_pars['overall'] = overall_nr_of_parameters

    return nr_of_pars

def freeze_parameters(shooting_block,parameters_to_freeze):

    # get all the parameters that we are optimizing over
    pars = shooting_block.state_dict()
    for pn in parameters_to_freeze:
        print('Freezing {}'.format(pn))
        pars[pn].requires_grad = False