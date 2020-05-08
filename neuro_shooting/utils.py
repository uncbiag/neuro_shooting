import torch

def setup_device(desired_gpu=None):
    if torch.cuda.is_available() and (desired_gpu is not None):
        device = torch.device('cuda:' + str(desired_gpu))
        print('Setting the default tensor type to torch.cuda.FloatTensor')
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print('Setting the cude device to {}'.format(desired_gpu))
        torch.cuda.set_device(2)
    else:
        device = 'cpu'
        print('Setting the default tensor type to torch.FloatTensor')
        torch.set_default_tensor_type(torch.FloatTensor)
        print('Device is {}'.format(device))

    return device