import torch
import torch.nn as nn

def softmax(x,epsilon = 1.0):
  return x*(torch.ones_like(x))/(torch.exp(-x*epsilon) + torch.ones_like(x))


def dsoftmax(x,epsilon = 1.0):
  return epsilon*softmax(x,epsilon)*(torch.ones_like(x))/(torch.exp(epsilon*x) + torch.ones_like(x)) + (torch.ones_like(x))/(torch.exp(-x*epsilon) + torch.ones_like(x))

def drelu(x):
    # derivative of relu
    res = (x>=0)
    res = res.type(x.type())
    return res

def dtanh(x):
    # derivative of tanh
    return 1.0-torch.tanh(x)**2

def identity(x):
    return x

def didentity(x):
    return torch.ones_like(x)

def get_nonlinearity(nonlinearity):
    """
    Returns the desired nonlinearity and its derivative as a tuple. Currently supported nonlinearities are:
    identity, relu, tanh, sigmoid, and softmax.

    :param nonlinearity: as a string: 'identity', 'relu', 'tanh', 'sigmoid', 'softmax'
    :return: tuple (nonlinearity,derivative of nonlinearity)
    """

    supported_nonlinearities = ['identity', 'relu', 'tanh', 'sigmoid', 'softmax']

    if nonlinearity is None:
        use_nonlinearity = 'identity'
    else:
        use_nonlinearity = nonlinearity.lower()

    if use_nonlinearity not in supported_nonlinearities:
        raise ValueError('Unsupported nonlinearity {}'.format(use_nonlinearity))

    if use_nonlinearity == 'relu':
        nl = nn.functional.relu
        dnl = drelu
    elif use_nonlinearity == 'tanh':
        nl = torch.tanh
        dnl = dtanh
    elif use_nonlinearity == 'identity':
        nl = identity
        dnl = didentity
    elif use_nonlinearity == 'sigmoid':
        nl = torch.sigmoid
        dnl = torch.sigmoid
    elif use_nonlinearity == 'softmax':
        nl = softmax
        dnl = dsoftmax
    else:
        raise ValueError('Unknown nonlinearity {}'.format(use_nonlinearity))

    return nl,dnl
