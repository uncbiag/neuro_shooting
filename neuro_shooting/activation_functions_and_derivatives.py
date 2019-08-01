import torch

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