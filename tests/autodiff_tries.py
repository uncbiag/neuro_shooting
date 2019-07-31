import torch

from torch.autograd import gradcheck
import torch.autograd as autograd

import torch.nn as nn
from torch.autograd import Function,Variable
import torch.nn.functional as F
import numpy as np

def softmax(x,epsilon = 1.0):
   return x*(torch.ones_like(x))/(torch.exp(-x*epsilon) + torch.ones_like(x))

def dsoftmax(x,epsilon = 1.0):
   return epsilon*softmax(x,epsilon)*(torch.ones_like(x))/(torch.exp(epsilon*x) + torch.ones_like(x)) + (torch.ones_like(x))/(torch.exp(-x*epsilon) + torch.ones_like(x))

def myFunction(x):
   #temp = torch.zeros_like(x)
   return x**3

def myFunctionDiff(x):
   temp = torch.zeros_like(x)
   #return 0.5*x + 0.5*(x-temp)
   return 3*x**2

def myFunction2Manual(p,x):
   return myFunctionDiff(x) * p

def myFunction2AutoDiff(p,x):
   #z = x
   #z = Variable(x, requires_grad=True)
   compute = torch.sum(p * myFunction(x))
   #compute.backward()
   #return x.grad

   xgrad, = autograd.grad(compute, x,
                              grad_outputs=compute.data.new(compute.shape).fill_(1),
                              create_graph=True)

   return xgrad


def myLoss(p,x):
   #return torch.sum(myFunction2AutoDiff(p,x)*x) #+ torch.sum(p*myFunction(x))
   return torch.sum(myFunction2AutoDiff(p,x)) + torch.sum(p*myFunction(x))

def myLossManual(p,x):
   #return torch.sum(myFunction2Manual(p,x) * x) #+ torch.sum(p * myFunction(x))
   return torch.sum(myFunction2Manual(p,x)) + torch.sum(p * myFunction(x))


if __name__ == '__main__':
   size =(2,4,5)
   z = torch.randn(size)
   x = Variable(z.data,requires_grad = True)
   xx = Variable(z.data, requires_grad=True)
   p = torch.randn(size, requires_grad=True)

   a = myFunction2AutoDiff(p,x)
   b = myFunction2Manual(p,x)
   print(torch.sum((a-b)**2))
   print(torch.sum(a**2))

   loss = myLoss(p,x)
   print("loss autodiff",loss)
   loss.backward()
   print(x.grad)

   loss2 = myLossManual(p,xx)
   print("loss manual",loss2)
   loss2.backward()
   print(xx.grad)

   print('difference:')
   print(x.grad-xx.grad)

   #test = gradcheck(myLoss,(x,p), eps=1e-4, atol=1e-10, raise_exception=True)
   #print(test)

