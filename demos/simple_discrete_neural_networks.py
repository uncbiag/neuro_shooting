import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

def replicate_modules(module,nr_of_layers, **kwargs):
    modules = OrderedDict()
    for i in range(nr_of_layers):
        modules['l{}'.format(i)] = module(**kwargs)

    return modules

class SimpleResNetBlock(nn.Module):

    def __init__(self):
        super(SimpleResNetBlock, self).__init__()

        self.l1 = nn.Linear(1,1,bias=True)

    def forward(self, x):
        x = x + self.l1(F.relu(x))

        return x

class UpDownResNetBlock(nn.Module):

    def __init__(self, inflation_factor=5):
        super(UpDownResNetBlock, self).__init__()

        self.l1 = nn.Linear(1,inflation_factor,bias=True)
        self.l2 = nn.Linear(inflation_factor,1,bias=True)

    def forward(self, x):
        y = self.l1(F.relu(x))
        z = self.l2(F.relu(y))

        return x + z

class UpDownDoubleResNetBlock(nn.Module):

    def __init__(self, inflation_factor=5):
        super(UpDownDoubleResNetBlock, self).__init__()

        self.l1 = nn.Linear(inflation_factor,1,bias=True)
        self.l2 = nn.Linear(1,inflation_factor,bias=False)

    def forward(self, x1x2):
        x1 = x1x2[0]
        x2 = x1x2[1]

        x1 = x1 + self.l1(F.relu(x2))
        x2 = x2 + self.l2(F.relu(x1)) # this is what an integrator would typically do
        #x2 = self.l2(F.relu(x1))

        return x1, x2

class DoubleResNetUpDown(nn.Module):

    def __init__(self, nr_of_layers=30, inflation_factor=5):
        super(DoubleResNetUpDown, self).__init__()
        print("nr_of_layers ",nr_of_layers)
        modules = replicate_modules(module=UpDownDoubleResNetBlock,nr_of_layers=nr_of_layers, inflation_factor=inflation_factor)
        self.model = nn.Sequential(modules)

    def forward(self, x1x2):
        return self.model(x1x2)

class ResNetUpDown(nn.Module):

    def __init__(self, nr_of_layers=10, inflation_factor=5):
        super(ResNetUpDown, self).__init__()
        modules = replicate_modules(module=UpDownResNetBlock, nr_of_layers=nr_of_layers, inflation_factor=inflation_factor)
        self.model = nn.Sequential(modules)

    def forward(self, x):
        return self.model(x)

class ResNet(nn.Module): # corresponds to our simple shooting model

    def __init__(self, nr_of_layers=10):
        super(ResNet, self).__init__()

        modules = replicate_modules(module=SimpleResNetBlock, nr_of_layers=nr_of_layers)
        self.model = nn.Sequential(modules)

    def forward(self, x):
        return self.model(x)

class DoubleResNetUpDownRNN(nn.Module): # corresponds to our simple shooting model

    def __init__(self, nr_of_layers=10, inflation_factor=5):
        super(DoubleResNetUpDownRNN, self).__init__()

        self.nr_of_layers = nr_of_layers
        print("use "+str(self) + " with " + str(self.nr_of_layers) + " nr of layers")
        self.l1 = UpDownDoubleResNetBlock(inflation_factor=inflation_factor)

    def forward(self, x1x2):
        x1 = x1x2[0]
        x2 = x1x2[1]

        for i in range(self.nr_of_layers):
            x1, x2 = self.l1((x1, x2))

        return x1, x2

class ResNetRNN(nn.Module):

    def __init__(self, nr_of_layers=10, inflation_factor=5):
        super(ResNetRNN, self).__init__()
        self.nr_of_layers = nr_of_layers
        print("use "+ str(self) + " nr of layers " + str(self.nr_of_layers))

        self.l1 = nn.Linear(inflation_factor,1,bias=True)
        self.l2 = nn.Linear(1,inflation_factor,bias=True)

    def forward(self, x):
        for i in range(self.nr_of_layers):
            x = x + 1./self.nr_of_layers * self.l1(F.relu(self.l2(x)))

        return x

class ODESimpleFunc(nn.Module):

    def __init__(self):
        super(ODESimpleFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 1),
            nn.Tanh(),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)
