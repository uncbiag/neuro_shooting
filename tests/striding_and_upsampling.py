import torch


def count_and_print_parameters(net):

    total_number_of_parameters = 0

    named_parameters = list(net.named_parameters())

    for p in named_parameters:
        par_name = p[0]
        par_tensor = p[1]
        cur_nr_of_parameters = par_tensor.numel()
        print('Name: {}; #-pars: {}'.format(par_name,cur_nr_of_parameters))
        total_number_of_parameters += cur_nr_of_parameters

    print('\n')
    print('---------------')
    print('Total number of parameters = {}\n\n'.format(total_number_of_parameters))

    return total_number_of_parameters

gpu = 0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
nonlinearity = 'tanh'


import torch.nn as nn
import torch.nn.functional as F

# create some random input
# random images of size 25x25 with 10 channels, and a batch of 15
images = torch.randn(3,1,11,11)


import neuro_shooting.res_net as res_net

# class BasicResNet(nn.Module):
#
#     def __init__(self,
#                  nr_of_image_channels,
#                  layer_channels=[64,128,256,512],
#                  nr_of_blocks_per_layer=[3,3,3,3],
#                  downsampling_stride=2,
#                  nonlinearity='tanh',
#                  particle_sizes=[[15,15],[11,11],[7,7],[5,5]],
#                  nr_of_particles=10,
#                  nr_of_classes=10
#                  ):

# create a minimal example so we can count the number of parameters that we expect

net = res_net.BasicResNet(nr_of_image_channels=1,
                          layer_channels=[2,3],
                          nr_of_blocks_per_layer=[2,2],
                          downsampling_stride=2,
                          particle_sizes=[[5,5],[3,3]],
                          nr_of_particles=2,
                          nr_of_classes=2)

# run it once before to dynamically allocate parameters
net(images)
net.parameters()
# once they have been allocated move everything to the GPU
net = net.to(device)

count_and_print_parameters(net)

print(list(net.named_parameters()))
print(net)




