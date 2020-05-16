import torch
import neuro_shooting.utils as utils
import torch.optim as optim

def count_print_and_compare_parameters(net,target_parameter_numbers):

    total_number_of_parameters = 0

    named_parameters = list(net.named_parameters())

    passed_tests = 0
    failed_tests = 0

    for p in named_parameters:
        par_name = p[0]
        par_tensor = p[1]
        cur_nr_of_parameters = par_tensor.numel()
        total_number_of_parameters += cur_nr_of_parameters

        if par_name in target_parameter_numbers:
            expected_parameter_numbers = target_parameter_numbers[par_name]
            if expected_parameter_numbers==cur_nr_of_parameters:
                passed_tests += 1
                print('Name: {}; #-pars: {}: passed'.format(par_name, cur_nr_of_parameters))
            else:
                failed_tests +=1
                print('Name: {}; #-pars: {}: failed, expected: {}'.format(par_name, cur_nr_of_parameters, expected_parameter_numbers))
        else:
            failed_tests += 1
            print('Name: {}; #-pars: {}: failed as target value was not specified'.format(par_name, cur_nr_of_parameters))

    print('\n')
    print('---------------')
    print('Total number of parameters = {}\n\n'.format(total_number_of_parameters))
    print('{} of {} tests passed'.format(passed_tests,passed_tests+failed_tests))

    return total_number_of_parameters


utils.setup_random_seed(seed=1234)
utils.setup_device(desired_gpu=0)
nonlinearity = 'tanh'

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

# net = res_net.BasicResNet(nr_of_image_channels=1,
#                           layer_channels=[2,3],
#                           nr_of_blocks_per_layer=[2,2],
#                           downsampling_stride=2,
#                           particle_sizes=[[5,5],[3,3]],
#                           nr_of_particles=2,
#                           nr_of_classes=2)

layer_channels = [2,3]
downsampling_stride = 2
#particle_sizes = [[5,5],[5,5]]
particle_sizes = [[5,5],[3,3]]
nr_of_particles = 2
nr_of_classes = 2


net = res_net.BasicResNet(nr_of_image_channels=1,
                          layer_channels=layer_channels,
                          #nr_of_blocks_per_layer=[2,2],
                          nr_of_blocks_per_layer=[1,1],
                          downsampling_stride=downsampling_stride,
                          particle_sizes=particle_sizes,
                          nr_of_particles=nr_of_particles,
                          nr_of_classes=nr_of_classes)

# run it once before to dynamically allocate parameters
net(images)
net.parameters()

conv_filter_size = 3*3

pars_layer_0 = nr_of_particles*layer_channels[0]*particle_sizes[0][0]*particle_sizes[0][1]
# parameters for the additional dimension
pars_layer_1_nd = nr_of_particles*(layer_channels[1]-layer_channels[0])*particle_sizes[1][0]*particle_sizes[1][1]
# parameters for padding (to bring the strided parameters back to the same spatial dimension as for the new ones
patch_size_after_downsampling = (particle_sizes[0][0]//downsampling_stride)*(particle_sizes[0][1]//downsampling_stride)
pars_layer_1_enlarge = (particle_sizes[1][0]*particle_sizes[1][1]-patch_size_after_downsampling)*nr_of_particles*layer_channels[0]

expected_parameters = {
    'initial_conv.weight': conv_filter_size*layer_channels[0],
    'initial_conv.bias': layer_channels[0],
    'batch_norm.weight': layer_channels[0],
    'batch_norm.bias': layer_channels[0],
    'shooting_layer_0_0.q1': pars_layer_0,
    'shooting_layer_0_0.q2': pars_layer_0,
    'shooting_layer_0_0.p_q1': pars_layer_0,
    'shooting_layer_0_0.p_q2': pars_layer_0,
    'shooting_layer_1_0.q1': pars_layer_1_nd,
    'shooting_layer_1_0.q2': pars_layer_1_nd,
    'shooting_layer_1_0.p_q1': pars_layer_1_nd,
    'shooting_layer_1_0.p_q2': pars_layer_1_nd,
    'shooting_layer_1_0.pt_enlarge_q1': pars_layer_1_enlarge,
    'shooting_layer_1_0.pt_enlarge_q2': pars_layer_1_enlarge,
    'shooting_layer_1_0.pt_enlarge_p_q1': pars_layer_1_enlarge,
    'shooting_layer_1_0.pt_enlarge_p_q2': pars_layer_1_enlarge,
    'last_linear.weight': layer_channels[-1]*nr_of_classes,
    'last_linear.bias': nr_of_classes
}

count_print_and_compare_parameters(net=net,target_parameter_numbers=expected_parameters)

# just to test that all the paramters here change indeed
optimizer = optim.SGD(net.parameters(), lr=1.0, momentum=0.9)

# zero the parameter gradients
optimizer.zero_grad()

# compute gradient and do a test update
ret = net(images)
loss = torch.mean(ret)
loss.backward()

params_before = dict()
for name,p in net.named_parameters():
    params_before[name] = p.clone()

optimizer.step()

# now check that they are different
for name,p in net.named_parameters():
    diff = torch.sum(torch.abs(p-params_before[name]))
    diff_same_elements = torch.sum(p==params_before[name])
    print('Diff {}: {}; # of elements that stayed the same = {}'.format(name,diff.item(), diff_same_elements.item()))

#print(list(net.named_parameters()))
#print(net)




