import torch
import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.shooting_models as shooting_models

gpu = 0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
nonlinearity = 'tanh'

# create some random input
# random images of size 25x25 with 10 channels, and a batch of 15
sample_batch = torch.randn(15,10,25,25)

shooting_model_1 = shooting_models.AutoShootingIntegrandModelSimpleConv2D(in_features=10, nonlinearity=nonlinearity,nr_of_particles=10,particle_size=[7,7],particle_dimension=10)
# this is simply pass through (as no new particles are being created)
shooting_model_2 = shooting_models.AutoShootingIntegrandModelSimpleConv2D(in_features=10, nonlinearity=nonlinearity,nr_of_particles=None)

shooting_block_1 = shooting_blocks.ShootingBlockBase(name='block1', shooting_integrand=shooting_model_1)
shooting_block_2 = shooting_blocks.ShootingBlockBase(name='block2', shooting_integrand=shooting_model_2)

shooting_block_1 = shooting_block_1.to(device)
shooting_block_2 = shooting_block_2.to(device)

ret1,state_dicts1,costate_dicts1,data_dicts1 = shooting_block_1(x=sample_batch)
ret2,state_dicts2,costate_dicts2,data_dicts2 = shooting_block_2(data_dict_of_dicts=data_dicts1,
                                                               pass_through_state_dict_of_dicts=state_dicts1,
                                                               pass_through_costate_dict_of_dicts=costate_dicts1)


ret1.sum() # there is only one value, so it returns a tensor
dummy_loss = ret2.sum()
#dummy_loss = ret2['block1']['q1'].sum() # there are multiple values from the different blocks. Need to specify which one is desired
dummy_loss.backward()
