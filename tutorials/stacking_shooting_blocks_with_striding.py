import torch
import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.shooting_models as shooting_models
import neuro_shooting.striding_block as striding_block
import numpy as np

gpu = 0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
nonlinearity = 'tanh'

# create some random input
#torch.Size([10, 10, 1, 2])
sample_batch = torch.randn(50,10,1,2)

shooting_model = shooting_models.AutoShootingIntegrandModelUpDown(nonlinearity=nonlinearity,nr_of_particles=10,particle_size=2,particle_dimension=1)

shooting_block_1 = shooting_blocks.ShootingBlockBase(name='block1', shooting_integrand=shooting_model)
shooting_block_2 = shooting_blocks.ShootingBlockBase(name='block2', shooting_integrand=shooting_model)

shooting_block_1 = shooting_block_1.to(device)
shooting_block_2 = shooting_block_2.to(device)

ret1,state_dicts1,costate_dicts1,data_dict1 = shooting_block_1(x=sample_batch)
ret2,state_dicts2,costate_dicts2,data_dict2 = shooting_block_2(data_dict=data_dict1,
                                                               pass_through_state_dict_of_dicts=state_dicts1,
                                                               pass_through_costate_dict_of_dicts=costate_dicts1)

dummy_loss = ret2.sum()
dummy_loss.backward()
