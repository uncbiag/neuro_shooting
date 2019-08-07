import torch
import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.shooting_models as shooting_models
import neuro_shooting.striding_block as striding_block

gpu = 0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
nonlinearity = 'tanh'

# create some random input
sample_batch = torch.randn(50,10,1,15)

shooting_model_1 = shooting_models.AutoShootingIntegrandModelUpDown(in_features=15, nonlinearity=nonlinearity,nr_of_particles=10,particle_size=15,particle_dimension=1)
shooting_model_2 = shooting_models.AutoShootingIntegrandModelUpDown(in_features=25, nonlinearity=nonlinearity,nr_of_particles=10,particle_size=10,particle_dimension=1)

shooting_block_1 = shooting_blocks.ShootingBlockBase(name='block1', shooting_integrand=shooting_model_1)
shooting_block_2 = shooting_blocks.ShootingBlockBase(name='block2', shooting_integrand=shooting_model_2)

shooting_block_1 = shooting_block_1.to(device)
shooting_block_2 = shooting_block_2.to(device)

ret1,state_dicts1,costate_dicts1,data_dicts1 = shooting_block_1(x=sample_batch)
ret2,state_dicts2,costate_dicts2,data_dicts2 = shooting_block_2(data_dict_of_dicts=data_dicts1,
                                                               pass_through_state_dict_of_dicts=state_dicts1,
                                                               pass_through_costate_dict_of_dicts=costate_dicts1)

# do some striding so we can test that this filter works
# this is not particularly useful for this example (other to show that the filter works)
# as we typically only want to stride in spatial dimensions. The convolutional examples will be better.

striding = striding_block.ShootingStridingBlock(stride=2,stride_dims=2)
s_state_dicts,s_costate_dicts,s_data_dicts = striding(state_dict_of_dicts=state_dicts2,
                                                      costate_dict_of_dicts=costate_dicts2,
                                                      data_dict_of_dicts=data_dicts2)


dummy_loss = ret2['q1'].sum()
