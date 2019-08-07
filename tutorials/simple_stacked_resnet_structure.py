import torch
import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.shooting_models as shooting_models
import neuro_shooting.striding_block as striding_block
import neuro_shooting.state_costate_and_data_dictionary_utils as scd_utils
import neuro_shooting.parameter_initialization as parameter_initialization

gpu = 0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
nonlinearity = 'tanh'

# Let's create some random input images of size 64x64, batch size 15
sample_image_batch = torch.randn(15,1,64,64)

nr_of_particles = 10
particle_sizes = [[15,15],[11,11],[7,7]]
nr_of_features = [10,20,40]

sz = [1]*len(sample_image_batch.shape)
sz[1] = nr_of_features[0]
sample_image_batch_multi_channel = sample_image_batch.repeat(sz)

state_initializer = parameter_initialization.ConvolutionEvolutionParameterInitializer(only_random_initialization=True, random_initialization_magnitude=0.5)
costate_initializer = parameter_initialization.ConvolutionEvolutionParameterInitializer(only_random_initialization=True, random_initialization_magnitude=0.5)

shooting_model_1_1 = shooting_models.AutoShootingIntegrandModelSimpleConv2D(in_features=nr_of_features[0], nonlinearity=nonlinearity,
                                                                            state_initializer=state_initializer,
                                                                            costate_initializer=costate_initializer,
                                                                            nr_of_particles=nr_of_particles,particle_size=particle_sizes[0],particle_dimension=nr_of_features[0])
shooting_model_1_2 = shooting_models.AutoShootingIntegrandModelSimpleConv2D(in_features=nr_of_features[0], nonlinearity=nonlinearity,nr_of_particles=None)
shooting_model_1_3 = shooting_models.AutoShootingIntegrandModelSimpleConv2D(in_features=nr_of_features[0], nonlinearity=nonlinearity,nr_of_particles=None)


shooting_model_2_1 = shooting_models.AutoShootingIntegrandModelSimpleConv2D(in_features=nr_of_features[1], nonlinearity=nonlinearity,
                                                                            state_initializer=state_initializer,
                                                                            costate_initializer=costate_initializer,
                                                                            keep_initial_state_parameters_at_zero=True,
                                                                            nr_of_particles=nr_of_particles,particle_size=particle_sizes[1],particle_dimension=nr_of_features[1]-nr_of_features[0])
shooting_model_2_2 = shooting_models.AutoShootingIntegrandModelSimpleConv2D(in_features=nr_of_features[1], nonlinearity=nonlinearity,nr_of_particles=None)
shooting_model_2_3 = shooting_models.AutoShootingIntegrandModelSimpleConv2D(in_features=nr_of_features[1], nonlinearity=nonlinearity,nr_of_particles=None)


shooting_model_3_1 = shooting_models.AutoShootingIntegrandModelSimpleConv2D(in_features=nr_of_features[2], nonlinearity=nonlinearity,
                                                                            state_initializer=state_initializer,
                                                                            costate_initializer=costate_initializer,
                                                                            keep_initial_state_parameters_at_zero=True,
                                                                            nr_of_particles=nr_of_particles,particle_size=particle_sizes[2],particle_dimension=nr_of_features[2]-nr_of_features[1])
shooting_model_3_2 = shooting_models.AutoShootingIntegrandModelSimpleConv2D(in_features=nr_of_features[2], nonlinearity=nonlinearity,nr_of_particles=None)
shooting_model_3_3 = shooting_models.AutoShootingIntegrandModelSimpleConv2D(in_features=nr_of_features[2], nonlinearity=nonlinearity,nr_of_particles=None)


shooting_block_1_1 = shooting_blocks.ShootingBlockBase(name='block1_1', shooting_integrand=shooting_model_1_1)
shooting_block_1_2 = shooting_blocks.ShootingBlockBase(name='block1_2', shooting_integrand=shooting_model_1_2)
shooting_block_1_3 = shooting_blocks.ShootingBlockBase(name='block1_3', shooting_integrand=shooting_model_1_3)

striding_block_1 = striding_block.ShootingStridingBlock(stride=[2,2],stride_dims=[2,3])

shooting_block_2_1 = shooting_blocks.ShootingBlockBase(name='block2_1', shooting_integrand=shooting_model_2_1)
shooting_block_2_2 = shooting_blocks.ShootingBlockBase(name='block2_2', shooting_integrand=shooting_model_2_2)
shooting_block_2_3 = shooting_blocks.ShootingBlockBase(name='block2_3', shooting_integrand=shooting_model_2_3)

striding_block_2 = striding_block.ShootingStridingBlock(stride=[2,2],stride_dims=[2,3])

shooting_block_3_1 = shooting_blocks.ShootingBlockBase(name='block3_1', shooting_integrand=shooting_model_3_1)
shooting_block_3_2 = shooting_blocks.ShootingBlockBase(name='block3_2', shooting_integrand=shooting_model_3_2)
shooting_block_3_3 = shooting_blocks.ShootingBlockBase(name='block3_3', shooting_integrand=shooting_model_3_3)

shooting_block_1_1 = shooting_block_1_1.to(device)
shooting_block_1_2 = shooting_block_1_2.to(device)
shooting_block_1_3 = shooting_block_1_3.to(device)

shooting_block_2_1 = shooting_block_2_1.to(device)
shooting_block_2_2 = shooting_block_2_2.to(device)
shooting_block_2_3 = shooting_block_2_3.to(device)

shooting_block_3_1 = shooting_block_3_1.to(device)
shooting_block_3_2 = shooting_block_3_2.to(device)
shooting_block_3_3 = shooting_block_3_3.to(device)


ret1_1,state_dicts1_1,costate_dicts1_1,data_dicts1_1 = shooting_block_1_1(x=sample_image_batch_multi_channel)
ret1_2,state_dicts1_2,costate_dicts1_2,data_dicts1_2 = shooting_block_1_2(data_dict_of_dicts=data_dicts1_1,
                                                               pass_through_state_dict_of_dicts=state_dicts1_1,
                                                               pass_through_costate_dict_of_dicts=costate_dicts1_1)

ret1_3,state_dicts1_3,costate_dicts1_3,data_dicts1_3 = shooting_block_1_3(data_dict_of_dicts=data_dicts1_2,
                                                               pass_through_state_dict_of_dicts=state_dicts1_2,
                                                               pass_through_costate_dict_of_dicts=costate_dicts1_2)

s_state_dicts1_3,s_costate_dicts1_3,s_data_dicts1_3 = striding_block_1(state_dict_of_dicts=state_dicts1_3,
                                                      costate_dict_of_dicts=costate_dicts1_3,
                                                      data_dict_of_dicts=data_dicts1_3)

ret2_1,state_dicts2_1,costate_dicts2_1,data_dicts2_1 = shooting_block_2_1(data_dict_of_dicts=s_data_dicts1_3,
                                                               pass_through_state_dict_of_dicts=s_state_dicts1_3,
                                                               pass_through_costate_dict_of_dicts=s_costate_dicts1_3)

ret2_2,state_dicts2_2,costate_dicts2_2,data_dicts2_2 = shooting_block_2_2(data_dict_of_dicts=data_dicts2_1,
                                                               pass_through_state_dict_of_dicts=state_dicts2_1,
                                                               pass_through_costate_dict_of_dicts=costate_dicts2_1)

ret2_3,state_dicts2_3,costate_dicts2_3,data_dicts2_3 = shooting_block_2_3(data_dict_of_dicts=data_dicts2_2,
                                                               pass_through_state_dict_of_dicts=state_dicts2_2,
                                                               pass_through_costate_dict_of_dicts=costate_dicts2_2)

s_state_dicts2_3,s_costate_dicts2_3,s_data_dicts2_3 = striding_block_2(state_dict_of_dicts=state_dicts2_3,
                                                      costate_dict_of_dicts=costate_dicts2_3,
                                                      data_dict_of_dicts=data_dicts2_3)

ret3_1,state_dicts3_1,costate_dicts3_1,data_dicts3_1 = shooting_block_3_1(data_dict_of_dicts=s_data_dicts2_3,
                                                               pass_through_state_dict_of_dicts=s_state_dicts2_3,
                                                               pass_through_costate_dict_of_dicts=s_costate_dicts2_3)

ret3_2,state_dicts3_2,costate_dicts3_2,data_dicts3_2 = shooting_block_3_2(data_dict_of_dicts=data_dicts3_1,
                                                               pass_through_state_dict_of_dicts=state_dicts3_1,
                                                               pass_through_costate_dict_of_dicts=costate_dicts3_1)

ret3_3,state_dicts3_3,costate_dicts3_3,data_dicts3_3 = shooting_block_3_3(data_dict_of_dicts=data_dicts3_2,
                                                               pass_through_state_dict_of_dicts=state_dicts3_2,
                                                               pass_through_costate_dict_of_dicts=costate_dicts3_2)

dummy_loss = ret3_3.sum()
dummy_loss.backward()
