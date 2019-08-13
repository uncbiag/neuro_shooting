import torch
import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.shooting_models as shooting_models
import neuro_shooting.parameter_initialization as parameter_initialization

gpu = 0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
nonlinearity = 'tanh'

# Let's create some random input images of size 64x64, batch size 15
sample_image_batch = torch.ones(15,1,64,64,device = device,requires_grad = True)
sample_image_batch2 = torch.ones(15,1,64,64,device = device,requires_grad = True)


nr_of_particles = 10
particle_sizes = [[15,15],[11,11],[7,7]]
nr_of_features = [10,20,40]

sz = [1]*len(sample_image_batch.shape)
sz[1] = nr_of_features[0]
sample_image_batch_multi_channel = sample_image_batch.repeat(sz)
sample_image_batch_multi_channel2 = sample_image_batch2.repeat(sz)

state_initializer = parameter_initialization.ConvolutionEvolutionParameterInitializer(only_random_initialization=True, random_initialization_magnitude=0.)
costate_initializer = parameter_initialization.ConvolutionEvolutionParameterInitializer(only_random_initialization=True, random_initialization_magnitude=0.)

shooting_model_1_1 = shooting_models.AutoShootingIntegrandModelSimpleConv2D(in_features=nr_of_features[0], nonlinearity=nonlinearity,
                                                                            state_initializer=state_initializer,
                                                                            costate_initializer=costate_initializer,
                                                                            nr_of_particles=nr_of_particles,particle_size=particle_sizes[0],particle_dimension=nr_of_features[0])




shooting_block_1_1 = shooting_blocks.ShootingBlockBase(name='block1_1', shooting_integrand=shooting_model_1_1,use_finite_difference = True)
shooting_block_1_1 = shooting_block_1_1.to(device)
ret1_1,state_dicts1_1,costate_dicts1_1,data_state_dicts1_1,data_costate_dicts1_1 = shooting_block_1_1(x=sample_image_batch_multi_channel)

dummy_loss = ret1_1.sum()
dummy_loss.backward()
print(dummy_loss)

print("gradient",sample_image_batch_multi_channel.grad)

state_initializer = parameter_initialization.ConvolutionEvolutionParameterInitializer(only_random_initialization=True, random_initialization_magnitude=0.)
costate_initializer = parameter_initialization.ConvolutionEvolutionParameterInitializer(only_random_initialization=True, random_initialization_magnitude=0.)

shooting_model_1_1 = shooting_models.AutoShootingIntegrandModelSimpleConv2D(in_features=nr_of_features[0], nonlinearity=nonlinearity,
                                                                            state_initializer=state_initializer,
                                                                            costate_initializer=costate_initializer,
                                                                            nr_of_particles=nr_of_particles,particle_size=particle_sizes[0],particle_dimension=nr_of_features[0])




shooting_block_1_1 = shooting_blocks.ShootingBlockBase(name='block1_1', shooting_integrand=shooting_model_1_1,use_finite_difference = False)
shooting_block_1_1 = shooting_block_1_1.to(device)
ret1_1,state_dicts1_1,costate_dicts1_1,data_state_dicts1_1,data_costate_dicts1_1 = shooting_block_1_1(x=sample_image_batch_multi_channel2)

dummy_loss = ret1_1.sum()
print(dummy_loss)
dummy_loss.backward()
print("gradient",sample_image_batch_multi_channel2.grad)