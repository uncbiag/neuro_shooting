import torch
import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.shooting_models as shooting_models
import neuro_shooting.parameter_initialization as parameter_initialization
import neuro_shooting.utils as utils
import neuro_shooting.generic_integrator as generic_integrator

utils.setup_random_seed(seed=1234)
utils.setup_device(desired_gpu=0)
nonlinearity = 'tanh'

# setup the integrator
integrator_options = dict()
integrator_options['step_size'] = 0.1
integrator = generic_integrator.GenericIntegrator(integrator_library = 'odeint', integrator_name = 'rk4',integrator_options=integrator_options)

# setup the initializers
state_initializer = parameter_initialization.ConvolutionEvolutionParameterInitializer(only_random_initialization=True, random_initialization_magnitude=0.5)
costate_initializer = parameter_initialization.ConvolutionEvolutionParameterInitializer(only_random_initialization=True, random_initialization_magnitude=0.5)

# Let's create some random input images of size 64x64, batch size 15
sample_image_batch = torch.randn(15,1,64,64)

nr_of_particles = 10
particle_sizes = [7,7]
nr_of_features = 10

sz = [1]*len(sample_image_batch.shape)
sz[1] = nr_of_features
sample_image_batch_multi_channel = sample_image_batch.repeat(sz)


shooting_model = shooting_models.AutoShootingIntegrandModelSimpleConv2D(in_features=nr_of_features, nonlinearity=nonlinearity,
                                                                        state_initializer=state_initializer,
                                                                        costate_initializer=costate_initializer,
                                                                        nr_of_particles=nr_of_particles,particle_size=particle_sizes,
                                                                        particle_dimension=nr_of_features)

shooting_block = shooting_blocks.ShootingBlockBase(name='test_block', shooting_integrand=shooting_model, integrator=integrator)

ret,state_dicts,costate_dicts,data_dicts = shooting_block(x=sample_image_batch_multi_channel)

dummy_loss = ret.sum()
dummy_loss.backward()
