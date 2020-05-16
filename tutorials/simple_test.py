import torch
import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.shooting_models as shooting_models
import neuro_shooting.generic_integrator as generic_integrator

# create some random input
sample_batch = torch.randn(50,10,1,15) # time-points, batch-size, particle-dimension (channels), particle-size
nonlinearity = 'relu'

integrator_options = dict()
integrator_options['step_size'] = 0.1
integrator = generic_integrator.GenericIntegrator(integrator_library = 'odeint', integrator_name = 'rk4',integrator_options=integrator_options)

shooting_model_1 = shooting_models.AutoShootingIntegrandModelUpDown(in_features=15, nonlinearity=nonlinearity,nr_of_particles=10,
                                                                    particle_size=15,particle_dimension=1,
                                                                    use_analytic_solution=True)

shooting_block_1 = shooting_blocks.ShootingBlockBase(name='block1', shooting_integrand=shooting_model_1, integrator=integrator)

ret1,state_dicts1,costate_dicts1,data_dicts1 = shooting_block_1(x=sample_batch)

dummy_loss = ret1.sum()