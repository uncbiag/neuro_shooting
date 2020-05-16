import torch
import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.shooting_models as shooting_models
import neuro_shooting.striding_block as striding_block

# create some random input
#sample_batch = torch.randn(50,10,1,15)
#sample_batch = torch.randn(25,50,1,5) # time-points, batch-size, particle-dimension (channels), particle-size

sample_batch = 0.0*torch.randn(25,200,1,5) # time-points, batch-size, particle-dimension (channels), particle-size


nonlinearity = 'relu'

import neuro_shooting.generic_integrator as generic_integrator
integrator_options = dict()
integrator_options['step_size'] = 0.1
integrator = generic_integrator.GenericIntegrator(integrator_library = 'odeint', integrator_name = 'rk4',integrator_options=integrator_options)
                                                 # use_adjoint_integration=use_adjoint, integrator_options=integrator_options, rtol=rtol, atol=atol,
                                                 #  nr_of_checkpoints=nr_of_checkpoints,
                                                 #  checkpointing_time_interval=checkpointing_time_interval)

shooting_model_1 = shooting_models.AutoShootingIntegrandModelUpDown(in_features=5, nonlinearity=nonlinearity,
                                                          parameter_weight=1.0,
                                                          inflation_factor=2,
                                                          nr_of_particles=10, particle_dimension=1,
                                                          particle_size=5,
                                                          use_analytic_solution=True,
                                                          use_rnn_mode=False,
                                                          optimize_over_data_initial_conditions=False,
                                                          optimize_over_data_initial_conditions_type='linear')

# shooting_model_1 = shooting_models.AutoShootingIntegrandModelUpDown(in_features=15, nonlinearity=nonlinearity,nr_of_particles=10,
#                                                                     particle_size=15,particle_dimension=1,
#                                                                     use_analytic_solution=True)

# shooting_model_1 = shooting_models.AutoShootingIntegrandModelUpDown(in_features=5, nonlinearity=nonlinearity,nr_of_particles=10,
#                                                                     parameter_weight=1.0,inflation_factor=100,
#                                                                     particle_size=5,particle_dimension=1,
#                                                                     use_analytic_solution=True)


# shooting_model_1 = shooting_models.AutoShootingIntegrandModelSimple(in_features=15, nonlinearity=nonlinearity,nr_of_particles=10,
#                                                                     particle_size=15,particle_dimension=1, use_analytic_solution=True)


#shooting_model_2 = shooting_models.AutoShootingIntegrandModelUpDown(in_features=25, nonlinearity=nonlinearity,nr_of_particles=10,particle_size=10,particle_dimension=1)

# shooting_block = shooting_blocks.ShootingBlockBase(name='simple', shooting_integrand=smodel,
#                                                    use_particle_free_rnn_mode=use_particle_free_rnn_mode,
#                                                    integrator=integrator)

shooting_block_1 = shooting_blocks.ShootingBlockBase(name='block1', shooting_integrand=shooting_model_1, integrator=integrator)
#shooting_block_2 = shooting_blocks.ShootingBlockBase(name='block2', shooting_integrand=shooting_model_2)

ret1,state_dicts1,costate_dicts1,data_dicts1 = shooting_block_1(x=sample_batch)
#ret2,state_dicts2,costate_dicts2,data_dicts2 = shooting_block_2(data_dict_of_dicts=data_dicts1,
#                                                               pass_through_state_dict_of_dicts=state_dicts1,
#                                                               pass_through_costate_dict_of_dicts=costate_dicts1)

# do some striding so we can test that this filter works
# this is not particularly useful for this example (other to show that the filter works)
# as we typically only want to stride in spatial dimensions. The convolutional examples will be better.

#striding = striding_block.ShootingStridingBlock(stride=2,stride_dims=2)
#s_state_dicts,s_costate_dicts,s_data_dicts = striding(state_dict_of_dicts=state_dicts2,
#                                                      costate_dict_of_dicts=costate_dicts2,
#                                                      data_dict_of_dicts=data_dicts2)

#dummy_loss = ret2['q1'].sum()

dummy_loss = ret1.sum()