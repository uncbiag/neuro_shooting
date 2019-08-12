import torch
import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.shooting_models as shooting_models

gpu = 0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
nonlinearity = 'tanh'

# create some random input
#sample_batch = torch.randn(50,10,1,2)
sample_batch_1 = torch.ones(10,1,2,requires_grad = True,device = device)
sample_batch_2 = torch.ones(10,1,2,requires_grad = True,device = device)

shooting_model_1 = shooting_models.AutoShootingIntegrandModelUpDown(in_features=2, nonlinearity=nonlinearity,nr_of_particles=10,particle_size=2,particle_dimension=1)
shooting_model_2 = shooting_models.AutoShootingIntegrandModelUpDown(in_features=2, nonlinearity=nonlinearity,nr_of_particles=10,particle_size=2,particle_dimension=1)
#shooting_model_2 = shooting_models.AutoShootingIntegrandModelUpDown(in_features=4, nonlinearity=nonlinearity,nr_of_particles=10,particle_size=2,particle_dimension=1)
#shooting_model_2 = shooting_models.AutoShootingIntegrandModelUpDown(in_features=2, nonlinearity=nonlinearity,nr_of_particles=None)

shooting_block_1 = shooting_blocks.ShootingBlockBase(name='block1', shooting_integrand=shooting_model_1,use_finite_difference =False)
shooting_block_2 = shooting_blocks.ShootingBlockBase(name='block2', shooting_integrand=shooting_model_2,use_finite_difference = True)

shooting_block_1 = shooting_block_1.to(device)
shooting_block_2 = shooting_block_2.to(device)

ret1,state_dicts1,costate_dicts1,data_state_dicts1,data_costate_dicts1 = shooting_block_1(x=sample_batch_1)
ret2,state_dicts2,costate_dicts2,data_state_dicts2,data_costate_dicts2 = shooting_block_2(x=sample_batch_2)
#ret2,state_dicts2,costate_dicts2,data_state_dicts2,data_costate_dicts2 = shooting_block_2(data_state_dict_of_dicts=data_state_dicts1,data_costate_dict_of_dicts = data_costate_dicts1,
                                                               #pass_through_state_dict_of_dicts=state_dicts1,
                                                               #pass_through_costate_dict_of_dicts=costate_dicts1)

a = torch.randn(ret1.shape,device = device)
dummy_loss1 =  torch.sum(ret1*a)# there is only one value, so it returns a tensor
dummy_loss2 = torch.sum(ret2*a)
print("loss ",dummy_loss1)
print("loss ",dummy_loss2)
dummy_loss1.backward()
dummy_loss2.backward()
print("gradient ",sample_batch_1.grad)
print("gradient finite difference",sample_batch_2.grad)
print("difference: ",sample_batch_2.grad - sample_batch_1.grad)
for k in state_dicts1:
    for l in state_dicts1[k]:
        print("toto",state_dicts1[k][l].grad)

for k in state_dicts2:
    for l in state_dicts2[k]:
        print("toto", state_dicts2[k][l].grad)

print("param1 ", shooting_block_1._parameters["p_q1"].grad)
print("param1 ", shooting_block_2._parameters["p_q1"].grad)