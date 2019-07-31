import torch
import torch.autograd as autograd
import torch.optim as optim

d = torch.load('fixed_point_test.pt')

nr_of_fixed_point_iterations = 100
learning_rate = 0.25
compute_simple_gradient_descent = False

inv_Kbar = d['inv_Kbar']
inv_Kbar_b = d['inv_Kbar_b']

nl = torch.tanh

def advect_q(q, theta1, bias1, theta2, bias2):
    """
    Forward equation which  is applied on the data. In principle similar to advect_q
    :param x:
    :param theta:
    :param bias:
    :return: \dot q_i = \theta \sigma(q_i) + b
    """
    temp_q_inner = torch.matmul(theta2, q) + bias2
    temp_q = nl(temp_q_inner)
    return torch.matmul(theta1, temp_q) + bias1


def compute_lagrangian(p, q, theta1, bias1, theta2, bias2):
    theta1_penalty = torch.mm(theta1.view(1, -1), torch.mm(inv_Kbar, theta1.view(-1, 1)))
    bias1_penality = torch.mm(bias1.t(), torch.mm(inv_Kbar_b, bias1))

    #theta2_penalty = torch.mm(theta2.view(1, -1), torch.mm(inv_Kbar, theta2.view(-1, 1)))

    #theta2_penalty = torch.norm((torch.mm(theta2, theta2.t()) - torch.eye(2)))

    theta2_penalty = ((torch.mm(theta2, theta2.t()) - torch.eye(2))**2).sum()

    #theta2_penalty = torch.mm((theta2-torch.eye(2)).view(1, -1), torch.mm(inv_Kbar, (theta2-torch.eye(2)).view(-1, 1)))

    bias2_penality = torch.mm(bias2.t(), torch.mm(inv_Kbar_b, bias2))

    kinetic_energy = 0.5 * (theta1_penalty + theta2_penalty + bias1_penality + bias2_penality)

    # this is really only how one propagates through the system given the parameterization
    potential_energy = torch.mean(p * advect_q(q, theta1, bias1, theta2, bias2))

    L = kinetic_energy - potential_energy

    return L


p = d['p']
q = d['q']
theta1 = d['theta1']
theta2 = d['theta2']
bias1 = d['bias1']
bias2 = d['bias2']

print(theta1)
print(theta2)
print(bias1)
print(bias2)

vars = [theta1, theta2, bias1, bias2]

#optimizer = optim.Adam(vars, lr=learning_rate)
optimizer = optim.SGD(vars, lr=learning_rate,momentum=0.9,nesterov=True)


for n in range(nr_of_fixed_point_iterations):

    optimizer.zero_grad()

    current_lagrangian = compute_lagrangian(p=p, q=q, theta1=theta1, bias1=bias1, theta2=theta2, bias2=bias2)

    print('{}: val = {}'.format(n,current_lagrangian.item()))

    theta_grad1, bias_grad1, theta_grad2, bias_grad2, = autograd.grad(current_lagrangian,
                                                                      (theta1, bias1, theta2, bias2),
                                                                      grad_outputs=current_lagrangian.data.new(
                                                                          current_lagrangian.shape).fill_(1),
                                                                      create_graph=True,
                                                                      retain_graph=True,
                                                                      allow_unused=True)

    if compute_simple_gradient_descent:

        theta1 = theta1 - learning_rate*theta_grad1
        bias1 = bias1 - learning_rate*bias_grad1

        theta2 = theta2 - learning_rate*theta_grad2
        bias2 = bias2 - learning_rate*bias_grad2

    else:

        theta1.grad = theta_grad1
        theta2.grad = theta_grad2
        bias1.grad = bias_grad1
        bias2.grad = bias_grad2

        optimizer.step()

    # print(theta)
    # print(bias)

print('theta1:')
print(theta1)
print('theta2:')
print(theta2)
print('bias1:')
print(bias1)
print('bias2:')
print(bias2)

# print('p:')
# print(p)
#
# print('q:')
# print(q)
