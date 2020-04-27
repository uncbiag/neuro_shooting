import torch
from sortedcontainers import SortedDict
import neuro_shooting.activation_functions_and_derivatives as ad
import torch.nn as nn

def compute_parameters(states):
    parameters = dict()

    # now compute the parameters
    q1 = states['q1']
    p1 = states['p1']
    q2 = states['q2'].transpose(1,2)
    p2 = states['p2'].transpose(1,2)

    l1 = torch.mean(torch.matmul(nl(q2), p1),dim = 0).t()
    l2 = torch.mean(torch.matmul(p2, q1),dim = 0)

    parameters["l2_weight"] = l2
    parameters["l1_weight"] = l1
    parameters["l2_bias"] = torch.mean(p2, dim=0).t()
    parameters["l1_bias"] = torch.mean(p1, dim=0)
    return parameters

def compute_hamiltonian(states):
    energy = compute_energy(states)
    lagrangian = compute_lagrangian(states)
    return (lagrangian - energy) * states["p1"].shape[0]

def compute_energy(states):
    result = 0
    parameters = compute_parameters(states)
    for key in parameters.keys():
        result+=0.5 * torch.sum(parameters[key]**2)
    return result

def compute_lagrangian(states):
    A = states["p1"].shape[0]
    rhs = compute_rhs(states)
    lagrangian = torch.sum(states["p1"] * rhs["q1"]) + torch.sum(states["p2"] * rhs["q2"])
    return lagrangian/A

nl = nn.functional.relu
dnl = ad.drelu
#nl = ad.softmax
#dnl = ad.dsoftmax

def compute_rhs(states):
    parameters = compute_parameters(states)

    rhs = SortedDict()
    A = states["p1"].shape[0]
    l1 = parameters["l1_weight"]
    l2 = parameters["l2_weight"]
    bias1 = parameters["l1_bias"]
    bias2 = parameters["l2_bias"]
    # now compute the parameters
    q1 = states['q1']
    q2 = states['q2']
    p1 = states['p1']
    p2 = states['p2']

    rhs["q1"] = torch.matmul(nl(q2),l1.t()) + bias1
    rhs["q2"] = torch.matmul(q1,l2.t()) + bias2
    rhs["q1"] = rhs["q1"]
    rhs["q2"] = rhs["q2"]

    dot_p2 = - dnl(q2) * torch.matmul(p1, l1)
    dot_p1 = - torch.matmul(p2, l2)
    rhs['p1'] = dot_p1
    rhs['p2'] = dot_p2

    return rhs


def advect(timestep, initial_states):
    result = SortedDict()
    states = SortedDict()
    states_temp = SortedDict()
    for name in initial_states.keys():
        states[name] = initial_states[name].clone()
        states_temp[name] = initial_states[name].clone()
        result[name] = []
    result["energy"] = []
    result["lagrangian"] = []
    n = int(1./timestep)

    for i in range(n):
        rhs = compute_rhs(states)
        for name in states.keys():
            states_temp[name] = states[name] + 0.5 * timestep * rhs[name]
        rhs2 = compute_rhs(states_temp)
        for name in states_temp:
            states[name] = states[name] + timestep * rhs2[name]
            result[name].append(states[name][0,0,0])
        result["energy"].append(compute_energy(states))
        result["lagrangian"].append(compute_lagrangian(states))

    return states,result

def finite_differences(initial_states, epsilon = 0.0001,state_name = "p1",costate_name = "q1"):
    states_temp = SortedDict()
    rhs = compute_rhs(initial_states)
    for name in initial_states.keys():
        states_temp[name] = initial_states[name].clone()

    l1 = compute_hamiltonian(initial_states)
    gradient = torch.zeros_like(initial_states[state_name])
    for i in range(initial_states[state_name].shape[0]):
        for j in range(initial_states[state_name].shape[2]):
            states_temp[state_name] = initial_states[state_name].clone()
            states_temp[state_name][i,0,j] = initial_states[state_name][i,0,j] + epsilon
            l2 = compute_hamiltonian(states_temp)
            gradient[i,0,j] = (l2 - l1)/epsilon
    return gradient,rhs[costate_name]



if __name__=="__main__":
    print("toto")
    initial_states = SortedDict()
    n_particles = 7
    size_q1 = 1
    size_q2 = 5
    timestep = 0.02

    initial_states["q1"] = torch.tensor([[[ 2.5894]],
        [[ 0.4932]],
        [[ 3.3355]],
        [[ 2.5192]],
        [[ 1.9769]],
        [[ 0.6604]],
        [[-1.4231]],
        [[-0.2764]]], requires_grad=True)
    initial_states["q2"] = torch.tensor([[[ 2.0540, -0.1818,  0.0143,  0.7345, -1.7673]],
        [[ 1.7371,  2.4269,  2.6769, -1.6054,  0.2951]],
        [[ 4.5079,  3.0657,  2.6508, -2.2877,  4.4365]],
        [[-2.0122, -0.0658, -4.7322, -0.6830, -2.0476]],
        [[ 0.9853,  2.0527, -4.9762,  0.6134,  3.2868]],
        [[-2.9411,  4.9192, -1.4423, -1.0320, -1.1641]],
        [[ 0.6226,  1.0918,  2.0503, -0.3876,  0.8046]],
        [[-2.2261, -2.4392, -1.4383,  3.8271, -1.4558]]], requires_grad=True)
    initial_states["p1"] = torch.tensor([[[ 1.7873]],
        [[ 1.9972]],
        [[ 1.7886]],
        [[ 8.7142]],
        [[ 2.3711]],
        [[ 1.4691]],
        [[ 2.4614]],
        [[-0.5894]]], requires_grad=True)
    initial_states["p2"] = torch.tensor([[[ 1.0656, -1.2352,  1.2794, -1.0476,  0.6527]],
        [[ 0.0115, -0.7171, -0.4078, -0.3260, -0.7074]],
        [[ 0.4306, -0.0866,  0.9549, -0.1474,  0.3941]],
        [[ 2.8053, -9.3032, -1.5642, -5.1650, -0.2377]],
        [[ 1.6282, -2.1628,  1.6657, -1.8801,  0.4627]],
        [[ 2.2930, -5.9964, -2.9206, -4.2974, -1.0469]],
        [[-1.3578, -1.2170, -4.9959, -0.1325, -1.5833]],
        [[ 1.2180, -3.6513, -7.8035, -4.9892,  1.6416]]], requires_grad=True)
    # initial_states["q1"] = torch.tensor([[[0.7501]],
    #         [[-0.5410]],
    #         [[1.4154]],
    #         [[-0.1148]],
    #         [[-0.3530]],
    #         [[-0.7293]],
    #         [[0.1357]],
    #         [[-0.1562]]], requires_grad=False)
    #
    # initial_states["q2"] = torch.tensor([[[-0.0199, 0.2058, -0.4887, -0.9527, 0.3126]],
    #         [[-0.7060, -0.6868, -0.1463, -0.1216, -0.3508]],
    #         [[-0.9320, -0.0854, -0.6451, -0.3805, -0.6714]],
    #         [[0.4771, -0.7824, -0.3193, -0.4775, 0.8443]],
    #         [[0.9689, -0.6867, 1.0794, -0.5185, -0.8415]],
    #         [[0.5998, 0.7136, -0.3300, -0.0570, -0.0405]],
    #         [[-0.4197, -0.4218, -0.3541, 0.9305, -0.2596]],
    #         [[-0.1699, -0.3413, -0.1462, 0.6848, -0.4991]]], requires_grad=False)
    #
    # initial_states["p1"] = torch.tensor([[[0.0575]],
    #         [[0.2434]],
    #         [[0.0372]],
    #         [[-0.0571]],
    #         [[-0.0068]],
    #         [[-0.0834]],
    #         [[-0.1962]],
    #         [[-0.0356]]],requires_grad=False)
    # initial_states["p2"] = torch.tensor([[[-0.0981, -0.4023, -0.1343, -0.2815, -0.1628]],
    #         [[0.0332, 0.0844, 0.1113, 0.3639, 0.0907]],
    #         [[-0.2875, -0.2781, -0.2701, -0.4343, -0.1542]],
    #         [[0.0787, -0.2295, -0.0869, 0.3211, 0.1882]],
    #         [[0.0905, 0.1138, -0.0857, 0.3379, -0.0622]],
    #         [[0.1477, 0.1533, 0.0687, 0.4282, 0.0529]],
    #         [[-0.2490, -0.2063, -0.3477, 0.1459, -0.2472]],
    #         [[0.1108, -0.0989, -0.2654, 0.3309, -0.0567]]],
    #        requires_grad=False)
    initial_states["q1"] =  torch.ones(n_particles,1,size_q1)
    initial_states["p1"] = torch.rand(n_particles,1,size_q1)
    initial_states["q2"] = torch.ones(n_particles,1,size_q2)
    initial_states["p2"] = torch.rand(n_particles,1,size_q2)
    for a in initial_states.keys():
        print(a,initial_states[a].shape)
    states,result = advect(timestep,initial_states)
    import pylab as pl

    pl.figure()
    pl.plot(range(len(result["energy"])), result["energy"])
    pl.plot(range(len(result["energy"])), result["lagrangian"],"o")
    pl.title("energy")
    pl.show()
    for name in result.keys():
        if name not in ["energy","lagrangian"]:
            pl.plot(range(len(result[name])),result[name])
    pl.title("trajectories")
    pl.show()

    fd,th = finite_differences(initial_states,state_name="q1",costate_name="p1",epsilon=0.001)
    print("finitediff",fd)
    print("analytic",th)
    print("difference",fd+th)