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

    temp = torch.matmul(nl(q2), p1)
    l1 = torch.mean(temp, dim=0).t()

    temp2 = torch.matmul(p2, q1)
    l2 = torch.mean(temp2, dim=0)
    parameters["l2_weight"] = l2
    parameters["l1_weight"] = l1
    parameters["l2_bias"] = torch.mean(p2, dim=0).t()
    parameters["l1_bias"] = torch.mean(p1, dim=0)

    return parameters

def compute_energy(states):
    result = 0
    parameters = compute_parameters(states)
    for key in parameters.keys():
        result+=torch.sum(parameters[key]**2)
    return result


nl = nn.functional.relu
dnl = ad.drelu
nl = ad.softmax
dnl = ad.dsoftmax
def compute_rhs(states):
    parameters = compute_parameters(states)

    rhs = SortedDict()

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

    return states,result


if __name__=="__main__":
    print("toto")
    initial_states = SortedDict()
    n_particles = 3
    size_q1 = 1
    size_q2 = 5
    timestep = 0.01

    initial_states["q1"] = torch.tensor([[[ 2.4148]],
        [[-1.2360]],
        [[-1.9907]],
        [[-0.1487]],
        [[ 0.6624]],
        [[-2.1214]],
        [[ 1.6731]],
        [[-1.7039]],
        [[ 2.2073]],
        [[-0.6990]]], requires_grad=True)
    initial_states["q2"]= torch.tensor([[[-2.1184, -1.6336, -0.6855,  2.3704,  1.6521]],
        [[-0.3499, -0.7457,  0.9467, -0.6741, -0.3296]],
        [[ 1.9163,  1.6425, -1.2606, -0.8166, -1.2797]],
        [[-0.0963, -0.1473,  0.0421,  1.5864,  1.6440]],
        [[-0.3622, -0.1828,  2.7314,  2.5283,  2.5077]],
        [[ 1.2432,  2.1364, -1.7182, -1.3671, -1.5856]],
        [[-1.5291, -0.9854, -0.0951, -0.0597, -0.0747]],
        [[-0.0811,  2.7605, -0.9513, -0.3492, -0.6446]],
        [[-2.0023, -2.6077, -0.1721, -0.5561,  2.2961]],
        [[ 2.6375,  2.5632, -0.4733, -0.4456, -0.8273]]], requires_grad=True)
    initial_states["p1"] = torch.tensor([[[-0.0143]],
        [[-0.0886]],
        [[ 0.0162]],
        [[-0.1913]],
        [[-0.0749]],
        [[ 0.0444]],
        [[ 0.0365]],
        [[-0.0239]],
        [[-0.1022]],
        [[ 0.1150]]])
    initial_states["p2"] = torch.tensor([[[ 0.5583,  0.6433, -4.1031, -4.1940, -4.5529]],
        [[-6.8445, -3.7016, -2.4826, -2.9398, -3.0272]],
        [[-7.5710, -8.3275,  0.6161, -0.1731, -0.0373]],
        [[-1.8342, -0.6965, -1.3962, -0.8821, -0.4642]],
        [[-0.8088, -0.0986, -1.9296, -1.6217, -1.6315]],
        [[-6.3130, -3.9559,  0.5125, -0.2458, -0.6392]],
        [[-0.3501,  0.1929, -3.7906, -3.0818, -2.2634]],
        [[-6.1522, -7.7471,  0.2019, -0.6502,  0.2622]],
        [[ 0.4161,  0.4097, -4.6389, -4.6341, -5.0500]],
        [[-2.0547, -4.1092, -0.5452, -0.6620, -0.0166]]], requires_grad=True)
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
    # initial_states["q1"] =  torch.ones(n_particles,1,size_q1)
    # initial_states["p1"] = torch.ones(n_particles,1,size_q1)
    # initial_states["q2"] = torch.ones(n_particles,1,size_q2)
    # initial_states["p2"] = 1 * torch.ones(n_particles,1,size_q2)
    for a in initial_states.keys():
        print(a,initial_states[a].shape)
    states,result = advect(timestep,initial_states)
    import pylab as pl

    pl.figure()
    pl.plot(range(len(result["energy"])), result["energy"])
    pl.title("energy")
    pl.show()
    for name in result.keys():
        if name is not "energy":
            pl.plot(range(len(result[name])),result[name])
    pl.title("trajectories")
    pl.show()
