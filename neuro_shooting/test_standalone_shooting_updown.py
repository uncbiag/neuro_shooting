import torch
from sortedcontainers import SortedDict
import neuro_shooting.activation_functions_and_derivatives as ad
import torch.nn as nn

def compute_parameters(states):
    parameters = dict()

    # now compute the parameters
    q1 = states['q1']
    p1 = states['p1']
    q2 = states['q2'].transpose(2,1)
    p2 = states['p2'].transpose(2,1)

    temp = torch.matmul(nl(q2), p1)
    l1 = torch.sum(temp, axis=0).t()

    temp2 = torch.matmul(p2, q1)
    l2 = torch.sum(temp2, dim=0)
    parameters["l2_weight"] = l2
    parameters["l1_weight"] = l1
    parameters["l2_bias"] = torch.sum(p2, dim=0).t()
    parameters["l1_bias"] = torch.sum(p1, dim=0)

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
    size_q2 = 3
    timestep = 0.025
    initial_states["q1"] =  torch.ones(n_particles,1,size_q1)
    initial_states["p1"] = torch.ones(n_particles,1,size_q1)
    initial_states["q2"] = torch.ones(n_particles,1,size_q2)
    initial_states["p2"] = 1 * torch.ones(n_particles,1,size_q2)
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
