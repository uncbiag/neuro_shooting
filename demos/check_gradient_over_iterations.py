import torch

#grad_iter_1_analytic_False.p

for itr in range(1,5):
    analytic_solution = torch.load('grad_iter_{}_analytic_{}.pt'.format(itr,True))
    autodiff_solution = torch.load('grad_iter_{}_analytic_{}.pt'.format(itr,False))

    print('Iter = {}\n---------'.format(itr))
    for k in analytic_solution:
        print('Analytic: {} = {}'.format(k,analytic_solution[k]))
        print('Autodiff: {} = {}'.format(k,autodiff_solution[k]))
        print('Ratio analytic/autodiff: {} = {}'.format(k,analytic_solution[k]/autodiff_solution[k]))

    print('\n')

print('Hello world')