import torch

#grad_iter_1_analytic_False.p

number_of_tests_passed = 0
number_of_tests_attempted = 0
tolerance = 5e-3

check_only_gradients = True

for itr in range(1,5+1):
    analytic_solution = torch.load('grad_iter_{}_analytic_{}.pt'.format(itr,True))
    autodiff_solution = torch.load('grad_iter_{}_analytic_{}.pt'.format(itr,False))

    print('Iter = {}\n---------'.format(itr))
    for k in analytic_solution:

        if (check_only_gradients and k.endswith('_grad')) or (not check_only_gradients):

            print('Analytic: {} = {}'.format(k,analytic_solution[k]))
            print('Autodiff: {} = {}'.format(k,autodiff_solution[k]))
            print('Ratio analytic/autodiff: {} = {}'.format(k,analytic_solution[k]/autodiff_solution[k]))

            number_of_tests_attempted += 1

            nz_autodiff = autodiff_solution[k] !=0
            nz_analytic = analytic_solution[k] !=0

            if torch.all(nz_analytic==nz_autodiff).item():
                rel_error = torch.max(torch.abs(autodiff_solution[k][nz_autodiff] / analytic_solution[k][nz_analytic] - 1))
                if rel_error < tolerance:
                    number_of_tests_passed += 1
                    print('\nPASSED with relative error of {}\n'.format(rel_error))
                else:
                    print('\n-------------------------------------')
                    print('FAILED with relative error of {}'.format(rel_error))
                    print('-------------------------------------\n')
            else:
                print('\n-------------------------------------')
                print('FAILED due to different ZERO pattern')
                print('-------------------------------------\n')

    print('\n')

print('\nOverall summary:')
print('-----------------\n')
print('Passed {}/{} tests.'.format(number_of_tests_passed,number_of_tests_attempted))
print('Failed {}/{} tests.'.format(number_of_tests_attempted-number_of_tests_passed,number_of_tests_attempted))

if number_of_tests_passed==number_of_tests_attempted:
    print('\nCongratulations, all tests passes at a tolerance level of {}'.format(tolerance))
