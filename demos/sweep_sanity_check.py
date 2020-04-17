import itertools
import os

for taskid in range(108):

    print('taskid: {}'.format(taskid))

    hyperparameter_config = {
        'nr_of_particles': [10,50,100],
        'pw': [0.1,0.5,1],
        'true_nonlinearity':['relu', 'tanh'],
        'assumed_nonlinearity': ['relu', 'tanh'],
        't_max': [1,10,100]
    }

    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # taskid = int(taskid[0])

    os.system("python sanity_check.py "
              "--log_to_file "
              "--nr_of_particles %s "
              "--pw %s "
              "--true_nonlinearity %s "
              "--assumed_nonlinearity %s "
              "--t_max %s"%
              (hyperparameter_experiments[taskid]['nr_of_particles'],
               hyperparameter_experiments[taskid]['pw'],
               hyperparameter_experiments[taskid]['true_nonlinearity'],
               hyperparameter_experiments[taskid]['assumed_nonlinearity'],
               hyperparameter_experiments[taskid]['t_max']))