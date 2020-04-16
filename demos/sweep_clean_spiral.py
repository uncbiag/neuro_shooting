import sys
import os
import itertools


def main(taskid):


    hyperparameter_config = {
        'shooting_model': ['simple','2nd_order','updown'],
        'nr_of_particles': [10,50,100],
        'pw': [0.1,0.5,1],
        'nonlinearity':['identity', 'relu', 'tanh', 'sigmoid']
    }

    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    taskid = int(taskid[0])

    os.system("python clean_spiral.py "
              "--viz "
              "--method dopri5 "
              "--shooting_model %s "
              "--nr_of_particles %s "
              "--pw %s "
              "--nonlinearity %s"%
              (hyperparameter_experiments[taskid]['shooting_model'],
               hyperparameter_experiments[taskid]['nr_of_particles'],
               hyperparameter_experiments[taskid]['pw'],
               hyperparameter_experiments[taskid]['nonlinearity']))

if __name__ == "__main__":
    main(sys.argv[1:])