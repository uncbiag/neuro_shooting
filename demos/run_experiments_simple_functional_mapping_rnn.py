# this script is simply to run many experimental conditions for subsequent evaluation
# this is the new style which only retains the core experimental settings
# TODO: refactor run_experiments_simple_functional_mapping.py in this style after runs are finished

import neuro_shooting.experiment_utils as eu

if __name__ == '__main__':

    args = eu.setup_cmdline_parsing(cmdline_type='simple_functional_mapping',cmdline_title='RNN simple functional mapping')

    run_args_template_particle_free_rnn = {
        'use_particle_free_rnn_mode': None,
        'shooting_model': args.shooting_model,
        'niters': 250,
        'unfreeze_parameters_at_iter': 50,
        'save_figures': None,
        'viz_freq': 300,
        'fcn': args.fcn,
        'optimize_over_data_initial_conditions': None #[True, False]  # we can add binary flags like this
    }

    run_args_template_particle_rnn = {
        'use_particle_rnn_mode': None,
        'shooting_model': args.shooting_model,
        'niters': 250,
        'unfreeze_parameters_at_iter': 50,
        'save_figures': None,
        'viz_freq': 300,
        'fcn': args.fcn,
        'optimize_over_data_initial_conditions': None  # [True, False]  # we can add binary flags like this
    }

    # we do not need to sweep over particles for the particle-free method, so just set it to 2
    run_args_to_sweep_particle_free_rnn = {
        'nr_of_particles': [2],  # number of particles needs to be at least 2
        'inflation_factor': [4, 8, 16, 32],
    }

    run_args_to_sweep_particle_rnn = {
        'nr_of_particles': [2,5,15,25,50], # number of particles needs to be at least 2
        'inflation_factor': [4,8,16,32],
    }

    # run for particle-free
    eu.sweep_parameters(args, run_args_to_sweep_particle_free_rnn, run_args_template_particle_free_rnn,
                        python_script='simple_functional_mapping_example.py',output_dir_prefix='particle_free_rnn_')
    # run with particles
    eu.sweep_parameters(args, run_args_to_sweep_particle_rnn, run_args_template_particle_rnn,
                        python_script='simple_functional_mapping_example.py',output_dir_prefix='particle_rnn_')

    print('Done processing')
