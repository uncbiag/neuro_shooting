# this script is simply to run many experimental conditions for subsequent evaluation
# this is the new style which only retains the core experimental settings

import neuro_shooting.experiment_utils as eu

if __name__ == '__main__':

    args = eu.setup_cmdline_parsing(cmdline_type='spiral',cmdline_title='RNN spiral')

    niters = 1500

    run_args_template_particle_free_rnn = {
        'use_particle_free_rnn_mode': None,
        'shooting_model': args.shooting_model,
        # 'viz': None,
        'niters': niters,
        'optional_weight': 10,  # only used for updown_universal model
        'custom_parameter_freezing': None,
        'unfreeze_parameters_at_iter': 50,
        'save_figures': None,
        'viz_freq': 3000,
        'custom_parameter_initialization': None,
        'validate_with_long_range': None,
        'optimize_over_data_initial_conditions': None  # [True, False]  # we can add binary flags like this
    }

    run_args_template_particle_free_time_dependent_rnn = {
        'use_particle_free_time_dependent_mode': None,
        'nr_of_particle_free_time_dependent_steps': 5,
        'shooting_model': args.shooting_model,
        # 'viz': None,
        'niters': niters,
        'optional_weight': 10,  # only used for updown_universal model
        'custom_parameter_freezing': None,
        'unfreeze_parameters_at_iter': 50,
        'save_figures': None,
        'viz_freq': 3000,
        'custom_parameter_initialization': None,
        'validate_with_long_range': None,
        'optimize_over_data_initial_conditions': None  # [True, False]  # we can add binary flags like this
    }

    run_args_template_particle_rnn = {
        'use_particle_rnn_mode': None,
        'shooting_model': args.shooting_model,
        # 'viz': None,
        'niters': niters,
        'optional_weight': 10,  # only used for updown_universal model
        'custom_parameter_freezing': None,
        'unfreeze_parameters_at_iter': 50,
        'save_figures': None,
        'viz_freq': 3000,
        'custom_parameter_initialization': None,
        'validate_with_long_range': None,
        'optimize_over_data_initial_conditions': None  # [True, False]  # we can add binary flags like this
    }

    # we do not need to sweep over particles for the particle-free method, so just set it to 2
    run_args_to_sweep_particle_free_rnn = {
        'nr_of_particles': [2],  # number of particles needs to be at least 2
        'inflation_factor': [16,32,64,128]
    }

    run_args_to_sweep_particle_free_time_dependent_rnn = {
        'nr_of_particles': [2],  # number of particles needs to be at least 2
        'inflation_factor': [16,32,64,128]
    }

    run_args_to_sweep_particle_rnn = {
        'nr_of_particles': [15,25,50], # number of particles needs to be at least 2
        'inflation_factor': [16,32,64,128]
    }

    # run for particle-free
    eu.sweep_parameters(args, run_args_to_sweep=run_args_to_sweep_particle_free_rnn,
                        run_args_template=run_args_template_particle_free_rnn,
                        python_script='spiral.py',output_dir_prefix='spiral_particle_free_rnn_',
                        do_not_recompute=not args.force_recompute)

    # run for particle-free time-dependent model
    eu.sweep_parameters(args, run_args_to_sweep=run_args_to_sweep_particle_free_time_dependent_rnn,
                        run_args_template=run_args_template_particle_free_time_dependent_rnn,
                        python_script='spiral.py', output_dir_prefix='spiral_particle_free_time_dependent_rnn_',
                        do_not_recompute=not args.force_recompute)

    # run with particles
    eu.sweep_parameters(args, run_args_to_sweep=run_args_to_sweep_particle_rnn,
                        run_args_template=run_args_template_particle_rnn,
                        python_script='spiral.py',output_dir_prefix='spiral_particle_rnn_',
                        do_not_recompute=not args.force_recompute)

    print('Done processing')
