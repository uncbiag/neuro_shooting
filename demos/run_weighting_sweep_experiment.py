import neuro_shooting.experiment_utils as eu

if __name__ == '__main__':

    args = eu.setup_cmdline_parsing(cmdline_type='simple_functional_weighting',cmdline_title='Weighting sweep simple functional mapping')

    run_args_template_updown = {
        'shooting_model': 'updown',
        'niters': 100,
        'custom_parameter_freezing': None,
        'unfreeze_parameters_at_iter': 25,
        'save_figures': None,
        'viz_freq': 300,
        'fcn': args.fcn,
        'optimize_over_data_initial_conditions': None #[True, False]  # we can add binary flags like this
    }

    run_args_template_updown_universal = {
        'shooting_model': 'updown_universal',
        'niters': 100,
        'custom_parameter_freezing': None,
        'unfreeze_parameters_at_iter': 25,
        'save_figures': None,
        'viz_freq': 300,
        'fcn': args.fcn,
        'optimize_over_data_initial_conditions': None  # [True, False]  # we can add binary flags like this
    }

    # we do not need to sweep over particles for the particle-free method, so just set it to 2
    run_args_to_sweep_updown = {
        'nr_of_particles': [10,20],  # number of particles needs to be at least 2
        'inflation_factor': [8, 16, 32],
    }

    run_args_to_sweep_updown_universal = {
        'nr_of_particles': [10,20], # number of particles needs to be at least 2
        'inflation_factor': [8,16,32],
        'optional_weight': [0.001, 0.01, 0.1, 1.0]
    }

    if args.sweep_updown:
        eu.sweep_parameters(args, run_args_to_sweep_updown, run_args_template_updown,
                            python_script='simple_functional_mapping_example.py',output_dir_prefix='test_updown_')

    if args.sweep_updown_univeral:
        eu.sweep_parameters(args, run_args_to_sweep_updown_universal, run_args_template_updown_universal,
                            python_script='simple_functional_mapping_example.py',output_dir_prefix='test_univeral_updown_')

    print('Done processing')
