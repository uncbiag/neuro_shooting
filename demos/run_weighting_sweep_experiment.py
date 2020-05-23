import neuro_shooting.experiment_utils as eu

if __name__ == '__main__':

    args = eu.setup_cmdline_parsing(cmdline_type='simple_functional_weighting',cmdline_title='Weighting sweep simple functional mapping')

    run_args_template_updown = {
        'shooting_model': 'updown',
        'niters': 150,
        'custom_parameter_freezing': None,
        'unfreeze_parameters_at_iter': 50,
        'save_figures': None,
        'viz_freq': 300,
        'fcn': args.fcn,
        'optimize_over_data_initial_conditions': None #[True, False]  # we can add binary flags like this
    }

    run_args_template_updown_universal = {
        'shooting_model': 'updown_universal',
        'niters': 150,
        'custom_parameter_freezing': None,
        'unfreeze_parameters_at_iter': 50,
        'save_figures': None,
        'viz_freq': 300,
        'fcn': args.fcn,
        'optimize_over_data_initial_conditions': None  # [True, False]  # we can add binary flags like this
    }

    # we do not need to sweep over particles for the particle-free method, so just set it to 2
    run_args_to_sweep_updown = {
        'nr_of_particles': [20,30],  # number of particles needs to be at least 2
        'inflation_factor': [8, 16, 32],
    }

    run_args_to_sweep_updown_universal = {
        'nr_of_particles': [20,30], # number of particles needs to be at least 2
        'inflation_factor': [8,16,32],
        'optional_weight': [1.0, 2.0, 5.0, 10.0, 50.0]
    }

    if args.sweep_updown:
        eu.sweep_parameters(args, run_args_to_sweep_updown, run_args_template_updown,
                            python_script='simple_functional_mapping_example.py',output_dir_prefix='test_updown_',
                            do_not_recompute=not args.force_recompute)

    if args.sweep_updown_universal:
        eu.sweep_parameters(args, run_args_to_sweep_updown_universal, run_args_template_updown_universal,
                            python_script='simple_functional_mapping_example.py',output_dir_prefix='test_univeral_updown_',
                            do_not_recompute=not args.force_recompute)

    print('Done processing')
