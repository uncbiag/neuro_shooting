# this script is simply to run many experimental conditions for subsequent evaluation

import neuro_shooting.experiment_utils as eu

if __name__ == '__main__':

    args = eu.setup_cmdline_parsing(cmdline_type='spiral',cmdline_title='Spiral')

    run_args_template = {
        'shooting_model': args.shooting_model,
        #'viz': None,
        'niters': 1500,
        'optional_weight': 10, # only used for updown_universal model
        'custom_parameter_freezing': None,
        'unfreeze_parameters_at_iter': 50,
        'save_figures': None,
        'viz_freq': 3000,
        'custom_parameter_initialization': None,
        'validate_with_long_range': None,
        'optimize_over_data_initial_conditions': None #[True, False]  # we can add binary flags like this
    }

    run_args_to_sweep = {
        'nr_of_particles': [15,25,50], # number of particles needs to be at least 2
        'inflation_factor': [16,32,64,128]
    }

    # run for particle-free
    eu.sweep_parameters(args, run_args_to_sweep=run_args_to_sweep,
                        run_args_template=run_args_template,
                        python_script='spiral.py', output_dir_prefix='spiral_particles_',
                        do_not_recompute=not args.force_recompute)

    print('Done processing')
