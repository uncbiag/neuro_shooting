# this script is simply to run many experimental conditions for subsequent evaluation

import neuro_shooting.experiment_utils as eu

if __name__ == '__main__':

    args = eu.setup_cmdline_parsing(cmdline_type='simple_functional_mapping',cmdline_title='Simple functional mapping')

    run_args_template = {
        'shooting_model': args.shooting_model,
        #'viz': None,
        'niters': 500,
        'optional_weight': 10, # only used for updown_universal model
        'custom_parameter_freezing': None,
        'unfreeze_parameters_at_iter': 50,
        'save_figures': None,
        'viz_freq': 600,
        'fcn': args.fcn,
        'do_not_plot_temporal_data': None,
        'optimize_over_data_initial_conditions': None #[True, False]  # we can add binary flags like this
    }

    run_args_to_sweep = {
        'nr_of_particles': [2,5,15,25], # number of particles needs to be at least 2
        'inflation_factor': [4,8,16,32,64,128]
    }

    # run for particle-free
    eu.sweep_parameters(args, run_args_to_sweep=run_args_to_sweep,
                        run_args_template=run_args_template,
                        python_script='simple_functional_mapping_example.py', output_dir_prefix='long_iteration_particles_',
                        do_not_recompute=not args.force_recompute)

    print('Done processing')
