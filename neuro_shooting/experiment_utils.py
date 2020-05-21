import argparse
import copy
import neuro_shooting.command_line_execution_tools as ce
import os

def setup_cmdline_parsing(cmdline_type='simple_functional_mapping',cmdline_title=None):

    if cmdline_title is None:
        cmdline_title = cmdline_type

    supported_types = ['simple_functional_mapping','simple_functional_weighting']
    if cmdline_type not in supported_types:
        raise ValueError('Unsupported command line type {}'.format(cmdline_type))

    if cmdline_type=='simple_functional_mapping':
        parser = argparse.ArgumentParser(cmdline_title)
        parser.add_argument('--gpu', type=int, default=0, help='Enable GPU computation on specified GPU.')
        parser.add_argument('--path_to_python', type=str, default=os.popen('which python').read().rstrip(), help='Full path to python in your conda environment.')
        parser.add_argument('--nr_of_seeds', type=int, default=1, help='Number of consecutive random seeds which we should run; i.e., number of random runs')
        parser.add_argument('--starting_seed_id', type=int, default=0, help='Seed that we start with.')
        parser.add_argument('--fcn', type=str, default='cubic', choices=['cubic','quadratic'])
        parser.add_argument('--shooting_model', type=str, default='updown_universal', choices=['updown_univeral', 'universal','periodic','dampened_updown','simple', '2nd_order', 'updown', 'general_updown'])
        parser.add_argument('--output_base_directory', type=str, default='sfm_results', help='Main directory that the results will be stored in')
        args = parser.parse_args()

        return args

    if cmdline_type=='simple_functional_weighting':
        parser = argparse.ArgumentParser(cmdline_title)
        parser.add_argument('--gpu', type=int, default=0, help='Enable GPU computation on specified GPU.')
        parser.add_argument('--path_to_python', type=str, default=os.popen('which python').read().rstrip(), help='Full path to python in your conda environment.')
        parser.add_argument('--nr_of_seeds', type=int, default=1, help='Number of consecutive random seeds which we should run; i.e., number of random runs')
        parser.add_argument('--starting_seed_id', type=int, default=0, help='Seed that we start with.')
        parser.add_argument('--fcn', type=str, default='cubic', choices=['cubic','quadratic'])
        parser.add_argument('--sweep_updown', action='store_true', default=False)
        parser.add_argument('--sweep_updown_universal', action='store_true', default=False)
        parser.add_argument('--output_base_directory', type=str, default='sfm_results', help='Main directory that the results will be stored in')
        args = parser.parse_args()

        return args

def create_experiment_name(basename,d):

    name = basename
    for k in d:
        name += '_{}_{}'.format(k,d[k])
    return name

def merge_args(run_args_template,add_args):

    merged_args = copy.deepcopy(run_args_template)

    for k in add_args:
        v = add_args[k]
        if (v is True) or (v is False): # check if v is binary
            if v:
                merged_args[k] = None # just add the flag
        else:
            merged_args[k] = v

    return merged_args

def sweep_parameters(args,run_args_to_sweep,run_args_template,python_script='simple_functional_mapping_example.py',output_dir_prefix=''):
    swept_parameter_list = ce.recursively_sweep_parameters(pars_to_sweep=run_args_to_sweep)

    # base settings
    seeds = list(range(0 + args.starting_seed_id,
                       args.nr_of_seeds + args.starting_seed_id))  # do 10 runs each, we can also specify this manually [1,20] # seeds we iterate over (for multiple runs)

    output_base_directory = output_dir_prefix + args.output_base_directory

    if not os.path.exists(output_base_directory):
        os.mkdir(output_base_directory)

    # now go over all these parameter structures and run the experiments
    for sidx, seed in enumerate(seeds):
        for d in swept_parameter_list:

            if 'shooting_model' in d: # we are sweeping over it
                current_shooting_model = d['shooting_model']
            elif 'shooting_model' in run_args_template:
                current_shooting_model = run_args_template['shooting_model']
            else:
                current_shooting_model = args.shooting_model

            if 'fcn' in d:  # we are sweeping over it
                current_fcn = d['fcn']
            elif 'fcn' in run_args_template:
                current_fcn = run_args_template['fcn']
            else:
                current_fcn = args.fcn

            basename = 'run_{:02d}_{}_{}'.format(sidx + args.starting_seed_id, current_fcn, current_shooting_model)
            experiment_name = create_experiment_name(basename, d)
            output_directory = os.path.join(output_base_directory, experiment_name)
            log_file = os.path.join(output_directory, 'runlog.log')

            if not os.path.exists(output_directory):
                os.mkdir(output_directory)

            run_args = merge_args(run_args_template=run_args_template, add_args=d)
            # add the output-directory
            run_args['output_directory'] = output_directory
            run_args['seed'] = seed

            print('Running {}'.format(experiment_name))

            ce.run_command_with_args(python_script=python_script,
                                     run_args=run_args,
                                     path_to_python=args.path_to_python,
                                     cuda_visible_devices=args.gpu,
                                     log_file=log_file)