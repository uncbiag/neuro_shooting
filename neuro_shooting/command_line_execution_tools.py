import subprocess
import os
import copy
from collections import OrderedDict

def recursively_sweep_parameters(pars_to_sweep,par_dict=OrderedDict(),swept_parameter_list=[]):

    cur_pars = copy.deepcopy(pars_to_sweep)
    # get first key and remove it

    cur_keys = list(cur_pars.keys())
    if len(cur_keys)==0:
        #print('pars: {}'.format(par_dict))
        swept_parameter_list.append(copy.deepcopy(par_dict))
        return par_dict

    first_key = cur_keys[0]
    # sweep over it
    sweep_vals = cur_pars[first_key]

    # remove the key from the dictionary
    cur_pars.pop(first_key)

    for v in sweep_vals:
        par_dict[first_key] = v
        recursively_sweep_parameters(pars_to_sweep=cur_pars,par_dict=par_dict,swept_parameter_list=swept_parameter_list)

    return swept_parameter_list

def run_command_with_args(python_script,run_args,conda_environment,cuda_visible_devices=0,log_file=None):

    # now run the command
    cmd_arg_list = _make_arg_list(run_args)
    current_python_script = python_script

    pre_command = 'conda activate {}'.format(conda_environment)

    entire_pre_command = get_bash_precommand(cuda_visible_devices=cuda_visible_devices,pre_command=pre_command)
    execute_python_script_via_bash(current_python_script, cmd_arg_list, pre_command=entire_pre_command, log_file=log_file)


def _make_arg_list(args):
    arg_list = []
    for k in args:
        arg_list.append('--' + str(k))
        if args[k] is not None:
            arg_list.append( str(args[k]) )

    return arg_list

def get_bash_precommand(cuda_visible_devices,pre_command=None):

    if (cuda_visible_devices is not None) and (pre_command is not None):
        ret = '{:s} && CUDA_VISIBLE_DEVICES={:d}'.format(pre_command,cuda_visible_devices)
    elif (cuda_visible_devices is not None) and (pre_command is None):
        ret = 'CUDA_VISIBLE_DEVICES={:d}'.format(cuda_visible_devices)
    elif (cuda_visible_devices is None) and (pre_command is not None):
        ret = '{:s} && '.format(pre_command)
    else:
        ret = ''
    return ret

def get_stage_log_filename(output_directory,stage,process_name=None):
    if process_name is not None:
        ret = os.path.join(output_directory,'log_stage_{:d}_{:s}.txt'.format(stage,process_name))
    else:
        ret = os.path.join(output_directory,'log_stage_{:d}.txt'.format(stage))
    return ret

def create_kv_string(kvs):
    ret = ''
    is_first = True
    for k in kvs:
        if is_first:
            ret += str(k)+'='+str(kvs[k])
            is_first = False
        else:
            ret += '\\;' + str(k) + '=' + str(kvs[k])

    return ret

def _escape_semicolons(s):
    s_split = s.split(';')
    if len(s_split)<2:
        return s
    else:
        ret = s_split[0]
        for c in s_split[1:]:
            ret += '\\;' + c
        return ret

def add_to_config_string(cs,cs_to_add):
    if (cs is None) or (cs==''):
        # we need to check if there are semicolons in the string to add, if so, these need to be escaped
        ret = _escape_semicolons(cs_to_add)
    elif cs_to_add is None:
        ret = cs
    else:
        cs_to_add_escaped = _escape_semicolons(cs_to_add)
        ret = cs + '\\;' + cs_to_add_escaped

    return ret

def get_string_argument_from_list( arguments_as_list ):
  arguments_as_one_string = ''
  for el in arguments_as_list:
    arguments_as_one_string += el
    arguments_as_one_string += ' '
  return arguments_as_one_string

def execute_command( command, arguments_as_list, verbose=True ):
  # all commands that require substantial computation
  # will be executed through this function.
  # This will allow a uniform interface also for various clusters
  arguments_as_one_string = get_string_argument_from_list( arguments_as_list )

  if verbose:
    print('\nExecuting command:')
    print('     ' + command + ' ' + arguments_as_one_string )
  # create list for subprocess
  command_list = [ command ]
  for el in arguments_as_list:
    command_list.append( el )
  subprocess.call( command_list )

def execute_python_script_via_bash( python_script, arguments_as_list, pre_command=None, log_file=None ):

  str = get_string_argument_from_list(['python'] + [python_script] + arguments_as_list)
  if pre_command is not None and log_file is not None:
    bash_command = 'bash -l -c "{:s} {:s} > >(tee {:s}) 2>&1"'.format(pre_command,str,log_file)
  elif pre_command is not None and log_file is None:
    bash_command = 'bash -l -c "{:s} {:s}"'.format(pre_command, str)
  elif pre_command is None and log_file is not None:
    bash_command = 'bash -l -c "{:s} > >(tee {:s}) 2>&1"'.format(str, log_file)
  else:
    bash_command = 'bash -l -c "{:s}"'.format(str)

  print('\nExecuting command:')
  print('         ' + bash_command)
  os.system(bash_command)

def call_python_script( cmd ):
  cmd_full = 'python ' + cmd
  p = subprocess.Popen(cmd_full, stdout=subprocess.PIPE, shell=True)
  out, err = p.communicate()
  result = out.split('\n')
  for lin in result:
    if not lin.startswith('#'):
        print(lin)
