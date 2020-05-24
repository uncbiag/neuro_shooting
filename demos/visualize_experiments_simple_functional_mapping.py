import torch
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import neuro_shooting.figure_settings as figure_settings
import neuro_shooting.figure_utils as figure_utils
import copy

import json

import argparse

def setup_cmdline_parsing():

    # Example configuration: --output_json_config simple_functional_mapping_experiments.json --output_base_directory current_results --figure_base_directory figure_results --shooting_model updown_universal --fcn cubic
    # --output_json_config spiral_experiments.json --output_base_directory current_spiral_results --figure_base_directory figure_results_spiral --shooting_model updown_universal

    parser = argparse.ArgumentParser('Simple functional mapping')
    parser.add_argument('--output_base_directory', type=str, default='sfm_results', help='Main directory that results have been stored in.')
    parser.add_argument('--output_json_config', type=str, default=None, help='Allows to specify a json configuration file for the outputs and the mappings to labels.')
    parser.add_argument('--figure_base_directory', type=str, default='results', help='Directory that the resulting figures will be in.')
    parser.add_argument('--fcn', type=str, default=None, choices=['cubic','quadratic'])
    parser.add_argument('--shooting_model', type=str, default='updown_universal', choices=['updown_universal','univeral','periodic','dampened_updown','simple', '2nd_order', 'updown', 'general_updown'])
    args = parser.parse_args()

    return args

def convert_to_flat_dictionary(d,d_keys=['args']):
    # creates a dictionary that contains all the keys, but flattens the entries of the keys in d_keys (as they are also assume to be dictionaries)

    d_ret = dict()
    for k in d:
        if k in d_keys:
            cur_data = d[k]
            if type(cur_data)==dict:
                for cdk in cur_data:
                    new_k = '{}.{}'.format(k,cdk)
                    d_ret[new_k] = cur_data[cdk]
            elif type(cur_data)==list:
                for current_tuple in cur_data:
                    current_key = current_tuple[0]
                    current_value = current_tuple[1]
                    new_k = '{}.{}'.format(k,current_key)
                    d_ret[new_k] = current_value
            else:
                raise ValueError('Expected dictionary, but got {}'.format(type(cur_data)))

        else:
            d_ret[k] = d[k]

    return d_ret

def get_plot_values_and_names(data,xname,yname,default_xval=None,ignore_list=None):

    # first get all the values there are for the query key (xname)
    all_unique_values = None

    for current_data,current_name in data:

        # plot sorted for name (we will average over the occurrences)
        unique_values = np.sort(current_data[xname].unique())

        if all_unique_values is None:
            all_unique_values = unique_values
        else:
            all_unique_values = np.concatenate((all_unique_values,unique_values),axis=0)

    all_unique_values = np.sort(np.unique(all_unique_values))

    # filter out all the values we want to ignore for plotting
    all_unique_values = ignore_values(all_unique_values,key=xname,ignore_list=ignore_list)

    all_vals = []
    all_names = []
    for current_data, current_name in data:

        vals = []
        for v in all_unique_values:
            # current_vals = data.loc[data[xname]==v][yname].to_numpy()
            current_vals = current_data.loc[current_data[xname]==v][yname].to_numpy()
            if (len(current_vals)==0) & (default_xval is not None):
                print('WARNING: replacing value for key {}: {} -> {}'.format(xname,v,default_xval))
                current_vals = current_data.loc[current_data[xname] == default_xval][yname].to_numpy()

            vals.append(current_vals)

        all_vals.append(vals)
        all_names.append(current_name)


    return all_vals,all_names,all_unique_values

def find_nonempty_values(vals):
    ne_vals = []
    for v in vals:
        if len(v)>0:
            ne_vals.append(v)

    return ne_vals

def find_nonempty_values_and_positions(vals,pos,ignore_outliers=False):

    ne_vals = []
    ne_pos = []
    ne_outliers = []

    nr_of_entries = len(vals)

    for itr in range(nr_of_entries):
        v = vals[itr]
        p = pos[itr]

        if ignore_outliers:
            v, nr_of_outliers = remove_outliers(v)
        else:
            nr_of_outliers = 0

        if len(v)>0:
            ne_vals.append(v)
            ne_pos.append(p)
            ne_outliers.append(nr_of_outliers)

    return ne_vals,ne_pos,ne_outliers

def _all_in_one_plot(vals,names,unique_values,do_printing,title_string,use_boxplot=False,ignore_outliers=True):

    # Create a figure instance
    fig = plt.figure( facecolor='white')
    ax = fig.add_subplot(111, frameon=True)

    nr_of_groups = len(names)
    nr_of_unique_values = len(unique_values)

    width = 1.0/(nr_of_groups+1)

    xlabel_name = _translate_label(xname)
    ylabel_name = _translate_label(yname)

    position_offsets = np.linspace(0+width/2+width/4,1-width/2-width/4,nr_of_groups)#(np.arange(0,nr_of_groups)-nr_of_groups/2)*width
    position_offsets = -(position_offsets-np.mean(position_offsets))
    default_positions = np.arange(1,nr_of_unique_values+1)

    vert_lines = (default_positions[0:-1]+default_positions[1:])/2

    if do_printing:
        xlabel_name = figure_utils.escape_latex_special_characters(xlabel_name)
        ylabel_name = figure_utils.escape_latex_special_characters(ylabel_name)

    ax.set_xlabel(xlabel_name)
    ax.set_ylabel(ylabel_name)
    ax.set_title(title_string)

    # Create the boxplot
    labels = [str(q) for q in unique_values]

    bps = []
    legend_names = []

    # fill with colors
    colors = ['silver', 'deepskyblue', 'seagreen']

    outlier_info = []

    for n in range(nr_of_groups):
        ne_vals, ne_pos, nr_of_outliers = find_nonempty_values_and_positions(vals=vals[n],pos=default_positions-position_offsets[n], ignore_outliers=ignore_outliers)

        if len(ne_vals)>0:
            if use_boxplot:
                bp = ax.boxplot(x=ne_vals, widths=width, positions=ne_pos, notch=False, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor(colors[n])
                bps.append(bp['boxes'][0])
            else:
                bp = ax.violinplot(dataset=ne_vals,widths=width, positions=ne_pos, showmedians=True, quantiles=[[0.25,0.75]]*len(ne_vals))
                bps.append(bp['bodies'][0])

            legend_names.append(names[n])

        outlier_info.append((ne_pos,nr_of_outliers))

    ax.set_xticks(np.arange(1, len(unique_values) + 1))
    ax.set_xticklabels(labels)
    ax.legend(bps, legend_names, loc='best')

    # create verticle lines
    for x in vert_lines:
        ax.axvline(x=x,color='k',linewidth=0.5)

    # now plot how many outliers there were if any (need to do this at the end so we can get the axis information)
    ylim = ax.get_ylim()
    desired_ypos = ylim[0] + 0.025*(ylim[1]-ylim[0])
    for ne_pos,nr_of_outliers in outlier_info:
        for nr_o, c_pos in zip(nr_of_outliers, ne_pos):
            if nr_o > 0:  # there was an outlier
                plt.text(c_pos, desired_ypos, '{}*'.format(nr_o), horizontalalignment='center')


def _side_by_side_plot(vals,names,unique_values,do_printing,title_string,use_boxplot=False):
    nr_of_subplots = len(names)
    subplot_value = 101 + 10 * nr_of_subplots

    # Create a figure instance
    fig = plt.figure(figsize=(4 * nr_of_subplots, 4), facecolor='white')

    ax = dict()

    # fill with colors
    colors = ['silver', 'deepskyblue', 'seagreen']
    #colors = ['pink', 'lightblue', 'lightgreen']

    for n in range(nr_of_subplots):

        # Create an axes instance
        ax[n] = fig.add_subplot(subplot_value + n, frameon=True)

        xlabel_name = _translate_label(xname)
        ylabel_name = _translate_label(yname)

        if do_printing:
            xlabel_name = figure_utils.escape_latex_special_characters(xlabel_name)
            ylabel_name = figure_utils.escape_latex_special_characters(ylabel_name)

        ax[n].set_xlabel(xlabel_name)
        ax[n].set_ylabel(ylabel_name)
        ax[n].set_title(names[n])

        # Create the boxplot
        labels = [str(q) for q in unique_values]

        ne_vals = find_nonempty_values(vals=vals[n])
        if len(ne_vals)>0:
            if use_boxplot:
                bp = ax.boxplot(x=ne_vals, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor(colors[n])
            else:
                bp = ax[n].violinplot(dataset=ne_vals, showmedians=True, quantiles=[[0.25,0.75]]*len(ne_vals))

        ax[n].set_xticks(np.arange(1, len(unique_values) + 1))
        ax[n].set_xticklabels(labels)

    fig.suptitle(title_string,y=1.0)

def _plot_data(data,xname,yname,default_xval=None,title_string='',visualize=True,save_figure_directory=None,save_figure_name=None,ignore_list=None):

    if not visualize and not save_figure_name and not save_figure_directory:
        return

    do_printing = (save_figure_directory is not None) or (save_figure_name is not None)

    if do_printing and visualize:
        raise ValueError('Cannot print and visualize at the same time')

    if do_printing:
        previous_backend, rcsettings = figure_settings.setup_pgf_plotting()

        if save_figure_directory is None:
            save_figure_directory = '.'
        if save_figure_name is None:
            save_figure_name = 'plot_{}_over_{}'.format(yname,xname)


    vals, names, unique_values = get_plot_values_and_names(data=data,xname=xname,yname=yname,default_xval=default_xval,ignore_list=ignore_list)

    #_side_by_side_plot(vals=vals,names=names,unique_values=unique_values,do_printing=do_printing,title_string=title_string)
    _all_in_one_plot(vals=vals,names=names,unique_values=unique_values,do_printing=do_printing,title_string=title_string)


    if do_printing:
        figure_utils.save_all_formats(output_directory=save_figure_directory,
                                      filename=save_figure_name)

        figure_settings.reset_pgf_plotting(backend=previous_backend, rcsettings=rcsettings)

    if visualize:
        plt.show()


def plot_data(data,xname,yname,title_string='',visualize=True,save_figure_directory=None,save_figure_name=None,default_xval=None,ignore_list=None):

    if len(data)==0:
        print('INFO: empty data for {}/{} -- not plotting'.format(xname,yname))
        return

    if not visualize and save_figure_directory is None and save_figure_name is None:
        _plot_data(data=data, xname=xname,yname=yname,default_xval=default_xval,title_string=title_string,visualize=True,ignore_list=ignore_list)
    else:
        if visualize:
            _plot_data(data=data, xname=xname, yname=yname, default_xval=default_xval, title_string=title_string, visualize=True, save_figure_directory=None, save_figure_name=None,ignore_list=ignore_list)
        if save_figure_directory is not None:
            _plot_data(data=data, xname=xname, yname=yname, default_xval=default_xval, title_string=title_string, visualize=False, save_figure_directory=save_figure_directory, save_figure_name=save_figure_name,ignore_list=ignore_list)

def load_JSON(filename):
    """
    Loads a JSON configuration file

    :param filename: filename of the configuration to be loaded
    """
    try:
        with open(filename) as data_file:
            print('Loading parameter file = {}'.format(filename))
            ret = json.load(data_file)
            return ret
    except IOError as e:
        print('Could not open file = {}; ignoring request.'.format(filename))


def write_JSON(filename, data):
    """
    Writes the JSON configuration to a file

    :param filename: filename to write the configuration to
    :param data: data to write out; should be a dictionary
    """

    with open(filename, 'w') as outfile:
        print('Writing parameter file = {}'.format(filename))
        json.dump(data, outfile, indent=4, sort_keys=True)


def _translate_label(label_name):
    # mapping of names between labels

    label_mappings = {
        'sim_loss': 'similarity loss',
        'args.nr_of_particles': '# of particles',
        'args.inflation_factor': 'inflation factor',
        'test_loss': 'loss',
        'norm_loss': 'parameter norm loss',
        'log_complexity_measures.log2_frobenius': 'log_2(frobenius norm complexity)',
        'log_complexity_measures.log2_nuclear': 'log_2(nuclear norm complexity)'
    }

    if label_name in label_mappings:
        return label_mappings[label_name]
    else:
        return label_name

def get_files(output_base_directory,output_config):
    # get all the result files
    if output_config is None:
        # just flat
        files = sorted(glob.glob(os.path.join(args.output_base_directory, '**', '*.pt'), recursive=True))
        return [(files,'')]
    else:
        all_files = []
        for k in output_config:
            print('Reading directory {} for {}'.format(k,output_config[k]))
            files = sorted(glob.glob(os.path.join(args.output_base_directory, k, '**', '*.pt'), recursive=True))
            all_files.append((files,output_config[k]))
        return all_files

def get_pandas_dataframes(files):

    all_data = []

    for current_files,current_name in files:

        data = None
        # now read the data and create a pandas data fram
        for f in current_files:
            current_data = torch.load(f)
            current_data = convert_to_flat_dictionary(current_data, ['args', 'nr_of_parameters', 'log_complexity_measures', 'short_range_log_complexity_measures'])
            if data is None:
                data = pd.DataFrame([current_data])
            else:
                data = data.append([current_data], ignore_index=True)

        all_data.append((data,current_name))

    return all_data

def get_data_range(data,key):

    vals = None
    for current_data,_ in data:
        current_vals = current_data[key].unique()
        if vals is None:
            vals = current_vals
        else:
            vals = np.concatenate((vals,current_vals),axis=0)

    # sort them
    vals = np.sort(np.unique(vals))
    return vals

def select_data(data,selection,default_val=None):

    selected_data = []

    for current_data,current_name in data:
        current_selection = None
        for s in selection:
            if current_selection is None:
                # e.g.,: data.loc[(data['args.shooting_model']==model_name) & (data['args.fcn']==fcn_name)]
                current_selection = current_data.loc[current_data[s]==selection[s]] # key s has the value
                if (len(current_selection)==0) & (default_val is not None):
                    print('WARNING: replacing value for {}: {} -> {}'.format(s,selection[s],default_val))
                    current_selection = current_data.loc[current_data[s] == default_val]  # key s has the value
            else:
                current_selection = current_selection.loc[current_selection[s]==selection[s]] # key s has the value
                if (len(current_selection) == 0) & (default_val is not None):
                    print('WARNING: replacing value for {}: {} -> {}'.format(s, selection[s], default_val))
                    current_selection = current_selection.loc[current_selection[s]==default_val] # key s has the value

        selected_data.append((current_selection,current_name))

    return selected_data


def ignore_values(np_array, key, ignore_list):

    if ignore_list is None:
        return np_array

    if key in ignore_list:
        values_to_ignore = ignore_list[key]
        ret = np_array
        for v in values_to_ignore:
            ret = ret[ret!=v]
        return ret
    else:
        return np_array

def remove_outliers(np_vals,max_vals=1e8):

    perc_25 = np.percentile(np_vals,25)
    perc_75 = np.percentile(np_vals,75)
    iqr = perc_75-perc_25

    # standard rule for boxplot outlier rejection
    outlier_multiplier = 1.5 # how many interquartile ranges away from 75th or 25th quantiles to count as an outlier; 1.5 is the MATLAB rule

    min_outlier_val = perc_25-outlier_multiplier*iqr
    max_outlier_val = perc_75+outlier_multiplier*iqr

    np_vals_outliers_removed = np_vals[(np_vals>=min_outlier_val) & (np_vals<=max_outlier_val) & (np_vals<=max_vals)]
    nr_of_outliers = len(np_vals)-len(np_vals_outliers_removed)

    return np_vals_outliers_removed,nr_of_outliers


def compute_statistics(pd_vals,determine_outliers=True):

    raw_vals = pd_vals.to_numpy()

    if np.max(raw_vals)>100000:
        print('{}'.format(raw_vals))
        print('Hello')

    if determine_outliers:
        vals, nr_of_outliers = remove_outliers(raw_vals)
    else:
        nr_of_outliers = 0
        vals = raw_vals

    stats = dict()
    stats['mean'] = np.mean(vals)
    stats['median'] = np.median(vals)
    stats['std'] = np.std(vals)
    stats['min'] = np.min(vals)
    stats['max'] = np.max(vals)
    stats['perc_25'] = np.percentile(vals,25)
    stats['perc_75'] = np.percentile(vals,75)
    stats['iqr'] = stats['perc_75']-stats['perc_25']
    stats['raw_vals'] = vals
    stats['nr_of_values'] = len(stats['raw_vals'])
    stats['nr_of_outliers'] = nr_of_outliers

    return stats

def get_stats_from_row(row,name):
    vals = []
    for d in row:
        vals.append(d[name])
    return vals

def create_interleaved_string(vals,format_str,delimiter):

    str = ''
    nr_of_values = len(vals[0])

    for itr in range(nr_of_values):
        current_vals = []
        for vs in vals: # this is a list
            current_vals.append(vs[itr])
        str += format_str.format(*current_vals) + delimiter
    return str

def print_table(table,table_name,row_name,row_vals,col_name,col_vals):

    print('{}'.format(table_name))
    print('{}: {}'.format(col_name,col_vals))

    for i,rv in enumerate(row_vals):
        current_row = table[i]

        # means = get_stats_from_row(current_row,'mean')
        # stds = get_stats_from_row(current_row,'std')
        #
        # interleaved_string = create_interleaved_string(means,stds,'{:2.2f}({:2.2f})','; ')

        medians = get_stats_from_row(current_row,'median')
        iqrs = get_stats_from_row(current_row,'iqr')
        nr_of_outliers = get_stats_from_row(current_row,'nr_of_outliers')

        interleaved_string = create_interleaved_string([medians,iqrs,nr_of_outliers],'{:2.2f}({:2.2f})({})','; ')

        print('{}={}: {}'.format(row_name,rv,interleaved_string))

    print('\nNumber of values for computations:\n')
    for i,rv in enumerate(row_vals):
        current_row = table[i]

        nr_of_values = get_stats_from_row(current_row,'nr_of_values')
        print('{}={}: {}'.format(row_name,rv,nr_of_values))



def print_tables(tables,row_name,row_vals,col_name,col_vals,measure):

    print('\nMeasure = {}:\n'.format(measure))
    for table,table_name in tables:
        print_table(table=table,table_name=table_name,row_name=row_name,row_vals=row_vals,col_name=col_name,col_vals=col_vals)
        print('\n')


def create_table(data,keys=['args.nr_of_particles','args.inflation_factors'],measure='test_loss', ignore_list=None, default_list=dict()):
    # creates a table for the specified keys which gives the statistics for the specified measure

    if len(keys)!=2:
        raise ValueError('Two key values were expected, but obtained {}'.format(len(keys)))

    row_name = keys[0]
    col_name = keys[1]

    row_vals = get_data_range(data=data, key=row_name)
    col_vals = get_data_range(data=data, key=col_name)

    # filter out in case we do not want some of them
    row_vals = ignore_values(row_vals, key=row_name, ignore_list=ignore_list)
    col_vals = ignore_values(col_vals, key=col_name, ignore_list=ignore_list)

    tables = []

    determine_outliers = True

    for current_data,current_name in data:
        table = []
        # create table
        for rv in row_vals:
            current_row = []
            cur_row_selection = current_data.loc[current_data[row_name]==rv]
            if len(cur_row_selection)==0:
                if row_name in default_list:
                    default_value = default_list[row_name]
                    print('WARNING: {}: {}: {}->{}'.format(current_name,row_name,rv,default_value))
                    cur_row_selection = current_data.loc[current_data[row_name] == default_value]

            for cv in col_vals:
                cur_col_selection = cur_row_selection.loc[cur_row_selection[col_name]==cv]
                if len(cur_col_selection)==0:
                    # try the default if there is one
                    if col_name in default_list:
                        default_value = default_list[col_name]
                        print('WARNING: {}: {}: {}->{}'.format(current_name, col_name, cv, default_value))
                        cur_col_selection = cur_row_selection.loc[cur_row_selection[col_name]==default_value]

                cur_vals = cur_col_selection[measure]
                stats = compute_statistics(cur_vals,determine_outliers=determine_outliers)
                current_row.append(stats)

            table.append(current_row)
        tables.append((table,current_name))

    print_tables(tables=tables,row_name=row_name,row_vals=row_vals,col_name=col_name,col_vals=col_vals,measure=measure)

if __name__ == '__main__':

    args = setup_cmdline_parsing()

    if args.output_json_config is not None:
        output_config = load_JSON(args.output_json_config)
    else:
        output_config = None

    # get all the result files
    files = get_files(output_base_directory=args.output_base_directory,output_config=output_config)

    # create all the pandas dataframes (for the different output file/name tuples)

    data = get_pandas_dataframes(files=files)

    all_nr_of_particles = get_data_range(data=data,key='args.nr_of_particles')
    all_inflation_factors = get_data_range(data=data,key='args.inflation_factor')

    # we can now do our desired plots

    # get the desired function and model_name
    fcn_name = args.fcn
    model_name = args.shooting_model

    # here we can add key-value pairs (with lists) of things we do not wish to plot
    ignore_list = dict()
    default_list = dict()

    uses_fcn = fcn_name is not None;

    if uses_fcn:
        save_figure_directory = '{}_figures_model_{}_fcn_{}'.format(args.figure_base_directory,model_name,fcn_name)

        selection = {
            'args.shooting_model': model_name,
            'args.fcn': fcn_name
        }
        ynames_to_plot = ['sim_loss','test_loss','norm_loss','log_complexity_measures.log2_frobenius','log_complexity_measures.log2_nuclear']

    else:
        save_figure_directory = '{}_figures_model_{}'.format(args.figure_base_directory, model_name)

        selection = {
            'args.shooting_model': model_name,
        }
        ynames_to_plot_template = ['sim_loss','test_loss','norm_loss','log_complexity_measures.log2_frobenius','log_complexity_measures.log2_nuclear']
        ynames_to_plot = copy.deepcopy(ynames_to_plot_template)
        for n in ynames_to_plot_template:
            ynames_to_plot.append('short_range_{}'.format(n))

        ignore_list['args.nr_of_particles'] = [2]
        default_list['args.nr_of_particles'] = 2

        all_nr_of_particles = ignore_values(all_nr_of_particles,key='args.nr_of_particles',ignore_list=ignore_list)

    model_data = select_data(data=data,selection=selection)

    # computing some statistics and output as LaTeX table
    create_table(model_data,keys=['args.nr_of_particles','args.inflation_factor'],measure='nr_of_parameters.overall', ignore_list=ignore_list, default_list=default_list)
    create_table(model_data,keys=['args.nr_of_particles','args.inflation_factor'],measure='test_loss', ignore_list=ignore_list, default_list=default_list)

    # creating some plots

    # create plots for a particular number of particles
    for nr_of_particles in all_nr_of_particles:
        # get the current data
        xname = 'args.inflation_factor'
        #data = model_data.loc[model_data['args.nr_of_particles'] == nr_of_particles]
        data = select_data(data=model_data,selection={'args.nr_of_particles': nr_of_particles},default_val=2)

        title_string = '{} particles'.format(nr_of_particles)
        ynames = ynames_to_plot
        for yname in ynames:
            save_figure_name = 'plot_{}_{}_{}_over_{}_for_{}_particles'.format(model_name,fcn_name,yname,xname,nr_of_particles)
            plot_data(data=data,xname=xname, yname=yname, title_string=title_string, save_figure_directory=save_figure_directory,save_figure_name=save_figure_name,ignore_list=ignore_list)

    # create plots for a particular inflation factor
    for inflation_factor in all_inflation_factors:
        # get the current data
        xname = 'args.nr_of_particles'
        #data = model_data.loc[model_data['args.inflation_factor'] == inflation_factor]
        data = select_data(data=model_data,selection={'args.inflation_factor': inflation_factor})

        title_string = 'inflation factor = {}'.format(inflation_factor)
        ynames =ynames_to_plot
        for yname in ynames:
            save_figure_name = 'plot_{}_{}_{}_over_{}_for_inflation_factor_{}'.format(model_name,fcn_name,yname,xname,inflation_factor)
            plot_data(data=data, xname=xname, yname=yname, title_string=title_string, save_figure_directory=save_figure_directory,save_figure_name=save_figure_name,default_xval=2,ignore_list=ignore_list)
