import torch
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import neuro_shooting.figure_settings as figure_settings
import neuro_shooting.figure_utils as figure_utils

import argparse

def setup_cmdline_parsing():
    parser = argparse.ArgumentParser('Simple functional mapping')
    parser.add_argument('--output_base_directory', type=str, default='sfm_results', help='Main directory that results have been stored in')
    parser.add_argument('--fcn', type=str, default='cubic', choices=['cubic','quadratic'])
    parser.add_argument('--shooting_model', type=str, default='updown', choices=['univeral','periodic','dampened_updown','simple', '2nd_order', 'updown', 'general_updown'])
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

def _plot_data(data,xname,yname,visualize=True,save_figure_directory=None,save_figure_name=None):

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


    # plot sorted for xname (we will average over the occurrences)
    unique_values = np.sort(data[xname].unique())

    vals = []
    for v in unique_values:
        current_vals = data.loc[data[xname]==v][yname].to_numpy()
        vals.append(current_vals)

    # Create a figure instance
    fig = plt.figure()

    # Create an axes instance
    ax = fig.add_subplot(111, frameon=True)

    xlabel_name = _translate_label(xname)
    ylabel_name = _translate_label(yname)

    if do_printing:
        xlabel_name = figure_utils.escape_latex_special_characters(xlabel_name)
        ylabel_name = figure_utils.escape_latex_special_characters(xlabel_name)

    ax.set_xlabel(xlabel_name)
    ax.set_ylabel(ylabel_name)

    # Create the boxplot
    labels = [str(q) for q in unique_values]

    ax.set_xticks(np.arange(1, len(unique_values) + 1))
    ax.set_xticklabels(labels)
    bp = ax.violinplot(dataset=vals)

    if do_printing:
        figure_utils.save_all_formats(output_directory=save_figure_directory,
                                      filename=save_figure_name)

        figure_settings.reset_pgf_plotting(backend=previous_backend, rcsettings=rcsettings)

    if visualize:
        plt.show()


def plot_data(data,xname,yname,visualize=True,save_figure_directory=None,save_figure_name=None):

    if len(data)==0:
        print('INFO: empty data for {}/{} -- not plotting'.format(xname,yname))
        return

    if not visualize and save_figure_directory is None and save_figure_name is None:
        _plot_data(data=data, xname=xname,yname=yname,visualize=True)
    else:
        if visualize:
            _plot_data(data=data, xname=xname, yname=yname, visualize=True, save_figure_directory=None, save_figure_name=None)
        if save_figure_directory is not None:
            _plot_data(data=data, xname=xname, yname=yname, visualize=False, save_figure_directory=save_figure_directory, save_figure_name=save_figure_name)


def _translate_label(label_name):
    # mapping of names between labels

    label_mappings = {
        'sim_loss': 'similarity loss',
        'args.nr_of_particles': '# of particles',
        'args.inflation_factor': 'inflation factor'
    }

    if label_name in label_mappings:
        return label_mappings[label_name]
    else:
        return label_name

if __name__ == '__main__':

    args = setup_cmdline_parsing()

    # get all the result files
    files = sorted(glob.glob(os.path.join(args.output_base_directory, '**', '*.pt'), recursive=True))

    data = None

    # now read the data and create a pandas data fram
    for f in files:
        current_data = torch.load(f)
        current_data = convert_to_flat_dictionary(current_data,['args','nr_of_parameters','log_complexity_measures'])
        if data is None:
            data = pd.DataFrame([current_data])
        else:
            data = data.append([current_data],ignore_index = True)


    # we can now do our desired plots

    # get cubic example
    all_nr_of_particles = np.sort(data['args.nr_of_particles'].unique())
    all_inflation_factors = np.sort(data['args.inflation_factor'].unique())

    fcn_name = args.fcn
    model_name = args.shooting_model

    save_figure_directory = 'figures_model_{}_fcn_{}'.format(model_name,fcn_name)

    model_data = data.loc[(data['args.shooting_model']==model_name) & (data['args.fcn']==fcn_name)]

    # create plots for a particular number of particles
    for nr_of_particles in all_nr_of_particles:
        # get the current data
        xname = 'args.inflation_factor'
        yname = 'sim_loss'
        save_figure_name = 'plot_{}_over_{}_for_{}_particles'.format(yname,xname,nr_of_particles)
        data = model_data.loc[model_data['args.nr_of_particles']==nr_of_particles]
        plot_data(data=data,xname=xname, yname=yname, save_figure_directory=save_figure_directory,save_figure_name=save_figure_name)

    # create plots for a particular inflation factor
    for inflation_factor in all_inflation_factors:
        # get the current data
        xname = 'args.nr_of_particles'
        yname = 'sim_loss'
        save_figure_name = 'plot_{}_over_{}_for_inflation_factor_{}'.format(yname,xname,inflation_factor)
        data = model_data.loc[model_data['args.inflation_factor'] == inflation_factor]
        plot_data(data=data, xname=xname, yname=yname, save_figure_directory=save_figure_directory,save_figure_name=save_figure_name)
