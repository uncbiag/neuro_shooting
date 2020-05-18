import matplotlib.pyplot as plt
import os

def escape_underscores(str):
    return str.replace('_','\_')

def save_all_formats(output_directory, filename):

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    formats = ['png','pdf','pgf']
    for f in formats:
        current_filename = '{}.{}'.format(filename,f)
        complete_filename = os.path.join(output_directory,current_filename)
        plt.savefig(complete_filename)
