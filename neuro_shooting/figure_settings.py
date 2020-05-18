import matplotlib
import matplotlib.pyplot as plt

def set_font_size_for_axis(ax, fontsize=18):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

def setup_pgf_plotting():
    previous_backend = matplotlib.get_backend()

    new_rcsettings = {'pgf.texsystem': "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'pgf.preamble': [
            '\\usepackage{amsmath}',  # load additional packages
            '\\usepackage{amssymb}',  # load additional packages
        ]
    }

    previous_rcsettings = dict()
    for k in new_rcsettings:
        previous_rcsettings[k] = matplotlib.rcParams[k]

    matplotlib.use('pgf')
    matplotlib.rcParams.update(new_rcsettings)

    return previous_backend, previous_rcsettings

def reset_pgf_plotting(backend,rcsettings):
    matplotlib.use(backend)
    matplotlib.rcParams.update(rcsettings)

def setup_plotting():

    TINY = 8
    SMALL = 10
    MEDIUM = 12
    LARGE = 16
    HUGE = 20

    plt.rc('font', size=LARGE)          # controls default text sizes
    plt.rc('axes', titlesize=LARGE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LARGE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=LARGE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=LARGE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LARGE)    # legend fontsize
    plt.rc('figure', titlesize=HUGE)  # fontsize of the figure title