import matplotlib.pyplot as plt
import os
import re

def escape_latex_special_characters(text):
    """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key=lambda item: - len(item))))
    escaped_string = regex.sub(lambda match: conv[match.group()], text)
    return escaped_string
    #return text.replace('_','\_')

def save_all_formats(output_directory, filename):

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    plt.tight_layout()
    formats = ['png','pdf','pgf']
    for f in formats:
        current_filename = '{}.{}'.format(filename,f)
        complete_filename = os.path.join(output_directory,current_filename)
        plt.savefig(complete_filename)
