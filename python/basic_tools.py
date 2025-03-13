# basic_tools.py

import os
import matplotlib.pyplot as plt

# Creates directory if it does not exist
def makeDir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# Save plot to png and pdf
def savePlot(plot_dir, plot_name):
    output_png = "{0}/{1}.png".format(plot_dir, plot_name)
    output_pdf = "{0}/{1}.pdf".format(plot_dir, plot_name)
    plt.savefig(output_png, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')

