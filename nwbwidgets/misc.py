import matplotlib.pyplot as plt
import numpy as np


def show_annotations(annotations):
    fig, ax = plt.subplots()
    ax.plot(annotations.timestamps, np.ones(len(annotations.timestamps)), "|")
    ax.set_xlabel('time (s)')
    return fig
