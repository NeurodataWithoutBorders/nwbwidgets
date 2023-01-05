"""Pytest configuration file."""

import matplotlib.pyplot as plt


def pytest_configure(config):
    """Set configurations to be run before all testing."""

    # Turn on interactive mode for matplotlib
    #   In practice, this defaults `block=False` for plt.show()
    #   This means that plots won't show and wait for input before proceeding
    #   To run tests while showing plots, comment this out, or replace with `plt.ioff`
    plt.ion()
