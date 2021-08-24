Getting Started
===============

NWB Widgets
-----------
A library of widgets for visualization NWB data in a Jupyter notebook (or lab). The widgets allow you to navigate through the hierarchical structure of the NWB file and visualize specific data elements. It is designed to work out-of-the-box with NWB 2.0 files and to be easy to extend.

Installation
------------

#. Using Pip:
    .. code-block:: shell

       $ pip install nwbwidgets

#. You can check for any missing packages by explicitly installing from the `requirements <https://github.com/NeurodataWithoutBorders/nwb-jupyter-widgets/blob/master/requirements.txt/>`_ file:
    .. code-block:: shell

        $ pip install -r requirements.txt

#. Cloning the github repo:
    .. code-block:: shell

        $ git clone https://github.com/NeurodataWithoutBorders/nwb-jupyter-widgets.git
        $ cd nwb-jupyter-widgets
        $ python setup.py install (or develop)

Usage
-----

.. code-block:: python

    from pynwb import NWBHDF5IO
    from nwbwidgets import nwb2widget

    io = NWBHDF5IO('path/to/file.nwb', mode='r')
    nwb = io.read()

    nwb2widget(nwb)

.. image:: https://drive.google.com/uc?export=download&id=1JtI2KtT8MielIMvvtgxRzFfBTdc41LiE