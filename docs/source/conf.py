# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NWB Widgets'
copyright = '2022, NeurodataWithoutBorders'
author = 'NeurodataWithoutBorders'

# -- Support building doc without install --------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))

import sys
import os

cwd = os.getcwd()
project_root = os.path.dirname(os.path.dirname(cwd))

# Insert the project root dir as the first element in the PYTHONPATH.
# This lets us ensure that the source package is imported, and that its
# version is used.
sys.path.insert(0, os.path.join(project_root, '../../nwbwidgets'))



# -- Autodoc configuration -----------------------------------------------------

autoclass_content = 'both'
autodoc_docstring_signature = True
autodoc_member_order = 'bysource'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # Markdown support
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.autodoc",  # Includes documentation from docstrings in docs/api
    "sphinx.ext.autosummary",  # To-add
    "sphinx_toggleprompt",  # Used to control >>> behavior in the conversion gallery example doctests
    "sphinx_copybutton",  # Used to control the copy button behavior in the conversion gallery doctsts
    "sphinx.ext.intersphinx",  # Allows links to other sphinx project documentation sites
    "sphinx_search.extension",  # Allows for auto search function the documentation
    "sphinx.ext.viewcode",  # Shows source code in the documentation
    "sphinx.ext.extlinks",  # Allows to use shorter external links defined in the extlinks variable.
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# TODO - not working- Include arguments with typehints from classes
# ref: https://stackoverflow.com/questions/5599254/how-to-use-sphinxs-autodoc-to-document-a-classs-init-self-method
# ref: https://stackoverflow.com/questions/63957326/sphinx-does-not-show-class-instantiation-arguments-for-generic-classes-i-e-par
autoclass_content = 'both'
autodoc_typehints = 'signature'

intersphinx_mapping = {
    'pynwb': ('https://pynwb.readthedocs.io/en/stable/', None)
}