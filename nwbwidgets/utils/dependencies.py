"""Utilities for managing dependencies."""

from functools import wraps
from importlib import import_module

from ipywidgets import widgets

###################################################################################################
###################################################################################################

def safe_import(*args):
    """Try to import a module, with a safety net for if the module is not available.

    Parameters
    ----------
    *args : str
        Module to import.

    Returns
    -------
    mod : module or False
        Requested module, if successfully imported, otherwise boolean (False).

    Notes
    -----
    The input, `*args`, can be either 1 or 2 strings, as pass through inputs to import_module:
    - To import a whole module, pass a single string, ex: ('matplotlib').
    - To import a specific package, pass two strings, ex: ('.pyplot', 'matplotlib')
    """

    try:
        mod = import_module(*args)
    except ImportError:
        mod = False

    return mod


def check_widget_dependency(dependency, name):
    """Decorator that checks as widget for required dependencies.

    Parameters
    ----------
    dependency : module or False
        Module, if successfully imported, or boolean (False) if not.
    name : str
        Full name of the module, to be printed in message.

    Returns
    -------
    wrap : callable
        The decorated function.

    Notes
    -----
    If the dependency is available this decorator passes through the widget.
    If the dependency is not available, a text widget is returned noting the missing module.
    """

    def wrap(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if not dependency:
                txt = "The " + name + " module is required for this widget."
                return widgets.Text(txt)
            else:
                return func(*args, **kwargs)
        return wrapped_func
    return wrap
