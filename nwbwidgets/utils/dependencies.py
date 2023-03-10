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


def check_widget_dependencies(dependencies):
    """Decorator that checks a widget for required dependencies.

    Parameters
    ----------
    dependencies : dict
        Each key should be the name of the module. This is printed if module is missing.
        Each value should be the the module, if successfully imported, or boolean (False) if not.

    Returns
    -------
    wrap : callable
        The decorated function.

    Notes
    -----
    If the dependencies are available this decorator passes through the widget.
    If the dependencies are not available, a text widget is returned noting the missing module(s).
    """

    def wrap(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            missing = [name for name, dependency in dependencies.items() if not dependency]
            if not missing:
                return func(*args, **kwargs)
            else:
                txt = "This widget requires the following extra dependencies: {}"
                return widgets.Label(txt.format(", ".join(missing)))

        return wrapped_func

    return wrap
