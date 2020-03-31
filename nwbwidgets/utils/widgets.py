from ipywidgets import Output
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots, clear_output


def interactive_output(f, controls, process_controls=lambda x: x):
    """Connect widget controls to a function.

    This function does not generate a user interface for the widgets (unlike `interact`).
    This enables customisation of the widget user interface layout.
    The user interface layout must be defined and displayed manually.
    """

    out = Output()

    def observer(change):
        kwargs = process_controls({k: v.value for k, v in controls.items()})
        show_inline_matplotlib_plots()
        with out:
            clear_output(wait=True)
            f(**kwargs)
            show_inline_matplotlib_plots()
    for k, w in controls.items():
        w.observe(observer, 'value')
    show_inline_matplotlib_plots()
    observer(None)
    return out