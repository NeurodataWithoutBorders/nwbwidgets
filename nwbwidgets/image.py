import matplotlib.pyplot as plt
import ipywidgets as widgets
import itkwidgets
import numpy as np

def show_index_series(index_series):
    indexed_timeseries = index_series.indexed_timeseries

    output = widgets.Output()
    def show_image(index=0):
        fig, ax = plt.subplots(subplot_kw={'xticks': [], 'yticks': []})
        ax.imshow(indexed_timeseries.data[index][:,:], cmap='gray')
        output.clear_output(wait=True)
        with output:
            plt.show(fig)
    def on_index_change(change):
        show_image(change.new)
    slider = widgets.IntSlider(value=0,
            min=0,
            max=indexed_timeseries.data.shape[0] - 1,
            orientation='horizontal')
    slider.observe(on_index_change, names='value')
    show_image()

    image_widgets = widgets.VBox([output, slider])

    info = []
    for key in ('description', 'comments', 'unit', 'resolution', 'conversion'):
        info.append(widgets.Text(value=repr(getattr(index_series, key)), description=key, disabled=True))
    text_widgets = widgets.VBox(info)

    return widgets.HBox([text_widgets, image_widgets])


def show_grayscale_image(grayscale_image):
    return itkwidgets.view(np.array(grayscale_image.data),
            ui_collapsed=True,
            cmap='Grayscale')
