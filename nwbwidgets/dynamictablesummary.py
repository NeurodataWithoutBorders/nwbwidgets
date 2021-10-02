from ipywidgets import widgets, fixed, FloatProgress, Layout

from pynwb.misc import AnnotationSeries, Units, DecompositionSeries, DynamicTable

from .utils.dynamictable import infer_categorical_columns

from .utils.widgets import interactive_output

import plotly.graph_objects as go

import matplotlib.pyplot as plt


field_lay = widgets.Layout(
        max_height="40px", max_width="500px", min_height="30px", min_width="180px"
    )


class DynamicTableSummary(widgets.VBox):
    def __init__(self, table: DynamicTable):
        super().__init__()

        self.dynamic_table = table

        self.categorical_cols = infer_categorical_columns(self.dynamic_table)
        self.col_names = list(self.dynamic_table.colnames)

        num_entries = len(self.dynamic_table)
        num_columns = len(self.dynamic_table.colnames)
        num_categorical = len(self.categorical_cols)


        self.name_text = widgets.Label(f"Table name: {self.dynamic_table.name}\n", layout=field_lay)
        self.entries_text = widgets.Label(f"Number of entries: {num_entries}\n", layout=field_lay)
        self.col_text = widgets.Label(f"Number of columns: {num_columns} - (categorical: {num_categorical})",
                                      layout=field_lay)

        self.summary_text = widgets.VBox([self.name_text, self.entries_text, self.col_text])

        self.column_dropdown = widgets.Dropdown(
                options=[None] + self.col_names,
                description="inspect column",
                layout=Layout(max_width="400px"),
                style={"description_width": "initial"},
                disabled=False,
            )

        self.controls = dict(col_name=self.column_dropdown)

        out_fig = interactive_output(self.plot_hist_bar, self.controls)

        # self.column_dropdown.observe(self.dropdown_callback, "value")

        # self.fig = go.FigureWidget()
        # self.fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        # self.output = widgets.Output()

        bottom_panel = widgets.HBox([self.column_dropdown, out_fig])

        # with self.output as out:
        #     self.fig, self.ax = plt.subplots()

        self.children = [self.summary_text, bottom_panel]

    def dropdown_callback(self, value):
        if value['new'] in self.categorical_cols:
            func = self.ax.bar
        else:
            func = self.ax.hist
        with self.output:
            func(self.dynamic_table[value['new']])
            plt.show()

    def plot_hist_bar(self, col_name):
        fig, ax = plt.subplots(figsize=(10, 7))

        if col_name is not None:
            if col_name in self.categorical_cols:
                ax.bar(self.dynamic_table[col_name])
            else:
                ax.hist(self.dynamic_table[col_name])
        return ax
