from ipywidgets import widgets, Layout
import plotly.express as px

from pynwb.misc import DynamicTable

from .utils.dynamictable import infer_columns_to_plot
from .utils.widgets import interactive_output


field_lay = widgets.Layout(
        max_height="40px", max_width="500px", min_height="30px", min_width="180px"
    )


class DynamicTableSummaryWidget(widgets.VBox):
    def __init__(self, table: DynamicTable):
        super().__init__()
        self.dynamic_table = table

        self.col_names, self.categorical_cols = infer_columns_to_plot(self.dynamic_table)

        num_entries = len(self.dynamic_table)
        num_columns = len(self.dynamic_table.colnames)
        num_columns_to_plot = len(self.col_names)
        num_categorical = len(self.categorical_cols)

        self.name_text = widgets.Label(f"Table name: {self.dynamic_table.name}\n", layout=field_lay)
        self.entries_text = widgets.Label(f"Number of entries: {num_entries}\n", layout=field_lay)
        self.col_text = widgets.Label(f"Number of columns: {num_columns} - real (r): {num_columns - num_categorical}, "
                                      f"categorical (c): {num_categorical}",
                                      layout=field_lay)
        self.col_plot_text = widgets.Label(f"Number of inspectable columns: {num_columns_to_plot}")

        self.summary_text = widgets.VBox(
            [self.name_text, self.entries_text, self.col_text, self.col_plot_text])

        self.col_names_display = {}
        for col in self.col_names:
            if col in self.categorical_cols:
                self.col_names_display[f"(c) {col}"] = col
            else:
                self.col_names_display[f"(r) {col}"] = col

        self.column_dropdown = widgets.SelectMultiple(
                options=list(self.col_names_display),
                description="Inspect columns",
                layout=Layout(max_width="400px"),
                style={"description_width": "initial"},
                disabled=False,
                tooltip="Select columns to inspect. You can select at most 1 categorical and 3 real columns."
            )
        self.column_dropdown.observe(self.max_selection)

        self.nbins = widgets.IntText(
            10, min=0, description="# bins", layout=Layout(max_width="400px")
        )
        self.nbins.layout.visibility = "hidden"

        self.show_labels = widgets.Checkbox(value=True, description="show labels")

        self.plot_controls = widgets.HBox(
            [self.column_dropdown, self.nbins, self.show_labels])

        self.controls = dict(
            col_names_display=self.column_dropdown, nbins=self.nbins, show_labels=self.show_labels)

        out_fig = interactive_output(self.plot_hist_bar, self.controls)
        bottom_panel = widgets.VBox([self.plot_controls, out_fig])

        self.children = [self.summary_text, bottom_panel]

    def max_selection(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            if len(self.column_dropdown.value) > 4:
                print("Maximum number of selected items reached! "
                      "You can select at most 4 items (1 categorical and 3 real)")
                self.column_dropdown.value = ()

    def reset_dropdown(self):
        self.column_dropdown.value = ()

    def plot_hist_bar(self, col_names_display, nbins, show_labels):
        fig = None
        df = self.dynamic_table.to_dataframe()
        # remove 's' from dynamic table name
        if show_labels:
            entry_names = [f"{self.dynamic_table.name[:-1]} {index}" for index in df.index.values]
        else:
            entry_names = None

        if len(col_names_display) > 0:
            if len(col_names_display) == 1:
                # display either 1d histogram (r) or barplot (c)
                col_name_display = col_names_display[0]
                col_name = self.col_names_display[col_name_display]
                if col_name in self.categorical_cols:
                    self.nbins.layout.visibility = "hidden"
                    fig = px.histogram(df, x=col_name, histfunc='count')
                else:
                    self.nbins.layout.visibility = "visible"
                    fig = px.histogram(
                        df, x=col_name, marginal="violin", nbins=nbins)
                fig.show()
            elif len(col_names_display) == 2:
                # display either scatterplot (2x r) or multiple histograms (1r + 1c)
                real_cols = [r for r in col_names_display if "(r)" in r]
                num_real = len(real_cols)
                if num_real == 2:
                    self.nbins.layout.visibility = "hidden"
                    col_name_0 = col_name = self.col_names_display[col_names_display[0]]
                    col_name_1 = col_name = self.col_names_display[col_names_display[1]]
                    fig = px.scatter(df, x=col_name_0, y=col_name_1, marginal_x="violin", marginal_y="violin",
                                     text=entry_names)
                    fig.show()
                elif num_real == 1:
                    self.nbins.layout.visibility = "visible"
                    col_real = self.col_names_display[real_cols[0]]
                    col_cat_name = [c for c in col_names_display if c not in real_cols][0]
                    col_cat = self.col_names_display[col_cat_name]
                    fig = px.histogram(
                        df, x=col_real, color=col_cat, marginal="violin", nbins=nbins)
                    fig.show()
                else:
                    print("Select at least one real variable")
                    self.reset_dropdown()
            elif len(col_names_display) == 3:
                # display either 3d scatterplot (3x r) or colored 2d scatterplot (2r + 1c)
                real_cols = [r for r in col_names_display if "(r)" in r]
                num_real = len(real_cols)
                if num_real == 3:
                    self.nbins.layout.visibility = "hidden"
                    col_name_0 = col_name = self.col_names_display[col_names_display[0]]
                    col_name_1 = col_name = self.col_names_display[col_names_display[1]]
                    col_name_2 = col_name = self.col_names_display[col_names_display[2]]
                    fig = px.scatter_3d(
                        df, x=col_name_0, y=col_name_1, z=col_name_2,
                        text=entry_names)
                    fig.show()
                elif num_real == 2:
                    self.nbins.layout.visibility = "hidden"
                    col_real_0 = col_name = self.col_names_display[real_cols[0]]
                    col_real_1 = col_name = self.col_names_display[real_cols[1]]
                    col_cat_name = [c for c in col_names_display if c not in real_cols][0]
                    col_cat = self.col_names_display[col_cat_name]
                    df[col_cat] = df[col_cat].astype("str")
                    fig = px.scatter(df, x=col_real_0,
                                     y=col_real_1, color=col_cat,
                                     text=entry_names)
                    fig.show()
                else:
                    print("Select at most one categorical variable")
                    self.reset_dropdown()
            elif len(col_names_display) == 4:
                real_cols = [r for r in col_names_display if "(r)" in r]
                num_real = len(real_cols)

                if num_real == 3:
                    self.nbins.layout.visibility = "hidden"
                    col_name_0 = col_name = self.col_names_display[real_cols[0]]
                    col_name_1 = col_name = self.col_names_display[real_cols[1]]
                    col_name_2 = col_name = self.col_names_display[real_cols[2]]
                    col_cat_name = [c for c in col_names_display if c not in real_cols][0]
                    col_cat = self.col_names_display[col_cat_name]
                    df[col_cat] = df[col_cat].astype("str")
                    fig = px.scatter_3d(df, x=col_name_0,
                                        y=col_name_1, z=col_name_2, color=col_cat,
                                        text=entry_names)
                    fig.show()
                else:
                    print("Select 3 real and one categorical variables")
                    self.reset_dropdown()
        return fig
