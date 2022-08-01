import numbers
from typing import Iterable

import numpy as np
import pandas as pd

from pynwb.core import DynamicTable


def extract_data_from_intervals(dynamic_table: DynamicTable):
    # solution: https://stackoverflow.com/a/50297200/11483674
    # data = [x if x == x else 'NaN' for x in dynamic_table[:]]
    data = [x for x in dynamic_table[:] if x == x]
    classes = pd.unique(data).tolist()
    try:
        data = [float(i) for i in data]
        classes = [float(i) for i in classes]
        classes.sort(key=float)
    except:
        pass
    return data, classes


def infer_categorical_columns(dynamic_table: DynamicTable, region: Iterable = None):
    """
    Parameters
    ----------
    dynamic_table: DynamicTable
    region: Iterable
        row indices to select

    Returns
    -------
    categorical_cols: dict()
        keys: as columns that are categorical, values as the unique values
    """
    categorical_cols = {}
    region = (
        np.array(region, dtype="uint")
        if region is not None
        else np.arange(len(dynamic_table), dtype="uint")
    )
    for name in dynamic_table.colnames:
        if len(dynamic_table[name].shape) == 1:
            try:
                if isinstance(
                    dynamic_table[name].data[0], (str, numbers.Number, bytes)
                ):
                    column_data = [dynamic_table[name].data[i] for i in region]
                elif hasattr(dynamic_table[name].data[0], "name"):
                    column_data = [dynamic_table[name].data[i].name for i in region]
                else:
                    continue
                unique_vals = np.unique(column_data)
                if not isinstance(unique_vals[0], str) and all(np.isnan(unique_vals)):
                    continue
                if 1 < len(unique_vals) <= (len(column_data) / 2):
                    unique_vals = [
                        x.decode() if isinstance(x, bytes) else x for x in unique_vals
                    ]  # handle h5py 3.0
                    categorical_cols[name] = np.array(column_data)
            except Exception as e:
                print(e)
    return categorical_cols


def infer_columns_to_plot(dynamic_table: DynamicTable):
    """Infer which columns can be plotted in summary widgets

    Parameters
    ----------
    dynamic_table : DynamicTable

    Returns
    -------
    column_names_to_plot: list
        Columns that can be plotted with the dynamic table summary
    """

    column_names_to_plot = []

    df = dynamic_table.to_dataframe()
    categorical_columns = []
    for name in dynamic_table.colnames:
        # if name not in categorical_cols.keys():  # categorical columns can always be plotted
        value = df[name].values[0]

        if isinstance(value, (int, float, np.integer)):
            column_names_to_plot.append(name)
        elif isinstance(value, (str, bool, bytes)):
            column_names_to_plot.append(name)
            categorical_columns.append(name)

    return column_names_to_plot, categorical_columns


def group_and_sort(
    group_vals=None, group_select=None, order_vals=None, keep_rows=None, limit=None
):
    """
    Logical flow:
    0) Apply discard_rows - throw out any listed rows
    1) Apply group select - Return only values that are within this group
    2) Apply order - If group is provided, items are sorted first by group and then by order
    3) Apply limit - Applied per group
    4) Apply window - Return only items [int] through [int]. Useful for plotting

    Parameters
    ----------
    group_vals: array-like of str
        ['a', 'b', 'b', 'a', 'a', 'b', 'b', 'c']
    group_select: array-like of str
        ['a', 'b']
    order_vals: array-like of ints
        [0, 3, 4, 1, 2, 5, 6, 7]
    limit: int

    Returns
    -------

    """

    if group_vals is not None:
        if order_vals is None:
            order = np.argsort(group_vals)
        else:
            order = np.lexsort([order_vals, group_vals])
        labels, group_inds = np.unique(group_vals[order], return_inverse=True)
    else:
        labels, group_inds = None, None
        if order_vals is None:
            # cannot give order because we don't know how many units there are
            raise ValueError("group_vals and order_vals cannot both be None")
        else:
            order = np.argsort(order_vals)

    # apply discard rows
    if keep_rows is not None:
        order = order[keep_rows]
        if group_inds is not None:
            group_inds = group_inds[keep_rows]

    # apply discard NaN categories
    try:
        if any(np.isnan(labels)):
            nan_labs = np.isnan(labels)
            keep = ~np.isin(group_inds, np.where(nan_labs)[0])
            group_inds = group_inds[keep]
            order = order[keep]
            labels = labels[~np.isnan(labels)]
    except TypeError:  # if labels are strings
        pass

    if labels is not None:
        # remove groups that are missing
        labels = labels[np.isin(range(len(labels)), group_inds)]
        _, group_inds = np.unique(group_inds, return_inverse=True)

        # apply discard groups (but keep labels)
        if group_select is not None:
            keep = np.isin(labels[group_inds], group_select)
            group_inds = group_inds[keep]
            order = order[keep]

        labels = np.array([x.decode() if isinstance(x, bytes) else x for x in labels])

    # apply limit
    inds = list()
    if limit is not None:
        if group_inds is not None:
            for i in range(len(labels)):
                inds.extend(np.where(group_inds == i)[0][:limit])
            order = order[inds]
            group_inds = group_inds[inds]
        else:
            order = order[:limit]

    return order, group_inds, labels
