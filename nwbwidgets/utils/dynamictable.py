from pynwb.core import DynamicTable
import numpy as np


def infer_categorical_columns(dynamic_table: DynamicTable):

    categorical_cols = {}
    for name in dynamic_table.colnames:
        if len(dynamic_table[name].shape) == 1:
            unique_vals = np.unique(dynamic_table[name].data)
            if 1 < len(unique_vals) < 10:
                categorical_cols[name] = unique_vals
    return categorical_cols


def get_group_inds_and_order(dt: DynamicTable, rows=None, group_by=None, order_by=None):

    if rows is None:
        rows = np.arange(len(dt))

    if group_by is not None:
        group_vals = np.array(dt[group_by][rows.tolist()])
        if order_by is None:
            order = np.argsort(group_vals)
        else:
            order_vals = dt[order_by][rows.tolist()]
            order = np.lexsort([order_vals, group_vals])
        labels, group_inds = np.unique(group_vals[order], return_inverse=True)
    else:
        labels, group_inds = None, None
        if order_by is None:
            order = np.arange(len(rows))
        else:
            order_vals = dt[order_by][rows.tolist()]
            order = np.argsort(order_vals)

    return order, group_inds, labels
