from pynwb.core import DynamicTable
import numpy as np


def infer_categorical_columns(dynamic_table: DynamicTable):

    categorical_cols = {}
    for name in dynamic_table.colnames:
        if len(dynamic_table[name].shape) == 1:
            try:  # TODO: fix this
                unique_vals = np.unique(dynamic_table[name].data)
                if 1 < len(unique_vals) <= (len(dynamic_table[name].data) / 2):
                    categorical_cols[name] = unique_vals
            except:
                pass
    return categorical_cols


def group_and_sort(group_vals=None, order_vals=None, limit=None, window=None):

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
            raise ValueError('group_vals and order_vals cannot both be None')
        else:
            order = np.argsort(order_vals)

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

    # apply window
    if window is not None:
        order = order[window[0]:window[1]]
        if group_inds is not None:
            group_inds = group_inds[window[0]:window[1]]

    return order, group_inds, labels
