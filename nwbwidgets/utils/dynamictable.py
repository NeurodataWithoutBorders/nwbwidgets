from pynwb.core import DynamicTable
import numpy as np


def infer_categorical_columns(dynamic_table: DynamicTable):

    categorical_cols = {}
    for name in dynamic_table.colnames:
        if len(dynamic_table[name].shape) == 1:
            try:  # TODO: fix this
                unique_vals = np.unique(dynamic_table[name].data)
                if 1 < len(unique_vals) <= (len(dynamic_table[name].data) / 2):
                    unique_vals = [x.decode() if isinstance(x, bytes) else x for x in unique_vals]  # handle h5py 3.0
                    categorical_cols[name] = unique_vals
            except:
                pass
    return categorical_cols


def group_and_sort(group_vals=None, group_select=None, order_vals=None, discard_rows=None, limit=None):
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
            raise ValueError('group_vals and order_vals cannot both be None')
        else:
            order = np.argsort(order_vals)

    # apply discard rows
    if discard_rows is not None:
        keep = np.logical_not(np.isin(order, discard_rows))
        order = order[keep]
        if group_inds is not None:
            group_inds = group_inds[keep]

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
