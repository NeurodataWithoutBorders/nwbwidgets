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
