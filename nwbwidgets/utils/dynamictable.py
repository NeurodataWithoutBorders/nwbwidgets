from pynwb.core import DynamicTable
import numpy as np


def infer_categorical_columns(dynamic_table: DynamicTable):

    categorical_cols = []
    for name in dynamic_table.colnames:
        if (len(dynamic_table[name].shape) == 1) and \
                (1 < len(np.unique(dynamic_table[name].data)) < 10):
            categorical_cols.append(name)
    return categorical_cols