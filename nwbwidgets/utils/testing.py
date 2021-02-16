import numpy as np


def dicts_exact_equal(first, second):
    """Return whether two dicts of arrays are exactly equal"""
    if first.keys() != second.keys():
        return False
    return all(np.array_equal(first[key], second[key]) for key in first)


def dicts_approximate_equal(first, second):
    """Return whether two dicts of arrays are roughly equal"""
    if first.keys() != second.keys():
        return False
    return all(np.allclose(first[key], second[key]) for key in first)
