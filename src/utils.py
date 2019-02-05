"""
Utility functions.
"""

import numpy as np

def is_iterable(obj):
    try:
        list(obj)
        return True
    except TypeError:
        return False
    
def one_hot_encode(y):
    us = np.unique(y)
    us = sorted(us)
    dc = dict(zip(us, range(len(us))))
    
    out = np.zeros((y.shape[0], len(us)))
    for i, label in enumerate(y):
        out[i, dc[label]] = 1
        
    return out, dc
   