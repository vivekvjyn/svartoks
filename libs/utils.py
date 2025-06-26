import math
import numpy as np

def pad(ts, max_length):
    if len(ts) > max_length:
        return ts[:max_length]
    else:
        return np.pad(ts, (0, max_length - len(ts)), mode='constant', constant_values=-2)

def create_mask(lengths, max_length):
    masks = [[False] * length + [True] * (max_length - length) for length in lengths]
    return np.array(masks)

def replace_nans(ts):
    ts = np.array(ts, dtype=np.float32)
    ts[np.isnan(ts)] = -4800
    ts = ts / 2400.0
    return ts.tolist()
