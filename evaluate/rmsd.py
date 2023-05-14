import numpy as np
from scipy.spatial.distance import cdist

import sys
sys.path.append('..')
from utils.geometry import kabsch_numpy


def compute_crmsd(X, Y, aligned=False):
    if not aligned:
        X_aligned, _, _ = kabsch_numpy(X, Y)
    else:
        X_aligned = X
    dist = np.sum((X_aligned - Y) ** 2, axis=-1)
    crmsd = np.sqrt(np.mean(dist))
    return float(crmsd)


def compute_irmsd(X, Y, seg, aligned=False, threshold=8.):
    X_re, X_li = X[seg == 0], X[seg == 1]
    Y_re, Y_li = Y[seg == 0], Y[seg == 1]
    dist = cdist(Y_re, Y_li)
    positive_re_idx, positive_li_idx = np.where(dist < threshold)
    positive_Y = np.concatenate((Y_re[positive_re_idx], Y_li[positive_li_idx]), axis=0)
    positive_X = np.concatenate((X_re[positive_re_idx], X_li[positive_li_idx]), axis=0)
    return float(compute_crmsd(positive_X, positive_Y, aligned))


