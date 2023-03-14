import numpy as np
from scipy.spatial.distance import cdist

import sys
sys.path.append('..')
from data.geometry import kabsch_numpy


def compute_crmsd(X, Y):
    X_aligned, _, _ = kabsch_numpy(X, Y)
    dist = np.sum((X_aligned - Y) ** 2, axis=-1)
    crmsd = np.sqrt(dist.sum() / dist.shape[0])
    return float(crmsd)


def compute_irmsd(X, Y, seg, threshold=8):
    X_aligned, _, _ = kabsch_numpy(X, Y)
    X_ab, X_ag = X_aligned[seg == 0], X_aligned[seg == 1]
    Y_ab, Y_ag = Y[seg == 0], Y[seg == 1]
    abag_dist = cdist(Y_ab, Y_ag)
    ab_idx, ag_idx = np.where(abag_dist < threshold)
    ab_dist = np.sum((X_ab[ab_idx] - Y_ab[ab_idx]) ** 2, axis=-1)
    ag_dist = np.sum((X_ag[ag_idx] - Y_ag[ag_idx]) ** 2, axis=-1)
    dist = np.vstack([ab_dist, ag_dist])
    irmsd = np.sqrt(dist.sum() / dist.shape[1])
    return float(irmsd)


