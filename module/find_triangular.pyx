import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_triangular_cython(np.ndarray[np.int64_t, ndim=2] edges):
    cdef int idx, k_idx
    cdef long i, j
    cdef list k, klist = []
    cdef np.ndarray[np.int64_t, ndim=1] source = edges[0]
    cdef np.ndarray[np.int64_t, ndim=1] target = edges[1]
    for idx in range(edges.shape[1]):
        k = []
        i = source[idx]
        j = target[idx]
        edge_starts_with_i = np.where(source == i)[0]
        edge_starts_with_j = np.where(source == j)[0]
        i_ends = target[edge_starts_with_i]
        j_ends = target[edge_starts_with_j]
        i_ends = np.tile(i_ends[:, np.newaxis], (1, edge_starts_with_j.shape[0]))
        j_ends = np.tile(j_ends[:, np.newaxis], (1, edge_starts_with_i.shape[0])).T
        i_idxs, j_idxs = np.nonzero(i_ends == j_ends)
        for k_idx in range(i_idxs.shape[0]):
            k.append([edge_starts_with_i[i_idxs[k_idx]], edge_starts_with_j[j_idxs[k_idx]]])
        klist.append(k)
    return klist
