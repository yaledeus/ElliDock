import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_triangular_cython(np.ndarray[np.int64_t, ndim=2] edges):
    cdef int idx
    cdef long i, j, max_k_num = 0
    cdef list klist = []
    cdef np.ndarray[np.int64_t, ndim=1] source = edges[0]
    cdef np.ndarray[np.int64_t, ndim=1] target = edges[1]
    cdef np.ndarray[np.int64_t, ndim=1] edge_starts_with_i, edge_starts_with_j, i_idxs, j_idxs
    cdef np.ndarray[np.int64_t, ndim=2] k
    cdef np.ndarray[np.int64_t, ndim=3] klist_redundant
    for idx in range(edges.shape[1]):
        i = source[idx]
        j = target[idx]
        edge_starts_with_i = np.where(source == i)[0]
        edge_starts_with_j = np.where(source == j)[0]
        i_ends = target[edge_starts_with_i]
        j_ends = target[edge_starts_with_j]
        i_ends = np.tile(i_ends[:, None], (1, edge_starts_with_j.shape[0]))
        j_ends = np.tile(j_ends[:, None], (1, edge_starts_with_i.shape[0])).T
        i_idxs, j_idxs = np.nonzero(i_ends == j_ends)
        k = np.vstack([edge_starts_with_i[i_idxs], edge_starts_with_j[j_idxs]])
        if max_k_num < k.shape[1]:
            max_k_num = k.shape[1]
        klist.append(k)
    klist_redundant = -1 * np.ones((edges.shape[1], 2, max_k_num), dtype=np.int64)
    for idx in range(edges.shape[1]):
        k = klist[idx]
        klist_redundant[idx, :, :k.shape[1]] = k
    return klist_redundant
