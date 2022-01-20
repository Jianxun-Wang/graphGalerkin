import numpy as np
from pyCaMOtk.tens_core import *
import pdb

porder, ndim = 2, 3
M0 = np.array(range((porder+1)**ndim))
shp = [porder+1 for k in range(ndim)]
M = M0.reshape(shp, order='F')

# Faces
#print M[0, :]
#print M[:, 0]
#print M[-1, :]
#print M[:, -1]

# Edges
colon = range(porder+1)
N = tensprod_vector_unif([0, -1], ndim-1, flatten=True)
for d in range(ndim):
    idx_into_idx = [j for j in range(ndim) if j!=d]
    for k in range(N.shape[1]):
        # want: idx with colon in dth entry and idx0[:, k] in other entries
        idx_linidx = np.zeros(porder+1, order='F', dtype=int)
        idx_mltidx = np.zeros(ndim, order='F', dtype=int)
        for j in range(porder+1):
            idx_mltidx[d] = colon[j]
            idx_mltidx[idx_into_idx] = N[:, k]
            idx_linidx[j] = linidx_from_mltidx(shp, idx_mltidx)
        print M0[idx_linidx]
