import numpy as np

#from pyCaMOtk.multinom_core import simpprod_vector
from pyCaMOtk.ndist_onedim import ndist_onedim
from pyCaMOtk.tens_core import tensprod_vector_unif

def ndist_mltdim_hcube(ndim, nv0, ndist0):
    """
    Compute nodal distribution over hypercube [-1, 1]^d; same number of nodes
    and nodal distribution in all dimensions.

    Input arguments
    ---------------
    ndim : int
      Number of dimensions
    nv0 : int
      Number of nodes in each dimension
    ndist0 : str
      Nodal distribution to use in all dimensions

    Output argument
    ---------------
    x : ndarray, shape (ndim, nv0**ndim)
      Nodal coordinates
    """
    x0 = ndist_onedim(nv0, ndist0)
    x = tensprod_vector_unif(x0, ndim, flatten=True)
    #shp0, x = tensprod_vector([x0 for k in range(ndim)])
    #shp = (shp0[0], int(np.prod(shp0[1:])))
    #x = np.array(x, dtype=float, order='F').reshape(shp, order='F')
    return x

def ndist_mltdim_simp(ndim, nv0, ndist0):
    """
    Compute nodal distribution over simplex [0, 1]^d; same number of nodes and
    nodal distribution in all dimensions (restricted to simplex).

    Input arguments
    ---------------
    ndim : int
      Number of dimensions
    nv0 : int
      Number of nodes in each dimension
    ndist0 : str
      Nodal distribution to use in all dimensions

    Output argument
    ---------------
    x : ndarray, shape (ndim, nv)
      Nodal coordinates
    """
    xk0 = 0.5*(1+ndist_onedim(nv0, ndist0))
    #xk = simpprod_vector(ndim, xk0)
    xk = tensprod_vector_unif(xk0, ndim, flatten=True, rstr2simp=True)
    return xk 

def ndist_mltdim(ndim, nv0, ndist0, etype):
    """
    Compute nodal distribution over ndim-dimensional regular geometry; same
    number of nodes and nodal distribution in all dimensions.

    Input arguments
    ---------------
    ndim : int
      Number of dimensions
    nv0 : int
      Number of nodes in each dimension
    ndist0 : str
      Nodal distribution to use in all dimensions
    etype : str
      Element type
    
    Output argument
    ---------------
    x : ndarray, shape (ndim, nv)
      Nodal coordinates
    """
    if etype == 'hcube':
        return ndist_mltdim_hcube(ndim, nv0, ndist0)
    elif etype == 'simp':
        return ndist_mltdim_simp(ndim, nv0, ndist0)
    else:
        raise ValueError('Element geometry not supported.')
