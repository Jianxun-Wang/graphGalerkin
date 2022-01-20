import numpy as np

from pyCaMOtk.qrule_onedim import qrule_onedim
from pyCaMOtk.tens_core import tensprod_scalar_unif, tensprod_vector_unif

def qrule_mltdim_hcube_from_onedim(ndim, w0, z0):
    """
    Quadrature rule for hypercube as tensor product of one-dimensional
    quadrature rule
    
    Input arguments
    ---------------
    ndim : int
      Number of spatial dimensions
    w0 : ndarray, size = (N0,)
      Quadrature weights of one-dimensional quadrature rule
    z0 : ndarray, size = (N0,)
      Quadrature nodes of one-dimensional quadrature rule

    Return values
    -------------
    w : ndarray, size = N = N0**ndim
      Quadrature weights
    z : ndarray, size = (ndim, N)
      Quadrature points
    
    Example
    -------
    >> w0, z0 = qrule_ondim(4, 'gl')
    >> w, z = qrule_mltdim_hcube_from_onedim(ndim, w0, z0)
    """
    w = tensprod_scalar_unif(w0, ndim, flatten=True)
    z = tensprod_vector_unif(z0, ndim, flatten=True)
    return w, z
    #_, w = tensprod_scalar([w0 for k in range(ndim)])
    #shp0, z = tensprod_vector([z0 for k in range(ndim)])
    #shp = (shp0[0], int(np.prod(shp0[1:])))
    #w = np.array(w, dtype=float, order='F')
    #z = np.array(z, dtype=float, order='F').reshape(shp, order='F')
    #return w, z

def qrule_mltdim_simp_from_hcube(w_hcube0, z_hcube0):
    """
    Quadrature rule for simplex [0, 1]**ndim from a quadrature rule
    for the hypercube [-1, 1]**ndim
    
    Input arguments
    ---------------
    w_hcube0 : ndarray, size = (N,)
      Quadrature weights of hypercube quadrature rule
    z_hcube0 : ndarray, size = (ndim, N)
      Quadrature nodes of hypercube quadrature rule

    Return values
    -------------
    w : ndarray, size = (N,)
      Quadrature weights
    z : ndarray, size = (ndim, N)
      Quadrature points
    
    Example
    -------
    >> w0, z0 = qrule_ondim(4, 'gl')
    >> w_hcube, z_hcube = qrule_mltdim_hcube_from_onedim(ndim, w0, z0)
    >> w, z = qrule_mltdim_simp_from_hcube(w_hcube, z_hcube)
    """

    # Transform quadrature over [-1, 1]^d hcube to [0, 1]^d hcube
    ndim, nq = z_hcube0.shape
    w_hcube = w_hcube0 * 0.5**ndim
    z_hcube = 0.5*(z_hcube0+1)

    # Initialize simplex quadrature
    w_simp = w_hcube.copy('F')
    z_simp = z_hcube.copy('F')
    
    # Map nodes to simplex and adjust weights
    for k in range(nq):
        for j in range(ndim):
            w_simp[k] *= (1-z_hcube[j, k])**(ndim-j-1)
        for j in range(ndim)[::-1]:
            for i in range(j):
                z_simp[j, k] *= (1-z_simp[i, k])
    return w_simp, z_simp

def qrule_mltdim_hcube(ndim, nq0, qrule0):
    """
    Quadrature rule for hypercube [-1, 1]^ndim
    
    Input arguments
    ---------------
    ndim : int
      Number of spatial dimensions
    nq0 : int
      Number of reference quadrature nodes
    qrule0 : str
      Name of reference quadrature rule

    Return values
    -------------
    w : ndarray, size = N = nq0**ndim
      Quadrature weights
    z : ndarray, size = (ndim, N)
      Quadrature points
    
    Example
    -------
    >> w, z = qrule_mltdim_hcube(2, 5, 'gl')
    """
    w0, z0 = qrule_onedim(nq0, qrule0)
    return qrule_mltdim_hcube_from_onedim(ndim, w0, z0)
    #_, w = tensprod_scalar([w0 for k in range(ndim)])
    #shp, z = tensprod_vector([z0 for k in range(ndim)])
    #w, z = np.array(w, order='F'), np.array(z, order='F')
    #z = z.reshape([shp[0], int(np.prod(shp[1:]))], order='F')
    #return w, z

def qrule_mltdim_simp(ndim, nq0, qrule0):
    """
    Quadrature rule for simplex [0, 1]^ndim
    
    Input arguments
    ---------------
    ndim : int
      Number of spatial dimensions
    nq0 : int
      Number of reference quadrature nodes
    qrule0 : str
      Name of reference quadrature rule

    Return values
    -------------
    w : ndarray, size = N
      Quadrature weights
    z : ndarray, size = (ndim, N)
      Quadrature points
    
    Example
    -------
    >> w, z = qrule_mltdim_simp(2, 5, 'gl')
    """
    w_hcube, z_hcube = qrule_mltdim_hcube(ndim, nq0, qrule0)
    w, z = qrule_mltdim_simp_from_hcube(w_hcube, z_hcube)
    return w, z

def qrule_mltdim(ndim, nq0, qrule0, etype):
    """
    Wrapper for multidimensional quadrature rules
    
    Input arguments
    ---------------
    ndim : int
      Number of spatial dimensions
    nq0 : int
      Number of reference quadrature nodes
    qrule0 : str
      Name of reference quadrature rule
    etype : str
      Element geometry

    Return values
    -------------
    w : ndarray, size = N
      Quadrature weights
    z : ndarray, size = (ndim, N)
      Quadrature points
    
    Example
    -------
    >> w, z = qrule_mltdim(2, 5, 'gl', 'simp')
    >> w, z = qrule_mltdim(2, 5, 'gl', 'hcube')
    """
    if etype == 'hcube':
        return qrule_mltdim_hcube(ndim, nq0, qrule0)
    elif etype == 'simp':
        return qrule_mltdim_simp(ndim, nq0, qrule0)
    else:
        raise ValueError('Element geometry not supported.')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    w, z = qrule_mltdim_simp(2, 5, 'gl')
    plt.plot(z[0, :], z[1, :], 'bo')
    plt.show()
