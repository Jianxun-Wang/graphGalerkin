from __future__ import print_function
import numpy as np
from pyCaMOtk.qrule_onedim import qrule_onedim_gl, qrule_onedim_gll

def ndist_onedim_unif(nv):
    """
    Compute one-dimensional uniform nodal distribution over interval [-1, 1]
    
    Input arguments
    ---------------
    nv : int
      Number of nodes
    
    Return value
    ------------
    x : iterable of number, size = nv
      Nodal coordinates 
    
    Examples
    --------
    >> x = ndist_onedim_unif(3) # [-1, 0, 1]
    """
    dx = 2.0/float(nv-1)
    x = np.array([-1.0+k*dx for k in range(nv)], dtype=float, order='F')
    return x

def ndist_onedim_cheb(nv):
    """
    Compute one-dimensional Chebyshev nodal distribution over interval [-1, 1]
    
    Input arguments
    ---------------
    nv : int
      Number of nodes
    
    Return value
    ------------
    x : iterable of number, size = nv
      Nodal coordinates 
    
    Examples
    --------
    >> x = ndist_onedim_cheb(4) # [-0.92388, -0.38268, 0.38268, 0.92388]
    """
    x = np.zeros(nv, dtype=float, order='F')
    for k in range(nv):
        x[nv-k-1] = np.cos(float(2*k+1)*np.pi/float(2*nv))
    return x

def ndist_onedim_gl(nv):
    """
    Compute one-dimensional Gauss-Legendre nodal distribution
    over interval [-1, 1]
    
    Input arguments
    ---------------
    nv : int
      Number of nodes
    
    Return value
    ------------
    x : iterable of number, size = nv
      Nodal coordinates 
    
    Examples
    --------
    >> x = ndist_onedim_gl(2)   # [-0.57735, 0.57735]
    """
    _, x = qrule_onedim_gl(nv)
    return x

def ndist_onedim_gll(nv):
    """
    Compute one-dimensional Gauss-Legendre-Lobotto nodal distribution
    over interval [-1, 1]
    
    Input arguments
    ---------------
    nv : int
      Number of nodes
    
    Return value
    ------------
    x : iterable of number, size = nv
      Nodal coordinates 
    
    Examples
    --------
    >> x = ndist_onedim_gll(4)  # [-1, -0.4472136, 0.4472136, 1]
    """
    _, x = qrule_onedim_gll(nv)
    return x

def ndist_onedim(nv, ndist):
    """
    Compute one-dimensional nodal distribution over interval [-1, 1] (wrapper)
    
    Input arguments
    ---------------
    nv : int
      Number of nodes
    ndist : str
      Nodal distribution 
    
    Return value
    ------------
    x : iterable of number, size = nv
      Nodal coordinates 
    
    Examples
    --------
    >> x = ndist_onedim(3, 'unif') # [-1, 0, 1]
    >> x = ndist_onedim(4, 'cheb') # [-0.92388, -0.38268, 0.38268, 0.92388]
    >> x = ndist_onedim(2, 'gl')   # [-0.57735, 0.57735]
    >> x = ndist_onedim(4, 'gll')  # [-1, -0.4472136, 0.4472136, 1]
    """
    if ndist == 'unif':
        return ndist_onedim_unif(nv)
    elif ndist == 'cheb':
        return ndist_onedim_cheb(nv)
    elif ndist == 'gl':
        return ndist_onedim_gl(nv)
    elif ndist == 'gll':
        return ndist_onedim_gll(nv)
    else:
        raise ValueType('Nodal distribution not supported.')

if __name__ == '__main__':
    print(ndist_onedim(3, 'unif')) # [-1, 0, 1]
    print(ndist_onedim(4, 'cheb')) # [-0.92388, -0.38268, 0.38268, 0.92388]
    print(ndist_onedim(2, 'gl'))   # [-0.57735, 0.57735]
    print(ndist_onedim(4, 'gll'))  # [-1, -0.4472136, 0.4472136, 1]
