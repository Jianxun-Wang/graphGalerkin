from __future__ import print_function
import numpy as np
import scipy.linalg

from pyCaMOtk.check import is_type

def qrule_onedim_gauss_from_recursion(a, b, c, mu0):
    """
    Compute Gaussian quadrature rules from orthogonal polynomial recursion:

         p_j(x) = (a_j*x+b_j)*p_{j-1}(x) - c_j*p_{j-2}(x)     (j = 1, ..., N)

    and integral of weighting function over domain: mu0 = int_a^b w(x) dx

    Input arguments
    ---------------
    a, b, c : iterable of number, size = N
      See definition above
    mu0 : number
      See definition above

    Return values
    -------------
    w, z : iterable of number, size = N
      Quadrature weights, points

    Example
    -------
    >> # See qrule_onedim_gl(*), qrule_onedim_gll(*)
    """
    if not is_type(a, 'iter_of_number'):
        raise TypeError('a must be iterable of number')
    if not is_type(b, 'iter_of_number'):
        raise TypeError('b must be iterable of number')
    if not is_type(c, 'iter_of_number'):
        raise TypeError('c must be iterable of number')
    if not is_type(mu0, 'number'):
        raise TypeError('mu0 must be number')
    a, b, c = np.array(a, dtype=float), \
              np.array(b, dtype=float), \
              np.array(c, dtype=float)
    diag, offdiag = -b/a, np.sqrt(c[1:]/a[:-1]/a[1:])
    A = np.diag(diag, 0) + np.diag(offdiag, -1) + np.diag(offdiag, 1)
    z, eigvec = scipy.linalg.eigh(A)
    w = mu0*eigvec[0, :]**2
    return w, z

def qrule_onedim_gl(n):
    """
    Compute Gauss-Legrendre quadruatre rule, i.e., Gaussian quadrature
    corresponding to Legendre orthogonal polynomials.

    Legrendre polynomials:
     - weighting function = 1
     - interval = [-1, 1]
     - recursion:
          p_j(x) = (2j-1)/j*x*p_{j-1}(x) - (j-1)/j*p_{j-2}(x)

    Input arguments
    ---------------
    n : int
      Number of quadrature nodes

    Return values
    -------------
    w, z : iterable of number, size = N
      Quadrature weights, points

    Example
    -------
    >> w, z = qrule_onedim_gl(5)
    """
    if not is_type(n, 'int'):
        raise TypeError('n must be int')
    mu0 = 2.0 # int_[-1, 1] 1 * dx = 2
    j = np.arange(1, n+1, dtype=float)
    a, c = (2*j-1)/j, (j-1)/j
    b = np.zeros(n, dtype=float, order='F')
    return qrule_onedim_gauss_from_recursion(a, b, c, mu0)

def qrule_onedim_gll(n):
    """
    Compute Gauss-Legrendre-Lobatto quadruatre rule, i.e., Gaussian quadrature
    corresponding to Legendre orthogonal polynomials, including [-1, 1].

    Legrendre polynomials:
     - weighting function = 1
     - interval = [-1, 1]
     - recursion:
          p_j(x) = (2j-1)/j*x*p_{j-1}(x) - (j-1)/j*p_{j-2}(x)

    Adapted from Greg von Winckel (lglnodes.m)

    Input arguments
    ---------------
    n : int
      Number of quadrature nodes

    Return values
    -------------
    w, z : iterable of number, size = N
      Quadrature weights, points

    Example
    -------
    >> w, z = qrule_onedim_gl(5)
    """
    # Use Chebyshev-Gauss-Lobatto nodes as the first guess
    N = n-1
    z = np.cos(np.pi*np.arange(N+1)/float(N))[::-1]
    zold = 0.0*z

    # The Legendre Vandermonde Matrix
    P = np.zeros((N+1, N+1), dtype=float, order='F')

    # Compute P_(N) using the recursion relation
    # Compute its first and second derivatives and 
    # update z using the Newton-Raphson method.
    while np.max(np.abs(z-zold)) > 1.0e-14:
        zold = np.copy(z)
        P[:, 0], P[:, 1] = 1.0, z
        for k in range(2, N+1):
            P[:, k] = ( (2*k-1)*z*P[:, k-1] - (k-1)*P[:, k-2] )/float(k)
        z = zold - (z*P[:, N] - P[:, N-1]) /((N+1)*P[:, N])
    w = 2.0/(N*(N+1)*P[:, N]**2)
    return w, z

def qrule_onedim(n, qrule):
    """
    Wrapper for quadrature rules implemented

    Input arguments
    ---------------
    n : int
      Number of quadrature nodes
    qrule : str
      Name of quadrature rule

    Return values
    -------------
    w, z : iterable of number, size = N
      Quadrature weights, points

    Example
    -------
    >> w, z = qrule_onedim(5, 'gl')
    """
    if qrule == 'gl':
        return qrule_onedim_gl(n)
    elif qrule == 'gll':
        return qrule_onedim_gll(n)
    else:
        raise ValueError('qrule not implemented')

if __name__ == '__main__':
    print(qrule_onedim_gl(4))
    print(qrule_onedim_gll(5)) 
