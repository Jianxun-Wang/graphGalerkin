from __future__ import print_function
import numpy as np

def get_monom_coeffs_expnts_onedim(porder, nderiv=0):
    """
    Get the coefficients and exponents of the monomials basis up to degree
    PORDER and their derivatives.

    Monomial basis
     0th deriv:    {1, x, x**2, ..., x**PORDER}
     1st deriv:    {0, 1,  2*x, ..., PORDER*x**(PORDER-1)}
     2nd deriv:    {0, 0,    2, ..., PORDER*(PORDER-1)*x**(PORDER-2)}

    Input arguments
    ---------------
    porder : int
      Polynomial order
    nderiv : int
      Number of derivatives to return

    Return values
    -------------
    coeffs : ndarray (porder+1, nderiv+1)
      Coefficients of monomial basis and its derivatives
    expnts : ndarray (porder+1, nderiv+1)
      Exponents of monomial basis and its derivatives

    Example
    -------
    >> coeffs, expnts = get_monom_coeffs_expnts_onedim(porder=3, nderiv=2)
    # coeffs = [1, 0, 0;
                1, 1, 0;
                1, 2, 2;
                1, 3, 6]
    # expnts = [0, 0, 0; 
                1, 0, 0;
                2, 1, 0;
                3, 2, 1]
    """
    coeffs, expnts = np.ones((porder+1, nderiv+1), dtype=float, order='F'), \
                     np.zeros((porder+1, nderiv+1), dtype=int, order='F')
    for k in range(nderiv+1):
        expnts[k+1:, k] = range(1, porder+1-k)
        if k > 0:
            coeffs[:k, k] = 0
            coeffs[k:, k] = expnts[k:, k-1]*coeffs[k:, k-1]
    return coeffs, expnts

def eval_monom_onedim(coeffs, expnts, x):
    """
    Evaluate monomials and derivatives given coefficients, exponents, and
    evaluation points.

    Input arguments
    ---------------
    coeffs : ndarray (nb, nderiv+1)
      Coefficients of monomials; COEFFS[i, j] = coefficient of jth derivative
      of ith monomial
    expnts : ndarray (nb, nderiv+1)
      Exponents of monomials; EXPNTS[i, j] = exponent of jth derivative of
      ith monomial
    x : ndarray (nx,)
      Points at which to evaluate monomials

    Return values
    -------------
    Q : ndarray (nb, nderiv+1, nx)
      Monomials and derivatives evaluated at points in X
    """
    # Extract information from input
    nb, nderiv, nx = expnts.shape[0], expnts.shape[1]-1, x.size

    # Evaluate monomials
    Q = np.zeros((nb, nderiv+1, nx), dtype=float, order='F')
    for k in range(nderiv+1):
        for j in range(nb):
            if coeffs[j, k] == 0: continue
            Q[j, k, :] = coeffs[j, k]*x**expnts[j, k]
    return Q

def eval_poly_onedim_jacobi(alpha, beta, porder, x, nderiv=0):
    """
    Evaluate Jacobi polynomials using recurrence relation

    Input arguments
    ---------------
    alpha, beta : number
      Jacobi polynomial parameters
    porder : int
      Polynomial order
    x : ndarray (nx,)
      Points at which to evaluate monomials
    nderiv : int
      Number of derivatives to return

    Return values
    -------------
    Q : ndarray (nb, nderiv+1, nx)
      Jacobi polynomials and derivatives evaluated at points in X

    Notes
    -----
    No unit tests
    """
    # Extract information from input
    nx = x.size

    # Preallocate polynomials
    Q = np.zeros((porder+1, nderiv+1, nx), dtype=float, order='F')

    # Evaluated 0th derivative (values)
    Q[0, 0, :], Q[1, 0, :] = 1.0, (alpha+1)+0.5*(alpha+beta+2)*(x-1)
    for j in range(2, porder+1):
        n = j+1
        s, s0 = 2*n+alpha+beta, 2*n*(n+alpha+beta)
        Q0 = (s-1)*(s*(s-2)*x+alpha**2-beta**2)*Q[j-1, 0, :]
        Q1 = -2*(n+alpha-1)*(n+beta-1)*(2*n+alpha+beta)*Q[j-2, 0, :]
        Q[j, 0, :] = (1.0/s0)*(Q0+Q1)
    if nderiv == 0: return Q

    # Higher order derivatives not implemented
    raise ValueError('Not implemented')

def eval_poly_onedim_cheb(porder, x, nderiv=0):
    """
    Evaluate Chebyshev polynomials using recurrence relation

    Input arguments
    ---------------
    porder : int
      Polynomial order
    x : ndarray (nx,)
      Points at which to evaluate monomials
    nderiv : int
      Number of derivatives to return

    Return values
    -------------
    Q : ndarray (nb, nderiv+1, nx)
      Chebyshev polynomials and derivatives evaluated at points in X

    Notes
    -----
    No unit tests
    """
    # Extract information from input
    nx = x.size

    # Preallocate polynomials
    Q = np.zeros((porder+1, nderiv+1, nx), dtype=float, order='F')

    # Evaluated 0th derivative (values)
    Q[0, 0, :], Q[1, 0, :] = 1.0, x
    for j in range(2, porder+1):
        Q[j, 0, :] = 2*x*Q[j-1, 0, :]-Q[j-2, 0, :]
    if nderiv == 0: return Q

    # Higher order derivatives not implemented
    raise ValueError('Not implemented')

def eval_poly_onedim_leg(porder, x, nderiv=0):
    """
    Evaluate Legendre polynomials using recurrence relation

    Input arguments
    ---------------
    porder : int
      Polynomial order
    x : ndarray (nx,)
      Points at which to evaluate monomials
    nderiv : int
      Number of derivatives to return

    Return values
    -------------
    Q : ndarray (nb, nderiv+1, nx)
      Legendre polynomials and derivatives evaluated at points in X

    Notes
    -----
    No unit tests
    """
    # Extract information from input
    nx = x.size

    # Preallocate polynomials
    Q = np.zeros((porder+1, nderiv+1, nx), dtype=float, order='F')

    # Evaluated 0th derivative (values)
    Q[0, 0, :], Q[1, 0, :] = 1.0, x
    for j in range(2, porder+1):
        n = j+1
        Q[j, 0, :] = (1.0/float(n+1))*((2*n+1)*x*Q[j-1, 0, :]-n*Q[j-2, 0, :])
    if nderiv == 0: return Q

    # Higher order derivatives not implemented
    raise ValueError('Not implemented')

def eval_poly_nodal_onedim_lagr(xk, x, nderiv=0):
    """
    Evaluate Lagrangian polynomial basis functions and derivatives
    from nodal coordinates (XK) and evaluation points (X)
    
    Input arguments
    ---------------
    xk : ndarray, size = (nv,)
      Coordinates of nodes
    x : ndarray, size = (nx,)
      Coordinates at which to evaluate the basis functions
    nderiv : int
      Number of derivatives to return
    
    Output arguments
    ----------------
    Q : ndarray, size = (nv, nderiv+1, nx)
      Lagrange polynomials and derivatives evaluated at points in X
    """
    # Extract information from input
    nb, nx = xk.size, x.size

    # 0th derivative (values) of Lagrange polynomials
    Q = np.ones((nb, nderiv+1, nx), dtype=float, order='F')
    for k in range(nx):
        for j in range(nb):
            for i in range(nb):
                if i == j: continue
                Q[j, 0, k] *= (x[k]-xk[i])/(xk[j]-xk[i]);
    if nderiv == 0: return Q

    # 1st derivative of Lagrange polynomials
    Q[:, 1, :] = 0.0
    for k in range(nx):
        for j in range(nb):
            for i in range(nb):
                if i == j: continue

                tmp = 1.0;
                for m in range(nb):
                    if m==j or m==i: continue
                    tmp *= (x[k]-xk[m])/(xk[j]-xk[m])
                Q[j, 1, k] += tmp/(xk[j]-xk[i]);
    if nderiv == 1: return Q

    # Higher order derivatives not implemented
    raise ValueError('Not implemented')

def eval_poly_nodal_onedim_herm(xk, x, nderiv=0):
    """
    Evaluate Hermite polynomial basis functions and derivatives
    from nodal coordinates (XK) and evaluation points (X)
    
    Input arguments
    ---------------
    xk : ndarray, size = (nv,)
      Coordinates of nodes
    x : ndarray, size = (nx,)
      Coordinates at which to evaluate the basis functions
    nderiv : int
      Number of derivatives to return
    
    Output arguments
    ----------------
    Q : ndarray, size = (2*nv, nderiv+1, nx)
      Hermite polynomials and derivatives evaluated at points in X
    """
    # Extract information from input
    nv, nx = xk.size, x.size

    # Evaluate Lagrange polynomials
    L, Lk = eval_poly_nodal_onedim_lagr(xk, x, nderiv), \
            eval_poly_nodal_onedim_lagr(xk, xk, nderiv)

    # Preallocate polynomials
    Q = np.ones((2, nv, nderiv+1, nx), dtype=float, order='F')

    # 0th derivative (values) of Lagrange polynomials
    Q[:, :, 0, :] = 1.0
    for k in range(nx):
        for i in range(nv):
            Q[0, i, 0, k] = (1-2*Lk[i,1,i]*(x[k]-xk[i]))*L[i,0,k]**2
            Q[1, i, 0, k] = (x[k]-xk[i])*L[i,0,k]**2
    if nderiv == 0:
        Q = Q.reshape((2*nv, nderiv+1, nx), order='F')
        return Q

    # 1st derivative of Lagrange polynomials
    Q[:, :, 1, :] = 0.0
    for k in range(nx):
        for i in range(nv):
            Q[0, i, 1, k] = 2*L[i,0,k]*(L[i,1,k]*(1-2*Lk[i,1,i]*(x[k]-xk[i])) -
                                        L[i,0,k]*Lk[i,1,i])
            Q[1, i, 1, k] = L[i,0,k]*(L[i,0,k]+2*(x[k]-xk[i])*L[i,1,k])
    if nderiv == 1:
        Q = Q.reshape((2*nv, nderiv+1, nx), order='F')
        return Q

    # Higher order derivatives not implemented
    raise ValueError('Not implemented')

def eval_poly_nodal_onedim_vander(xk, x, nderiv=0, glob_cont=0):
    """
    Evaluate (generalized) nodal polynomial basis functions and derivatives
    from nodal coordinates (XK) and evaluation points (X)
    
    Input arguments
    ---------------
    xk : ndarray, size = (nv,)
      Coordinates of nodes
    x : ndarray, size = (nx,)
      Coordinates at which to evaluate the basis functions
    nderiv : int
      Number of derivatives to return
    glob_cont : int
      Global continuity of polynomial, i.e., number of derivatives to
      match at each node

    Output arguments
    ----------------
    Q : ndarray, size = (nb, nderiv+1, nx), where nb = nv*(glob_cont+1)
      Generalized nodal polynomials and derivatives evaluated at points in X

    Notes
    -----
    No unit tests
    """

    # Extract information from input
    nv, nx = xk.size, x.size
    nb = (glob_cont+1)*nv
    porder = nb-1

    # Evalute Vandermonde matrices
    coeffsH, expntsH = get_monom_coeffs_expnts_onedim(porder, nderiv=glob_cont)
    coeffsT, expntsT = get_monom_coeffs_expnts_onedim(porder, nderiv=nderiv)
    Vh, Vt = eval_monom_onedim(coeffsH, expntsH, xk), \
             eval_monom_onedim(coeffsT, expntsT, x)
    Vh = Vh.reshape((nb, (glob_cont+1)*nv), order='F')
    Vt = Vt.reshape((nb, (nderiv+1)*nx), order='F')

    # Solve linear system to obtain polynomial basis and derivatives
    Q = np.linalg.solve(Vh, Vt)
    Q = Q.reshape((nb, nderiv+1, nx), order='F')
    return Q

def eval_poly_onedim(porder, x, nderiv=0, wbasis='monom', **kwargs):
    """
    Evaluate polynomial basis in one dimension

    Input arguments
    ---------------
    porder : int
      Polynomial order
    x : ndarray (nx,)
      Points at which to evaluate monomials
    nderiv : int
      Number of derivatives to return
    wbasis : str
      Which polynomial basis

    Return values
    -------------
    Q : ndarray (nb, nderiv+1, nx)
      Polynomial and derivatives evaluated at points in X
    """
    if wbasis == 'monom':
        coeffs, expnts = get_monom_coeffs_expnts_onedim(porder, nderiv)
        return eval_monom_onedim(coeffs, expnts, x)
    elif wbasis == 'jacobi':
        alpha, beta = kwargs['alpha'], kwargs['beta']
        return eval_poly_onedim_jacobi(alpha, beta, porder, x, nderiv)
    elif wbasis == 'cheb':
        return eval_poly_onedim_cheb(porder, x, nderiv)
    elif wbasis == 'leg':
        return eval_poly_onedim_leg(porder, x, nderiv)
    else:
        raise ValueError('Not implemented')

def eval_poly_nodal_onedim(xk, x, nderiv=0, wbasis='lagr'):
    """
    Evaluate nodal polynomial basis in one dimension

    Input arguments
    ---------------
    xk : ndarray, size = (nv,)
      Coordinates of nodes
    x : ndarray (nx,)
      Points at which to evaluate monomials
    nderiv : int
      Number of derivatives to return
    wbasis : str
      Which nodal polynomial basis

    Return values
    -------------
    Q : ndarray (nb, nderiv+1, nx)
      Polynomials and derivatives evaluated at points in X
    """
    if wbasis == 'lagr':
        return eval_poly_nodal_onedim_lagr(xk, x, nderiv)
    elif wbasis in ['herm', 'herm1']:
        return eval_poly_nodal_onedim_herm(xk, x, nderiv)
    elif 'herm' in wbasis:
        glob_cont = int(wbasis.lstrip('herm'))
        return eval_poly_nodal_onedim_vander(xk, x, nderiv, glob_cont)

if __name__ == '__main__':
    coeffs, expnts = get_monom_coeffs_expnts_onedim(porder=3, nderiv=2)
    print(coeffs) 
    print(expnts) 

    xk = np.linspace(-1, 1, 2)
    x = np.linspace(-1, 1, 100)
    #Q = eval_poly_nodal_onedim_vander(xk, x, 2, 0)
    #print Q.shape
    Q = eval_poly_nodal_onedim_vander(xk, x, 2, 1)
    print(Q.shape) 
    #Q = eval_poly_nodal_onedim_vander(xk, x, 2, 2)
    #print Q.shape

    import matplotlib.pyplot as plt
    for k in range(Q.shape[0]):
        plt.plot(x, Q[k, 0, :], lw=2)
    plt.show()
