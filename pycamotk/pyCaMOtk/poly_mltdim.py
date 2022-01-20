from __future__ import print_function
import numpy as np
from scipy.special import comb as nchoosek
try:
    from scipy.special import factorial
except:
    from scipy.misc import factorial
    

from pyCaMOtk.poly_onedim import eval_poly_nodal_onedim_lagr
from pyCaMOtk.tens_core import tensprod_scalar,      \
                               tensprod_scalar_unif, \
                               tensprod_vector_unif

def get_number_pderivs(ndim, nderiv):
    """
    Get the (cumulative) number of partial derivatives of a function in
    NDIM variables when taking up to and including NDERIV derivatives

                 npderivs = 1 + NDIM + NDIM**2 + ...

    Input arguments
    ---------------
    ndim : int
      Numbr of variables
    nderiv : int
      Number of derivatives

    Return value
    ------------
    npderivs : int
      Number of partial derivatives

    Examples
    --------
    >> get_number_pderivs(3, 2) # 1 + 3**1 + 3**2 = 13
    >> get_number_pderivs(2, 4) # 1 + 2**1 + 2**2 + 2**3 + 2**4 = 31
    >> get_number_pderivs(4, 4) # 1 + 4**1 + 4**2 + 4**3 + 4**4 = 341
    """
    if ndim == 1: return int(nderiv+1)
    return int(int(ndim**(nderiv+1)-1)/int(ndim-1))

def get_dim_polysp(ndim, porder, polysp='P'):
    """
    Dimension of polynomial space

    Polynomial spaces
       P  : {x1**i1 + x2**i2 + ... | i1+ i2+ ... <= PORDER}
       Q  : {x1**i1 + x2**i2 + ... | i1, i2, ... <= PORDER}
       PQ : {x1**i1 + x2**i2 + ... | i1+ i2+ ... idm1 <= PORDER, id <= PORDER}

    Input arguments
    ---------------
    ndim : int
      Numbr of variables
    porder : int
      Polynomial order
    polysp : str {'P', 'Q', 'PQ'}
      Polynomial space

    Return value
    ------------
    N : int
      Dimension of polynomial space

    Examples
    --------
    >> get_dim_polysp(2, 2, 'P')  #  6
    >> get_dim_polysp(2, 3, 'P')  # 10
    >> get_dim_polysp(2, 2, 'Q')  #  9
    >> get_dim_polysp(2, 3, 'Q')  # 16
    >> get_dim_polysp(3, 2, 'P')  # 10
    >> get_dim_polysp(3, 3, 'P')  # 20
    >> get_dim_polysp(3, 2, 'Q')  # 27
    >> get_dim_polysp(3, 3, 'Q')  # 64
    >> get_dim_polysp(3, 2, 'PQ') # 18
    >> get_dim_polysp(3, 3, 'PQ') # 40
    """
    if polysp == 'P':
        return int(nchoosek(porder+ndim, porder))
    elif polysp == 'Q':
        return (porder+1)**ndim
    elif polysp == 'PQ':
        dimP, dimQ = get_dim_polysp(ndim-1, porder, 'P'), \
                     get_dim_polysp(1, porder, 'Q')
        return dimP*dimQ
    else:
        raise ValueError('Polynomial space not supported')

def compute_deriv_monom_mltdim(coeffs0, expnts0, nderiv=0):
    """
    Compute derivatives of multidimensional monomials

    Input arguments
    ---------------
    coeffs0 : ndarray (nb,)
      Coefficients of monomials
    expnts0 : ndarray (ndim, nb)
      Exponents of monomials
    nderiv : int
      Number of derivatives to return

    Return values
    -------------
    coeffs : ndarray (nb, npderivs)
      Coefficients of monomials from coeffs0 and its partial derivatives
    expnts : ndarray (ndim, nb, npderivs)
      Exponents of monomials from coeffs0 and its partial derivatives
    """
    # Extract information from input
    ndim, nb = expnts0.shape
    npderivs = get_number_pderivs(ndim, nderiv)

    # Pre-allocate coefficients/exponents; set 0th derivative
    coeffs = np.ones((nb, npderivs), dtype=float, order='F')
    expnts = np.zeros((ndim, nb, npderivs), dtype=int, order='F')
    coeffs[:, 0], expnts[:, :, 0] = coeffs0, expnts0

    idx0, idx1 = 0, 1
    for k in range(1, nderiv+1):
        # Update index to extract entries corresponding to previous deriv order
        if k > 1:
            idx0, idx1 = get_number_pderivs(ndim, k-2), \
                         get_number_pderivs(ndim, k-1)

        # Extract coefficient/exponents for previous deriv order
        coeffsP = coeffs[:, idx0:idx1]
        expntsP = expnts[:, :, idx0:idx1]

        # Compute all partial derivatives
        for d in range(ndim):
            idx0, idx1 = idx1, idx1+ndim**(k-1)
            coeffs[:, idx0:idx1] = coeffsP*expntsP[d, :, :]
            expnts[:, :, idx0:idx1] = expntsP
            expnts[d, :, idx0:idx1] -= 1
    expnts[expnts<0] = 0
    return coeffs, expnts

def get_monom_coeffs_expnts_mltdim(ndim, porder, polysp='P', nderiv=0):
    """
    Get the coefficients and exponents of the monomials basis of a
    polynomial space and their derivatives.

    Input arguments
    ---------------
    ndim : int
      Numbr of variables
    porder : int
      Polynomial order
    polysp : str {'P', 'Q', 'PQ'}
      Polynomial space
    nderiv : int
      Number of derivatives to return

    Return value
    ------------
    coeffs : ndarray (N, npderivs)
      Coefficients of monomial basis of polynomial space and their
      derivatives, where N = get_dim_polysp(ndim, porder, polysp)
      and npderivs = get_number_pderivs(ndim, nderiv)
    expnts : ndarray (ndim, N, npderivs)
      Exponents of monomial basis of polynomial space and their derivatives

    Examples
    --------
    >> coeffs, expnts = get_monom_coeffs_expnts_mltdim(3, 2, 'P')
    >> coeffs, expnts = get_monom_coeffs_expnts_mltdim(3, 2, 'Q')
    >> coeffs, expnts = get_monom_coeffs_expnts_mltdim(3, 2, 'PQ')
    """
    if polysp == 'P':
        # Create tensor product of [0, ..., porder+1] as exponents
        # of monomials of a basis for Q
        v0 = range(0, porder+1)
        expnts00 = tensprod_vector_unif(v0, ndim, flatten=True)

        # Pre-allocate exponents for P
        N = get_dim_polysp(ndim, porder, 'P')
        expnts0 = np.zeros((ndim, N), order='F', dtype=int)

        # Restrict exponents to only those whose sum is at most PORDER
        cnt = -1
        for k in range(expnts00.shape[1]):
            if np.sum(expnts00[:, k]) <= porder:
                cnt += 1
                expnts0[:, cnt] = expnts00[:, k]
    elif polysp == 'Q':
        # Create tensor product of [0, ..., porder+1] as exponents
        # of monomials of a basis for Q
        v0 = range(0, porder+1)
        expnts0 = tensprod_vector_unif(v0, ndim, flatten=True)
        N = get_dim_polysp(ndim, porder, 'Q')
        #shp, expnts0 = tensprod_vector([v0 for d in range(ndim)])
        #expnts0 = create_ndarray_from_tens([shp[0], int(np.prod(shp[1:]))],
        #                                   expnts0)
    elif polysp == 'PQ':
        # Get dimensions of relevant polynomial spaces
        N, N0, N1 = get_dim_polysp(ndim, porder, 'PQ'),  \
                    get_dim_polysp(ndim-1, porder, 'P'), \
                    get_dim_polysp(1, porder, 'Q')

        # Pre-allocate exponents and form exponents of relevant P/Q spaces
        expnts0 = np.zeros((ndim, N), order='F', dtype=int)
        _, expntsP = get_monom_coeffs_expnts_mltdim(ndim-1, porder, 'P')
        _, expntsQ = get_monom_coeffs_expnts_mltdim(1, porder, 'Q')

        # Form exponents as a tensor product of the exponents corresponding
        # to the relevant P/Q spaces
        for k in range(N1):
            expnts0[:-1, k*N0:(k+1)*N0] = expntsP[:, :, 0]
            expnts0[-1, k*N0:(k+1)*N0] = expntsQ[0, k, 0]
    else:
        raise ValueError('Polynomial space not supported.')

    # Arbitrarily select coefficients of 1 for each monomial term in basis
    coeffs0 = np.ones(N, dtype=float, order='F')

    # Compute partial derivatives of monomial basis
    coeffs, expnts = compute_deriv_monom_mltdim(coeffs0, expnts0, nderiv)
    return coeffs, expnts

def eval_monom_mltdim(coeffs, expnts, x):
    """
    Evaluate monomials and derivatives given coefficients, exponents, and
    evaluation points.

    Input arguments
    ---------------
    coeffs : ndarray (nb, npderiv)
      Coefficients of monomials; COEFFS[i, j] = coefficient of jth derivative
      of ith monomial
    expnts : ndarray (ndim, nb, npderiv)
      Exponents of monomials; EXPNTS[:, i, j] = exponents of jth partial
      derivative of ith monomial
    x : ndarray (ndim, nx)
      Points at which to evaluate monomials

    Return values
    -------------
    Q : ndarray (nb, npderiv, nx)
      Monomials and partial derivatives evaluated at points in X
    """
    # Extract information from input
    nx = x.shape[1]
    ndim, nb, npderiv = expnts.shape

    # Evaluate monomials
    Q = np.zeros((nb, npderiv, nx), dtype=float, order='F')
    for k in range(npderiv):
        for j in range(nb):
            if coeffs[j, k] == 0: continue
            Q[j, k, :] = coeffs[j, k]
            for d in range(ndim):
                Q[j, k, :] *= x[d, :]**expnts[d, j, k]
    return Q

def eval_poly_nodal_mltdim_lagr_tensprod(xk0, x, nderiv=0):
    """
    Evaluate Lagrangian polynomial basis functions in multiple variables
    and derivatives from 1D nodal coordinates (XK0) and evaluation points
    (X) as a tensor product of 1D Lagrangian polynomials.

    Input arguments
    ---------------
    xk0 : ndarray (nv0,)
      Nodes in 1D (tensor product of XK0 defines all nodes)
    x : ndarray (ndim, nx)
      Points at which to evaluate polynomials
    nderiv : int
      Number of derivatives to return

    Return values
    -------------
    Q : ndarray (nb, npderiv, nx)
      Lagrange polynomials and partial derivatives evaluated at points in X
    """

    # Extract information from input
    nv0, porder = xk0.size, xk0.size-1
    ndim, nx = x.shape
    nb = nv0**ndim
    npderiv = get_number_pderivs(ndim, nderiv)

    # Evaluate one-dimensional functions
    x0, x0idx, x0inv = np.unique(x.flatten('F'),
                                 return_index=True,
                                 return_inverse=True)
    x0inv = x0inv.reshape((ndim, nx), order='F')
    Q0 = eval_poly_nodal_onedim_lagr(xk0, x0, nderiv)
 
    # Pre-allocate for basis functions and derivatives
    Q = np.zeros((nb, npderiv, nx), dtype=float, order='F')

    # Compute basis functions using tensor product (value)
    for k in range(nx):
        vlst = [Q0[:, 0, x0inv[s, k]] for s in range(ndim)]
        Q00 = tensprod_scalar(vlst)
        Q[:, 0, k] = np.array(Q00, dtype=float, order='F')
    if nderiv == 0: return Q

    # Compute basis functions using tensor product (1st derivative)
    for k in range(nx):
        for d in range(ndim):
            vlst = [Q0[:, 0, x0inv[s, k]] for s in range(ndim)]
            vlst[d] = Q0[:, 1, x0inv[d, k]]
            Q00 = tensprod_scalar(vlst, flatten=True)
            Q[:, 1+d, k] = np.array(Q00, dtype=float, order='F')
    if nderiv == 1: return Q

    # Higher order derivatives not implemented
    raise ValueError('Not implemented')

def eval_poly_nodal_mltdim_vander(xk, x, polysp='P', nderiv=0, glob_cont=0):
    """
    Evaluate (generalized) nodal polynomial basis functions in multiple
    variables and derivatives from nodal coordinates (XK) and evaluation
    points (X)
    
    Input arguments
    ---------------
    xk : ndarray, size = (ndim, nv)
      Coordinates of nodes
    x : ndarray, size = (ndim, nx)
      Coordinates at which to evaluate the basis functions
    polysp : str {'P', 'Q', 'PQ'}
      Polynomial space
    nderiv : int
      Number of derivatives to return
    glob_cont : int
      Global continuity of polynomial, i.e., number of derivatives to
      match at each node

    Output arguments
    ----------------
    Q : ndarray, size = (nb, npderiv, nx)
      Generalized nodal polynomials and derivatives evaluated at points in X,
      where nb = nv*get_number_pderivs(ndim, glob_cont) and
      npdriv = get_number_pderivs(ndim, nderiv)

    Notes
    -----
    No unit test
    """

    # Extract information from input
    nx = x.shape[1]
    ndim, nv = xk.shape
    npderiv, npderiv_glob_cont = get_number_pderivs(ndim, nderiv), \
                                 get_number_pderivs(ndim, glob_cont)
    nb = npderiv_glob_cont*nv

    # Determine the polynomial order that leads to a space of the
    # appropriate dimension
    dim_found = False
    for porder in range(100):
        if nb == get_dim_polysp(ndim, porder, polysp):
            dim_found = True
            break
    if not dim_found:
        raise ValueError('Inputs incompatible; number of constraints and ' \
                         'dimension of polynomial space are not equal')

    # Evalute Vandermonde matrices
    coeffsH, expntsH = get_monom_coeffs_expnts_mltdim(ndim, porder, polysp,
                                                      nderiv=glob_cont)
    coeffsT, expntsT = get_monom_coeffs_expnts_mltdim(ndim, porder, polysp,
                                                      nderiv=nderiv)
    Vh, Vt = eval_monom_mltdim(coeffsH, expntsH, xk), \
             eval_monom_mltdim(coeffsT, expntsT, x)
    Vh = Vh.reshape((nb, npderiv_glob_cont*nv), order='F')
    Vt = Vt.reshape((nb, npderiv*nx), order='F')

    # Solve linear system to obtain polynomial basis and derivatives
    Q = np.linalg.solve(Vh, Vt)
    Q = Q.reshape((nb, npderiv, nx), order='F')
    return Q

def compute_intg_monom_unit_simp(expnts):
    """
    Evaluate exact integral of monomial over unit simplex
    
         int_{unit simp in d-dims} x(0)^r(0)*...*x(d-1)^r(d-1) dx =
    
                   r(0)! ... r(d-1)! / (r(0)+...+r(d-1)+d)!
    
    Input arguments
    ---------------
    expnts : ndarray of int, shape (d,)
      Exponents for multinomial
    
    Return values
    -------------
    I : number
      Integral of multinomial over unit simplex
    """

    expnts = np.array(expnts)
    ndim = expnts.size
    s = int(np.sum(expnts))

    # Denominator: (sum(expnts)+ndim)!
    denom = int(1)# python2.7, the default is 1L, Han Gao changed it to int(1)
    denom *= factorial(s+ndim, exact=True)

    # Numerator: sum(expnts(k)!)
    num = int(1)# python2.7, the default is 1L, Han Gao changed it to int(1)
    for k in range(ndim):
        if expnts[k]>0:
            num *= factorial(expnts[k], exact=True)

    return float(num)/float(denom)

def compute_intg_monom_unit_hcube(expnts):
    """
    Evaluate exact integral of multinomial over unit hypercube [0, 1]^d
    
         int_{unit hcube in d-dims} x(0)^r(0)*...*x(d-1)^r(d-1) dx =
    
                       [(r(0)+1)*...*(r(d-1)+1)]^(-1)
    
    Input arguments
    ---------------
    expnts : ndarray of int, shape (d,)
      Exponents for multinomial
    
    Return values
    -------------
    I : number
      Integral of multinomial over unit hypercube
    """
    expnts = np.array(expnts)
    return 1.0/np.prod(expnts+1)

if __name__ == '__main__':

    print(get_dim_polysp(2, 2, 'P')) 
    print(get_dim_polysp(2, 3, 'P')) 
    print(get_dim_polysp(2, 2, 'Q')) 
    print(get_dim_polysp(2, 3, 'Q')) 
    print(get_dim_polysp(3, 2, 'P')) 
    print(get_dim_polysp(3, 3, 'P'))
    print(get_dim_polysp(3, 2, 'Q')) 
    print(get_dim_polysp(3, 3, 'Q')) 
    print(get_dim_polysp(3, 2, 'PQ')) 
    print(get_dim_polysp(3, 3, 'PQ')) 

    print(get_monom_expnts_mltdim(3, 2, 'P')) 
    print(get_monom_expnts_mltdim(3, 2, 'Q')) 
    print(get_monom_expnts_mltdim(3, 2, 'PQ')) 

    coeffs0 = np.ones(9, dtype=float, order='F')
    expnts0 = np.array([[0, 1, 2, 0, 1, 2, 0, 1, 2],
                        [0, 0, 0, 1, 1, 1, 2, 2, 2]],
                       dtype=int, order='F')
    coeffs, expnts = compute_deriv_monom_mltdim(coeffs0, expnts0, 2)
