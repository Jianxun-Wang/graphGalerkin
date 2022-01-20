from __future__ import print_function
from copy import copy, deepcopy
import numpy as np

from pyCaMOtk.estr import estr
from pyCaMOtk.check import is_type, is_within_bounds

def get_strides(dims):
    """
    Get strides into a d-dimensional array with dimensions dims such that
    the (i0, i1, i2, ...) entry is the kth linear entry (Fortran order)

        k := i0*strd[0] + i1*strd[1] + ....

    Input arguments
    ---------------
    dims : iterable of int, size = d
      Shape of array

    Return value
    ------------
    strd : iterable of int, size = d+1
      Strides into d-dimensional array

    Example
    -------
    >> get_strides((2, 3, 4)) # (1, 2, 6, 24)
    """
    if not is_type(dims, 'iter_of_int'):
       raise TypeError('dims must be iterable of int')
    ndim = len(dims)
    strd = [1 for k in range(ndim+1)]
    for k in range(ndim):
        strd[k+1] = strd[k]*dims[k]
    return strd

def linidx_from_mltidx(dims, mltidx):
    """
    Get the linear index into a d-dimensional array with dimensions dims
    from an index into each dimension. Fortran ordering, negative indices
    allowed.

    Input arguments
    ---------------
    dims : iterable of int, size = d
      Shape of array
    mltidx : iterable of int, size = d
      mltidx[k] is the index into the kth dimension of the array

    Return value
    ------------
    linidx : int
      Linear index corresponding to multi-index mltidx into d-dimensional array

    Example
    -------
    >> linidx_from_mltidx((2, 3, 4), (1, 2, 2)) # 17
    """
    if not is_type(dims, 'iter_of_int'):
       raise TypeError('dims must be iterable of int')
    if not is_type(mltidx, 'iter_of_int'):
       raise TypeError('mltidx must be iterable of int')
    linidx, ndim, strd = 0, len(dims), get_strides(dims)
    if not is_within_bounds(dims, mltidx):
        ValueError('mltidx out of range')
    mltidx = [(k if k>-1 else k+dims[i]) for i, k in enumerate(mltidx)]
    for k in range(ndim):
        linidx += strd[k]*mltidx[k]
    return linidx

def mltidx_from_linidx(dims, linidx):
    """
    Get the index into each dimension of a d-dimensional array with
    dimensions dims from the linear index. Fortran ordering, negative
    indices allowed.

    Input arguments
    ---------------
    dims : iterable of int, size = d
      Shape of array
    linidx : int
      Linear index into array

    Return value
    ------------
    mltidx : iterable of int, size = d
      Multi-index corresponding to linear index linidx into d-dimensional array

    Example
    -------
    >> mltidx_from_linidx((2, 3, 4), 17) # (1, 2, 2)
    """
    if not is_type(dims, 'iter_of_int'):
       raise TypeError('dims must be iterable of int')
    if not is_type(linidx, 'int'):
       raise TypeError('linidx must be int')
    strd = get_strides(dims)
    if not is_within_bounds(strd[-1], linidx):
        ValueError('mltidx out of range')
    linidx = linidx if linidx>-1 else linidx+strd[-1]
    mltidx = [0 for k in range(len(dims))]
    for k in range(len(dims))[::-1]:
        mltidx[k] = int(linidx / strd[k]) # default linidx / strd[k]
        linidx -= mltidx[k]*strd[k]
    return mltidx
def tensprod_scalar(vlst, flatten=True):
    """
    Return the entries of the scalar tensor product of the arrays
    in vlst. If len(vlst) = ndim, then the shape of the scalar tensor
    product is (len(vlst[0]), ..., len(vlst[ndim-1]) and the kth entry
    is vlst[0][i0]*vlst[1][i1]*...

    Input arguments
    ---------------
    vlst : iterable of iterable of str or number
      vlst[k] is the kth one-dimensional array in the tensor product

    flatten : bool
      Whether to flatten output into vector

    Return value
    ------------
    vtp : ndarray of str/number, size = prod(len(vlst[k]))
      Entries of resulting tensor from tensor product

    Examples
    --------
    >> A = create_ndarray_of_estr([2], ['a', 'b'])
    >> B = create_ndarray_of_estr([3], ['c', 'd', 'e'])
    >> tensprod_scalar([A, B], flatten=True)
    # tens = ['a*c', 'b*c', 'a*d', 'b*d', 'a*e', 'b*e']
    >> tensprod_scalar([[1, 2], [3, 4, 5]], flatten=True)
    # tens = [3, 6, 4, 8, 5, 10]
    """
    if not is_type(vlst, 'iter_of_iter_of_str_number'):
       raise TypeError('vlst must be iterable of iterable of str/number')
    vtp = vlst[0]
    for k in range(1, len(vlst)):
        v_ = vlst[k]
        vtp = np.outer(vtp, v_)
        vtp = vtp.flatten('F')
    if not flatten:
        dims = [len(v_) for v_ in vlst]
        vtp = vtp.reshape(dims, order='F')
    return vtp

def tensprod_scalar_unif(v0, N, flatten=True, rstr2simp=False):
    """
    Return the entries of the scalar tensor product of the array v0
    with itself N times.

    Input arguments
    ---------------
    v0 : iterable of str or number
      One-dimensional array in tensor product

    N : int
      Number of dimensions, i.e., times to tensor product v0 with itself

    flatten : bool
      Whether to flatten output into vector

    rstr2simp : bool
      Whether to restrict tensor product to simplex

    Return value
    ------------
    vtp : ndarray of str/number
      Entries of resulting tensor from tensor (simplex) product

    Examples
    --------
    >> A = create_ndarray_of_estr([2], ['a', 'b'])
    >> tensprod_scalar_unif(A, 2, flatten=True, rstr2simp=False)
    # tens = ['a*a', 'a*b', 'a*b', 'b*b']
    >> tensprod_scalar_unif(A, 2, flatten=True, rstr2simp=True)
    # tens = ['a*a', 'a*b', 'a*b']
    """
    vlst = [v0 for k in range(N)]
    if not rstr2simp:
        return tensprod_scalar(vlst, flatten=flatten)
    v = tensprod_scalar(vlst, flatten=True)
    n = len(v0)
    dims = [n for k in range(N)]
    idx = [mltidx_from_linidx(dims, k) for k in range(n**N)]
    idx = [idx0 for idx0 in idx if sum(idx0)<n]
    idx = [linidx_from_mltidx(dims, idx0) for idx0 in idx]
    return v[idx]

def tensprod_vector(vlst, flatten=True):
    """
    Return the entries of the vector tensor product of the arrays
    in vlst. If len(vlst) = ndim, then the shape of the vector tensor
    product is (ndim, len(vlst[0]), ..., len(vlst[ndim-1]) and the entries
    are (vlst[0][i0], vlst[1][i1], ...)

    Input arguments
    ---------------
    vlst : iterable of iterable of str or number
      vlst[k] is the kth one-dimensional array in the tensor product

    flatten : bool
      Whether to flatten output into vector

    Return value
    ------------
    vtp : ndarray of str/number, size = len(vlst)*prod(len(vlst[k]))
      Entries of resulting tensor from tensor product

    Examples
    --------
    >> A = create_ndarray_of_estr([2], ['a', 'b'])
    >> B = create_ndarray_of_estr([3], ['c', 'd', 'e'])
    >> tens = tensprod_vector([A, B], flatten=True)
    # tens = [['a', 'b', 'a', 'b', 'a', 'b']
              ['c', 'c', 'd', 'd', 'e', 'e']]
    """
    if not is_type(vlst, 'iter_of_iter_of_str_number'):
       raise TypeError('vlst must be iterable of iterable of str/number')
    N = len(vlst)
    V = np.meshgrid(*vlst, indexing='ij')
    vtp = np.vstack([V[k].flatten('F')[None, :] for k in range(N)])
    if not flatten:
        dims = [len(vlst)] + [len(v_) for v_ in vlst]
        vtp = vtp.reshape(dims, order='F')
    return vtp

def tensprod_vector_unif(v0, N, flatten=True, rstr2simp=False):
    """
    Return the entries of the vector tensor product of the arrays
    in vlst.

    Input arguments
    ---------------
    v0 : iterable of str or number
      One-dimensional array in tensor product

    N : int
      Number of dimensions, i.e., times to tensor product v0 with itself

    flatten : bool
      Whether to flatten output into vector

    rstr2simp : bool
      Whether to restrict tensor product to simplex

    Return value
    ------------
    vtp : ndarray of str/number, size = len(vlst)*prod(len(vlst[k]))
      Entries of resulting tensor from tensor product

    Examples
    --------
    >> A = create_ndarray_of_estr([2], ['a', 'b'])
    >> tens = tensprod_vector_unif(A, 2, flatten=True, rstr2simp=False)
    # tens = [['a', 'b', 'a', 'b']
              ['a', 'a', 'b', 'b']]
    >> tens = tensprod_vector_unif(A, 2, flatten=True, rstr2simp=True)
    # tens = [['a', 'b', 'a']
              ['a', 'a', 'b']]
    """
    vlst = [v0 for k in range(N)]
    if not rstr2simp:
        return tensprod_vector(vlst, flatten=flatten)
    v = tensprod_vector(vlst, rstr2simp)
    n = len(v0)
    dims = [n for k in range(N)]
    idx = [mltidx_from_linidx(dims, k) for k in range(n**N)]
    idx = [idx0 for idx0 in idx if sum(idx0)<n]
    idx = [linidx_from_mltidx(dims, idx0) for idx0 in idx]
    return v[:, idx]

if __name__ == '__main__':
    from pyCaMOtk.estr import estr, create_ndarray_of_estr
    A = create_ndarray_of_estr([2], ['a', 'b'])
    B = create_ndarray_of_estr([3], ['c', 'd', 'e'])
    print(tensprod_scalar([A, B], flatten=True))
    print(tensprod_scalar([[1, 2], [3, 4, 5]], flatten=True))
    print(tensprod_scalar_unif(A, 2, flatten=True, rstr2simp=False))
    print(tensprod_scalar_unif(A, 2, flatten=True, rstr2simp=True))
    print(tensprod_vector([A, B], flatten=True))
    print(tensprod_vector_unif(A, 2, flatten=True, rstr2simp=False))
    print(tensprod_vector_unif(A, 2, flatten=True, rstr2simp=True))

#def simpprod_scalar(vlst, as_ndarray=True):
#    """
#    """
#
#    # Ensure all entries in vlst have same size (required from simpprod)
#    len0 = len(vlst[0])
#    for k in range(1, len(vlst)):
#        if len(vlst[k]) != len0:
#            raise ValueError('All entries of vlst must be the same size')
#
#    # Compute tensor product of vlst
#    shp, vtp = tensprod_scalar(vlst)
#
#    # Restrict tensor product to simplex
#    vsp, sz = [], int(np.prod(shp))
#    for k in range(sz):
#        dims = mltidx_from_linidx(shp, k)
#        if sum(dims) >= len0: continue
#        vsp.append(vtp[k])
#    return (len(vsp),), vsp
#
#def simpprod_vector(vlst, as_ndarray=True):
#    """
#    """
#
#    # Ensure all entries in vlst have same size (required from simpprod)
#    len0 = len(vlst[0])
#    for k in range(1, len(vlst)):
#        if len(vlst[k]) != len0:
#            raise ValueError('All entries of vlst must be the same size')
#
#    # Compute tensor product of vlst
#    shp, vtp = tensprod_vector(vlst)
#
#    # Restrict tensor product to simplex
#    vsp, ndim, sz = [], shp[0], int(np.prod(shp))
#    for k in range(sz):
#        dims = mltidx_from_linidx(shp, k)
#        if sum(dims[1:]) >= len0: continue
#        vsp.append(vtp[k])
#    return (ndim, len(vsp)/ndim), vsp
#
#def tensprod_scalar(vlst, as_ndarray=True):
#    """
#    Return the shape and entries of the scalar tensor product of the arrays
#    in vlst. If len(vlst) = ndim, then the shape of the scalar tensor
#    product is (len(vlst[0]), ..., len(vlst[ndim-1]) and the kth entry
#    is vlst[0][i0]*vlst[1][i1]*...
#
#    Input arguments
#    ---------------
#    vlst : iterable of iterable of str or number
#      vlst[k] is the kth one-dimensional array in the tensor product
#
#    Return value
#    ------------
#    dims : iterable of int, size = d
#      Dimensions of resulting tensor from tensor product
#    vtp : iterable of str/number, size = prod(len(vlst[k]))
#      Entries of resulting tensor from tensor product
#
#    Examples
#    --------
#    >> tensprod_scalar([['a', 'b'], ['c', 'd', 'e']])
#    # dims = (2, 3)
#    # tens = ['a*c', 'b*c', 'a*d', 'b*d', 'a*e', 'b*e']
#    >> tensprod_scalar([[1, 'b'], ['c', 2, 'e']])
#    # dims = (2, 3)
#    # tens = ['c', 'b*c', 2, '2*b', 'e', 'b*e']
#    """
#    if not is_type(vlst, 'iter_of_iter_of_str_number'):
#       raise TypeError('vlst must be iterable of iterable of str/number')
#    vlstl = deepcopy(vlst)
#    for j in range(len(vlst)):
#        for k in range(len(vlst[j])):
#            vlstl[j][k] = estr(vlstl[j][k]) if is_type(vlstl[j][k], 'str') \
#                                            else vlstl[j][k]
#    vtp = []
#    ndim, dims = len(vlstl), [len(v) for v in vlstl]
#    for k in range(0, int(prod(dims))):
#        mltidx = mltidx_from_linidx(dims, k)
#        vtpk = copy(vlstl[0][mltidx[0]])
#        for j in range(1, ndim):
#            vtpk *= vlstl[j][mltidx[j]]
#        vtp.append(vtpk)
#    return dims, vtp
#
#def tensprod_vector(vlst, as_ndarray=True):
#    """
#    Return the shape and entries of the vector tensor product of the arrays
#    in vlst. If len(vlst) = ndim, then the shape of the vector tensor
#    product is (ndim, len(vlst[0]), ..., len(vlst[ndim-1]) and the entries
#    are (vlst[0][i0], vlst[1][i1], ...)
#
#    Input arguments
#    ---------------
#    vlst : iterable of iterable of str or number
#      vlst[k] is the kth one-dimensional array in the tensor product
#
#    Return value
#    ------------
#    dims : iterable of int, size = d+1
#      Dimensions of resulting tensor from tensor product
#    vtp : iterable of str/number, size = len(vlst)*prod(len(vlst[k]))
#      Entries of resulting tensor from tensor product
#
#    Examples
#    --------
#    >> dims, tens = tensprod_vector([['a', 'b'], ['c', 'd', 'e']])
#    # dims = (2, 2, 3)
#    # tens = ['a', 'c', 'b', 'c', 'a', 'd', 'b', 'd', 'a', 'e', 'b', 'e']
#    >> tensprod_vector([[1, 'b'], ['c', 2, 'e']])
#    # dims = (2, 2, 3)
#    # tens = [1, 'c', 'b', 'c', 1, 2, 'b', 2, 1, 'e', 'b', 'e']
#    """
#    if not is_type(vlst, 'iter_of_iter_of_str_number'):
#       raise TypeError('vlst must be iterable of iterable of str/number')
#    vtp = []
#    ndim, dims = len(vlst), [len(v) for v in vlst]
#    for k in range(0, int(prod(dims))):
#        mltidx = mltidx_from_linidx(dims, k)
#        for j in range(ndim):
#            vtp.append(copy(vlst[j][mltidx[j]]))
#    dims = [ndim] + dims
#    return dims, vtp
#
#def create_ndarray_from_tens(dims, A):
#    """
#    Create ndarray from list of entries and array shape.
#
#    Input arguments
#    ---------------
#    dims : iterable of int
#      Dimensions of array
#    A : iterable of str/number, size = prod(dims)
#      Entries of array
#
#    Return value
#    ------------
#    B : ndarray, shape = dims
#      Array consisting of the entries in A
#    """
#    typ = type(A[0])
#    B = array(A, order='F', dtype=typ).reshape(dims, order='F')
#    return B
